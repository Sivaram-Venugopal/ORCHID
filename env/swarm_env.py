import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import List, Optional
from env.physics import propagate, circular_state, rtn_to_eci, eci_to_rtn, tsiolkovsky, RE

# ── Constants ──────────────────────────────────────────────────────────────────
DRY_MASS        = 500.0
INIT_FUEL       = 50.0
MAX_DV          = 0.015
COOLDOWN_S      = 600.0
DT_STEP         = 30.0
BOX_KM          = 5.0
K_THREATS       = 7

# Debris size classes
DEBRIS_LARGE    = 2   # 5-10cm, tracked
DEBRIS_MEDIUM   = 1   # 2-5cm, partially tracked
DEBRIS_SMALL    = 0   # 1-2cm, untracked

# Safety distances per size class
SAFE_KM = {DEBRIS_LARGE: 1.0, DEBRIS_MEDIUM: 0.5, DEBRIS_SMALL: 0.3}
CONJ_KM = {DEBRIS_LARGE: 0.2, DEBRIS_MEDIUM: 0.1, DEBRIS_SMALL: 0.05}

# Collision penalties per size class
COLLISION_PENALTY = {DEBRIS_LARGE: -100.0, DEBRIS_MEDIUM: -50.0, DEBRIS_SMALL: -20.0}
SAT_COLLISION_PENALTY = -100.0

@dataclass
class SwarmConfig:
    n_controlled:   int   = 10
    n_partners:     int   = 5
    n_debris_large: int   = 7
    n_debris_medium:int   = 7
    n_debris_small: int   = 6
    alt_km:         float = 450.0
    inc_deg:        float = 28.5
    episode_steps:  int   = 120
    dt_step:        float = DT_STEP

@dataclass 
class AgentState:
    eci:        np.ndarray
    fuel:       float = INIT_FUEL
    last_burn:  float = -9999.0
    alive:      bool  = True

@dataclass
class DebrisObject:
    eci:        np.ndarray
    size_class: int    # DEBRIS_LARGE, DEBRIS_MEDIUM, DEBRIS_SMALL
    tracked:    bool   # False for small debris

class SwarmOrbitEnv(gym.Env):
    """
    ORCHID v2: 10 controlled satellites + 5 partner satellites + 20 debris objects.
    Decentralised MARL with shared policy, diverse partners, classified debris.
    """
    metadata = {"render_modes": []}

    def __init__(self, config: SwarmConfig = None, partner_policies: list = None):
        super().__init__()
        self.cfg = config or SwarmConfig()
        self.partner_policies = partner_policies or self._default_partners()

        # Observation: own(8) + K_THREATS*9 + time(1) + box(1) = 73
        obs_dim = 8 + K_THREATS * 9 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.n_controlled, obs_dim),
            dtype=np.float32
        )
        # Action: 3D thrust per controlled satellite
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.cfg.n_controlled, 3),
            dtype=np.float32
        )

        self.agents   = []
        self.partners = []
        self.debris   = []
        self._step    = 0
        self._time    = 0.0

    def _default_partners(self):
        from partners.policy_pool import get_random_training_policy
        return [get_random_training_policy() for _ in range(5)]

    def _spawn_satellite(self, base_state, offset_km=None):
        state = base_state.copy()
        if offset_km is not None:
            offset = rtn_to_eci(offset_km, base_state)
            state[:3] += offset
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.cfg
        rng = np.random.default_rng(seed)

        base = circular_state(cfg.alt_km, cfg.inc_deg)

        # Spawn 10 controlled satellites in a loose formation
        self.agents = []
        for i in range(cfg.n_controlled):
            offset = rng.uniform(-2.0, 2.0, 3)
            offset[0] = abs(offset[0]) + 0.3  # keep radial separation
            state = self._spawn_satellite(base, offset)
            self.agents.append(AgentState(eci=state, fuel=INIT_FUEL))

        # Spawn 5 partner satellites with slight orbital variations
        self.partners = []
        for i in range(cfg.n_partners):
            alt_offset = rng.uniform(-10, 10)
            partner_base = circular_state(cfg.alt_km + alt_offset, cfg.inc_deg)
            offset = rng.uniform(-3.0, 3.0, 3)
            state = self._spawn_satellite(partner_base, offset)
            self.partners.append(AgentState(eci=state, fuel=INIT_FUEL))
            if i < len(self.partner_policies):
                if hasattr(self.partner_policies[i], "reset"):
                    self.partner_policies[i].reset()

        # Spawn debris by size class
        self.debris = []
        debris_spec = [
            (cfg.n_debris_large,  DEBRIS_LARGE,  True,  4.0),
            (cfg.n_debris_medium, DEBRIS_MEDIUM, True,  3.0),
            (cfg.n_debris_small,  DEBRIS_SMALL,  False, 2.0),
        ]
        for count, size_class, tracked, spread in debris_spec:
            for _ in range(count):
                alt_offset = rng.uniform(-15, 15)
                debris_base = circular_state(cfg.alt_km + alt_offset, cfg.inc_deg)
                offset = rng.uniform(-spread, spread, 3)
                state = self._spawn_satellite(debris_base, offset)
                # Add small random velocity perturbation for small debris
                if size_class == DEBRIS_SMALL:
                    state[3:] += rng.uniform(-0.001, 0.001, 3)
                self.debris.append(DebrisObject(eci=state, size_class=size_class, tracked=tracked))

        self._step = 0
        self._time = 0.0
        return self._get_obs_all(), {}

    def _get_obs_single(self, agent: AgentState) -> np.ndarray:
        cfg = self.cfg
        pos = agent.eci[:3]
        vel = agent.eci[3:]
        fuel_norm = agent.fuel / INIT_FUEL
        cooldown_norm = min((self._time - agent.last_burn) / COOLDOWN_S, 1.0)

        # Collect all threats: other agents + partners + debris
        threats = []
        for other in self.agents:
            if other is not agent and other.alive:
                rel_p = other.eci[:3] - pos
                rel_v = other.eci[3:] - vel
                dist = np.linalg.norm(rel_p)
                threats.append((dist, rel_p, rel_v, 1.0, 1.0))  # size=large, tracked

        for partner in self.partners:
            rel_p = partner.eci[:3] - pos
            rel_v = partner.eci[3:] - vel
            dist = np.linalg.norm(rel_p)
            threats.append((dist, rel_p, rel_v, 1.0, 1.0))

        for d in self.debris:
            rel_p = d.eci[:3] - pos
            rel_v = d.eci[3:] - vel
            dist = np.linalg.norm(rel_p)
            size_norm = d.size_class / 2.0
            tracked_norm = 1.0 if d.tracked else 0.0
            threats.append((dist, rel_p, rel_v, size_norm, tracked_norm))

        # Sort by distance, take K_THREATS closest
        threats.sort(key=lambda x: x[0])
        threat_obs = np.zeros(K_THREATS * 9)
        for j, (dist, rel_p, rel_v, size_norm, tracked_norm) in enumerate(threats[:K_THREATS]):
            threat_obs[j*9:(j+1)*9] = [
                *rel_p / BOX_KM,
                *rel_v / 0.01,
                dist / BOX_KM,
                size_norm,
                tracked_norm
            ]

        r0 = RE + cfg.alt_km
        in_box = float(abs(np.linalg.norm(pos) - r0) < BOX_KM)
        time_norm = self._step / cfg.episode_steps

        obs = np.concatenate([
            pos / 7000.0, vel / 8.0,
            [fuel_norm, cooldown_norm],
            threat_obs,
            [in_box, time_norm]
        ]).astype(np.float32)
        return obs

    def _get_obs_all(self):
        return np.array([self._get_obs_single(a) for a in self.agents], dtype=np.float32)

    def step(self, actions):
        cfg = self.cfg
        actions = np.clip(actions, -1.0, 1.0)

        # Apply actions to controlled agents
        fuel_used_total = 0.0
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                continue
            action = actions[i]
            dv_mag = np.linalg.norm(action) * MAX_DV
            cooldown_ok = (self._time - agent.last_burn) >= COOLDOWN_S

            if dv_mag > 1e-6 and cooldown_ok and agent.fuel > 0:
                total_mass = DRY_MASS + agent.fuel
                dm = tsiolkovsky(dv_mag, total_mass)
                dm = min(dm, agent.fuel)
                agent.fuel -= dm
                fuel_used_total += dm
                dv_eci = rtn_to_eci(action * MAX_DV, agent.eci)
                agent.eci[3:] += dv_eci
                agent.last_burn = self._time

        # Apply partner policies
        for i, partner in enumerate(self.partners):
            if i < len(self.partner_policies):
                obs = self._get_obs_single(AgentState(eci=partner.eci, fuel=partner.fuel))
                try:
                    p_action = np.clip(self.partner_policies[i].act(obs, None), -1.0, 1.0)
                except TypeError:
                    p_action = np.clip(self.partner_policies[i].act(obs), -1.0, 1.0)
                p_dv_mag = np.linalg.norm(p_action) * MAX_DV
                if p_dv_mag > 1e-6 and partner.fuel > 0:
                    p_mass = DRY_MASS + partner.fuel
                    p_dm = tsiolkovsky(p_dv_mag, p_mass)
                    partner.fuel -= min(p_dm, partner.fuel)
                    p_dv_eci = rtn_to_eci(p_action * MAX_DV, partner.eci)
                    partner.eci[3:] += p_dv_eci

        # Propagate all objects
        for agent in self.agents:
            agent.eci = propagate(agent.eci, cfg.dt_step)
        for partner in self.partners:
            partner.eci = propagate(partner.eci, cfg.dt_step)
        for d in self.debris:
            d.eci = propagate(d.eci, cfg.dt_step)

        self._step += 1
        self._time += cfg.dt_step

        # Compute rewards and check collisions
        rewards = np.zeros(cfg.n_controlled)
        collisions = 0
        agents_alive = sum(1 for a in self.agents if a.alive)

        for i, agent in enumerate(self.agents):
            if not agent.alive:
                continue

            r_safe = 1.0
            collision = False

            # Check vs other agents
            for j, other in enumerate(self.agents):
                if i == j or not other.alive:
                    continue
                dist = np.linalg.norm(agent.eci[:3] - other.eci[:3])
                if dist < CONJ_KM[DEBRIS_LARGE]:
                    collision = True
                    collisions += 1
                else:
                    r_safe = min(r_safe, np.clip(
                        (dist - CONJ_KM[DEBRIS_LARGE]) / (SAFE_KM[DEBRIS_LARGE] - CONJ_KM[DEBRIS_LARGE]), 0, 1))

            # Check vs partners
            for partner in self.partners:
                dist = np.linalg.norm(agent.eci[:3] - partner.eci[:3])
                if dist < CONJ_KM[DEBRIS_LARGE]:
                    collision = True
                    collisions += 1
                else:
                    r_safe = min(r_safe, np.clip(
                        (dist - CONJ_KM[DEBRIS_LARGE]) / (SAFE_KM[DEBRIS_LARGE] - CONJ_KM[DEBRIS_LARGE]), 0, 1))

            # Check vs debris
            for d in self.debris:
                dist = np.linalg.norm(agent.eci[:3] - d.eci[:3])
                conj = CONJ_KM[d.size_class]
                safe = SAFE_KM[d.size_class]
                if dist < conj:
                    collision = True
                    collisions += 1
                    rewards[i] += COLLISION_PENALTY[d.size_class]
                else:
                    r_safe = min(r_safe, np.clip((dist - conj) / (safe - conj), 0, 1))

            if collision:
                agent.alive = False
                rewards[i] += SAT_COLLISION_PENALTY
            else:
                r_fuel = agent.fuel / INIT_FUEL
                r0 = RE + cfg.alt_km
                r_box = float(abs(np.linalg.norm(agent.eci[:3]) - r0) < BOX_KM)
                r_team = sum(1 for a in self.agents if a.alive) / cfg.n_controlled
                rewards[i] = 0.5*r_safe + 0.3*r_fuel + 0.1*r_box + 0.1*r_team

        terminated = all(not a.alive for a in self.agents)
        truncated  = self._step >= cfg.episode_steps

        info = {
            'collisions':       collisions,
            'agents_alive':     sum(1 for a in self.agents if a.alive),
            'fuel_remaining':   [a.fuel for a in self.agents],
            'fuel_used_total':  fuel_used_total,
            'mean_reward':      float(np.mean(rewards)),
            'debris_counts': {
                'large':  cfg.n_debris_large,
                'medium': cfg.n_debris_medium,
                'small':  cfg.n_debris_small,
            }
        }
        return self._get_obs_all(), rewards, terminated, truncated, info

