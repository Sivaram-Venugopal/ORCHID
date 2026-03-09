import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from env.physics import propagate, circular_state, rtn_to_eci, tsiolkovsky, RE
import json, os

# ── Constants ──────────────────────────────────────────────────────────────────
DRY_MASS          = 500.0
INIT_FUEL         = 50.0
MAX_DV            = 0.015
COOLDOWN_S        = 600.0
DT_STEP           = 30.0
BOX_KM            = 5.0
K_THREATS         = 10   # increased for denser environment

DEBRIS_LARGE      = 2    # 5-10cm tracked
DEBRIS_MEDIUM     = 1    # 2-5cm partial
DEBRIS_SMALL      = 0    # 1-2cm untracked

SAFE_KM           = {DEBRIS_LARGE: 1.0, DEBRIS_MEDIUM: 0.5, DEBRIS_SMALL: 0.3}
CONJ_KM           = {DEBRIS_LARGE: 0.2, DEBRIS_MEDIUM: 0.1, DEBRIS_SMALL: 0.05}
COLLISION_PENALTY = {DEBRIS_LARGE: -100.0, DEBRIS_MEDIUM: -50.0, DEBRIS_SMALL: -20.0}
SAT_COLLISION_PENALTY = -100.0

@dataclass
class SwarmConfigV4:
    n_controlled:    int   = 50
    n_partners:      int   = 50
    n_debris_large:  int   = 400
    n_debris_medium: int   = 600
    n_debris_small:  int   = 1000
    alt_km:          float = 450.0
    inc_deg:         float = 28.5
    episode_steps:   int   = 120
    dt_step:         float = DT_STEP
    use_real_tles:   bool  = True

class AgentState:
    def __init__(self, eci, fuel=INIT_FUEL):
        self.eci        = eci.copy()
        self.fuel       = fuel
        self.last_burn  = -9999.0
        self.alive      = True
        self.collisions = 0

class DebrisObject:
    def __init__(self, eci, size_class, tracked):
        self.eci        = eci.copy()
        self.size_class = size_class
        self.tracked    = tracked

class SwarmOrbitEnvV4(gym.Env):
    """
    ORCHID v4: 500 real TLE satellites + 2000 simulated debris.
    Uses real satellite positions from SatNOGS/Space-Track.
    Decentralised MARL with shared PPO policy.
    """
    metadata = {"render_modes": []}

    def __init__(self, config: SwarmConfigV4 = None, partner_policies=None):
        super().__init__()
        self.cfg = config or SwarmConfigV4()
        self.partner_policies = partner_policies or self._default_partners()

        # obs: own(8) + K_THREATS*9 + extras(2) = 101
        obs_dim = 8 + K_THREATS * 9 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.n_controlled, obs_dim),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.cfg.n_controlled, 3),
            dtype=np.float32
        )

        # Load real TLE data if available
        self._real_states = []
        if self.cfg.use_real_tles:
            self._load_real_tles()

        self.agents   = []
        self.partners = []
        self.debris   = []
        self._step    = 0
        self._time    = 0.0

    def _load_real_tles(self):
        paths = [
            '/home/siva/orchid_project/orchid/data/real_leo_satellites.json',
            '/kaggle/working/real_leo_satellites.json',
            'data/real_leo_satellites.json',
        ]
        for path in paths:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                self._real_states = [np.array(s['state']) for s in data]
                print(f'Loaded {len(self._real_states)} real satellite states')
                return
        print('No real TLE file found, using synthetic initialisation')

    def _default_partners(self):
        from partners.policy_pool import get_random_training_policy
        return [get_random_training_policy() for _ in range(self.cfg.n_partners)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.cfg
        rng = np.random.default_rng(seed)

        # ── Controlled agents from real TLE states ───────────────────────────
        self.agents = []
        if self._real_states and cfg.use_real_tles:
            indices = rng.choice(len(self._real_states),
                                 size=min(cfg.n_controlled, len(self._real_states)),
                                 replace=False)
            for idx in indices:
                state = self._real_states[idx].copy()
                # Small perturbation so agents aren't exactly on top of each other
                state[:3] += rng.uniform(-0.5, 0.5, 3)
                self.agents.append(AgentState(eci=state))
        else:
            base = circular_state(cfg.alt_km, cfg.inc_deg)
            for i in range(cfg.n_controlled):
                offset = rng.uniform(-3.0, 3.0, 3)
                state  = base.copy()
                state[:3] += rtn_to_eci(offset, base)
                self.agents.append(AgentState(eci=state))

        # ── Partner satellites from remaining real TLE states ────────────────
        self.partners = []
        if self._real_states and cfg.use_real_tles:
            used = set(indices.tolist())
            remaining = [i for i in range(len(self._real_states)) if i not in used]
            p_indices = rng.choice(remaining,
                                   size=min(cfg.n_partners, len(remaining)),
                                   replace=False)
            for idx in p_indices:
                state = self._real_states[idx].copy()
                self.partners.append(AgentState(eci=state))
        else:
            base = circular_state(cfg.alt_km, cfg.inc_deg)
            for i in range(cfg.n_partners):
                offset = rng.uniform(-5.0, 5.0, 3)
                state  = base.copy()
                state[:3] += rtn_to_eci(offset, base)
                self.partners.append(AgentState(eci=state))

        for i, p in enumerate(self.partners):
            if i < len(self.partner_policies):
                if hasattr(self.partner_policies[i], 'reset'):
                    self.partner_policies[i].reset()

        # ── Simulated debris (2000 objects) ──────────────────────────────────
        self.debris = []
        debris_spec = [
            (cfg.n_debris_large,  DEBRIS_LARGE,  True,  8.0),
            (cfg.n_debris_medium, DEBRIS_MEDIUM, True,  6.0),
            (cfg.n_debris_small,  DEBRIS_SMALL,  False, 5.0),
        ]
        for count, size_class, tracked, spread in debris_spec:
            for _ in range(count):
                alt_var  = rng.uniform(-30, 30)
                d_base   = circular_state(cfg.alt_km + alt_var, cfg.inc_deg)
                offset   = rng.uniform(-spread, spread, 3)
                state    = d_base.copy()
                state[:3] += rtn_to_eci(offset, d_base)
                if size_class == DEBRIS_SMALL:
                    state[3:] += rng.uniform(-0.003, 0.003, 3)
                elif size_class == DEBRIS_MEDIUM:
                    state[3:] += rng.uniform(-0.001, 0.001, 3)
                self.debris.append(DebrisObject(
                    eci=state, size_class=size_class, tracked=tracked))

        self._step = 0
        self._time = 0.0
        return self._get_obs_all(), {}

    def _get_obs_single(self, agent: AgentState) -> np.ndarray:
        pos           = agent.eci[:3]
        vel           = agent.eci[3:]
        fuel_norm     = agent.fuel / INIT_FUEL
        cooldown_norm = min((self._time - agent.last_burn) / COOLDOWN_S, 1.0)

        threats = []
        for other in self.agents:
            if other is not agent and other.alive:
                rel_p = other.eci[:3] - pos
                dist  = np.linalg.norm(rel_p)
                threats.append((dist, rel_p, other.eci[3:]-vel, 1.0, 1.0))

        for partner in self.partners:
            rel_p = partner.eci[:3] - pos
            dist  = np.linalg.norm(rel_p)
            threats.append((dist, rel_p, partner.eci[3:]-vel, 1.0, 1.0))

        for d in self.debris:
            rel_p      = d.eci[:3] - pos
            dist       = np.linalg.norm(rel_p)
            size_norm  = d.size_class / 2.0
            track_norm = 1.0 if d.tracked else 0.0
            threats.append((dist, rel_p, d.eci[3:]-vel, size_norm, track_norm))

        threats.sort(key=lambda x: x[0])
        threat_obs = np.zeros(K_THREATS * 9)
        for j, (dist, rel_p, rel_v, size_n, track_n) in enumerate(threats[:K_THREATS]):
            threat_obs[j*9:(j+1)*9] = [
                *rel_p / BOX_KM, *rel_v / 0.01,
                dist / BOX_KM, size_n, track_n
            ]

        r0     = RE + self.cfg.alt_km
        in_box = float(abs(np.linalg.norm(pos) - r0) < BOX_KM)
        t_norm = self._step / self.cfg.episode_steps

        return np.concatenate([
            pos / 7000.0, vel / 8.0,
            [fuel_norm, cooldown_norm],
            threat_obs, [in_box, t_norm]
        ]).astype(np.float32)

    def _get_obs_all(self):
        return np.array([self._get_obs_single(a) for a in self.agents],
                        dtype=np.float32)

    def step(self, actions):
        cfg     = self.cfg
        actions = np.clip(actions, -1.0, 1.0)

        # ── Agent actions ─────────────────────────────────────────────────────
        fuel_used_total = 0.0
        for i, agent in enumerate(self.agents):
            if not agent.alive: continue
            dv_mag  = np.linalg.norm(actions[i]) * MAX_DV
            cool_ok = (self._time - agent.last_burn) >= COOLDOWN_S
            if dv_mag > 1e-6 and cool_ok and agent.fuel > 0:
                dm = min(tsiolkovsky(dv_mag, DRY_MASS + agent.fuel), agent.fuel)
                agent.fuel -= dm
                fuel_used_total += dm
                agent.eci[3:] += rtn_to_eci(actions[i] * MAX_DV, agent.eci)
                agent.last_burn = self._time

        # ── Partner actions ───────────────────────────────────────────────────
        for i, partner in enumerate(self.partners):
            if i >= len(self.partner_policies): continue
            obs = self._get_obs_single(AgentState(eci=partner.eci, fuel=partner.fuel))
            try:
                p_act = np.clip(self.partner_policies[i].act(obs, None), -1.0, 1.0)
            except TypeError:
                p_act = np.clip(self.partner_policies[i].act(obs), -1.0, 1.0)
            p_dv = np.linalg.norm(p_act) * MAX_DV
            if p_dv > 1e-6 and partner.fuel > 0:
                dm = min(tsiolkovsky(p_dv, DRY_MASS + partner.fuel), partner.fuel)
                partner.fuel -= dm
                partner.eci[3:] += rtn_to_eci(p_act * MAX_DV, partner.eci)

        # ── Propagate ─────────────────────────────────────────────────────────
        for a in self.agents:   a.eci = propagate(a.eci, cfg.dt_step)
        for p in self.partners: p.eci = propagate(p.eci, cfg.dt_step)
        for d in self.debris:   d.eci = propagate(d.eci, cfg.dt_step)

        self._step += 1
        self._time += cfg.dt_step

        # ── Rewards ───────────────────────────────────────────────────────────
        rewards    = np.zeros(cfg.n_controlled)
        collisions = 0
        n_alive    = sum(1 for a in self.agents if a.alive)

        for i, agent in enumerate(self.agents):
            if not agent.alive: continue
            r_safe    = 1.0
            collision = False

            for j, other in enumerate(self.agents):
                if i == j or not other.alive: continue
                dist = np.linalg.norm(agent.eci[:3] - other.eci[:3])
                if dist < CONJ_KM[DEBRIS_LARGE]:
                    collision = True; collisions += 1
                else:
                    r_safe = min(r_safe, np.clip(
                        (dist - CONJ_KM[DEBRIS_LARGE]) /
                        (SAFE_KM[DEBRIS_LARGE] - CONJ_KM[DEBRIS_LARGE]), 0, 1))

            for partner in self.partners:
                dist = np.linalg.norm(agent.eci[:3] - partner.eci[:3])
                if dist < CONJ_KM[DEBRIS_LARGE]:
                    collision = True; collisions += 1
                else:
                    r_safe = min(r_safe, np.clip(
                        (dist - CONJ_KM[DEBRIS_LARGE]) /
                        (SAFE_KM[DEBRIS_LARGE] - CONJ_KM[DEBRIS_LARGE]), 0, 1))

            for d in self.debris:
                dist = np.linalg.norm(agent.eci[:3] - d.eci[:3])
                c, s = CONJ_KM[d.size_class], SAFE_KM[d.size_class]
                if dist < c:
                    collision = True; collisions += 1
                    rewards[i] += COLLISION_PENALTY[d.size_class]
                else:
                    r_safe = min(r_safe, np.clip((dist-c)/(s-c), 0, 1))

            if collision:
                agent.alive = False
                rewards[i] += SAT_COLLISION_PENALTY
            else:
                r_fuel = agent.fuel / INIT_FUEL
                r0     = RE + cfg.alt_km
                r_box  = float(abs(np.linalg.norm(agent.eci[:3]) - r0) < BOX_KM)
                r_team = n_alive / cfg.n_controlled
                rewards[i] = 0.5*r_safe + 0.3*r_fuel + 0.1*r_box + 0.1*r_team

        terminated = all(not a.alive for a in self.agents)
        truncated  = self._step >= cfg.episode_steps

        fuel_remaining = [a.fuel for a in self.agents]
        used           = [INIT_FUEL - f for f in fuel_remaining]

        info = {
            'collisions':      collisions,
            'agents_alive':    sum(1 for a in self.agents if a.alive),
            'survival_rate':   sum(1 for a in self.agents if a.alive) / cfg.n_controlled,
            'fuel_remaining':  fuel_remaining,
            'fuel_used_total': fuel_used_total,
            'mean_fuel_used':  float(np.mean(used)),
            'fuel_efficiency': float(1 - np.mean(used) / INIT_FUEL),
            'mean_reward':     float(np.mean(rewards[rewards != 0])) if any(rewards != 0) else 0.0,
            'debris_total':    cfg.n_debris_large + cfg.n_debris_medium + cfg.n_debris_small,
        }
        return self._get_obs_all(), rewards, terminated, truncated, info
