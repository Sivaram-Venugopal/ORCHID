import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import KDTree
from dataclasses import dataclass, field
from typing import List, Optional
from env.physics import (
    propagate, circular_state, rtn_to_eci,
    tsiolkovsky, RE, MU
)

# ── Constants ─────────────────────────────────────────────────────────────────
DRY_MASS        = 500.0    # kg
INIT_FUEL       = 50.0     # kg
MAX_DV          = 0.015    # km/s per burn
COOLDOWN_S      = 600.0    # seconds between burns
CONJUNCTION_KM  = 0.100    # hard collision threshold (km)
SAFE_KM         = 0.500    # desired miss distance (km)
BOX_KM          = 5.0      # station-keeping box half-width (km)
DT_STEP         = 30.0     # simulation step size (seconds)
K_THREATS       = 5        # number of closest threats in observation

@dataclass
class EnvConfig:
    n_controlled:          int   = 1
    n_partner:             int   = 1
    n_debris:              int   = 5
    episode_len_s:         float = 3600.0
    guaranteed_conjunction: bool = True
    alt_km:                float = 450.0

@dataclass
class SatState:
    state:      np.ndarray        # ECI [x,y,z,vx,vy,vz]
    fuel:       float = INIT_FUEL
    cooldown:   float = 0.0
    is_controlled: bool = True

class OrchestratedOrbitEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig, partner_policy=None, seed=None):
        super().__init__()
        self.cfg     = config
        self.partner = partner_policy
        self.rng     = np.random.default_rng(seed)

        # Observation: [rel_pos(3), rel_vel(3), fuel(1),
        #               K threats * 7, n_partner * 6, box_flag(1), time(1)]
        obs_dim = 50
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-MAX_DV, high=MAX_DV, shape=(3,), dtype=np.float32
        )
        self.controlled: List[SatState] = []
        self.partners:   List[SatState] = []
        self.debris:     List[np.ndarray] = []
        self.t = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.controlled = []
        self.partners   = []
        self.debris     = []

        # Spawn controlled satellites
        for i in range(self.cfg.n_controlled):
            s = circular_state(self.cfg.alt_km, nu_deg=i*10.0)
            self.controlled.append(SatState(state=s))

        # Spawn partner satellites
        for i in range(self.cfg.n_partner):
            s = circular_state(self.cfg.alt_km, nu_deg=180.0 + i*10.0)
            self.partners.append(SatState(state=s, is_controlled=False))

        # Spawn debris
        for i in range(self.cfg.n_debris):
            nu = self.rng.uniform(0, 360)
            alt = self.cfg.alt_km + self.rng.uniform(-20, 20)
            s = circular_state(alt, nu_deg=nu)
            self.debris.append(s)

        # Guarantee at least one conjunction
        if self.cfg.guaranteed_conjunction and self.debris:
            offset = np.array([0.05, 0.05, 0.0, 0.0, 0.0, 0.0])
            self.debris[0] = self.controlled[0].state + offset

        obs  = self._get_obs(self.controlled[0])
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -MAX_DV, MAX_DV)

        # Apply action to controlled satellite
        sat = self.controlled[0]
        reward = 0.0
        collision = False

        if sat.cooldown <= 0 and np.linalg.norm(action) > 1e-6:
            dv_eci = rtn_to_eci(action, sat.state)
            fuel_used = tsiolkovsky(np.linalg.norm(action),
                                    DRY_MASS + sat.fuel)
            sat.fuel     = max(0.0, sat.fuel - fuel_used)
            sat.state[3:] += dv_eci
            sat.cooldown  = COOLDOWN_S
            reward -= 0.5 * np.linalg.norm(action)

        # Propagate all objects
        sat.state    = propagate(sat.state, DT_STEP)
        sat.cooldown = max(0.0, sat.cooldown - DT_STEP)

        for p in self.partners:
            p_obs = self._get_obs(p)
            p_act = self.partner.act(p_obs, p.state) if self.partner else np.zeros(3)
            p_act = np.clip(p_act, -MAX_DV, MAX_DV)
            if np.linalg.norm(p_act) > 1e-6:
                dv_eci = rtn_to_eci(p_act, p.state)
                p.state[3:] += dv_eci
            p.state = propagate(p.state, DT_STEP)

        for i, d in enumerate(self.debris):
            self.debris[i] = propagate(d, DT_STEP)

        self.t += DT_STEP

        # Collision detection
        all_others = [p.state for p in self.partners] + self.debris
        if all_others:
            positions = np.array([s[:3] for s in all_others])
            tree = KDTree(positions)
            hits = tree.query_ball_point(sat.state[:3], CONJUNCTION_KM)
            if hits:
                collision = True
                reward -= 100.0

        # Safety reward
        if not collision:
            reward += 0.5

        # Station keeping
        r0 = RE + self.cfg.alt_km
        alt_err = abs(np.linalg.norm(sat.state[:3]) - r0)
        if alt_err < BOX_KM:
            reward += 0.5
        else:
            reward -= 0.1

        done     = self.t >= self.cfg.episode_len_s
        obs      = self._get_obs(sat)
        info     = {"collision": collision, "fuel": sat.fuel, "time": self.t}
        return obs, float(reward), done, False, info

    def _get_obs(self, sat: SatState) -> np.ndarray:
        pos = sat.state[:3]
        vel = sat.state[3:]
        fuel_norm = sat.fuel / INIT_FUEL

        # K closest threats
        all_others = [p.state for p in self.partners] + self.debris
        threat_obs = np.zeros(K_THREATS * 7)
        if all_others:
            positions = np.array([s[:3] for s in all_others])
            dists = np.linalg.norm(positions - pos, axis=1)
            idx   = np.argsort(dists)[:K_THREATS]
            for j, i in enumerate(idx):
                rel_p = all_others[i][:3] - pos
                rel_v = all_others[i][3:] - vel
                dist  = dists[i]
                threat_obs[j*7:(j+1)*7] = [*rel_p, *rel_v, dist]

        # Partner observations
        partner_obs = np.zeros(self.cfg.n_partner * 6)
        for j, p in enumerate(self.partners):
            rel_p = p.state[:3] - pos
            rel_v = p.state[3:] - vel
            partner_obs[j*6:(j+1)*6] = [*rel_p, *rel_v]

        # Station keeping box flag
        r0 = RE + self.cfg.alt_km
        in_box = float(abs(np.linalg.norm(pos) - r0) < BOX_KM)
        time_norm = self.t / self.cfg.episode_len_s

        obs = np.concatenate([
            pos / 7000.0, vel / 8.0,
            [fuel_norm],
            threat_obs,
            partner_obs,
            [in_box, time_norm]
        ]).astype(np.float32)
        return obs
