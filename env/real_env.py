import numpy as np
import sys
sys.path.insert(0, '/home/siva/orchid_project/orchid')
from env.tle_loader import fetch_tle, tle_to_state, fetch_leo_conjunction_pair
from env.orbital_env import OrchestratedOrbitEnv, EnvConfig
from env.physics import eci_to_rtn

# Known LEO satellite NORAD IDs for real scenarios
REAL_SATELLITES = {
    'ISS':          25544,
    'CSS':          48274,   # Chinese Space Station
    'Sentinel-6':   46984,
    'Landsat-9':    49260,
    'NOAA-20':      43013,
}

class RealTLEEnv(OrchestratedOrbitEnv):
    """
    ORCHID environment initialised from real TLE data.
    Uses actual satellite positions as episode starting states.
    """
    def __init__(self, config=None, partner_policy=None,
                 primary_norad=25544, partner_norad=48274):
        super().__init__(config or EnvConfig(), partner_policy)
        self.primary_norad = primary_norad
        self.partner_norad = partner_norad
        self._real_data = None

    def _fetch_real_states(self):
        """Fetch current real satellite states from TLE."""
        try:
            data = fetch_leo_conjunction_pair(self.primary_norad, self.partner_norad)
            return data['satellite_a']['state'], data['satellite_b']['state']
        except Exception as e:
            print(f"TLE fetch failed ({e}), using synthetic initialisation")
            return None, None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Try to override with real TLE states
        state_a, state_b = self._fetch_real_states()
        if state_a is not None:
            from env.orbital_env import SatState
            self.agent = SatState(
                state=state_a,
                fuel=self.cfg.init_fuel if hasattr(self.cfg, 'init_fuel') else 50.0
            )
            self.partners[0] = SatState(
                state=state_b,
                fuel=self.cfg.init_fuel if hasattr(self.cfg, 'init_fuel') else 50.0
            )
            obs = self._get_obs(self.agent)
            print(f"Real TLE initialised: alt={np.linalg.norm(state_a[:3])-6378.137:.1f}km")
        
        return obs, info

if __name__ == "__main__":
    from partners.policy_pool import ProgradePolicy
    
    print("Creating real TLE environment...")
    env = RealTLEEnv(
        partner_policy=ProgradePolicy(),
        primary_norad=25544,   # ISS
        partner_norad=48274,   # CSS
    )
    
    print("Resetting environment with real data...")
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"First 6 obs (agent RTN pos/vel): {obs[:6]}")
    
    print("\nRunning 5 steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f} fuel={info['fuel']:.2f}kg collision={info['collision']}")
    
    print("\nReal TLE environment working!")
