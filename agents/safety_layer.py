import numpy as np
from env.physics import rtn_to_eci, tsiolkovsky

PC_THRESHOLD  = 1e-3
FUEL_RESERVE  = 2.5
MAX_DV        = 0.015

def collision_probability(miss_km, rel_vel_km_s, sigma_km=0.050, combined_radius_km=0.100):
    if miss_km < combined_radius_km:
        return 1.0
    exponent = -0.5 * (miss_km / sigma_km) ** 2
    Pc = np.exp(exponent) * (combined_radius_km / miss_km)
    return float(np.clip(Pc, 0.0, 1.0))

class SafetyLayer:
    def __init__(self):
        self.override_count = 0

    def filter(self, action, sat_state, threats, fuel, cooldown):
        if fuel <= FUEL_RESERVE:
            return np.zeros(3)
        max_pc = 0.0
        worst_threat = None
        for threat_state in threats:
            miss_km = np.linalg.norm(threat_state[:3] - sat_state[:3])
            rel_vel = np.linalg.norm(threat_state[3:] - sat_state[3:])
            pc = collision_probability(miss_km, rel_vel)
            if pc > max_pc:
                max_pc = pc
                worst_threat = threat_state
        if max_pc < PC_THRESHOLD:
            return action
        self.override_count += 1
        return np.array([0.0, -MAX_DV, 0.0])

    def reset(self):
        self.override_count = 0
