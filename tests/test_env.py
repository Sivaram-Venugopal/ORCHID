import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from env.physics import (
    propagate, circular_state, rtn_to_eci,
    eci_to_rtn, tsiolkovsky, MU, RE
)

def test_rk4_propagation():
    """Altitude drift after 1 orbit should be under 0.001 km."""
    state = circular_state(alt_km=450.0)
    r0 = np.linalg.norm(state[:3])
    period = 2 * np.pi * np.sqrt((RE + 450.0)**3 / MU)
    state_final = propagate(state, period)
    r1 = np.linalg.norm(state_final[:3])
    drift = abs(r1 - r0)
    assert drift < 0.001, f"Drift too large: {drift:.4f} km"
    print(f"[OK] RK4 propagation: altitude drift after 1 orbit = {drift:.4f} km")

def test_rtn_eci_roundtrip():
    """RTN to ECI and back should have zero error."""
    state = circular_state(alt_km=450.0)
    dv_rtn = np.array([0.001, 0.005, 0.002])
    dv_eci = rtn_to_eci(dv_rtn, state)
    dv_rtn_back = eci_to_rtn(dv_eci, state)
    error = np.linalg.norm(dv_rtn - dv_rtn_back)
    assert error < 1e-10, f"Round-trip error too large: {error:.2e}"
    print(f"[OK] RTN<->ECI round-trip error = {error:.2e}")

def test_tsiolkovsky():
    """10 m/s burn on 550 kg satellite should consume ~1.866 kg."""
    fuel = tsiolkovsky(0.010, 550.0)
    assert abs(fuel - 1.866) < 0.01, f"Fuel calc wrong: {fuel:.3f} kg"
    print(f"[OK] Tsiolkovsky: 10 m/s burn consumes {fuel:.3f} kg")

def test_circular_state():
    """Generated state should be at correct altitude."""
    state = circular_state(alt_km=450.0)
    alt = np.linalg.norm(state[:3]) - RE
    assert abs(alt - 450.0) < 0.001, f"Altitude wrong: {alt:.3f} km"
    print(f"[OK] Random state generation: alt = {alt:.3f} km")

if __name__ == "__main__":
    print("Running ORCHID physics tests...\n")
    test_rk4_propagation()
    test_rtn_eci_roundtrip()
    test_tsiolkovsky()
    test_circular_state()
    print("\n[OK] All physics tests passed!")
