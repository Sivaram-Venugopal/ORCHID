import numpy as np

# ── Physical Constants ────────────────────────────────────────────────────────
MU   = 398600.4418   # Earth gravitational parameter (km^3/s^2)
RE   = 6378.137      # Earth equatorial radius (km)
J2   = 1.08263e-3    # Earth's second zonal harmonic
G0   = 9.80665e-3    # Standard gravity (km/s^2)
ISP  = 300.0         # Specific impulse (seconds)

def deriv(t, state):
    """Compute state derivative with J2 perturbation."""
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)

    # Two-body gravity
    a_grav = -(MU / r**3) * np.array([x, y, z])

    # J2 perturbation
    f = 1.5 * J2 * MU * RE**2 / r**5
    zr2 = (z / r)**2
    a_j2 = f * np.array([
        x * (5*zr2 - 1),
        y * (5*zr2 - 1),
        z * (5*zr2 - 3)
    ])

    a = a_grav + a_j2
    return np.array([vx, vy, vz, a[0], a[1], a[2]])

def rk4_step(state, dt):
    """Single RK4 integration step."""
    k1 = deriv(0, state)
    k2 = deriv(0, state + 0.5*dt*k1)
    k3 = deriv(0, state + 0.5*dt*k2)
    k4 = deriv(0, state + dt*k3)
    return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def propagate(state, dt_total, dt_step=30.0):
    """Propagate state forward by dt_total seconds using RK4."""
    t = 0.0
    while t < dt_total - 1e-9:
        dt = min(dt_step, dt_total - t)
        state = rk4_step(state, dt)
        t += dt
    return state

def rtn_to_eci(dv_rtn, state):
    """Convert delta-v from RTN frame to ECI frame."""
    r_vec = state[:3]
    v_vec = state[3:]
    r_hat = r_vec / np.linalg.norm(r_vec)
    n_vec = np.cross(r_vec, v_vec)
    n_hat = n_vec / np.linalg.norm(n_vec)
    t_hat = np.cross(n_hat, r_hat)
    return dv_rtn[0]*r_hat + dv_rtn[1]*t_hat + dv_rtn[2]*n_hat

def eci_to_rtn(dv_eci, state):
    """Convert delta-v from ECI frame to RTN frame."""
    r_vec = state[:3]
    v_vec = state[3:]
    r_hat = r_vec / np.linalg.norm(r_vec)
    n_vec = np.cross(r_vec, v_vec)
    n_hat = n_vec / np.linalg.norm(n_vec)
    t_hat = np.cross(n_hat, r_hat)
    return np.array([
        np.dot(dv_eci, r_hat),
        np.dot(dv_eci, t_hat),
        np.dot(dv_eci, n_hat)
    ])

def tsiolkovsky(dv_km_s, mass_kg):
    """Return fuel mass consumed for a given delta-v."""
    return mass_kg * (1 - np.exp(-dv_km_s / (ISP * G0)))

def circular_state(alt_km, inc_deg=28.5, raan_deg=0.0, nu_deg=0.0):
    """Generate ECI state vector for a circular orbit."""
    r = RE + alt_km
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    nu = np.radians(nu_deg)
    v_circ = np.sqrt(MU / r)

    # Position in orbital plane
    rx = r * np.cos(nu)
    ry = r * np.sin(nu)

    # Rotate to ECI
    cos_r, sin_r = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(inc),  np.sin(inc)
    cos_u, sin_u = np.cos(nu),   np.sin(nu)

    x  =  cos_r*rx - sin_r*cos_i*ry
    y  =  sin_r*rx + cos_r*cos_i*ry
    z  =  sin_i*ry
    vx = -v_circ*(cos_r*sin_u + sin_r*cos_i*cos_u)
    vy = -v_circ*(sin_r*sin_u - cos_r*cos_i*cos_u)
    vz =  v_circ*sin_i*cos_u

    return np.array([x, y, z, vx, vy, vz])