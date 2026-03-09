import urllib.request
import json
import numpy as np
from sgp4.api import Satrec, WGS84
from datetime import datetime, timezone

def fetch_tle(norad_id):
    """Fetch real TLE data from SatNOGS for a given NORAD catalog ID."""
    url = f'https://db.satnogs.org/api/tle/?norad_cat_id={norad_id}'
    req = urllib.request.Request(url, headers={'User-Agent': 'ORCHID/1.0'})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode())
    if not data:
        raise ValueError(f"No TLE found for NORAD ID {norad_id}")
    return data[0]['tle1'], data[0]['tle2'], data[0]['tle0'].replace('0 ', '')

def tle_to_state(tle1, tle2, epoch=None):
    """
    Convert TLE to ECI state vector [x,y,z,vx,vy,vz] in km and km/s.
    Uses current time if epoch not specified.
    """
    sat = Satrec.twoline2rv(tle1, tle2)
    if epoch is None:
        epoch = datetime.now(timezone.utc)
    
    # Convert to Julian date
    jd = 2451545.0 + (epoch - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
    jd_i = int(jd)
    jd_f = jd - jd_i
    
    e, r, v = sat.sgp4(jd_i, jd_f)
    if e != 0:
        raise ValueError(f"SGP4 propagation error: {e}")
    
    return np.array([r[0], r[1], r[2], v[0], v[1], v[2]])

def fetch_leo_conjunction_pair(norad_id_1=25544, norad_id_2=None):
    """
    Fetch a pair of real satellites for conjunction scenario.
    Default: ISS + a random LEO satellite.
    """
    # Fetch primary satellite
    tle1_a, tle2_a, name_a = fetch_tle(norad_id_1)
    state_a = tle_to_state(tle1_a, tle2_a)
    
    if norad_id_2 is None:
        # Use a known LEO satellite close to ISS orbit
        norad_id_2 = 48274  # CSS (Chinese Space Station)
    
    try:
        tle1_b, tle2_b, name_b = fetch_tle(norad_id_2)
        state_b = tle_to_state(tle1_b, tle2_b)
    except:
        # Fallback: generate synthetic partner near primary
        state_b = state_a.copy()
        state_b[:3] += np.random.uniform(-5, 5, 3)
        name_b = "Synthetic Partner"
    
    return {
        'satellite_a': {'name': name_a, 'norad_id': norad_id_1, 'state': state_a},
        'satellite_b': {'name': name_b, 'norad_id': norad_id_2, 'state': state_b},
    }

if __name__ == "__main__":
    print("Fetching real ISS TLE data...")
    tle1, tle2, name = fetch_tle(25544)
    print(f"Satellite: {name}")
    print(f"TLE Line 1: {tle1}")
    print(f"TLE Line 2: {tle2}")
    
    state = tle_to_state(tle1, tle2)
    alt = np.linalg.norm(state[:3]) - 6378.137
    speed = np.linalg.norm(state[3:])
    
    print(f"\nCurrent ECI State:")
    print(f"Position: [{state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f}] km")
    print(f"Velocity: [{state[3]:.4f}, {state[4]:.4f}, {state[5]:.4f}] km/s")
    print(f"Altitude: {alt:.1f} km")
    print(f"Speed:    {speed:.4f} km/s")
