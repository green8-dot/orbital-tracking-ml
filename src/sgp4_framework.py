import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import pymongo

client = pymongo.MongoClient('localhost', 27017)
db = client.orbitscope_ml

def parse_tle(line1, line2):
    """Parse real TLE data and create propagator"""
    satellite = Satrec.twoline2rv(line1, line2)
    return satellite

def propagate_orbit(satellite, start_time, duration_hours=24, step_minutes=5):
    """Generate real orbital positions using SGP4"""
    positions = []
    
    for minutes in range(0, duration_hours * 60, step_minutes):
        t = start_time + timedelta(minutes=minutes)
        
        jd, fr = jday(t.year, t.month, t.day, 
                      t.hour, t.minute, t.second)
        
        error, position, velocity = satellite.sgp4(jd, fr)
        
        if error == 0:
            # Convert to lat/lon/alt
            positions.append({
                'timestamp': t,
                'position_teme': position,  # TEME coordinates
                'velocity': velocity,
                'error_code': error
            })
    
    return positions

def validate_propagation():
    """Test SGP4 with real ISS TLE"""
    # Real ISS TLE (you'll get fresh ones from Space-Track)
    line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-2 0  9999"
    line2 = "2 25544  51.6416 208.4590 0002744  35.3073  80.5436 15.50381554427823"
    
    satellite = parse_tle(line1, line2)
    positions = propagate_orbit(satellite, datetime.utcnow())
    
    print(f"Generated {len(positions)} positions")
    return positions