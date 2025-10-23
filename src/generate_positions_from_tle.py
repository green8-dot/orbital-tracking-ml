import pymongo
from sgp4.api import Satrec, jday, WGS72
from datetime import datetime, timedelta
import numpy as np

def tle_to_positions(tle_line1, tle_line2, start_time, duration_hours=24, step_minutes=10):
    """Convert TLE to position predictions using SGP4"""
    
    # Create satellite object from TLE
    satellite = Satrec.twoline2rv(tle_line1, tle_line2, WGS72)
    
    positions = []
    current_time = start_time
    end_time = start_time + timedelta(hours=duration_hours)
    
    while current_time <= end_time:
        # Convert to Julian date
        jd, fr = jday(
            current_time.year, current_time.month, current_time.day,
            current_time.hour, current_time.minute, current_time.second
        )
        
        # Propagate satellite position
        error_code, r, v = satellite.sgp4(jd, fr)
        
        if error_code == 0:  # Success
            # Convert from TEME to lat/lon/alt
            # r is position in km, v is velocity in km/s
            
            # Calculate magnitude for altitude
            r_mag = np.linalg.norm(r)
            altitude_km = r_mag - 6371.0  # Earth radius
            
            # Convert to lat/lon (simplified - proper conversion needs coordinate transforms)
            lat = np.degrees(np.arcsin(r[2] / r_mag))
            lon = np.degrees(np.arctan2(r[1], r[0]))
            
            # Calculate velocity magnitude
            v_mag = np.linalg.norm(v)
            
            positions.append({
                'timestamp': current_time,
                'timestamp_unix': current_time.timestamp(),
                'latitude': lat,
                'longitude': lon,
                'altitude_km': altitude_km,
                'velocity_km_s': v_mag,
                'position_teme': list(r),
                'velocity_teme': list(v),
                'propagation_error': error_code
            })
        
        current_time += timedelta(minutes=step_minutes)
    
    return positions

def main():
    print("=" * 60)
    print("GENERATING ISS POSITIONS FROM TLE DATA")
    print("=" * 60)
    
    # Connect to MongoDB
    client = pymongo.MongoClient('localhost', 27017)
    db = client.orbitscope_ml
    
    # Get latest TLE from database
    latest_tle = db.iss_tles_latest.find_one(
        {},
        sort=[('EPOCH', pymongo.DESCENDING)]
    )
    
    if not latest_tle:
        print("[ERROR] No TLE data found in database!")
        return
    
    print(f"/nUsing TLE from epoch: {latest_tle.get('EPOCH')}")
    print(f"Object: {latest_tle.get('OBJECT_NAME', 'ISS')}")
    
    # Extract TLE lines
    tle_line1 = latest_tle.get('TLE_LINE1')
    tle_line2 = latest_tle.get('TLE_LINE2')
    
    if not tle_line1 or not tle_line2:
        print("[ERROR] TLE lines not found in database!")
        return
    
    print(f"/nTLE Line 1: {tle_line1}")
    print(f"TLE Line 2: {tle_line2}")
    
    # Generate positions for next 24 hours
    start_time = datetime.now()
    print(f"/nGenerating positions from {start_time.isoformat()}Z")
    print("Duration: 24 hours, Step: 10 minutes")
    
    positions = tle_to_positions(
        tle_line1, tle_line2,
        start_time,
        duration_hours=24,
        step_minutes=10
    )
    
    print(f"/n[SUCCESS] Generated {len(positions)} position predictions")
    
    # Store in MongoDB
    if positions:
        # Add metadata to each position
        for pos in positions:
            pos['norad_id'] = 25544  # ISS
            pos['object_name'] = 'ISS'
            pos['data_source'] = 'sgp4_propagation'
            pos['tle_epoch'] = latest_tle.get('EPOCH')
            pos['generated_at'] = datetime.now()
        
        # Insert into database
        result = db.iss_positions_predicted.insert_many(positions)
        print(f"/n[SUCCESS] Stored {len(result.inserted_ids)} positions in database")
        
        # Display sample position
        sample = positions[0]
        print(f"/nSample position:")
        print(f"  Time: {sample['timestamp'].isoformat()}Z")
        print(f"  Latitude: {sample['latitude']:.4f}[?]")
        print(f"  Longitude: {sample['longitude']:.4f}[?]")
        print(f"  Altitude: {sample['altitude_km']:.2f} km")
        print(f"  Velocity: {sample['velocity_km_s']:.3f} km/s")
    
    # Database summary
    print("/n" + "=" * 60)
    print("Database summary:")
    collections = ['iss_tles_latest', 'iss_tles_historical', 'iss_positions_predicted']
    for collection_name in collections:
        if collection_name in db.list_collection_names():
            count = db[collection_name].count_documents({})
            print(f"  {collection_name}: {count} documents")

if __name__ == "__main__":
    main()