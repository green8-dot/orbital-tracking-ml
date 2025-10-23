import pymongo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

client = pymongo.MongoClient('localhost', 27017)
db = client.orbitscope_ml

def prepare_training_data():
    """Prepare ISS data for ML training"""
    
    # Get all positions sorted by time
    positions = list(db.iss_positions.find().sort('timestamp_unix', 1))
    
    print(f"Loading {len(positions)} ISS positions...")
    
    # Create features and targets
    X = []  # Features (current position)
    y_lat = []  # Target (next latitude)
    y_lon = []  # Target (next longitude)
    
    for i in range(len(positions) - 1):
        current = positions[i]
        next_pos = positions[i + 1]
        
        # Features: current position + altitude
        features = [
            current['latitude'],
            current['longitude'],
            current.get('altitude_km', 410),
            i  # Time index
        ]
        X.append(features)
        
        # Targets: next position
        y_lat.append(next_pos['latitude'])
        y_lon.append(next_pos['longitude'])
    
    return np.array(X), np.array(y_lat), np.array(y_lon)

def train_models():
    """Train Random Forest models for lat/lon prediction"""
    
    # Prepare data
    X, y_lat, y_lon = prepare_training_data()
    
    if len(X) < 20:
        print("Need at least 20 data points for training!")
        return
    
    print(f"/n[OK] Prepared {len(X)} training samples")
    print(f"  Features: current_lat, current_lon, altitude, time_index")
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_lat_train, y_lat_test = train_test_split(
        X, y_lat, test_size=0.2, random_state=42
    )
    _, _, y_lon_train, y_lon_test = train_test_split(
        X, y_lon, test_size=0.2, random_state=42
    )
    
    # Train latitude predictor
    print("/nTraining latitude predictor...")
    lat_model = RandomForestRegressor(n_estimators=50, random_state=42)
    lat_model.fit(X_train, y_lat_train)
    
    # Train longitude predictor
    print("Training longitude predictor...")
    lon_model = RandomForestRegressor(n_estimators=50, random_state=42)
    lon_model.fit(X_train, y_lon_train)
    
    # Evaluate models
    lat_pred = lat_model.predict(X_test)
    lon_pred = lon_model.predict(X_test)
    
    lat_mae = mean_absolute_error(y_lat_test, lat_pred)
    lon_mae = mean_absolute_error(y_lon_test, lon_pred)
    
    print("/n" + "=" * 40)
    print("Model Performance:")
    print(f"  Latitude MAE: {lat_mae:.4f}[?]")
    print(f"  Longitude MAE: {lon_mae:.4f}[?]")
    
    # Calculate distance error in km (rough approximation)
    km_per_degree = 111  # At equator
    distance_error = np.sqrt(lat_mae**2 + lon_mae**2) * km_per_degree
    print(f"  Approximate distance error: {distance_error:.2f} km")
    
    # Save models
    joblib.dump(lat_model, 'iss_latitude_predictor.pkl')
    joblib.dump(lon_model, 'iss_longitude_predictor.pkl')
    print("/n[OK] Models saved to disk")
    
    # Store model metadata in MongoDB
    model_info = {
        'model_name': 'iss_position_predictor_v1',
        'training_date': pd.Timestamp.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'latitude_mae': lat_mae,
        'longitude_mae': lon_mae,
        'distance_error_km': distance_error
    }
    db.model_metadata.insert_one(model_info)
    print("[OK] Model metadata saved to MongoDB")
    
    return lat_model, lon_model

def make_prediction(lat_model, lon_model):
    """Make a sample prediction"""
    
    # Get latest position
    latest = db.iss_positions.find_one(sort=[('timestamp_unix', -1)])
    
    if latest:
        # Prepare features
        features = [[
            latest['latitude'],
            latest['longitude'],
            latest.get('altitude_km', 410),
            99  # Dummy time index
        ]]
        
        # Predict next position
        next_lat = lat_model.predict(features)[0]
        next_lon = lon_model.predict(features)[0]
        
        print("/n" + "=" * 40)
        print("Sample Prediction:")
        print(f"  Current position: ({latest['latitude']:.4f}, {latest['longitude']:.4f})")
        print(f"  Predicted next: ({next_lat:.4f}, {next_lon:.4f})")
        
        # Store prediction
        prediction = {
            'current_lat': latest['latitude'],
            'current_lon': latest['longitude'],
            'predicted_lat': next_lat,
            'predicted_lon': next_lon,
            'model_version': 'v1',
            'prediction_time': pd.Timestamp.now().isoformat()
        }
        db.predictions.insert_one(prediction)
        print("/n[OK] Prediction saved to MongoDB")

if __name__ == "__main__":
    import pandas as pd
    
    print("ISS Position Prediction Model Training")
    print("=" * 40)
    
    # Check if we have data first
    count = db.iss_positions.count_documents({})
    if count == 0:
        print("/n[FAIL] No data in database!")
        print("Please run these commands in order:")
        print("1. python create_sample_data.py")
        print("2. python import_data.py")
        print("3. python train_iss_predictor.py")
    else:
        print(f"[OK] Found {count} ISS positions")
        
        # Train models
        result = train_models()
        
        if result:
            lat_model, lon_model = result
            # Make a prediction
            make_prediction(lat_model, lon_model)
            
            print("/n" + "=" * 40)
            print("[OK] ML Framework Successfully Initialized!")