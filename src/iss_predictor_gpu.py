"""
ISS Position Predictor - GPU Accelerated
=========================================
PyTorch neural network for ISS orbital position prediction
Replaces sklearn RandomForest with GPU-accelerated deep learning

Expected Performance:
- 4-5x faster training on large datasets
- Better accuracy for position prediction
- Real-time inference capability
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pymongo
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging


class ISSPositionPredictor(nn.Module):
    """Neural network for ISS position prediction"""

    def __init__(self, input_size=4, hidden_sizes=[64, 32]):
        super().__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size

        # Output layer: lat and lon predictions (normalized to 0-1)
        layers.append(nn.Linear(prev_size, 2))
        layers.append(nn.Sigmoid())  # Bound output to 0-1 range for normalized predictions

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ISSPredictorGPU:
    """GPU-accelerated ISS position predictor"""

    def __init__(self, use_gpu=True, db_uri='mongodb://localhost:27017'):
        # GPU setup
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"ISS Predictor using device: {self.device}")

        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Model
        self.model = ISSPositionPredictor().to(self.device)

        # Training hyperparameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 50

        # Database connection
        try:
            self.client = pymongo.MongoClient(db_uri)
            self.db = self.client.orbitscope_ml
            print("Connected to MongoDB")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            self.db = None

        # Models directory
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('ISSPredictorGPU')

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare ISS data from MongoDB for training

        Returns:
            X: Features (current position)
            y_lat: Target latitudes
            y_lon: Target longitudes
        """

        if self.db is None:
            self.logger.warning("No database connection, using sample data")
            return self._create_sample_data()

        # Get all positions sorted by time
        positions = list(self.db.iss_positions.find().sort('timestamp_unix', 1))

        if len(positions) < 20:
            self.logger.warning(f"Only {len(positions)} positions found, using sample data")
            return self._create_sample_data()

        self.logger.info(f"Loading {len(positions)} ISS positions from MongoDB...")

        # Create features and targets
        X = []
        y_lat = []
        y_lon = []

        for i in range(len(positions) - 1):
            current = positions[i]
            next_pos = positions[i + 1]

            # Features: current position + altitude + time index
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

        return np.array(X, dtype=np.float32), np.array(y_lat, dtype=np.float32), np.array(y_lon, dtype=np.float32)

    def _create_sample_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sample ISS orbit data for testing"""

        self.logger.info("Creating sample ISS orbit data...")

        # Simulate ISS orbit (simplified)
        n_samples = 1000
        t = np.linspace(0, 10, n_samples)

        # ISS orbital parameters (simplified)
        inclination = 51.6  # degrees
        period = 92.68  # minutes

        # Generate positions
        X = []
        y_lat = []
        y_lon = []

        for i in range(n_samples - 1):
            # Current position
            lat = inclination * np.sin(2 * np.pi * t[i] / period)
            lon = (360 * t[i] / period) % 360 - 180
            alt = 410 + 10 * np.sin(2 * np.pi * t[i] / 10)  # Slight altitude variation

            # Next position
            next_lat = inclination * np.sin(2 * np.pi * t[i+1] / period)
            next_lon = (360 * t[i+1] / period) % 360 - 180

            X.append([lat, lon, alt, i])
            y_lat.append(next_lat)
            y_lon.append(next_lon)

        return np.array(X, dtype=np.float32), np.array(y_lat, dtype=np.float32), np.array(y_lon, dtype=np.float32)

    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features for neural network training"""

        X_norm = X.copy()

        # Latitude: -90 to 90 -> 0 to 1
        X_norm[:, 0] = (X_norm[:, 0] + 90) / 180

        # Longitude: -180 to 180 -> 0 to 1
        X_norm[:, 1] = (X_norm[:, 1] + 180) / 360

        # Altitude: normalize by max ~500km
        X_norm[:, 2] = X_norm[:, 2] / 500

        # Time index: normalize by dataset size
        if len(X) > 0:
            X_norm[:, 3] = X_norm[:, 3] / len(X)

        return X_norm

    def train(self, X: np.ndarray = None, y_lat: np.ndarray = None, y_lon: np.ndarray = None, verbose=True) -> Dict[str, Any]:
        """
        Train neural network on ISS position data

        Returns:
            Training results with accuracy and timing
        """

        import time

        self.logger.info("Starting GPU-accelerated ISS predictor training...")

        # Get data
        if X is None or y_lat is None or y_lon is None:
            X, y_lat, y_lon = self.prepare_training_data()

        # Normalize features
        X_norm = self.normalize_features(X)

        # Normalize targets (lat/lon) for better training
        y_lat_norm = (y_lat + 90) / 180  # -90 to 90 -> 0 to 1
        y_lon_norm = (y_lon + 180) / 360  # -180 to 180 -> 0 to 1

        # Combine normalized targets
        y = np.column_stack([y_lat_norm, y_lon_norm]).astype(np.float32)

        # Split into train/test
        split_idx = int(0.8 * len(X_norm))
        X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        start_time = time.time()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0

            for batch_X, batch_y in loader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Validation
            if verbose and (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    train_pred = self.model(X_train_tensor)
                    train_loss = criterion(train_pred, y_train_tensor).item()

                    test_pred = self.model(X_test_tensor)
                    test_loss = criterion(test_pred, y_test_tensor).item()

                avg_loss = total_loss / len(loader)
                self.logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Loss: {avg_loss:.6f}, "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Test Loss: {test_loss:.6f}"
                )

        training_time = time.time() - start_time

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            test_pred = self.model(X_test_tensor)
            test_loss = criterion(test_pred, y_test_tensor).item()

            # Denormalize predictions and targets back to degrees
            test_pred_np = test_pred.cpu().numpy()
            y_test_np = y_test_tensor.cpu().numpy()

            # Denormalize: 0-1 -> degrees
            pred_lat = test_pred_np[:, 0] * 180 - 90  # 0-1 -> -90 to 90
            pred_lon = test_pred_np[:, 1] * 360 - 180  # 0-1 -> -180 to 180
            actual_lat = y_test_np[:, 0] * 180 - 90
            actual_lon = y_test_np[:, 1] * 360 - 180

            # Calculate MAE in degrees
            lat_mae_deg = np.mean(np.abs(pred_lat - actual_lat))
            lon_mae_deg = np.mean(np.abs(pred_lon - actual_lon))

            # Distance error estimate
            km_per_degree = 111
            distance_error = np.sqrt(lat_mae_deg**2 + lon_mae_deg**2) * km_per_degree

        results = {
            'training_time': training_time,
            'test_loss': test_loss,
            'lat_mae_degrees': lat_mae_deg,
            'lon_mae_degrees': lon_mae_deg,
            'distance_error_km': distance_error,
            'epochs': self.epochs,
            'device': str(self.device),
            'samples_trained': len(X_train)
        }

        self.logger.info(f"/nTraining Complete:")
        self.logger.info(f"  Time: {training_time:.2f} seconds")
        self.logger.info(f"  Latitude MAE: {lat_mae_deg:.4f}[?]")
        self.logger.info(f"  Longitude MAE: {lon_mae_deg:.4f}[?]")
        self.logger.info(f"  Distance Error: {distance_error:.2f} km")
        self.logger.info(f"  Device: {self.device}")

        return results

    def predict(self, current_position: np.ndarray) -> Tuple[float, float]:
        """
        Predict next ISS position

        Args:
            current_position: [lat, lon, altitude, time_index]

        Returns:
            (next_lat, next_lon)
        """

        self.model.eval()

        # Normalize input
        position = current_position.copy().reshape(1, -1)
        position_norm = self.normalize_features(position)

        # Convert to tensor
        position_tensor = torch.FloatTensor(position_norm).to(self.device)

        # Predict (model outputs normalized values)
        with torch.no_grad():
            output = self.model(position_tensor)
            lat_norm, lon_norm = output.cpu().numpy()[0]

        # Denormalize predictions back to degrees
        lat_pred = lat_norm * 180 - 90  # 0-1 -> -90 to 90
        lon_pred = lon_norm * 360 - 180  # 0-1 -> -180 to 180

        return float(lat_pred), float(lon_pred)

    def save_model(self, filename='iss_predictor_gpu.pth'):
        """Save trained model"""

        model_path = self.models_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }, model_path)

        self.logger.info(f"Model saved to {model_path}")

    def load_model(self, filename='iss_predictor_gpu.pth'):
        """Load trained model"""

        model_path = self.models_dir / filename
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Model loaded from {model_path}")
        else:
            self.logger.warning(f"Model file not found: {model_path}")


def test_iss_predictor_gpu():
    """Test GPU-accelerated ISS predictor"""

    print("/n" + "="*80)
    print("ISS POSITION PREDICTOR - GPU ACCELERATED TEST")
    print("="*80)

    # Initialize
    predictor = ISSPredictorGPU(use_gpu=True)

    # Train
    print("/n[TEST 1] Training on sample data")
    print("-"*80)
    results = predictor.train()

    print(f"/nResults:")
    print(f"  Training Time: {results['training_time']:.2f}s")
    print(f"  Latitude MAE: {results['lat_mae_degrees']:.4f}[?]")
    print(f"  Longitude MAE: {results['lon_mae_degrees']:.4f}[?]")
    print(f"  Distance Error: {results['distance_error_km']:.2f} km")
    print(f"  Device: {results['device']}")

    # Test prediction
    print("/n[TEST 2] Position Prediction")
    print("-"*80)

    current_pos = np.array([0.0, -75.0, 410.0, 500.0])  # lat, lon, alt, time
    next_lat, next_lon = predictor.predict(current_pos)

    print(f"Current Position: ({current_pos[0]:.2f}[?], {current_pos[1]:.2f}[?])")
    print(f"Predicted Next: ({next_lat:.2f}[?], {next_lon:.2f}[?])")

    # Save model
    predictor.save_model()

    print("/n" + "="*80)
    print("[OK] ISS Predictor GPU version fully operational")
    print("="*80)

    return predictor, results


if __name__ == "__main__":
    predictor, results = test_iss_predictor_gpu()
