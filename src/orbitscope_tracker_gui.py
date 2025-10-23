"""
OrbitScope Tracker - GUI Wrapper
==================================
Real-time ISS position tracking and trajectory prediction

Author: Framework Baseline Team
Date: 2025-10-03
"""

import tkinter as tk
from tkinter import ttk
import sys
from pathlib import Path
import requests
from datetime import datetime, timedelta

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / "BC_Healthcare_Portfolio" / "personal_finance"))

try:
    from gui_framework import APIDashboardGUI, ColorScheme, MetricCard
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure gui_framework.py is in BC_Healthcare_Portfolio/personal_finance/")
    sys.exit(1)


class OrbitScopeTrackerGUI(APIDashboardGUI):
    """GUI Dashboard for OrbitScope ISS Tracking"""

    def __init__(self, api_url: str = "http://localhost:8002"):
        super().__init__("OrbitScope - ISS Tracker", api_url, 1000, 750)

        # Tracking data
        self.current_position = None
        self.trajectory_history = []

        # Build dashboard
        self.build_dashboard()

        # Check API health
        self.check_api_health()

        # Get initial position
        self.get_current_position()

        # Auto-refresh every 10 seconds
        self._refresh_interval = 10000  # 10 seconds
        self.auto_refresh(self.refresh_dashboard)

    def build_dashboard(self):
        """Build main dashboard layout"""
        # Current position section
        self.build_current_position_section()

        # Prediction section
        self.build_prediction_section()

        # Trajectory section
        self.build_trajectory_section()

        # Control buttons
        self.build_controls()

    def build_current_position_section(self):
        """Build current position section"""
        card = self.create_card(self.main_frame, "Current ISS Position")

        content = ttk.Frame(card, style='Card.TFrame')
        content.pack(fill=tk.X, padx=15, pady=10)

        self.position_frame = ttk.Frame(content, style='Card.TFrame')
        self.position_frame.pack(fill=tk.X, pady=10)

        ttk.Label(self.position_frame, text="Loading...",
                 style='CardLabel.TLabel').pack()

        # Refresh button
        self.create_button(content, "Refresh Position",
                          self.get_current_position).pack(pady=5)

    def build_prediction_section(self):
        """Build position prediction section"""
        card = self.create_card(self.main_frame, "Position Prediction")

        content = ttk.Frame(card, style='Card.TFrame')
        content.pack(fill=tk.X, padx=15, pady=10)

        # Time input
        inputs_frame = ttk.Frame(content, style='Card.TFrame')
        inputs_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(inputs_frame, text="Minutes from now:", style='CardLabel.TLabel',
                 width=20).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.minutes_var = tk.StringVar(value="30")
        ttk.Entry(inputs_frame, textvariable=self.minutes_var,
                 width=10).grid(row=0, column=1, padx=5, pady=5)

        # Predict button
        self.create_button(content, "Predict Future Position",
                          self.predict_position).pack(pady=10)

        # Prediction result
        self.prediction_frame = ttk.Frame(content, style='Card.TFrame')
        self.prediction_frame.pack(fill=tk.X, pady=10)

        ttk.Label(self.prediction_frame, text="No prediction yet",
                 style='CardLabel.TLabel').pack()

    def build_trajectory_section(self):
        """Build trajectory section"""
        card = self.create_card(self.main_frame, "Trajectory History")

        # Scrollable frame
        canvas = tk.Canvas(card, height=150, bg=ColorScheme.card_bg,
                          highlightthickness=0)
        scrollbar = ttk.Scrollbar(card, orient="vertical", command=canvas.yview)
        self.trajectory_frame = ttk.Frame(canvas, style='Card.TFrame')

        self.trajectory_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.trajectory_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def build_controls(self):
        """Build control buttons"""
        button_frame = ttk.Frame(self.main_frame, style='Main.TFrame')
        button_frame.pack(pady=20)

        self.create_button(button_frame, "Check API Health",
                          lambda: self.check_and_show_health()).pack(side=tk.LEFT, padx=5)
        self.create_button(button_frame, "Refresh All",
                          self.refresh_dashboard).pack(side=tk.LEFT, padx=5)
        self.create_button(button_frame, "Quit",
                          self.quit, "Danger").pack(side=tk.LEFT, padx=5)

    def get_current_position(self):
        """Get current ISS position from API"""
        try:
            self.update_status("Getting ISS position...")

            response = requests.get(
                f"{self.api_url}/api/v1/current-position",
                headers={"X-API-Key": "dev-key-12345"},
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                self.current_position = result
                self.display_current_position(result)
                self.update_status("Position updated")
            else:
                self.show_message("Error", f"API error: {response.status_code}", "error")

        except Exception as e:
            self.show_message("Error", f"Failed to get position: {str(e)}", "error")
            self.update_status("Position update failed")

    def display_current_position(self, result: dict):
        """Display current ISS position"""
        # Clear
        for widget in self.position_frame.winfo_children():
            widget.destroy()

        position = result.get('position', {})
        latitude = position.get('latitude', 0)
        longitude = position.get('longitude', 0)
        altitude = position.get('altitude_km', 0)
        velocity = position.get('velocity_kmph', 0)

        # Metrics
        metrics_frame = ttk.Frame(self.position_frame, style='Card.TFrame')
        metrics_frame.pack(fill=tk.X, pady=10)

        MetricCard(metrics_frame, "Latitude",
                  f"{latitude:.2f}", "[?]", "primary")

        MetricCard(metrics_frame, "Longitude",
                  f"{longitude:.2f}", "[?]", "primary")

        MetricCard(metrics_frame, "Altitude",
                  f"{altitude:.1f}", "km", "accent")

        MetricCard(metrics_frame, "Velocity",
                  f"{velocity:.0f}", "km/h", "success")

        # Timestamp
        timestamp = result.get('timestamp', 'Unknown')
        ttk.Label(self.position_frame,
                 text=f"Updated: {timestamp}",
                 style='CardLabel.TLabel').pack(pady=5)

    def predict_position(self):
        """Predict future ISS position"""
        try:
            minutes = int(self.minutes_var.get())

            if not self.current_position:
                self.show_message("Error", "Get current position first", "error")
                return

            # Calculate target timestamp
            target_time = datetime.now() + timedelta(minutes=minutes)

            data = {
                "target_timestamp": target_time.isoformat(),
                "reference_position": self.current_position.get('position', {})
            }

            self.update_status(f"Predicting position in {minutes} minutes...")

            response = requests.post(
                f"{self.api_url}/api/v1/predict/position",
                json=data,
                headers={"X-API-Key": "dev-key-12345"},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                self.display_prediction(result, minutes)
                self.add_to_trajectory(result, minutes)
                self.update_status("Prediction complete")
            else:
                self.show_message("Error", f"API error: {response.status_code}", "error")

        except ValueError:
            self.show_message("Input Error", "Please enter valid minutes", "error")
        except Exception as e:
            self.show_message("Error", str(e), "error")

    def display_prediction(self, result: dict, minutes: int):
        """Display position prediction"""
        # Clear
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()

        prediction = result.get('prediction', {})
        latitude = prediction.get('predicted_latitude', 0)
        longitude = prediction.get('predicted_longitude', 0)
        confidence = prediction.get('confidence', 0)

        # Title
        ttk.Label(self.prediction_frame,
                 text=f"Position in {minutes} minutes:",
                 style='CardTitle.TLabel').pack(anchor=tk.W, pady=(0, 10))

        # Metrics
        metrics_frame = ttk.Frame(self.prediction_frame, style='Card.TFrame')
        metrics_frame.pack(fill=tk.X, pady=10)

        MetricCard(metrics_frame, "Predicted Lat",
                  f"{latitude:.2f}", "[?]", "accent")

        MetricCard(metrics_frame, "Predicted Lon",
                  f"{longitude:.2f}", "[?]", "accent")

        MetricCard(metrics_frame, "Confidence",
                  f"{confidence*100:.0f}", "%",
                  "success" if confidence > 0.9 else "warning")

    def add_to_trajectory(self, result: dict, minutes: int):
        """Add prediction to trajectory history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction = result.get('prediction', {})

        entry = {
            'timestamp': timestamp,
            'minutes_ahead': minutes,
            'latitude': prediction.get('predicted_latitude', 0),
            'longitude': prediction.get('predicted_longitude', 0)
        }

        self.trajectory_history.insert(0, entry)
        self.trajectory_history = self.trajectory_history[:10]  # Keep last 10

        self.refresh_trajectory_display()

    def refresh_trajectory_display(self):
        """Refresh trajectory history display"""
        # Clear
        for widget in self.trajectory_frame.winfo_children():
            widget.destroy()

        # Header
        header = ttk.Frame(self.trajectory_frame, style='Card.TFrame')
        header.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(header, text="Time", style='CardLabel.TLabel',
                 width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(header, text="Ahead (min)", style='CardLabel.TLabel',
                 width=12).pack(side=tk.LEFT, padx=5)
        ttk.Label(header, text="Latitude", style='CardLabel.TLabel',
                 width=12).pack(side=tk.LEFT, padx=5)
        ttk.Label(header, text="Longitude", style='CardLabel.TLabel',
                 width=12).pack(side=tk.LEFT, padx=5)

        # History rows
        for entry in self.trajectory_history:
            row = ttk.Frame(self.trajectory_frame, style='Card.TFrame')
            row.pack(fill=tk.X, pady=2)

            ttk.Label(row, text=entry['timestamp'], style='CardTitle.TLabel',
                     width=20).pack(side=tk.LEFT, padx=5)
            ttk.Label(row, text=str(entry['minutes_ahead']),
                     style='CardTitle.TLabel', width=12).pack(side=tk.LEFT, padx=5)
            ttk.Label(row, text=f"{entry['latitude']:.2f}[?]",
                     style='CardLabel.TLabel', width=12).pack(side=tk.LEFT, padx=5)
            ttk.Label(row, text=f"{entry['longitude']:.2f}[?]",
                     style='CardLabel.TLabel', width=12).pack(side=tk.LEFT, padx=5)

    def check_and_show_health(self):
        """Check and show API health"""
        is_healthy = self.check_api_health()
        status = "Healthy" if is_healthy else "Unavailable"
        msg_type = "info" if is_healthy else "error"
        self.show_message("API Status", f"API is {status}", msg_type)

    def refresh_dashboard(self):
        """Refresh dashboard data"""
        self.check_api_health()
        self.get_current_position()
        self.update_status("Dashboard refreshed")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    app = OrbitScopeTrackerGUI()
    app.run()
