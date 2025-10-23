# Orbital Tracking ML

ISS position prediction and satellite tracking with machine learning.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SGP4](https://img.shields.io/badge/SGP4-Orbital_Propagation-blue)](https://pypi.org/project/sgp4/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-blue)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)

## Features

- ISS position prediction (GPU-accelerated)
- SGP4 orbital propagation
- TLE data processing
- Satellite tracking GUI
- ML model training for orbit prediction

## Installation

```bash
git clone https://github.com/green8-dot/orbital-tracking-ml
cd orbital-tracking-ml
pip install -r requirements.txt
```

## Usage

### Train ISS Predictor

```bash
python src/train_iss_predictor.py
```

### Generate Positions from TLE

```bash
python src/generate_positions_from_tle.py
```

### Run Tracking GUI

```bash
python src/orbitscope_tracker_gui.py
```

### SGP4 Orbital Propagation

```bash
python src/sgp4_framework.py
```

## Components

- `iss_predictor_gpu.py` - GPU-accelerated ISS position prediction
- `train_iss_predictor.py` - ML model training
- `sgp4_framework.py` - SGP4 orbital propagation implementation
- `generate_positions_from_tle.py` - TLE to position conversion
- `orbitscope_tracker_gui.py` - Satellite tracking interface

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for GPU acceleration)
- Dependencies in requirements.txt

## Data Sources

- TLE data: Space-Track.org, CelesTrak
- ISS tracking data: NASA APIs

## License

MIT License

Copyright (c) 2025 Derek Green

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

See [LICENSE](LICENSE) file for full details.

## Author

Derek Green
- GitHub: [@green8-dot](https://github.com/green8-dot)
- LinkedIn: [derek-green-44723323a](https://www.linkedin.com/in/derek-green-44723323a/)
