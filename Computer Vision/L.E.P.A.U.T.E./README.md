# Lie Equivariant Perception Algebraic Unified Transform Embedding Framework (L.E.P.A.U.T.E. Framework)

A Python package for processing webcam images with a Lie group-based Transformer model and accessing the resulting data.

## Installation

### Install from The Python Package Index (PyPI) (currently unavailable)

```bash
pip3 install lepaute

```

### Install from source (recommended)

1. Make the installation script executable:

```bash
chmod +x install_dependencies.sh

```

2. Run the script to initialize the local isolation build environment:

```bash
./install_dependencies.sh

```

### Check System Requirements

1. Make the verification script executable:

```bash
chmod +x check_system_requirements.sh

```

2. Run the script to verify the environment baseline:

```bash
./check_system_requirements.sh

```

## Usage

### Using `example_pip.py`

Run the local pipeline subsystem verification:

```bash
python3 example_pip.py

```

* Executes the pipeline in mock mode, saves JSON telemetry, and prints verified 3D SE(3) transform parameters alongside detected object IDs.

### Using `example.py`

Run the full training validation:

```bash
python3 example.py

```

* Initiates structural telemetry collection (CI/CD Mock Mode), constructs an `EquivariantDataset` from continuous frame transitions, and executes a multi-task spatial contrastive training loop.

### Execution Modes

1. **GUI/Realtime Mode**:
```bash
python main.py

```

* Displays the camera feed (or synthetic industrial mock feed if hardware is unavailable).
* Shows detected objects, confidence scores, SE(3) displacement, and RMSE directly on the UI.
* Automatically saves telemetry to `lepaute_data.json` and images to the `frames` directory.

2. **JSON/Headless Mode**:
```python
from main import run_pipeline
run_pipeline(display_mode="json", unlimited=False, save_json=True)

```


* Runs inference without rendering a CV2 window and saves structured data directly to the disk.


3. **Custom Configurations**:
```python
from module import LepauteConfig
from main import run_pipeline

custom_cfg = LepauteConfig(device="cuda", dl_inference_freq=2)
run_pipeline(config=custom_cfg)

```

## Requirements

* `torch>=2.3.0`
* `torchvision>=0.18.0`
* `transformers>=4.40.0`
* `opencv-python>=4.8.0`
* `numpy>=1.24.0`
* `pytorch-metric-learning>=2.0.0`
* `pillow>=10.0.0`
* `timm>=0.9.0`
* `pydantic>=2.7.0`
* `pydantic-settings>=2.2.0`

## Additional Contributors

### PyCon HK 2025

* Primary contributor: [shz2](https://twitter.com/shivvor2)
* Special thanks to: [BenBenCHAK](https://github.com/BenBenCHAK), [usertam](https://github.com/usertam)

## Notes

* **Hardware Failover**: Ensure webcam access for real-time data collection. If the camera feed is lost or unavailable, `CameraIOStream` will automatically fall back to a synthetic mock generator to keep the pipeline running.
* **Environment Variables**: Advanced configurations (intrinsic camera parameters, target object categories, maximum disk files) can be configured safely via environment variables prefixed with `LEPAUTE_` (e.g., `LEPAUTE_DEVICE="mps"`).
* **Debugging**: Debug logs can be enabled by setting `logging.basicConfig(level=logging.DEBUG)` in your execution script.
* **Continuous Execution**: The `DiskManager` automatically limits stored frames to the most recent 2,000 files (configurable) to prevent memory saturation during prolonged deployments.

