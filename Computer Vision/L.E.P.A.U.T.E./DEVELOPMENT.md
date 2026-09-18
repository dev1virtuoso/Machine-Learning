# Development and Integration Guide

## 1. Overview

This document provides high-level architectural guidance for integrating the Lie Equivariant Perception Algebraic Unified Transform Embedding (L.E.P.A.U.T.E.) Framework into external developer systems. It covers environment orchestration, pipeline execution, and programmatic integration strategies for embedding the SE(3) tracking and classification subsystems into custom robotics or spatial perception environments.

The framework exposes a configuration-driven API (`LepauteConfig`) and a primary entry point (`run_pipeline`). All heavy inference is isolated in a background process (`InferenceWorker`) so that the main tracking loop remains non-blocking. Developers should interact exclusively through the public interfaces rather than modifying internal worker queues or state locks.

## System Integration Tutorial

Integrating the framework into an existing architecture requires interacting with the decoupled multi-process components through the exposed configuration API, rather than relying solely on the CLI wrapper.

### Programmatic Execution

External systems can directly invoke the perception pipeline using the `run_pipeline` interface and capture the returned structured payloads for downstream logic.

```python
from pipeline_and_config import LepauteConfig, DisplayMode
from main import run_pipeline

custom_config = LepauteConfig(
    device="cuda:0",
    fx=600.0,
    fy=600.0,
    cx=320.0,
    cy=240.0,
    object_names=["industrial_arm", "conveyor_belt", "target_widget"]
)

telemetry_results = run_pipeline(
    config=custom_config,
    display_mode=DisplayMode.HEADLESS,
    unlimited=False,
    save_json=False
)

for payload in telemetry_results:
    print(f"Frame {payload['frame_id']} | Object: {payload['category']} | Pose: {payload['xi']}")
```

Returned payloads contain at minimum:
- `frame_id`
- `category` (detected object label)
- `xi` (global SE(3) pose in tangent-space coordinates)
- `tracking_score`
- `fusion_mode` (`Refined`, `Tracker`, or `Recovery`)

### Custom Data Ingestion Pipelines

To route custom video streams (ROS Image topics, RTSP streams, or synthetic simulation buffers) into the system, subclass and override the `CameraIOStream.read()` method. The override must:

1. Return a boolean success flag.
2. Provide a frame as a standard `(H, W, 3)` uint8 RGB NumPy array.
3. Inject accurate metadata containing at least `timestamp` (float seconds) and `frame_id` (integer).

```python
from pipeline_and_config import CameraIOStream, LepauteConfig
import numpy as np
import time

class ROSImageStream(CameraIOStream):
    def __init__(self, config: LepauteConfig, topic_buffer):
        super().__init__(config, mock=True)  # bypass physical camera
        self.buffer = topic_buffer
        self._frame_id = 0

    def read(self):
        self._frame_id += 1
        frame = self.buffer.get_latest()  # must be RGB uint8
        if frame is None:
            return False, np.zeros(0), {}
        meta = {
            "timestamp": time.time(),
            "frame_id": self._frame_id
        }
        return True, frame, meta
```

Pass the custom stream instance into a modified initialization path or replace the stream construction inside `init_components` when embedding the pipeline.

### Configuration Parameters That Must Be Set Manually

Several values are hardware- or scene-specific and should be configured before production deployment.

**Camera Intrinsics**  
Loaded from `camera_config.json` (or the environment variable `LEPAUTE_CAMERA_CONFIG_PATH`). Fallback defaults are:

```json
{
  "fx": 250.0,
  "fy": 250.0,
  "cx": 160.0,
  "cy": 120.0
}
```

Override at construction:

```python
config = LepauteConfig(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
```

Incorrect intrinsics produce systematic scale and pose drift.

**Object Scale Priors**  
Metric scale priors (meters) are loaded from `object_config.json` (or `LEPAUTE_OBJECT_CONFIG_PATH`). The classifier label is mapped directly to these priors for monocular depth projection. Add or edit entries to match the objects present in the target environment.

**Compute Device**  
Automatically detected (`cuda` > `mps` > `cpu`). Force a specific device via `LepauteConfig(device=...)` or the CLI `--device` flag.

**Performance Profile**  
`PerformanceMode.LOW`, `MEDIUM`, or `HIGH` adjusts pyramid levels, Gauss-Newton iteration count, and optional frame-rate throttling. Select according to available compute and required latency.

**Model Checkpoint**  
The inference worker loads `./checkpoints/best_model.pth` by default. Ensure a trained checkpoint exists at this location; otherwise the refiner falls back to random initialization.

**SQLite Database Path**  
Defaults to `lepaute_data.db`. Override with `LepauteConfig(data_store=...)` or the CLI `--db` flag.

### Dataset Preparation and Offline Training

For systems that require custom training of the SE(3) Residual Refiner:

Obtain the dataset. A pre-converted LEPAUTE dataset is available at: [https://huggingface.co/datasets/dev1virtuoso/lepaute-dataset](dev1virtuoso/lepaute-dataset)

Alternatively, download and extract the YCB-V BOP dataset, then convert it:

```bash
python convert_bop_to_lepaute.py \
  --bop_dir ./dataset/bop_datasets/ycbv \
  --output_dir ./dataset/lepaute_dataset \
  --splits train_pbr train_real \
  --stride 1 \
  --workers 8
```

3. Launch training:

```bash
python train.py \
  --dataset_dir ./dataset/lepaute_dataset \
  --checkpoint_dir ./checkpoints \
  --epochs 50 \
  --seed 42 \
  --device cuda
```

The trainer supports resume via `--resume_mode resume|scratch|ask` and automatically handles train/test splits when the corresponding subdirectories exist.

### Benchmarking and CI Stress Testing (`benchmark.py`)

The framework includes a dedicated benchmark suite (`benchmark.py`) for deterministic stress testing, geometric stability validation, latency profiling, and CI integration. It evaluates the `SE3ResidualRefiner` under controlled image perturbations and reports mean geometric MSE, mode arbitration statistics, and per-module latency.

#### High-Level Workflow

The benchmark executes five sequential stages:

1. **Deterministic Environment Setup** – Locks random seeds, resolves the target device (CPU / CUDA / MPS), and enforces reproducibility flags.
2. **SE(3) Boundary Condition Tests** – Verifies numerical stability of `se3_exp_map` / `se3_log_map` near \(\pi\) rotations and sub-pixel micro-angles.
3. **Data Ingestion & Perturbation** – Loads transition frames from the SQLite database and applies a severity-controlled Albumentations pipeline (Gaussian noise, motion blur, brightness/contrast).
4. **Inference & Stress Loop** – Runs the residual refiner on every frame, records MSE against ground-truth Lie parameters, classifies fusion mode (`refined` / `tracker` / `recovery`), and profiles latency.
5. **Report Generation** – Produces a JSON report containing model SHA-256 hash, environment metadata, aggregated metrics, and latency statistics. Suitable for CI artifact upload.

#### Command-Line Usage

```bash
python benchmark.py [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-d`, `--device` | `str` | `both` | Compute target: `cpu`, `gpu` (auto-detect CUDA/MPS), `cuda`, `mps`, or `both` (runs CPU then GPU sequentially). |
| `-p`, `--perf` | `str` | `high` | `LepauteConfig` performance profile: `low`, `medium`, or `high`. |
| `-db`, `--db` | `str` | `lepaute_data.db` | Path to the SQLite sequence transition database. |
| `-s`, `--severity` | `float` | `1.5` | Perturbation degradation severity factor (controls noise std and motion-blur kernel size). |
| `-o`, `--output` | `str` | `benchmark_report.json` | Output path for the generated JSON report. When `--device both` is used, device suffixes are automatically appended (`_cpu`, `_gpu`). |
| `-m`, `--model-path` | `str` | `lepaute_refiner.onnx` | Path to the model artifact whose SHA-256 hash will be bound into the report. |
| `-n`, `--limit` | `int` | `None` | Maximum number of frames to process (useful for quick smoke tests). |
| `--seed` | `int` | `42` | Deterministic random seed lock for reproducibility. |

#### Example Invocations

```bash
# Full two-stage (CPU + GPU) stress test with default settings
python benchmark.py

# CPU-only run with reduced severity and frame limit
python benchmark.py -d cpu -s 0.8 -n 200 -o cpu_smoke.json

# High-severity GPU benchmark using a custom database and model
python benchmark.py \
  -d gpu \
  -p high \
  -db ./datasets/stress_sequences.db \
  -s 2.0 \
  -m ./checkpoints/best_model.pth \
  -o gpu_stress_report.json
```

#### Report Contents

The generated JSON contains:

- `ci_timestamp` – Unix timestamp of the run.
- `model_version_hash` – SHA-256 of the supplied model artifact (or `"N/A"`).
- `environment` – Selected device and performance mode.
- `metrics` – Mean geometric MSE and boundary-test results (`pi_angle_mse`, `tiny_angle_mse`).
- `latency_profiling_ms` – Per-module mean and max latency (currently focused on `SE3ResidualRefiner_Inference`).
- `mode_arbitration` – Counts of frames classified as deep-learning refined, direct-alignment tracker, or ORB kinematic recovery.

#### Programmatic Usage

```python
from pipeline_and_config import LepauteConfig, PerformanceMode
from models import SE3ResidualRefiner
from benchmark import LepauteBenchmark

config = LepauteConfig(performance_mode=PerformanceMode.HIGH)
refiner = SE3ResidualRefiner(config, feature_dim=256, max_resolution=64)
benchmark = LepauteBenchmark(config)

# Single-device pass
report = benchmark.run_single_pass(
    refiner_model=refiner,
    target_device="cuda",
    args=...  # argparse.Namespace or equivalent
)
```

The suite automatically creates an empty SQLite schema if the supplied database does not exist, ensuring the benchmark can be run safely in fresh CI environments.

### Embedding Individual Subsystems

Developers may also import and use isolated components:

```python
from geometry import se3_exp_map, se3_log_map, compose_poses
from vision_tracking import MonocularDirectTracker, YOLOClassifier, ManifoldKinematicForecaster
from models import SE3ResidualRefiner
from pipeline_and_config import LepauteConfig

config = LepauteConfig(device="cuda")
tracker = MonocularDirectTracker(config)
classifier = YOLOClassifier(config)
refiner = SE3ResidualRefiner(config)
forecaster = ManifoldKinematicForecaster()
```

When constructing custom tracking loops outside `main.py`, always detach intermediate SE(3) tensors (`.detach().cpu().numpy()`) before publishing or storing them.

## High-Level Troubleshooting

**Thread Blocking in Asynchronous Architectures**  
Avoid polling `get_latest_resolved_state()` with high-frequency blocking calls when integrating `InferenceWorker` into a broader async framework such as `asyncio` or ROS spin loops. Ensure the external main loop pulls this state asynchronously relative to the worker queue to prevent UI locking or telemetry stream starvation.

**VRAM Leaks in Custom Execution Loops**  
If constructing custom tracking loops outside the provided `main.py` entry point, ensure that intermediate SE(3) tensors generated by `se3_exp_map` and `se3_log_map` are detached (`.detach().cpu().numpy()`) before being appended to external publishers or lists. Failure to detach keeps the entire backpropagation computational graph alive in memory, rapidly saturating GPU VRAM.

**Concurrency Collisions on Metal Performance Shaders (MPS)**  
When deploying on Apple Silicon, utilizing the isolated `InferenceWorker` process alongside heavy main-thread operations can occasionally cause Metal command-buffer crashes. If this occurs, rely on `LepauteConfig`'s built-in fallback to `cpu` routing, or enforce the provided `_mps_lock` / `mps_safe` context manager around custom model inference blocks when forcing MPS execution.

**Camera Acquisition Failures**  
The capture thread attempts multiple platform-specific backends (AVFoundation on macOS, DSHOW/MSMF on Windows, V4L2 on Linux). Persistent connection failures usually indicate device permission issues or exclusive locks held by another process. For offline testing, instantiate `CameraIOStream` with `mock=True`.

**Empty or Stalled Job Queue**  
Under sustained high load the inference worker silently drops frames when the bounded queue (size 5) is full. Reduce input frame rate or lower the performance profile if drop counts become excessive.

**Checkpoint Loading Shape Mismatches**  
When resuming training or loading a refined model, `load_compiled_state_dict` strips `_orig_mod.` prefixes and tolerates shape mismatches by falling back to fresh initialization for incompatible layers. Verify that the checkpoint was produced with matching `feature_dim` and `max_resolution` settings.
