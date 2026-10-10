# Usage Guide

## 1. Overview

L.E.P.A.U.T.E. provides two deployment paths that share the same C geometry and photometric optimization core.

- **Standard version** (`main/`): Python pipeline with dense direct tracking, residual refinement network, object classification, and trajectory fusion. Suitable for research and desktop use.
- **MCU version** (`mcu/`): Pure-C thin wrapper around the shared core, designed for static-memory embedded targets.

Both versions call the identical SE(3) and Levenberg-Marquardt kernels. The standard version loads the shared library through ctypes; the MCU version links it at compile time under the `LEPAUTE_STATIC_MEM` flag.

## 2. Install Requirements

### Shared C Core

Build the core library once; both front-ends consume it.

```bash
cd core
mkdir -p build && cd build
cmake ..
make
```

The resulting shared object (`liblepaute_core.so` / `.dylib` / `.dll`) is expected under `core/build/` or `core/`. The Python geometry module searches these locations automatically.

### Standard Version (Python)

```bash
# Recommended Python 3.10+
pip install torch torchvision
pip install opencv-python numpy pydantic pydantic-settings
pip install albumentations ultralytics safetensors tqdm
```

Optional but recommended for accelerated tracking:

- CUDA-capable GPU or Apple Silicon (MPS)
- Compiled `liblepaute_core` (see above)

Place a trained residual-refiner checkpoint at `main/checkpoints/best_model.pth` if residual refinement is required. YOLO weights (`yolov8n.pt`) are loaded automatically by the classification module.

### MCU Version (C)

```bash
cd mcu
mkdir -p build && cd build
cmake -DLEPAUTE_STATIC_MEM=ON -DCMAKE_BUILD_TYPE=Release ..
make
```

The MCU build requires only a C99 compiler, `libm`, and the shared core sources. No dynamic allocation is performed when `LEPAUTE_STATIC_MEM` is defined.

## Parameters That Must Be Set Manually

The following values are not inferred at runtime and should be adjusted for each camera or target platform.

### Camera Intrinsics

| Parameter | Location | Description |
|---|---|---|
| `fx`, `fy` | `camera_config.json` or `LepauteConfig` / `lepaute_cfg_t.K` | Focal lengths in pixels |
| `cx`, `cy` | same | Principal-point offsets |

If `camera_config.json` is absent, the framework falls back to defaults (`fx=fy=250`, `cx=160`, `cy=120`). For production use, supply calibrated values matching the actual sensor resolution.

### Object Scale Priors

| Parameter | Location | Description |
|---|---|---|
| `object_scales` | `object_config.json` or `LepauteConfig.object_scales` | Metric depth priors (metres) keyed by object class |

Monocular tracking uses these priors to resolve absolute scale. Incorrect values produce systematic translation bias.

### Photometric Optimizer Settings

| Parameter | Default (standard) | Default (MCU) | Notes |
|---|---|---|---|
| `num_levels` / `pyramid_levels` | 3 | 2–3 | Higher levels improve convergence but increase cost |
| `max_iters_per_level` / `gn_max_iter` | 15 | 5–6 | Reduce on constrained devices |
| `huber_delta` | 10.0 | 10.0 | Robust loss threshold |
| `min_grad_thresh` | 25.0 (Python) / 5.0 (C) | 4.0–5.0 | Lower for low-texture scenes |
| `initial_lm_lambda` | 1e-3 | 1e-3 | LM damping start value |
| `scale_prior` | per-object | 1.0–1.2 | Constant depth used when no depth map is supplied |

### Static-Memory Limits (MCU only)

| Macro | Default | Purpose |
|---|---|---|
| `LEPAUTE_MAX_PYRAMID_LEVELS` | 8 | Maximum pyramid depth |
| `LEPAUTE_MAX_IMAGE_PIXELS` | 320×240 | Maximum pixels per image in the static pool |

Increase or decrease these macros according to available RAM before compiling the MCU binary.

### Performance Mode (standard only)

| Mode | Effect |
|---|---|
| `low` | Fewer pyramid levels, reduced iterations, frame-rate throttle |
| `medium` | Default balance |
| `high` | Extra pyramid level and more iterations |

Set via `--perf low|medium|high` or by assigning `LepauteConfig.performance_mode`.

## Standard Version Usage

### Running the Live Pipeline

```bash
cd main
python main.py --mode realtime --perf medium
```

Common flags:

| Flag | Description |
|---|---|
| `--mode headless\|gui\|realtime\|json\|detailedgui` | Output style |
| `--perf low\|medium\|high` | Performance profile |
| `--db PATH` | SQLite database path for transition storage |
| `--limit` | Restrict run to 50 frames |
| `--no_save` | Disable database writes |
| `--log_level general\|detailed` | Logging verbosity |

### Benchmarking

```bash
python benchmark.py -d both -p high -db lepaute_data.db -s 1.5 -n 200
```

### Unit Tests

```bash
python -m unittest test_module.py
```

## MCU Version Usage

### Demo Binary

After building:

```bash
./lepaute_mcu_demo
```

The supplied `main.c` generates a synthetic image pair, runs photometric refinement, and prints the recovered tangent vector and cost.

### Integrating into Application Code

```c
#include "lepaute.h"

lepaute_cfg_t cfg;
lepaute_default_cfg(&cfg);

/* Override for actual sensor */
cfg.K.fx = 320.0f;
cfg.K.fy = 320.0f;
cfg.K.cx = 160.0f;
cfg.K.cy = 120.0f;
cfg.scale_prior = 1.0f;
cfg.gn.num_levels = 2;
cfg.gn.max_iters_per_level = 5;

lepaute_se3_t T;
lepaute_se3_identity(&T);

double cost = lepaute_refine(ref_gray, cur_gray, width, height, &cfg, &T);

double xi[6];
lepaute_xi_from_se3(&T, xi);
```

`ref_gray` and `cur_gray` must be contiguous uint8 grayscale buffers of size `width * height`. The pose matrix `T` is updated in-place.

## 3. Troubleshooting

### C Library Not Found (Python)

Symptom: log message “Native library not found – using pure PyTorch fallback”.

- Confirm `liblepaute_core` exists under `core/build/` or `core/`.
- Rebuild the core with the correct platform extension (`.so`, `.dylib`, `.dll`).
- Verify the search paths listed in `geometry.py` match the actual location.

### High Final Cost / Tracking Failure

- Lower `min_grad_thresh` for low-texture environments.
- Increase pyramid levels or iterations if motion is large.
- Ensure camera intrinsics match the image resolution.
- Supply a realistic `scale_prior` (or object-scale entry) for the scene.

### MCU Static-Memory Errors

- Return codes `-2` / `-3` from pyramid creation indicate the static pool is exhausted or already occupied.
- Reduce image resolution or lower `LEPAUTE_MAX_IMAGE_PIXELS` / `LEPAUTE_MAX_PYRAMID_LEVELS` and recompile.
- Ensure only one pair of pyramids is live at a time (the pool supports two concurrent slots).

### YOLO or Refiner Initialization Failure

- Confirm `yolov8n.pt` is reachable and network access is available on first download.
- Place a valid checkpoint at `main/checkpoints/best_model.pth`; absence falls back to random weights.
- On Apple Silicon, ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set (handled automatically by the config loader).

### Process Hang on Shutdown

- The pipeline uses a multi-stage signal handler. Press Ctrl+C once for graceful exit; repeated presses force `os._exit` if a native extension is blocked.
- Verify the InferenceWorker heartbeat; a stalled worker triggers an automatic halt after the health-check interval.

### Database Lock or Corruption

- SequenceDataCollector uses SQLite WAL mode. On abnormal termination, run a manual checkpoint:
  ```sql
  PRAGMA wal_checkpoint(TRUNCATE);
  ```
- Avoid concurrent writers from external processes while the collector is active.
