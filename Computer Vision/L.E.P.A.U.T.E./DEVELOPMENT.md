# Development and Integration Guide

## 1. Overview

L.E.P.A.U.T.E. provides a shared C core (`core/`) that implements SE(3) Lie-group primitives and photometric Levenberg-Marquardt pose refinement. Two consumer layers sit on top of this core:

- **Standard (Python) layer** (`main/`): full monocular pipeline with PyTorch residual refinement, YOLO classification, asynchronous inference worker, SQLite persistence and optional GUI.
- **MCU (pure-C) layer** (`mcu/`): minimal static-memory wrapper that calls the same core under the `LEPAUTE_STATIC_MEM` compile flag.

Both layers share identical mathematical behaviour for the photometric solver. Developers who need to embed the framework into an existing system only have to link against the core (or call the thin MCU API) and supply the required configuration values listed below.

## 2. Parameters That Must Be Set Manually

The following values are not auto-detected and must be supplied by the integrator.

### Shared core / MCU (`lepaute_cfg_t` / `lepaute_gn_config_t`)

| Parameter | Type | Typical range | Description |
|----|---|---|---|
| `K.fx`, `K.fy` | float | camera focal length | Horizontal / vertical focal length in pixels |
| `K.cx`, `K.cy` | float | image centre | Principal point |
| `gn.num_levels` | int | 2–4 | Pyramid levels (lower = faster, less accurate) |
| `gn.max_iters_per_level` | int | 4–10 | LM iterations per level |
| `gn.huber_delta` | double | 5–15 | Huber robust threshold |
| `gn.min_grad_thresh` | double | 4–25 | Minimum squared gradient magnitude to accept a pixel |
| `scale_prior` | float | object size in metres | Constant depth prior used when no depth map is supplied |

### Standard Python layer (`LepauteConfig`)

| Parameter | Source | Notes |
|---|---|---|
| `fx/fy/cx/cy` | `camera_config.json` or environment `LEPAUTE_CAMERA_CONFIG_PATH` | Falls back to 250/250/160/120 |
| `object_scales` | `object_config.json` or environment `LEPAUTE_OBJECT_CONFIG_PATH` | Metric scale priors per class |
| `pyramid_levels` / `gn_max_iter` | constructor or `--perf` flag | Overridden by LOW/MEDIUM/HIGH profiles |
| `device` | auto-detected (CUDA → MPS → CPU) | Can be forced via environment |
| `data_store` | path to SQLite file | Default `lepaute_data.db` |

Any of these values that are left at the compiled-in defaults will produce incorrect metric scale or poor convergence on a real camera.

## 3. High-Level Integration Tutorial

### 3.1 Linking the shared C core

```cmake
# In your project's CMakeLists.txt
add_subdirectory(path/to/lepaute/core)
target_link_libraries(your_target PRIVATE lepaute_core)
target_include_directories(your_target PRIVATE
    path/to/lepaute/core/include)
```

For a pure MCU build that must not use the heap:

```cmake
target_compile_definitions(your_target PRIVATE LEPAUTE_STATIC_MEM)
# Optionally tighten the static pool size
target_compile_definitions(your_target PRIVATE
    LEPAUTE_MAX_IMAGE_PIXELS=19200)   # e.g. 160×120
```

### 3.2 Calling the MCU API from C/C++

```c
#include "lepaute.h"

lepaute_cfg_t cfg;
lepaute_default_cfg(&cfg);

/* Override with your camera intrinsics and desired depth prior */
cfg.K.fx = 320.0f;
cfg.K.fy = 320.0f;
cfg.K.cx = 160.0f;
cfg.K.cy = 120.0f;
cfg.scale_prior = 1.5f;          /* metres */

cfg.gn.num_levels          = 2;
cfg.gn.max_iters_per_level = 6;
cfg.gn.min_grad_thresh     = 5.0;

lepaute_se3_t T;
lepaute_se3_identity(&T);

double cost = lepaute_refine(ref_gray, cur_gray, width, height, &cfg, &T);

double xi[6];
lepaute_xi_from_se3(&T, xi);
/* xi = [tx, ty, tz, wx, wy, wz] */
```

The function returns the final average photometric cost. Values below approximately 80 indicate good alignment for typical 8-bit imagery.

### 3.3 Calling the core from Python (standard layer)

```python
from geometry import lm_refine_pose_pyramid, has_c_optimization
import numpy as np

assert has_c_optimization(), "C core not loaded"

xi, score = lm_refine_pose_pyramid(
    ref_gray, cur_gray,
    fx=320.0, fy=320.0, cx=160.0, cy=120.0,
    num_levels=3,
    max_iters=10,
    scale_prior=1.5,
    min_grad_thresh=5.0
)
```

### 3.4 Embedding the full Python pipeline

```python
from pipeline_and_config import LepauteConfig, DisplayMode
from main import run_pipeline

cfg = LepauteConfig(
    data_store="/path/to/your.db",
    # camera intrinsics are read from camera_config.json
    # or set explicitly:
    # fx=320.0, fy=320.0, cx=160.0, cy=120.0
)
cfg.performance_mode = "medium"   # or "low" / "high"

results = run_pipeline(
    cfg,
    display_mode=DisplayMode.HEADLESS,
    unlimited=True,
    save_json=True,
    mock=False          # True = synthetic frames, no camera needed
)
```

### 3.5 Building the MCU demo (reference)

```bash
cd mcu
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
./lepaute_mcu
```

On macOS the linker flag `--gc-sections` is automatically omitted; on other platforms it is applied for size reduction.

## 4. High-Level Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `ld: unknown options: --gc-sections` | AppleClang does not accept the flag | Ensure CMakeLists.txt contains `if(NOT APPLE)` around the flag (already present in the shipped file) |
| Final cost remains > 250 | Incorrect intrinsics or scale_prior | Verify `fx/fy/cx/cy` match the real camera; set `scale_prior` to the approximate object distance in metres |
| `C optimization symbols not available` | `liblepaute_core` not found by `geometry.py` | Build the core first (`cd core && mkdir build && cmake .. && make`) and ensure the resulting shared library is on the search path |
| Static-memory pyramid allocation returns -2 / -3 | Image larger than `LEPAUTE_MAX_IMAGE_PIXELS` or two pyramids already allocated | Increase the compile-time constant or free the previous pyramid before allocating another |
| Tracking score collapses after a few frames | Scale drift or illumination change | Enable photo-affine parameters (default) and keep `scale_prior` updated via the kinematic forecaster |
| Python process hangs on Ctrl+C | InferenceWorker still running | Press Ctrl+C three times; the third press forces `os._exit` |
| MPS / CUDA out-of-memory | High-resolution frames + large pyramid | Switch to `PerformanceMode.LOW` or reduce `orig_h` / `orig_w` |
| Linker error on unused `static_alloc` | Warning only (function is guarded by `#ifdef`) | Safe to ignore; can be silenced by compiling with `-Wno-unused-function` |

When integrating into a production system, always validate the photometric cost on a short recorded sequence before enabling live camera input. A sudden jump in cost usually indicates a change in lighting or an incorrect depth prior rather than a failure of the SE(3) solver itself.
