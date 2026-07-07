# Development and Integration Guide

## Overview
This document provides high-level architectural guidance for integrating the Lie Equivariant Perception Algebraic Unified Transform Embedding (L.E.P.A.U.T.E.) Framework into external developer systems. It covers environment orchestration, pipeline execution, and programmatic integration strategies for embedding the SE(3) tracking and classification subsystems into custom robotics or spatial perception environments.

## System Integration Tutorial
Integrating the framework into an existing architecture requires interacting with the decoupled multi-threaded components through the exposed configuration API, rather than relying solely on the CLI wrapper.

### Programmatic Execution
External systems can directly invoke the perception pipeline using the `run_pipeline` interface and capturing the returned structured payloads for downstream logic.

```python
from module import LepauteConfig, DisplayMode
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

(Code snippet adapted from programmatic interfaces)

### Custom Data Ingestion Pipelines

To route custom video streams (e.g., ROS Image topics, RTSP streams, or synthetic simulation buffers) into the system, developers should subclass and override the `CameraIOStream.read()` method. Ensure your pipeline proxy formats incoming frames into standard `(H, W, 3)` uint8 numpy arrays and injects accurate `timestamp` metadata before passing them into the `MonocularDirectTracker`.

### Dataset Preparation

The dataset pipeline requires downloading the BOP benchmark data and converting it into the proper format.

1. **Download the YCB-V dataset** from the Hugging Face BOP benchmark using the provided download script.

```bash
python ycb-v_download.py
```

2. **Extract the downloaded `.zip` files** located in the `./bop_datasets/ycbv` directory before running the converter.


3. **Convert the extracted BOP dataset** into the monocular dataset format, which calculates exact SE(3) relative poses across specified splits.

```bash
python3 convert_bop_to_lepaute.py --bop_dir ./dataset/bop_datasets/ycbv --output_dir ./dataset/lepaute_dataset --splits train_pbr train_real --stride 1 --workers 8
```

### Model Training

With the dataset compiled and the database log manifest generated, you can initiate the offline training sequence.

* **Launch the training script** pointing to your generated dataset directory and specifying your manifest name to train the Deep SE(3) Residual Refiner Subsystem.

```bash
python3 train.py --dataset_dir ./dataset/lepaute_dataset --manifest_name lepaute_data.json --checkpoint_dir ./checkpoints --epochs 50 --seed 42
```

* Append `--device cuda` (or `mps` / `cpu`) to the command if you need to manually override the compute target.

* Append `--no_compile` to the command to disable PyTorch 2.x optimizations if you run into compatibility or compilation issues.

* Append `--resume_mode ask` to configure state snapshot restoration preferences when restarting an interrupted run.

### Running the Main Pipeline

Execute the application in your requested display and performance modes:

* **GUI Mode:** Execute the application with a standard real-time graphical interface.

```bash
python main.py --mode gui --perf medium
```

* **Detailed GUI Mode:** Run the application with an advanced real-time HUD and live 2D trajectory map overlay.

```bash
python main.py --mode detailedgui --perf high
```

* **Headless Mode:** Run the application in the background without rendering a visual display.

```bash
python main.py --mode headless --perf low
```

* **JSON Mode:** Run the pipeline and output tracking and pose estimation results directly as structured metrics.

```bash
python main.py --mode json
```

### Command-Line Arguments Reference

#### `main.py` Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `gui` | Selects the system execution display framework (`headless`, `gui`, `json`, or `detailedgui`). ||
| `--perf` | `medium` | Selects the operational performance profile orchestration layer (`low`, `medium`, or `high`). ||
| `--db` | `None` | Explicitly sets a custom file path for the underlying local SQLite transition storage database. ||
| `--limit` | `False` | An evaluation testing toggle that limits telemetry mapping execution to a ceiling of 50 total frames. ||

#### `train.py` Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_dir` | **Required** | The target filesystem path containing the database log manifest along with its accompanying image assets. |
| `--manifest_name` | `None` | Specifies the targeted file name of the database log manifest JSON. |
| `--epochs` | `15` | Specifies the absolute maximum duration boundary for model training iterations. |
| `--checkpoint_dir` | `./checkpoints` | The destination folder location where training milestones and optimized weights are exported. |
| `--device` | `None` | Manually forces device routing targeting (e.g., `cuda`, `mps`, `cpu`). |
| `--no_compile` | `False` | Prevents the backend from running PyTorch 2.x ahead-of-time compilation optimizations. |
| `--seed` | `42` | Sets the random seed base to guarantee deterministic consistency across identical runs. |
| `--resume_mode` | `None` | Specifies checkpoint tracking state restoration behavior (e.g., `ask`). |

#### `convert_bop_to_lepaute.py` Arguments

| Argument | Default | Description |
|---|---|---|
| `--bop_dir` | **Required** | The local file reference source folder mapping where downloaded data is hosted. |
| `--output_dir` | `./lepaute_dataset` | The target directory destination for exporting formatted assets. |
| `--splits` | `None` | Specifies space-separated target dataset splits to filter and process (e.g., `train_pbr train_real`). |
| `--stride` | `1` | Sets the index jumping frame interval frequency gap during relative pose calculation tracking pairs. |
| `--obj_ids` | `None` | Accepts an enumerated whitespace-separated mask array listing specified identifier integers to target specific objects (Example: `--obj_ids 1 5 12`). |
| `--workers` | `4` | Determines the count of parallel processing worker threads designated for batch transformation. |
| `--scale` | `1000.0` | The geometric division normalization scaling coefficient to properly downscale physical matrix translation dimensions. |

## High-Level Troubleshooting

* **Thread Blocking in Asynchronous Architectures:** Avoid polling `get_latest_resolved_state()` with high-frequency blocking calls when integrating `InferenceWorker` into a broader async framework like `asyncio` or ROS spin loops. Ensure your external main loop pulls this state asynchronously relative to the worker's queue to prevent UI locking or telemetry stream starvation.


* **VRAM Leaks in Custom Execution Loops:** If constructing custom tracking loops outside of the provided `main.py` entry point, ensure that intermediate SE(3) tensors generated by `se3_exp_map` and `se3_log_map` are detached (`.detach().cpu().numpy()`) before being appended to external publishers or lists. Failure to detach keeps the entire backpropagation computational graph alive in memory, rapidly saturating GPU VRAM.


* **Concurrency Collisions on Metal Performance Shaders (MPS):** When deploying on Apple Silicon, utilizing the isolated `InferenceWorker` threading model alongside heavy main-thread operations can occasionally cause Metal command buffer crashes. If this occurs, rely on `LepauteConfig`'s built-in fallback to `cpu` routing, or enforce strict threading locks (`_mps_lock`) around custom model inference blocks if forcing `mps` execution during multi-threaded operation.