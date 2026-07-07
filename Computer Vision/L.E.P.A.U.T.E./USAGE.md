# L.E.P.A.U.T.E. Usage Guide

This document provides the standard operating procedures for executing the Lie Equivariant Perception Algebraic Unified Transform Embedding (L.E.P.A.U.T.E.) Framework. All execution wrapper shell scripts (`.sh`) have been deprecated and removed; the system is now invoked directly via Python.

## 1. Dataset Preparation

The dataset pipeline requires downloading the BOP benchmark data and converting it into the framework's monocular dataset format.

1. **Download the Data**: Fetch the YCB-V dataset from the Hugging Face BOP benchmark.

```bash
python ycb-v_download.py
```

2. **Extract Archives**: Extract the downloaded `.zip` files located in the `./bop_datasets/ycbv` directory before running the converter.

3. **Convert Dataset**: Format the data to calculate exact SE(3) relative poses across specified splits.

```bash
python convert_bop_to_lepaute.py --bop_dir ./dataset/bop_datasets/ycbv --output_dir ./dataset/lepaute_dataset --splits train_pbr train_real --stride 1 --workers 8
```

### Dataset Conversion Arguments (`convert_bop_to_lepaute.py`)

* **`--bop_dir`**: The local file reference source folder mapping where downloaded data is hosted (Required).
* **`--output_dir`**: The target directory destination for exporting formatted assets (Default: `./lepaute_dataset`).
* **`--splits`**: Specifies space-separated target dataset splits to filter and process.
* **`--stride`**: Sets the index jumping frame interval frequency gap (Default: `1`).
* **`--obj_ids`**: Enumerated whitespace-separated mask array to target specific objects (e.g., `--obj_ids 1 5 12`).
* **`--workers`**: Count of parallel processing worker threads (Default: `4`).
* **`--scale`**: Geometric division normalization scaling coefficient (Default: `1000.0`).

## 2. Model Training

Once the dataset is compiled and the database log manifest is generated, initiate the offline training sequence for the Deep SE(3) Residual Refiner Subsystem.

```bash
python train.py --dataset_dir ./dataset/lepaute_dataset --manifest_name lepaute_data.json --checkpoint_dir ./checkpoints --epochs 50 --seed 42
```

### Training Arguments (`train.py`)

* **`--dataset_dir`**: Target filesystem path containing the database log manifest and images (Required).
* **`--manifest_name`**: File name of the database log manifest JSON.
* **`--epochs`**: Maximum duration boundary for training iterations (Default: `15`).
* **`--checkpoint_dir`**: Destination folder for optimized weights (Default: `./checkpoints`).
* **`--device`**: Manually forces device routing targeting (e.g., `cuda`, `mps`, `cpu`).
* **`--no_compile`**: Prevents the backend from running PyTorch 2.x ahead-of-time compilation optimizations.
* **`--seed`**: Sets the random seed base for deterministic consistency (Default: `42`).
* **`--resume_mode`**: Specifies checkpoint state restoration behavior (e.g., `ask`).

## 3. Running the Main Pipeline

The primary perception pipeline can be launched using the `main.py` entry point.

* **GUI Mode**: Standard real-time graphical interface.

```bash
python main.py --mode gui --perf medium
```

* **Detailed GUI Mode**: Advanced real-time HUD and live 2D trajectory map overlay.

```bash
python main.py --mode detailedgui --perf high
```

* **Headless Mode**: Background execution without rendering a visual display.

```bash
python main.py --mode headless --perf low
```

* **JSON Mode**: Outputs tracking and pose estimation results directly as structured metrics.


```bash
python main.py --mode json
```

### Main Pipeline Arguments (`main.py`)

* **`--mode`**: Selects the system execution display framework (`headless`, `gui`, `json`, or `detailedgui`).
* **`--perf`**: Selects the operational performance profile orchestration layer (`low`, `medium`, or `high`).
* **`--db`**: Explicitly sets a custom file path for the underlying local SQLite transition storage database.
* **`--limit`**: An evaluation testing toggle that limits telemetry mapping execution to a ceiling of 50 total frames.

## 4. Programmatic Integration

For embedding the SE(3) tracking and classification subsystems into custom robotics architectures, bypass the CLI and invoke the framework programmatically.

Available imports from the core module include `LepauteConfig`, `DisplayMode`, `run_pipeline`, `MonocularDirectTracker`, `SigLIPClassifier`, and other Lie group algebraic mappings (`se3_exp_map`, `compose_poses`, etc.).

**Example Implementation:**

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

## 5. High-Level Troubleshooting

* **Thread Blocking in Asynchronous Architectures**: Avoid polling `get_latest_resolved_state()` with high-frequency blocking calls when integrating into external asynchronous frameworks. Ensure your main loop pulls states asynchronously to prevent stream starvation.

* **VRAM Leaks in Custom Loops**: If building custom loops outside of `main.py`, intermediate SE(3) tensors generated by Lie operations must be explicitly detached (`.detach().cpu().numpy()`) to prevent saturating GPU VRAM.

* **Concurrency Collisions on Apple Silicon (MPS)**: Heavy main-thread operations alongside isolated `InferenceWorker` threading can cause Metal command buffer crashes. Rely on the built-in CPU fallback or enforce strict threading locks (`_mps_lock`) around custom inference blocks.