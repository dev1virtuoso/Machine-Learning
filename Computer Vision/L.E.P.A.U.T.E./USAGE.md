# Usage Guide

## Overview

The L.E.P.A.U.T.E. (Lie Equivariant Perception Algebraic Unified Transform Embedding) Framework processes webcam feeds or mock video streams through a Lie group-based geometric Transformer model. It extracts 3D spatial telemetry, computes optical flow, registers SE(3) tracking transformations, performs real-time depth mapping, and uses zero-shot classifiers for target monitoring. This document serves as a comprehensive runtime operations manual, clarifying configuration parameters and functional workflows.

## Install Requirements

Before processing data or training, you must establish the local isolated build environment and verify your dependencies.

1.  **Execute the installation script** to create a virtual environment (`venv`) and install the core geometric Deep Learning backbones.
    ```bash
    bash install_dependencies.sh
    ```
2.  **Activate the newly created virtual environment.**
    ```bash
    source venv/bin/activate
    ```
3.  **Verify that your system meets the minimum package constraints** for modules like PyTorch and OpenCV.
    ```bash
    bash check_system_requirements.sh
    ```

## Manual Configuration Parameters

The system exposes a unified configurations head through `LepauteConfig` derived from Pydantic settings. These values must be declared manually via environment variables (prefixed with `LEPAUTE_`) or modified directly inside a custom instantiation script to fit your physical sensor topography.

### Hardware and Resource Parameters
* **`device`** (Default: Automatically selects `cuda`, `mps`, or `cpu`): Force setting to a specific processor layout if navigating multi-GPU infrastructure.
    * *Environment Variable*: `LEPAUTE_DEVICE="cuda:0"`
* **`dl_inference_freq`** (Type: `int`, Default: `5`): Sets the frequency barrier for heavy deep learning inference cycles. Processing every frame through complex backbones throttles loops; raising this value bypasses evaluation frames to maintain real-time interface throughput.
    * *Environment Variable*: `LEPAUTE_DL_INFERENCE_FREQ=5`

### Sensor Intrinsic Matrix Subsystem
For valid mathematical projection inside the SE(3) dense tracking layer, physical camera sensor properties require explicit configuration matching your optics hardware.
* **`fx`** (Type: `float`, Default: `250.0`): Focal length along the X-axis mapping pixel metrics.
* **`fy`** (Type: `float`, Default: `250.0`): Focal length along the Y-axis mapping pixel metrics.
* **`cx`** (Type: `float`, Default: `160.0`): Principal point X-coordinate matching half of your horizontal resolution center.
* **`cy`** (Type: `float`, Default: `120.0`): Principal point Y-coordinate matching half of your vertical resolution center.

### Classification Categories and Workspace Contexts
* **`object_names`** (Type: `List[str]`): Sets the open-vocabulary category strings parsed by the SigLIP vision-language head during processing. Customize this parameter manually to match real-world tracking targets.
    * *Example Initialization*: `LepauteConfig(object_names=["table", "cup", "keyboard"])`
* **`data_store`** (Type: `str`, Default: `"lepaute_data.db"`): File location handling downstream storage of localized spatial telemetry packets.

## Usage Guide

### Dataset Preparation
The dataset pipeline requires downloading the BOP benchmark data and converting it into the proper LEPAUTE format.

1.  **Download the YCB-V dataset** from the Hugging Face BOP benchmark using the provided download script.
    ```bash
    python ycb-v_download.py
    ```
2.  **Extract the downloaded `.zip` files** located in the `./bop_datasets/ycbv` directory before running the converter.
3.  **Convert the extracted BOP dataset** into the LEPAUTE monocular dataset format, which calculates exact SE(3) relative poses.
    ```bash
    python convert_bop_to_lepaute.py --bop_dir ./bop_datasets/ycbv --output_dir ./lepaute_dataset
    ```

### Model Training
With the dataset compiled and the `lepaute_data.json` manifest generated, you can initiate the offline training sequence.

* Launch the training script pointing to your generated dataset directory to train the Deep SE(3) Residual Refiner Subsystem.
    ```bash
    python train.py --dataset_dir ./lepaute_dataset --checkpoint_dir ./checkpoints --epochs 15
    ```
    * Append `--device cuda` (or `mps` / `cpu`) to the command if you need to manually override the compute target.
    * Append `--no_compile` to the command to disable PyTorch 2.x optimizations if you run into compatibility issues.

### Running the Main Pipeline
Execute the pipeline in your requested display modes using the `main.py` entry point:

* **GUI Mode:** Execute the application with a graphical user interface for real-time visualization.
    ```bash
    python main.py --mode gui
    ```
* **Headless Mode:** Run the application in the background without rendering a visual display.
    ```bash
    python main.py --mode headless
    ```
* **JSON Mode:** Run the pipeline and output the tracking and pose estimation results directly to a structured JSON file.
    ```bash
    python main.py --mode json
    ```

### Command-Line Arguments Reference

#### `main.py` Arguments
This script handles execution pipeline routing and runtime mode settings.
* `--mode`: Selects the system execution display framework. Choices: `['headless', 'gui', 'json']`. (Default: `gui`)
* `--db`: Explicitly sets a custom file path for the underlying local SQLite transition storage database. (Default: System configuration default)
* `--limit`: An evaluation testing toggle that limits telemetry mapping execution to a ceiling of 50 total frames. (Default: `False`)

#### `train.py` Arguments
This script orchestrates deterministic offline neural network optimization loops.
* `--dataset_dir`: **[Required]** The target filesystem path containing the `lepaute_data.json` database log manifest along with its accompanying `frames/` image folder.
* `--epochs`: Specifies the absolute maximum duration boundary for model training iterations. (Default: `15`)
* `--checkpoint_dir`: The destination folder location where training milestones and optimized weights are exported. (Default: `"./checkpoints"`)
* `--device`: Manually forces device routing targeting, letting you select a specific back-end compute environment. Choices: Any valid device string (e.g., `'cuda'`, `'mps'`, `'cpu'`).
* `--no_compile`: Prevents the backend from running PyTorch 2.x ahead-of-time compilation optimizations. (Default: `False`)
* `--seed`: Sets the random seed base to guarantee deterministic consistency across identical runs. (Default: `42`)

#### `convert_bop_to_lepaute.py` Arguments
This data ingestion module transforms raw datasets from standard external configurations into target spatial matrices.
* `--bop_dir`: **[Required]** The local file reference source folder mapping where downloaded data is hosted.
* `--output_dir`: The target directory destination for exporting formatted assets. (Default: `"./lepaute_dataset"`)
* `--stride`: Sets the index jumping frame interval frequency gap during relative pose calculation tracking pairs. (Default: `1`)
* `--obj_ids`: Accepts an enumerated whitespace-separated mask array listing specified identifier integers to target specific objects. (Example: `--obj_ids 1 5 12`)
* `--workers`: Determines the count of parallel processing worker threads designated for batch transformation. (Default: `4`)
* `--scale`: The geometric division normalization scaling coefficient to properly downscale physical matrix translation dimensions. (Default: `1000.0`)

## Troubleshooting

### Problem 1: System Freezes or Drops Massive Frame Batches in Realtime Mode
* **Root Cause**: The processing hardware cannot maintain runtime frame processing through deep vision and optical flow neural pathways concurrently.
* **Resolution**: Manually increase your execution skip factor inside the configuration settings to shift computational strain away from non-blocking threads:
    ```python
    from module import LepauteConfig
    config = LepauteConfig(dl_inference_freq=10)
    ```

### Problem 2: Optical Flow Matrix Outputs Invalid (NaN) or Unstable Lie Transformations
* **Root Cause**: Deep optical flow tracking pipelines calculate empty gradients when reading completely dark, over-exposed, or static monochrome frames.
* **Resolution**: Ensure active hardware lenses have proper lighting conditions. If running inside continuous validation or container integration test suites without a physical lens interface attached, explicitly enable the software mock generation protocol inside your script run arguments:
    ```python
    run_pipeline(mock=True)
    ```

### Problem 3: Multi-Threaded Queue Exhaustion Warning Logs in Terminals
* **Root Cause**: The main I/O camera acquisition routine is collecting incoming sensor frames faster than the isolated inference engine worker thread can empty its bounded buffer stacks.
* **Resolution**: Set processing properties away from CPU layouts toward accelerator options (`cuda` or `mps`). If restricted to CPU profiles, lower input resolution settings within your platform camera configuration parameters to reduce spatial dimensionality.