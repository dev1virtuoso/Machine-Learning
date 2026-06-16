# Development and Integration Guide

## 1. Overview

The Vision-based Distance and Object Analysis (ViDiAD) framework is a production-grade stereo vision engine designed for Automated Guided Vehicles (AGVs). It provides a complete spatial perception pipeline that fuses YOLOv8 object detection with StereoSGBM depth estimation and temporal Kalman filtering. For high-level developers, the system is designed as a decoupled background service that translates raw optical data into actionable kinematic commands—specifically linear and angular velocity vectors.

## Folder Overview

* **`module.py`**: The core implementation layer. It contains hardware drivers (`RobustStereoStream`), neural runtimes, spatial filters (`TrackKalmanFilter`), and the kinematic decision logic (`ProportionalKinematicController`).
* **`__init__.py`**: The package entry point that exports key classes for clean ingestion into external systems.
* **`example.py`**: A reference implementation demonstrating how to orchestrate the hardware capture, calibration, and real-time execution loops.

## Core Functionality

* **`RobustStereoStream`**: An asynchronous worker thread that manages dual-camera hardware hooks and frame buffering with automated recovery logic.
* **`TrackKalmanFilter`**: A stateful 4D filter that predicts and smooths target coordinates (Z and X) while rejecting sensor noise.
* **`ProportionalKinematicController`**: A logic engine that evaluates tracked hazards against a defined safety corridor to output proportional steering and velocity scaling.
* **`ExecutionWatchdog`**: A safety monitor that triggers emergency protocols if the processing loop encounters a hardware or software hang.

## Usage Guide

### System Requirements

The framework requires Python 3 and the following primary dependencies:

* `numpy`: Array mathematics and geometric transforms.
* `opencv-python` & `opencv-contrib-python`: Image processing and advanced stereo matching.
* `ultralytics`: YOLOv8 inference and tracking.

### Operational Modes

The system supports three command-line modes via `example.py`:

1. **Data Collection**:
`python3 example.py --mode capture`
Automatically logs synchronized left/right frames when a calibration pattern is detected.
2. **Stereo Calibration**:
`python3 example.py --mode calibrate`
Computes camera intrinsics and creates the `stereo_calibration.npz` matrix required for depth calculation.
3. **Production Execution**:
`python3 example.py --mode run`
Boots the full headless pipeline for real-time obstacle avoidance and velocity control.

## Higher Level Tutorial: Integrating to a Developer System

To integrate ViDiAD as a subsystem within your own application, follow this architectural pattern to bypass the CLI and ingest control vectors directly.

### 1. Configure the Perception Environment

Import the `CONFIG` dictionary and modify parameters to match your vehicle's physical footprint and performance requirements.

```python
from module import CONFIG, RobustStereoStream, ProportionalKinematicController, TrackKalmanFilter

CONFIG["TARGET_TRAVEL_CORRIDOR_W"] = 0.60
CONFIG["MAX_LINEAR_VELOCITY"] = 1.0
```

### 2. Initialize Hardware and Processing Nodes

Instantiate the threaded camera driver and the kinematic controller. Ensure the `ExecutionWatchdog` is fed within your loop to prevent emergency shutdowns.

```python
cameras = RobustStereoStream(0, 1)
cameras.start()

controller = ProportionalKinematicController()
active_filters = {} 
```

### 3. Implement the Perception Ingestion Loop

In your main execution loop, retrieve the synchronized frames and pass identified hazards to the controller.

```python
while True:
    ret, frame_l, frame_r = cameras.read()
    if not ret: continue
    
    hazards = [{"z": f.filtered_values[0], "x": f.filtered_values[1]} for f in active_filters.values()]
    
    state, velocity, steering = controller.calculate_velocities(hazards, "LEVEL_1_FULL_STEREO")
    
    my_hardware_interface.send(velocity, steering)
```

## ## high level Troubleshooting

* **Hardware Disconnection Loop**: If the system repeatedly attempts to reset, check the physical USB/MIPI connections. The `RobustStereoStream` uses an exponential backoff (up to 15s) to handle hardware resets.
* **Monocular Fallback**: If `stereo_calibration.npz` is missing or corrupt, the system defaults to "LEVEL_3_MONOCULAR_FALLBACK." This uses `CLASS_HEIGHT_PRIORS` to estimate distance. For accurate 3D mapping, re-run the `calibrate` mode.
* **Thermal Throttling (Low FPS)**: If the loop falls below 18 FPS, the system increases the `stereo_interval` to reduce CPU load. If this persists, increase the `DISPARITY_DOWNSCALE` value in `CONFIG` to reduce resolution.
* **Watchdog Trigger**: A `[CRITICAL SHUTDOWN]` log indicates the processing loop is blocking for more than 1.2 seconds. Investigate background processes or reduce the complexity of the object detection model.