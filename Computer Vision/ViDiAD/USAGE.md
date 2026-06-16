# Usage Guide

### 1. Overview
This documentation acts as the operational reference manual for configuring, managing, and managing the Vision-based Distance and Object Analysis framework within remote or headless environments. The engine decouples multi-threaded hardware streaming capture interfaces from the central processing and kinematic execution modules, allowing developers to execute operations as an independent process block.

### 2. Install Requirements
The execution framework requires a standard UNIX system environment configured with a Python 3 runtime interpreter alongside the following third-party dependencies:
* `numpy`: Used for high-speed spatial coordinate remapping, multi-dimensional geometric transformations, and structural array math.
* `opencv-python`: Provides baseline processing functions, coordinate mapping abstractions, and multi-threaded video stream grab tools.
* `opencv-contrib-python`: Required to supply advanced structural filters, including the DisparityWLSFilter and createRightMatcher utility nodes.
* `ultralytics`: Provides the real-time deep learning framework for object tracking, class prediction, and bounding box evaluations.

### ## 3. Troubleshooting
* **Hardware Disconnection Loop**: If logs continuously issue a `[RECOVERY] Hardware disconnected` warning, verify the structural integrity of the USB camera connections. The camera thread automatically initiates an exponential backoff loop, scaling recovery limits up to 15 seconds to handle hot-plug resets.
* **Calibration File Load Failure**: A missing file warning forces a fallback to monocular tracking. Ensure that structural parameter sets generated during the calibration phase are properly compressed and stored matching the file path configured within the module.
* **Low Main Frame Rates**: If performance metrics fall below 18 FPS, the engine shifts to long-interval processing modes to minimize system stress. To resolve this, adjust the downscale multiplier parameter to reduce calculation load across the block matcher.
* **Watchdog Emergency Coils**: If the terminal displays `[CRITICAL SHUTDOWN] WATCHDOG TIMEOUT EXCEEDED!`, the processing execution pipeline has encountered a block exceeding standard real-time capacity parameters. Check for background computational spikes or thread blocking inside the system modules.

### Manual Parameter Configuration
System behaviors must be managed manually by modifying fields located within the global `CONFIG` dictionary defined inside `module.py`. The available properties include:

| Configuration Property | Expected Datatype | Functional Purpose and Operational Bounds |
|---|---|---|
| `MODEL_PATH` | String | Path or identifier tag pointing to the preferred pre-trained weights file. |
| `CALIB_FILE` | String | Target file path designating the location of the compressed calibration parameter database. |
| `LEFT_CAM_INDEX` | Integer | System hardware identification index assigned to the physical left optical sensor array. |
| `RIGHT_CAM_INDEX` | Integer | System hardware identification index assigned to the physical right optical sensor array. |
| `TARGET_FPS` | Float | Maximum system runtime loop constraint enforcing the global target processing rate. |
| `BASE_STEREO_INTERVAL` | Integer | Frame step interval defining when full SGBM block matching calculations are executed during normal operations. |
| `THERMAL_STEREO_INTERVAL` | Integer | Degradation frame interval enforcing sparse tracking during low-frequency performance drops. |
| `DISPARITY_DOWNSCALE` | Float | Scaling multiplier determining the raw processing width and height reduction before block matching. |
| `MIN_VALID_DISTANCE` | Float | Lower distance perimeter cut-off filtering out ground plane artifacts and glare near the chassis. |
| `MAX_VALID_DISTANCE` | Float | Upper tracking envelope boundary isolating remote peripheral targets from collision calculations. |
| `CRITICAL_ZONE_METERS` | Float | Absolute safety boundary threshold triggering a kinematic fallback or system emergency stop command. |
| `MAX_LINEAR_VELOCITY` | Float | Maximum forward propulsion rate capability applied when traveling corridors are clear. |
| `TARGET_TRAVEL_CORRIDOR_W` | Float | Lateral safety corridor width offset defining the vehicle footprint tracking area. |
| `TRACK_EXPIRY_TIMEOUT` | Float | Maximum duration threshold allowed before an untracked target ID filter is deleted from state memory. |
| `CLASS_HEIGHT_PRIORS` | Dictionary | Structured height estimations mapped against specific class labels to validate fallback monocular pinhole estimations. |

### Usage Guide
The execution module supports three targeted modes of operation depending on system operational phases passed through the CLI parser interface within `example.py`:

1. **Automated Target Pair Image Capture**: Runs a headless pattern synchronization routine that monitors left and right camera feeds simultaneously for chess pattern matches, logging snapshots automatically onto disk when parameters settle.
```bash
python3 example.py --mode capture --left-dir calib_left --right-dir calib_right --grid-w 9 --grid-h 6
```

2. **Matrix Intrinsic Calibration**: Computes high-precision stereo remapping parameters based on collected target images, generating undistortion transforms saved to the deployment bundle.
```bash
python3 example.py --mode calibrate --left-dir calib_left --right-dir calib_right --output stereo_calibration.npz --square-size 0.025
```


3. **Headless Real-time Execution Pipeline**: Boots the default operation stream loop, pulling frames and applying tracking matrix models continuously, outputting navigation parameters down to the command console.
```bash
python3 example.py --mode run
```

### Common Use Cases

* **Autonomous AGV Corridor Navigation**: Enforces strict trajectory tracking within storage or manufacturing plant layouts using automated obstacle deceleration.
* **Headless Dataset Assembly**: Executes script-driven multi-sensor image collections on embedded deployments without requiring remote X11 graphics context setups.
* **Independent Depth Subsystem Integration**: Operates as a background process reporting tracking vectors to secondary motor controllers via low-overhead custom protocols.

### Folder Overview

The functional layout of the system module directory consists of the following components:

* `__init__.py`: Handles package namespace initialization, binding core abstractions cleanly for ingestion by external calling layers.
* `module.py`: Houses the foundational implementation logic, multi-threaded hardware handlers, state filters, and utility functions.
* `example.py`: Serves as the central command-line entry point orchestrating runtime modes and system log reporting.

### Core Functionality

* `RobustStereoStream`: Operates an isolated execution context background worker thread dedicated to grabbing, matching, and buffering image frames.
* `TrackKalmanFilter`: Implements multi-dimensional state vector predictions to manage spatial velocities and target coordinates while rejecting measurement anomalies.
* `ProportionalKinematicController`: Parses tracking arrays against active traveling parameters to calculate appropriate steering outputs and scaled motion trajectories.
* `ExecutionWatchdog`: Runs as an isolated hardware monitor verifying processing heartbeat check-ins to handle frozen pipelines or unexpected runtime hangs.