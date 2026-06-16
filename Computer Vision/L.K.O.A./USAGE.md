# Usage Guide

## 1. Overview
This directory comprises a modularized spatial perception framework designed to run autonomously on edge equipment. The logic is separated cleanly across three functional nodes:
* `module.py`: The core implementation layer holding hardware drivers, neural runtimes, analytical tracking filters, and the decision FSM.
* `__init__.py`: Package entry point handling standard structural symbol exports.
* `example.py`: A ready-made execution template managing high-level runtime setups and synchronized loops.

## Folder Overview
* **Configuration Layer (`ProductionSystemConfig`):** Dictates target performance frameworks, sensor dimensions, serial hardware definitions, safe thermal ceilings, and structural class priority assumptions.
* **Hardware Interactivity Layer (`ThreadSafeCameraReader`, `RobustUARTCommandPublisher`):** Houses the double-buffered video capture socket buffers and manages the low-level serial communication packet strings.
* **Analytical Perception Pipelines (`MultiBandRANSACSolver`, `MultiCueBottomRefiner`, `LightweightObstacleFilter`):** Solves floor pitch variables, extracts high-fidelity object ground points, and fills perception tracking windows.
* **Executive Coordination Layer (`ProductionVisionEngine`):** Integrates independent sub-modules, performs minimum-variance statistics calculations, checks thermal states, and handles vehicle state selections.

## Core Functionality
The application leverages multi-threaded processing boundaries to parse input channels independently. The camera thread manages device hardware interactions directly, maintaining a fast frame buffer cache. The main execution thread isolates single frames, pipelines them across the ONNX runtime model engines, computes localized spatial distances, determines the correct finite state, and builds a checksum-validated command packet. 

This tracking pipeline features adaptive multi-frame consensus rules to prevent steering oscillations, combined with a persistent 200ms background thread heartbeat ensuring safe physical failure handling if the tracking software loop stalls.

## 2. Install Requirements
The application expects a standard Python 3 runtime containing the following dependencies:
* `opencv-python`
* `numpy`
* `onnxruntime`
* `pyserial` (required if deployed in standard hardware production configurations)

## Usage Guide
To execute the tracking system, interact directly with the provided executable template. The internal logic follows this structure:

1. **Instantiate Parameter Structures:** Create a configuration container defining operational variables:
   ```python
   from module import ProductionSystemConfig
   sys_config = ProductionSystemConfig(production_mode=True, target_fps=12)

```

2. **Launch Hardware Subsystems:** Initialize the double-buffered capture loop and background command interfaces:
```python
from module import ThreadSafeCameraReader, RobustUARTCommandPublisher
hardware_reader = ThreadSafeCameraReader(sys_config).start()
control_publisher = RobustUARTCommandPublisher(sys_config)
```

3. **Run Process Loop Architecture:** Pass active memory matrices to the analytical navigation engine inside a frequency-locked synchronization loop:
```python
from module import ProductionVisionEngine
engine = ProductionVisionEngine(sys_config)

frame_fetched, active_frame = hardware_reader.get_latest_frame()
current_state, linear_v, steer_w = engine.process_frame(active_frame)
control_publisher.transmit(current_state, linear_v, steer_w)
```

## Common Use Cases

* **Automated Guided Vehicles (AGVs):** Autonomous industrial warehouse platforms monitoring predefined safety corridors while transporting factory pallets.
* **Logistics Inspection Systems:** Mobile telemetric tracking robots using continuous vision profiling to map ground clear zones and avoid facility hazards.
* **Visual Telemetry Logging Nodes:** Testing setups collecting raw spatial depth tracking profiles and serial command responses under laboratory conditions.

## ## 3. Troubleshooting

### Camera Frame Acquisition Failures

* **Symptom:** Driver loops continuously trace backoffs without registering valid image frames.
* **Mitigation:** Ensure the physical connection matches the device index specified in `camera_idx`. On Linux systems, verify that proper V4L2 device sockets are exposed by checking `/dev/video*`.

### Serial Bus Timeouts or Stalls

* **Symptom:** System defaults to immediate fallback stops and signals serial communication write timeout faults.
* **Mitigation:** Verify system hardware permissions allow access to your target communication node (e.g., `/dev/ttyAMA0`). Ensure the user profile is registered within the local `dialout` system permissions group.

### Low Frame Processing Rates

* **Symptom:** System tracking latency peaks, missing target frame execution windows.
* **Mitigation:** Ensure that the underlying ONNX execution model configurations match the hardware capabilities. If the hardware core temperatures climb beyond nominal thresholds, the framework intentionally steps down processing rates to manage thermal levels. Verify that adequate physical cooling components are functioning.

### Unintended Safety Interventions

* **Symptom:** Vehicle enters premature emergency deceleration frames on wide open floor planes.
* **Mitigation:** Highly reflective floor planes or rapid environmental light variations can alter standard disparity map calculations. Fine-tune the structural `BLOCK_SIGMA` threshold constraints inside the logic module or update the initialization profile to adjust camera hardware position heights.

### Manual Parameter Configuration

To adapt the L.K.O.A. system to specific vehicular hardware, factory settings, or testing environments, several parameters must be configured manually inside `ProductionSystemConfig`, `CameraHardwareProfile`, and `ThermalProfile`.

#### 1. Hardware & System Environment (`ProductionSystemConfig`)

These properties handle basic I/O mapping, file pathways, and execution modes. They must match your physical deployment layout:

| Parameter | Type | Default Value | Description |
|---|---|---|---|
| `production_mode` | `bool` | `True` | Set to `False` to enable debugging metrics and console log readouts. Set to `True` for maximum execution speed. |
| `target_fps` | `int` | `12` | The target processing loop frequency. Adjust this based on your embedded platform's processing capabilities. |
| `camera_idx` | `int` | `0` | The hardware index assigned to your V4L2 device (e.g., `0` for `/dev/video0`). |
| `uart_port` | `str` | `"/dev/ttyAMA0"` | The serial port pathway connecting the edge machine to the underlying micro-controller. |
| `uart_baud` | `int` | `115200` | The baud rate matching the peripheral serial interface configuration. |
| `yolo_onnx_path` | `str` | `"yolov8n.onnx"` | Local file path pointing to your exported YOLOv8 model file. |
| `midas_onnx_path` | `str` | `"midas_small.onnx"` | Local file path pointing to your exported MiDaS model file. |

#### 2. Camera Spatial Calibration (`CameraHardwareProfile`)

The precision of the minimum-variance distance estimation depends directly on providing accurate geometric properties. **Do not use default parameters without validating your physical rig dimensions:**

* **`height_m` (Default: `0.290`):** The precise vertical distance measured from the ground plane to the physical center of the camera lens (optical center), in meters.
* **`focal_y_px` (Default: `714.2`):** The vertical focal length ($f_y$) of your specific camera lens, expressed in pixels. This value must be derived from a standard chessboard camera calibration process.
* **`center_y_px` (Default: `241.8`):** The vertical principal point ($c_y$) of your camera sensor, in pixels. This represents the optical center along the vertical image plane.
* **`min_valid_distance` & `max_valid_distance` (Default: `0.25` / `8.00`):** The physical safety boundaries (in meters) for tracking operations. Any calculated distance falling outside this envelope is bounded or filtered out as telemetry noise.

#### 3. Thermal Management Profiles (`ThermalProfile`)

If deployed on passably cooled hardware or inside demanding thermal environments, adjust these thresholds to dictate when the processing loop lowers its execution rate:

* **`nominal_temp` (Default: `68.0`):** The temperature threshold (in Celsius) where the system begins step-down scaling, shifting the object detection interval to run every 2 or 3 frames instead of every frame.
* **`warning_temp` (Default: `73.0`):** The threshold where heavy depth inference steps are minimized to preserve processor cycles.
* **`critical_temp` (Default: `78.0`):** The maximum safe limit. Crossing this value triggers the tightest inference degradation window and extends physical vehicle braking margins (widening emergency stop distances from `0.95m` to `1.20m`).
* **`recovery_hysteresis` (Default: `3.0`):** The temperature drop (in Celsius) required below a threshold before the system recovers from fallback state limitations back into full-performance mode.

#### 4. Custom Asset Dimensions (`class_priors`)

If tracking specialized factory assets, you must manually update the `class_priors` dictionary mapping index IDs to physical asset dimensions:

```python
# Format: class_id: [Standard Real-World Width (meters), Detection Confidence Weight]
class_priors: Dict[int, List[float]] = field(default_factory=lambda: {
    0: [0.45, 0.92],   # Factory Personnel (Width: 0.45m)
    56: [0.50, 0.85],  # Logistics Pallets/Bins (Width: 0.50m)
    11: [0.35, 0.70]   # Industrial Traffic Cones (Width: 0.35m)
})
```