# L.K.O.A. Development and Integration Guide

## 1. Overview

The Lane-Keeping and Obstacle-Avoidance System (L.K.O.A.) is an edge-optimized, pure-vision navigation control pipeline. It ingests standard BGR camera frames and uses embedded deep learning runtimes along with geometric analysis to output explicit vehicle state decisions and motion vectors.

For a developer integrating this framework into a custom autonomous vehicle architecture, the system acts as a software abstraction layer that replaces complex spatial engineering pipelines. It handles frame capture synchronization, depth inference, and tracking internally, yielding three clean downstream primitives: a categorical vehicle safety state, a normalized linear velocity scalar, and a lateral steering rate.

## System Integration Tutorial

This tutorial guides you through embedding the L.K.O.A. engine directly into your custom robotic control loop rather than using the default standalone execution script.

### Step 1: Configuration Adaptation

To make the vision engine match your physical machine layout, instantiate and modify the configuration profile. You must match the camera mounting geometry and target thread timing specifications exactly.

```python
from module import ProductionSystemConfig, CameraHardwareProfile

custom_config = ProductionSystemConfig(
    production_mode=True,
    target_fps=15,
    camera_idx=0,
    hardware=CameraHardwareProfile(
        height_m=0.350,
        focal_y_px=714.2,
        center_y_px=241.8
    )
)

```

### Step 2: Ingesting Subsystem Nodes

Initialize the background data acquisition stream and the core coordination processor. The camera worker thread handles memory double-buffering independently to isolate execution timing.

```python
from module import ThreadSafeCameraReader, ProductionVisionEngine

camera_stream = ThreadSafeCameraReader(custom_config).start()

vision_engine = ProductionVisionEngine(custom_config)

```

### Step 3: Implementing the Execution Loop

Incorporate the frame processing logic into your system's cyclic control loop. Ensure you calculate and apply accurate pacing constraints to maintain deterministic control cycles.

```python
import time

loop_interval = 1.0 / custom_config.target_fps

try:
    while True:
        start_tick = time.time()
        
        frame_ready, frame_matrix = camera_stream.get_latest_frame()
        if not frame_ready or frame_matrix is None:
            time.sleep(0.002)
            continue
            
        vehicle_state, linear_velocity, steering_command = vision_engine.process_frame(frame_matrix)
        
        elapsed = time.time() - start_tick
        delay_needed = loop_interval - elapsed
        if delay_needed > 0:
            time.sleep(delay_needed)

except KeyboardInterrupt:
    print("Integration loop shutdown requested.")
finally:
    camera_stream.terminate()

```

### Step 4: Interpreting Control Primitives

When mapping the values returned by `process_frame` to your physical motor controllers, use the following structural definitions:

| Output Primitive | Type / Range | System Interpretation |
| --- | --- | --- |
| `vehicle_state` | `IntEnum` (0, 1, 2) | Returns `0` (FORWARD_CRUISE), `1` (CRITICAL_AVOIDANCE), or `2` (EMERGENCY_STOP). Use this to flag global vehicle behavioral changes. |
| `linear_velocity` | `float` [0.0, 1.0] | A normalized throttle scaling factor. Translate this to your maximum safe linear platform speed. |
| `steering_command` | `float` [-1.0, 1.0] | Lateral directional vector. Negative values indicate steering left; positive values indicate steering right. |

## high level Troubleshooting

### Latency Accumulation and Control Loop Lag

* **Symptom:** The vehicle reacts slowly to physical obstacles, and the processing cycle misses its targeted frame-rate intervals.
* **Mitigation:** The architecture intentionally degrades execution rates of heavy deep-learning blocks when processor core temperatures exceed safety thresholds. Check that your embedded hardware platform has adequate passive or active cooling. If temperatures are nominal, reduce the `target_fps` variable in your configuration to match the real-world processing ceiling of your hardware.

### False-Positive Emergency Interventions

* **Symptom:** The vehicle drops into an unexpected `EMERGENCY_STOP` state on empty paths or uniform flat surfaces.
* **Mitigation:** This behavior is typically caused by ground-plane miscalculations. Ensure that the values assigned to `height_m` and `focal_y_px` within the configuration profile exactly match your physical camera installation. Even minor deviations in camera height measurement will distort the distance estimation matrices.

### Indeterminate or Stalled Control Vectors

* **Symptom:** Actuators receive frozen or missing steering variables while the system is running.
* **Mitigation:** The execution framework uses a three-frame temporal consensus filter to minimize rapid steering oscillations. If your camera stream drops below half its rated performance, the engine may fail to achieve state consensus. Verify that your capture loop does not contain external blocking function calls that could starve the driver of processing cycles.