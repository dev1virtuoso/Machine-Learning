# Vision-based Distance and Object Analysis (ViDiAD)

## Abstract
Vision-based Distance and Object Analysis (ViDiAD) is an embedded, production-grade real-time stereo vision engine and object analytics suite designed specifically for Automated Guided Vehicles (AGVs) deploying on resource-constrained or headless computing hardware such as the Raspberry Pi. By fusing high-accuracy deep learning object detection with low-latency stereo matching, temporal sparse optical flow warping, and stateful 4D Kalman coordinate filtering, the system acts as an independent obstacle evaluation subsystem. It establishes automated lateral safety corridors and dynamic velocity scaling profiles to enforce physical safety constraints headlessly under varying hardware thermal states.

## System Overview
The structural orchestration of the processing nodes, hardware abstractions, and spatial filters within the runtime engine is illustrated below:

```mermaid
graph TD
    A[RobustStereoStream Worker Thread] -->|Left Camera Frame| B[Image Rectification Remap]
    A -->|Right Camera Frame| B
    B -->|Rectified Frame Pair| C{Compute Loop Schedulers}
    C -->|Heavy Stereo Compute Trigger| D[StereoSGBM / WLS Filter Spatial Engine]
    C -->|Inter-frame Interval Flow| E[Lucas-Kanade Sparse Optical Flow Warping]
    D -->|Generated 3D Point Coordinates| F[Spatial Matrix Representation]
    E -->|Warped Perspective Coordinates| F
    B -->|Rectified Primary Frame| G[YOLOv8 Multi-Class Object Tracker]
    G -->|Bounding Boxes and Target IDs| H[Sensor Fusion & Spatial Profiler]
    F -->|Depth and Lateral Map Overlays| H
    H -->|Raw Target Metric Z and X Coordinates| I[Stateful 4D TrackKalmanFilter Node]
    I -->|Smoothed Spatial Vector and Trend Map| J[ProportionalKinematicController]
    J -->|Calculated Linear / Angular Velocities| K[AGV Propulsion Drive Digital Output]
    L[ExecutionWatchdog Heartbeat Monitor] -->|Host Loop Refresh Validation| K
```

## Features and Capabilities

* **Adaptive Depth Computation Scheduling**: Dynamically shifts between intensive StereoSGBM block matching calculations and lightweight Lucas-Kanade sparse optical flow perspective warping depending on host processor refresh rates and thermal performance parameters.
* **Headless Telemetry Interface**: Stripped of standard graphical dependencies and window hooks, the architecture provides console logging abstractions tailored for seamless execution over remote terminal sessions or Secure Shell (SSH) links.
* **Stateful 4D Temporal Filtering**: Uses object-tracked Kalman filters to manage persistent tracking vectors containing dynamic target positions and speeds, reducing data capture anomalies and pixel noise.
* **Proportional Decoupled Kinematics**: Computes responsive steering and deceleration profiles relative to structural coordinates mapped against configurable traveling corridor buffer bounds.
* **Hardware System Watchdog**: Operates an isolated execution monitor thread to trigger safety recovery procedures if the processing loop blocks or drops below real-time performance thresholds.

## License

MIT License.

## Author

* [Carson Wu](https://github.com/dev1virtuoso/Documentation/blob/main/dev1virtuoso/Attachment/dev1virtuoso/carson-wu.md#Contact)
* Jonathan Tse
* Marcus Tong
* Hose Wong
* Wilber Lee
