# Heracles Humanoid Robot Platform V6

## Project Status: Open-Source Transition and Cessation of Internal Development

Due to limited personal resources and the imminent arrival of commercial low‑cost humanoid robots that can meet comparable technical specifications, further internal investment in physical production runs and virtual simulation tracks has been completely ceased. The Heracles V6 architecture is now officially frozen and transitioned into a permanent open-source asset.

By uploading the complete design assets to GitHub under the MIT License, the platform is handed over to the global robotics community. Teams and individuals are welcome to build, experiment with, modify, and improve the system independently.

For community members constructing or modifying the Heracles V6 platform, non-commercial technical support remains available upon request regarding component selection, circuit topologies, assembly procedures, or test scripts. The long-term vision is for Heracles V6 to serve as a lasting open resource for education, academic research, makerspace innovation, and expressive dance performances.

## Technical Overview and Repository Assets

The baseline Heracles V6 platform represents a software-defined, bionic architecture engineered to bridge highly articulate kinetic performance with decoupled logical reasoning.

### Core Hardware Architecture

* **Bionic Joint Design**: Features a multi-segment Bionic Shrimp-Shell Joint mechanism. This framework utilizes a rigid-soft coupled underactuated structure operating via non-linear, non-conjugate surfaces (Reverse Conjugate Surface, or RCS Geometry) to distribute heavy impacts and accommodate structural strain.
* **Actuator Topology**: Utilizes 9 primary GIM6010-8 24V star-winding joint motors paired with GDS68 drivers, supplemented by tendon-driven routing links utilizing 1 mm Dyneema lines and precision pulleys.
* **Computing Cluster**: Configured with a decentralized architecture consisting of a dual Raspberry Pi high-level perception system interfacing with 13 distributed Arduino Mega 2560 low-level motor controllers.
* **Centralized Power Distribution**: Designed around a single core 6S Lithium Polymer (LiPo) battery pack (3000mAh – 5000mAh, 30C minimum discharge rate) forming a high-current parallel power bus, keeping high-power lines separated from logic circuitry via a mandatory common ground alignment to the microcontroller GND pins.

### Algorithmic Strategy

* **Motion Control**: Decomposes intricate trajectories into discrete, optimized movement primitives termed Micro-Operations, minimizing low-level runtime computational overhead.
* **Control Hierarchy**: Bridges a specialized motion policy with an active balance safety policy using Center of Mass (COM) and Center of Pressure (COP) tracking to override performance inputs when stability boundaries are breached.

## CAD Model Release

The full 3D assembly model containing structural dimensions, component placements, and bionic shrimp-shell joint geometry is available via the official release link:

* **Download CAD Model**: [Heracles V6 STEP File](https://github.com/dev1virtuoso/Assets/releases/tag/Heracles_20260519.STEP)

## License

This project is licensed under the **MIT License**. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software and hardware design files, subject to providing original attribution.

## Contact and Support

For questions, community collaborations, or non-commercial technical support regarding electrical wiring, part acquisition, or kinematic simulation modeling, reach out via email:

* **Developer**: Carson Wu
* **Email**: carson.developer1125@gmail.com