> [!IMPORTANT]
> - This project is considered deprecated and abandoned. It is no longer actively maintained or updated. Please use it with caution and consider alternative solutions for your needs.
> - Development of K.R.I.S.T.Y. v5.0 was stopped suddenly and the project remains incomplete.
> - Development of K.R.I.S.T.Y. v6.0 and K.R.I.S.T.Y. v7.0 has also been discontinued.

# K.R.I.S.T.Y.

## Table of Contents

- [K.R.I.S.T.Y.](#kristy)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
  - [Project History and Timeline](#project-history-and-timeline)
    - [A.B.C. v1.0 (2022): Technical Details](#abc-v10-2022-technical-details)
    - [K.R.I.S.T.Y. v1.0 (2022): Technical Details](#kristy-v10-2022-technical-details)
    - [A.R.I.E.L. v2.0 (2022–2023): Technical Details](#ariel-v20-20222023-technical-details)
    - [K.R.I.S.T.Y. v3.0 (2023–2024): Technical Details](#kristy-v30-20232024-technical-details)
    - [K.R.I.S.T.Y. v4.0 (2024–2025): Technical Details](#kristy-v40-20242025-technical-details)
    - [K.R.I.S.T.Y. v5.0 (2025): Technical Details](#kristy-v50-2025-technical-details)
  - [System Architecture](#system-architecture)
  - [Key Documents and References](#key-documents-and-references)
  - [Installation and Usage](#installation-and-usage)
  - [Ethical Considerations and Limitations](#ethical-considerations-and-limitations)
  - [Contributors](#contributors)
  - [License](#license)

## Project Description

K.R.I.S.T.Y. (Knowledge Retrieval and Inference System for Transformative Yield) is an advanced AI platform that has evolved from a simple offline chatting bot into a sophisticated multimodal generative system and physical AI interface. It leverages natural language processing (NLP), symbolic calculation, graph neural networks, and physical simulation to enable creative content generation and embodied control across text, motion, vision, and robotics domains.

The project emphasizes a modular architecture, deterministic logic verification, safety gating, and continuous integration with knowledge bases and physical simulators.

Key features:
- **Natural Language Interaction**: Processes conversational prompts and structured queries to perform retrieval, reasoning, and multi-modal synthesis.
- **Multimodal Generation**: Supports text, 46-DoF motion trajectories, 64×64 images, BVH exports, and ROS 2 physical control commands.
- **Knowledge-Driven Architecture**: Uses structured SQLite knowledge graphs, BM25 retrieval, word co-occurrence PyG graphs, and multi-hop GraphRAG for accurate semantic grounding.
- **Safety and Hallucination Control**: Incorporates Rational Gates, Reasoning Gates with LogitWarper, zero-shot classifiers, and PyBullet physical feasibility validation.
- **Embodied AI Integration**: Integrates MoE kinematics routing, real2sim2real physical alignment loops, and potential field collision avoidance.

## Project History and Timeline

The project originated as a personal research and engineering initiative by Carson Wu and has undergone several architecture shifts, version iterations, and scope expansions between 2022 and 2025. Below is a chronological overview:

- **2022**: **A.B.C. v1.0** (Advance Bot for Chatting v1.0) re-implemented as a clean baseline offline rule-matching chatbot using regular expressions and a minimal `SessionContext` dictionary.
- **2022**: **K.R.I.S.T.Y. v1.0** developed for symbolic calculation, using SQLite, NetworkX directed graph topologies, and SymPy for precise product pricing and exact tax algebra.
- **2022–2023**: **A.R.I.E.L. v2.0** introduced neural networks to the core platform, employing bidirectional/unidirectional GRU architectures, SQLite/Redis/Celery infrastructure, and a Rational Gate for hallucination risk control.
- **2023–2024**: **K.R.I.S.T.Y. v3.0** matured the RAG pipeline by integrating BM25 document retrieval, PyTorch Geometric word co-occurrence graphs, BERT feature extraction, a Reasoning Gate, and LogitWarper decoding.
- **2024–2025**: **K.R.I.S.T.Y. v4.0** introduced unified multimodal generation in a single forward pass using FiLM layer conditioning, RoBERTa/GPT-2 encoders, and Watts–Strogatz GCN routing to simultaneously produce text, 46-DoF motion, and 64×64 images.
- **2025**: **K.R.I.S.T.Y. v5.0** shifted to Physical AI and Embodiment, featuring persistent multi-hop GraphRAG, MoE kinematics routing, PyBullet closed-loop physical validation, and ROS 2 control topic publishing before development was halted.

Development Timeline: 2022 – 2025 (Deprecated / Maintenance Halted).

---

### A.B.C. v1.0 (2022): Technical Details

This foundational baseline focused on creating a completely offline, zero-dependency chatbot with sub-millisecond startup times.

- **Core Paradigm**: Rule-based system and regular expression pattern matching.
- **Architecture**: Core logic built on python regex with named capture groups coupled with a lightweight `SessionContext` dictionary for simple session variables.
- **Dialogue & Logic**: Matched queries directly into template variables or callable functions (e.g., fetching local time or simple conditional logic).
- **Interface & Storage**: Deployment over Tkinter GUI or single-threaded Flask server; zero model loading delays and near-zero memory footprint.
- **Limitations**: Inflexible to phrasing/synonym variations; lack of true context tracking for multi-user concurrency.

### K.R.I.S.T.Y. v1.0 (2022): Technical Details

This version focused on symbolic computation and exact mathematical reasoning, demonstrating that deterministic problems should rely on formal logic rather than neural generation.

- **Core Paradigm**: Graph topology paired with symbolic algebra.
- **Knowledge Base & Topology**: Stored product and tax metadata in SQLite, constructed into a NetworkX directed graph where product nodes linked to tax-rate nodes via `subject_to` edges.
- **Symbolic Execution**: Utilized NLTK for entity tokenization and SymPy for algebraic expression modeling (e.g., `p*(1+t)`), preventing floating-point rounding errors and model hallucination.
- **Deployment**: Single-threaded Flask service optimized for precise financial and tax arithmetic.
- **Limitations**: Primitive natural language understanding limited to string match/containment; required manual code additions for schema updates.

### A.R.I.E.L. v2.0 (2022–2023): Technical Details

This update marked the first introduction of neural network models into the main development path.

- **Core Paradigm**: BiGRU Encoder + UniGRU Decoder Seq2Seq model with rule integration.
- **Hallucination Risk Control**: Introduced a **Rational Gate**—a 2-layer MLP accepting the final encoder hidden state to output a risk score between 0 and 1, enforcing hard refusals above 0.6.
- **Knowledge Retrieval**: Longest N-gram matching on SQLite entity and fact tables to assemble prompt context.
- **Infrastructure & Asynchronous Pipeline**: Integrated N-gram grammar correction, Redis caching, and Celery task queues for asynchronous processing.
- **Limitations**: Out-of-vocabulary tokens collapsed into `<unk>` due to frequency truncation; greedy decoding caused repetition; SQLite lock conflicts under high concurrency.

### K.R.I.S.T.Y. v3.0 (2023–2024): Technical Details

Version 3.0 matured the RAG (Retrieval-Augmented Generation) pipeline, introducing hybrid graph-neural representations and decoding-time probability warping.

- **Core Paradigm**: Hybrid BERT feature extraction + PyTorch Geometric GCN + T5 text polishing.
- **Retrieval Pipeline**: Combined spaCy entity extraction with BM25 passage retrieval over `index.jsonl` corpora.
- **Graph Neural Integration**: Constructed local word co-occurrence graphs over retrieved contexts using PyTorch Geometric, injecting GCN graph embeddings into BERT hidden states via residual connections.
- **Dual Safety Mechanism**:
  - **Reasoning Gate**: Hard threshold rejection for queries scoring above 0.7 in logic conflict.
  - **LogitWarper**: Dynamically suppressed high-uncertainty tokens at each T5 decoding step based on real-time logic scores.
- **Deployment**: Multi-threaded FastAPI microservice with multi-task training and early stopping.

### K.R.I.S.T.Y. v4.0 (2024–2025): Technical Details

This version expanded the system into unified multimodal creative synthesis, outputting three heterogeneous modalities in a single pass.

- **Core Paradigm**: RoBERTa encoder + GPT-2 backbone + FiLM layer modulation + Watts–Strogatz GCN routing.
- **Multimodal Generation**: Single forward pass synthesizes text, 46-DoF motion trajectories (via GRU DanceDecoder), and 64×64 pixel images (via transposed convolution ImageDecoder).
- **Knowledge Topology**: Utilized Watts–Strogatz small-world graph networks to route latent vectors `z` through FiLM layers.
- **Loss Balancing & Post-Processing**: Automated loss balancing across 5 heterogeneous tasks using Kendall's learnable uncertainty weighting; applied exponential velocity smoothing for motion and Laplacian sharpening filters for synthesized images.
- **Safety**: Regular expression jailbreak filters combined with BART zero-shot classification gates.

### K.R.I.S.T.Y. v5.0 (2025): Technical Details

The final iteration transitioned the platform toward Physical AI, Real2Sim2Real alignment, and embodied robotic control.

- **Core Paradigm**: Multi-hop GraphRAG + MoE Kinematics Routing + PyBullet Closed-Loop Simulation + ROS 2 Execution.
- **Persistent Knowledge Graph**: `PersistentGraphStore` executing multi-hop Breadth-First Search (BFS) over SQLite entities.
- **MoE Kinematics Router**: Mixture-of-Experts routing latent vectors `z` to specialized experts:
  - **RoboticsExpert**: Outputs joint angles clamped to ±0.5 rad.
  - **DanceExpert**: Dynamically scaled range motion.
  - **MusicExpert**: Audio feature alignment.
- **Physical Verification Loop**: Trajectories injected into PyBullet under `POSITION_CONTROL` to evaluate joint tracking error and base-tipping penalties, allowing differentiable gradient refinement of latent vectors.
- **Physical Safety & Hardware Output**: Joint-space artificial potential fields for collision avoidance, supervised by an `AgentGraphSupervisor` circuit breaker. Published commands directly to ROS 2 `/joint_commands` topics at 30 Hz, with optional BVH and ONNX export.
- **Status**: Development halted; maintained as an experimental record.

---

## System Architecture

The architecture is structured across modular layers to ensure scalability and cross-modal reasoning:
- **Interaction & Parsing Layer**: Flask/FastAPI interface for natural language parsing, regex intent routing, and parameter extraction.
- **Knowledge & Retrieval Core**: Persistent SQLite multi-hop GraphRAG, BM25 retriever, and NetworkX / PyTorch Geometric graph engines.
- **Inference & Safety Layer**: Rational/Reasoning Gates, LogitWarper, BART zero-shot filters, and Agentic Supervisor circuit-breakers.
- **Multimodal & Kinematics Engine**: FiLM-modulated decoders and MoE Kinematics Routers driving text, motion, vision, and audio synthesis.
- **Physical AI & Execution Layer**: PyBullet simulation closed loop with artificial potential fields, exporting BVH, ONNX models, and ROS 2 topic commands.

## Key Documents and References

- **20260921_01.md**: Comprehensive technical retrospective and architectural notes on personal experiments (A.B.C., A.R.I.E.L., L.A.K.E.S., A.L.L.E.S., H.A.R.P.E.R., K.R.I.S.T.Y. v1.0–v5.0).
- **20251006_01.md**: Detailed specification for K.R.I.S.T.Y. v4.0 multimodal generation architecture.
- **20230622_02.md**: Research documentation on K.R.I.S.T.Y. v2.0 knowledge-based chatbot design and SymPy math modules.
- **20250731_01.md**: Documentation for "Evil A.R.I.E.L." passive user profiling expert systems.
- **20240905_01.md**: Technical overview of A.R.I.E.L. v3.0 retrieval-augmented generation pipelines.

## Installation and Usage

1. **Prerequisites**:
   - Python 3.8+ with PyTorch, PyTorch Geometric, PyBullet, SymPy, NetworkX, spaCy, NLTK.
   - ROS 2 (Humble or newer) for physical hardware topic publishing.
   - GPU recommended for neural inference and simulation pipelines.

2. **Setup**:
```bash
git clone [https://github.com/dev1virtuoso/Machine-Learning.git](https://github.com/dev1virtuoso/Machine-Learning.git)
cd Machine-Learning/Natural Language Processing/K.R.I.S.T.Y.
pip install -r requirements.txt

```

3. **Running the System**:
```bash
python main.py --mode=generate --prompt="Synthesize 46-DoF joint motion for dance routine" --output-format=bvh

```



## Ethical Considerations and Limitations

* **Deprecation**: The repository is deprecated and no longer updated.
* **Latency Constraints**: High-frequency CPU physical simulation in PyBullet introduces latency under heavy load.
* **Cross-Modal Consistency**: Early resolution limits (64×64) in image synthesis and kinematic approximations require post-processing filters.
* **Safety Enforcements**: Physical joint clamping and potential field collision avoidance mitigate unsafe execution, but hardware deployment must be supervised.

## Contributors

* **Author**: Carson Wu
* Contributions welcome for historical archiving! See [CONTRIBUTING.md](https://www.google.com/search?q=CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License, see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

For questions or historical inquiries, contact Carson Wu via GitHub issues or [contact channels](https://www.google.com/search?q=https://github.com/dev1virtuoso/Documentation/blob/main/dev1virtuoso/Attachment/dev1virtuoso/carson-wu.md%23contact).
