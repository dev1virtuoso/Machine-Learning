# Usage Guide

## 1. Overview

This document provides instructions on navigating the configuration, code structures, and operational execution of the Artificial Linguistic Learning and Emulation System (A.L.L.E.S.). This codebase offers tools to train recurrent model variations on translation, text-style alignment, and contextual text generation.

## 2. Install Requirements

The system requires a standard Python environment outfitted with numerical processing libraries and a framework equipped for automated backpropagation and CUDA acceleration. Ensure the following dependencies are available before initiating execution:

* Python 3.8 or greater
* PyTorch 1.10 or greater (CUDA compute capabilities recommended for training acceleration)

## ## 3. Troubleshooting

* **File Descriptor Allocation Failures**: If multi-process child processes collapse during initialization, confirm your operating system file descriptor limits (`ulimit -n`) are sufficiently configured to manage the memory maps allocated by `DiskIndexedALLESDataset`.
* **NaN Losses During Discriminator Pass**: Extreme gradient updates from the Style Discriminator may destabilize the encoder hidden states. Lower the `alpha` parameters or increase the generator updates relative to the discriminator steps inside the training iteration loop.
* **CUDA Out Of Memory (OOM)**: The memory bank module keeps a historical hidden representation vector tensor. If memory errors trigger, scale down your training batch size flag or shorten the max history parameters defined inside the context bank initialization.
* **Index Errors in Embedding Layers**: If execution fails with an index out-of-range flag from the embedding lookups, delete your current `.pkl` vocabulary cache to force the dataset system to rebuild structural tokens matching your newest numerical labels.

## Folder Overview

* `main.py`: The entry executable component containing structural pipelines for model data ingestion, curriculum scheduling logic, Elastic Weight Consolidation execution, validation cycles, and training loops.
* `module.py`: The deep learning primitives housing custom PyTorch model modules including the Vocabulary builder, Gradient Reversal wrapper, Style Discriminator network, Attention engine, Encoder, and conditional Decoders.

## Core Functionality

The A.L.L.E.S. framework operates on an advanced Encoder-Decoder core wrapped with supportive modules to control text delivery style and structural coherence over long conversation streams. Data flows from raw split-text input streams directly through local files mapped to virtual memory maps. Training switches between an auto-encoding pretraining scheme (which injects random unk masks to enforce denoising autoencoding robustness) and a fine-tuning mechanism focused on specific translation or stylistic response outputs.

## Usage Guide

The execution environment relies on command-line parameter passing to provision source text tracking, metadata matching boundaries, and model parameter checkpoints.

### Command Line Arguments

* `--src_data` (str, Required): File path leading to line-delimited tokenized source text.
* `--trg_data` (str, Required): File path leading to line-delimited tokenized target translation text.
* `--lang_data` (str, Required): File path leading to line-delimited integer indices representing language identifiers matching the source text rows.
* `--style_data` (str, Required): File path leading to line-delimited integer indices representing stylistic or identity persona classifications matching the source text rows.
* `--vocab_path` (str, Optional): Desired path to serialize or load the generated global shared vocabulary pickle. Defaults to `shared_vocab.pkl`.
* `--task_mode` (str, Optional): Selection constraint determining step behavior. Choose between `pretrain` or `fine_tune`. Defaults to `fine_tune`.
* `--epochs` (str/int, Optional): Total tracking iterations allocated for training cycles. Defaults to `10`.
* `--resume_checkpoint` (str, Optional): Path pointing to a previously saved checkpoint model payload to re-establish saved parameters, optimizers, and tracking weights.

### Manual Parameter Configuration

Several configurations are established directly inside the script components or layer instantiations and require manual modifications within the source code files if customizations are needed:

* **Model Layer Dimensioning (`main.py`)**: The dimensions for sequence embeddings and hidden memory states are set inside the initialization sequence of the model architecture:
```python
encoder = Encoder(len(vocab), 256, 512, 512, 0.4)
decoder = Decoder(len(vocab), 256, 512, 512, num_langs, num_styles, 0.4)
```


Modify the values `256` (Embedding Dimensions) and `512` (RNN Hidden dimensions) to scale model capacities up or down.
* **Optimizer Rates (`main.py`)**: Initial learning rates for the models are isolated inside the Adam class calls:
```python
optimizer = optim.Adam(model.parameters(), lr=0.0005)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
```


Adjust the `lr` field to alter learning velocity or optimize convergence behaviors.
* **Denoising Noise Ratios (`main.py`)**: The base token masking perturbation profile inside `DiskIndexedALLESDataset` uses a static parameter value:
```python
self.noise_ratio = 0.15
```


This controls token masking frequency during the model pretraining stage.
* **Curriculum Boundaries (`main.py`)**: Steps managing length scaling over consecutive training cycles are restricted by boundaries defined inside the scheduler class:
```python
class CurriculumScheduler:
    def __init__(self, initial_max_len=15, progression_step=5):
```


Alter `initial_max_len` to alter sentence length constraints at structural step zero.
* **EWC Penalty Weight (`main.py`)**: The structural scaling parameter that balances the Elastic Weight Consolidation penalty loss against general cross-entropy error is managed within the training loop step function:
```python
if ewc is not None:
    loss += 5000.0 * ewc.penalty(model)
```


Change the static scale coefficient `5000.0` to balance model rigidity against plastic adaptability on new language domains.
* **History Context Depth (`main.py`)**: The conversational long-term memory capacity window parameter is locked within the execution context initialization:
```python
memory_bank = PersistentMemoryBank(len(dataset), decoder.rnn.hidden_size, max_history=5)
```


Tune `max_history` to adjust the maximum number of multi-turn conversational interactions tracked simultaneously in memory.

## Common Use Cases

* **Stylistic Chatbot Persona Generation**: Emulating custom textual behaviors (e.g., historical authors, modern dialect datasets) while preserving the original conversational meaning across multi-turn exchanges.
* **Domain Adaptation via Continuous Learning**: Moving an initialized translation pipeline across distinct jargon domains or separate target corpora without overwriting existing translation vocabulary foundations.
* **Resource-Constrained Sequence Matching**: Testing non-Transformer translation strategies on constrained edge computing infrastructure lacking large-scale multi-GPU arrangements.