# Usage Guide

## 1. Overview
This document provides instructions for operating the H.A.R.P.E.R. framework pipeline. The system processes text dialogue histories, identifies grammatical and rhetorical structures, tracks contextual flows over multiple turns, optimizes generation parameters using a policy-gradient Reinforcement Learning from Human Feedback (RLHF) strategy, and maps output token configurations directly into detailed acoustic waveforms via a location-sensitive neural speech synthesizer.

## 2. Install Requirements
Before executing the pipeline, verify that the required baseline deep learning environments and processing packages are available. Execute the following commands in the terminal context to install dependencies:

```bash
pip install torch
pip install numpy
pip install scipy
pip install spacy
pip install librosa
```

Additionally, download the required language models for the syntactic parsing layer:

```bash
python -m spacy download en_core_web_sm
```

## 3. Configuration and Parameter Settings

To run the system successfully across your specific text and audio datasets, multiple variable dimensions, paths, and hyperparameters must be set manually within `main.py` or passing dictionaries before initialization.

### Hardcoded Constants

The following architecture dimensions are specified in the structural execution initialization block and must align with your token dictionaries and targeting features:

* `VOCAB_SIZE`: Total number of individual language tokens available in your language dictionary (configured to `500` inside the factory mockup layer).
* `EMBEDDING_DIM`: Token mapping vector size, default configured to `128`.
* `HIDDEN_SIZE`: Size of the recurrent layer memory matrices for the encoders, text decoders, and style classifiers, default configured to `128`.
* `TONE_DIM`: Embedded size allocated to discrete tone variables, default configured to `32`.
* `NUM_TONES`: Quantized count of target speaking profiles or classes, default configured to `3`.
* `RHETORIC_DIM`: Vector size mapping high-dimensional syntactic styles, default configured to `256`.
* `MEL_DIM`: Spectral channel count passed into the acoustic synthesis module, default configured to `80`.

### Manual System Modifications

To migrate the code base from the baseline execution simulation toward functional application environments, configure these configurations directly within the source classes:

#### Vocabulary Dictionary Mapping

The rule-based and deep statistical language layers rely on text-to-index transitions. Supply a populated word-to-token index lookup table during initialization:

```python
custom_vocab = {"standard": 2, "evaluation": 3, "optimization": 4}
analyzer = RhetoricAnalyzer(spacy_model='en_core_web_sm', vocab_dict=custom_vocab)

```

#### Pre-trained Syntactic Model Weight Paths

The text sequence structural feature analyzer leverages a deep Bidirectional LSTM to build continuous rhetorical representations. Update the placeholder constructor values with your specific localized checkpoints:

```python
analyzer = RhetoricAnalyzer(
    spacy_model='en_core_web_sm',
    ml_model_path='/absolute/path/to/rhetoric_bilstm_weights.pt',
    vocab_dict=custom_vocab
)
```

#### Token Definitions

The RLHF policy search and generation optimization loop checks for precise sequence delimiters to maintain matrix alignments. Match these initialization arguments to your true vocabulary indices:

```python
rlhf_agent = PolicyGradientRLHF(
    generator_model=seq2seq_model,
    discriminator=discriminator,
    style_classifier=style_classifier,
    optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer,
    pad_token=0,
    sos_token=1
)
```

## 4. Troubleshooting

### Runtime CUDA Execution Errors

* **Symptom**: `RuntimeError: Expected object of backend CUDA but got backend CPU` or associated sequence dimensional device mismatches.
* **Mitigation**: The execution pipeline leverages automated context determination variables (`torch.cuda.is_available()`). If you use explicit variable conversions or enforce manual GPU selection via `.cuda()`, confirm that incoming custom validation sets, tensor dimensions, or target matrices generated outside the data factories are cast to `Variable(tensor.cuda())` if `USE_CUDA` resolves to true.

### Index Collapse inside the Policy Gradient Sample Loops

* **Symptom**: Multinomial distribution sampling failure alerts or NaN metrics appearing inside the RLHF advantage step logs.
* **Mitigation**: Check your manual parameter declaration for `VOCAB_SIZE`. If the token output projection layer maps indexes beyond the true boundaries of your input file tensors, or if individual log-probabilities contain unmasked zero states, `torch.multinomial` will fail. Ensure that `pad_token_id` is passed correctly to clear empty configurations before calculation.

### Phase Realignment Distortions during Speech Reconstruction

* **Symptom**: High audio artifact noises or robotic metallic sounds occurring across the final synthesized waveform sequences.
* **Mitigation**: The system leverages an algebraic `GriffinLimVocoder` pass. The default iterations parameter within `main.py` is initialized to a low processing pass (`n_iter=5`) to confirm code path correctness. For standard high-fidelity audio generation targets, override the invocation within the structural code block to raise the alignment iteration ceiling:

```python
waveform = GriffinLimVocoder.synthesize(linear_spec_np, n_iter=50)
```

### Zero-length Sequence Failures in Syntactic Feature Calculations

* **Symptom**: Division by zero errors or empty array slice warnings inside `calculate_prosody_correlation` or `_detect_anaphora`.
* **Mitigation**: Ensure that the sequence inputs include standard punctuation or text tokens. If data samples are packed completely with padding mask variables (index value `0`), the Dynamic Time Warping matrix tracker `_dtw_alignment` receives empty matrices, preventing valid path computation. Enforce minimum filter constraints (length greater than or equal to two valid tokens) on source frames before calling the evaluation layer.
* 