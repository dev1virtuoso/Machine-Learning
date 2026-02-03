# next-token_predictor

A from-scratch character-level **GPT** (small transformer) trained on the Tiny Shakespeare dataset.  
Implements the core "Attention is All You Need" architecture with multi-head self-attention, feed-forward layers, residual connections, and layer normalization — very close to the original nanoGPT style.

<img src="https://img.shields.io/badge/PyTorch-2.x-orange?style=flat&logo=pytorch"> <img src="https://img.shields.io/badge/dataset-Tiny%20Shakespeare-red"> <img src="https://img.shields.io/badge/parameters-~10.5M-blue">

## Features

- Pure PyTorch implementation (no external trainer libraries)
- Character-level tokenization
- Multi-head self-attention with causal masking
- Train + inference scripts separated
- Supports CUDA / MPS / CPU automatically
- Temperature-controlled sampling during generation
- Saves model and tokenizer for easy reuse

## Model Architecture (at a glance)

| Component | Value |
|----|---|
| Embedding dim (`n_embd`) | 512 |
| Heads | 8 |
| Layers | 12 |
| Block size (context) | 64 |
| Dropout | 0.1 |
| Total parameters | ~10.5 million |

## Files

```text
next-token_predictor/
├── train.py               # Training script
├── demo.py                # Load model + generate text
├── input.txt              # Tiny Shakespeare (downloaded if missing)
├── tiny_shakespeare_char_gpt.pt   # Trained weights (gitignored recommended)
└── tokenizer.pkl          # Character -> id mapping
```

## Quick Start

### 1. Train the model

```bash
python train.py
```

- ~30,000 steps (adjust `max_iters` to taste)
- Prints train/val loss every 500 steps
- Saves model -> `tiny_shakespeare_char_gpt.pt`
- Saves tokenizer -> `tokenizer.pkl`

Training takes:

- ≈ 15–40 minutes on a decent GPU
- Several hours on CPU / Mac M1/M2 (MPS)

### 2. Generate text

After training (or after downloading a checkpoint):

```bash
python demo.py
```

Example output (temperature = 0.92):

```
JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

...
```

You can easily change the starting prompt in `demo.py`:

```python
prompt = "ROMEO:\nO, she doth teach the torches to burn bright!"
context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
```

## Requirements

```text
torch >= 1.13
requests
```

Install:

```bash
pip install torch requests
```

## Tips & Tricks

- Increase `max_iters` to 50k–100k for noticeably better quality
- Try temperature 0.7–1.0 (lower = more coherent, higher = more creative/random)
- Larger `block_size` (128–256) usually helps if you have memory
- You can experiment with learning rate decay or cosine scheduler

## Acknowledgments

Heavily inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his excellent "Neural Networks: Zero to Hero" YouTube series.

Enjoy Shakespeare in the style of a tiny transformer! 🧙‍♂️