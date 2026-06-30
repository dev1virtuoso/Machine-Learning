# Usage Guide

## 1. Overview

This guide explains how to use the L.U.M.I.N.A. components for model initialization, training, and inference. The core files are `diffusion_core.py`, `lora_moe.py`, and `parallel_decoder.py`.

## 2. Install Requirements

Ensure the following dependencies are installed:

- Python 3.10+
- PyTorch 2.0+ with CUDA support (recommended)
- Additional packages: `torch.nn.functional`, `math`, `typing`

## Model Configuration

Key parameters that must be set manually in `LuminaConfig`:

```python
from diffusion_core import LuminaConfig

config = LuminaConfig(
    vocab_size=32000,
    hidden_dim=2048,
    num_layers=24,
    num_heads=16,
    head_dim=128,
    num_experts=8,
    top_k=2,
    inter_dim=5632,
    mask_token_id=31999,
    pad_token_id=0,
    model_size_params=5e9,
    label_smoothing=0.1,
    reweight_factor=1.0,
    z_loss_coeff=0.001,
    load_coeff=0.01
)
```

## Training Usage

```python
import torch
from diffusion_core import LuminaModel, MaskedDiffusionProcess, LuminaConfig

config = LuminaConfig(
    vocab_size=32000,
    hidden_dim=2048,
    num_layers=24,
    num_heads=16,
    head_dim=128,
    num_experts=8,
    top_k=2,
    inter_dim=5632,
    mask_token_id=31999,
    pad_token_id=0,
    model_size_params=5e9,
    label_smoothing=0.1,
    reweight_factor=1.0,
    z_loss_coeff=0.001,
    load_coeff=0.01
)
model = LuminaModel(config)
diffusion_process = MaskedDiffusionProcess(config)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    x_0 = batch["input_ids"].to(device)
    t = diffusion_process.schedule.sample_timesteps(x_0.size(0), device)
    
    x_t, mask_condition, padding_mask, p_keep = diffusion_process.forward_process(x_0, t)
    
    logits, aux_loss = model(x_t, t)
    
    loss = diffusion_process.compute_loss(logits, x_0, mask_condition, p_keep)
    total_loss = loss + aux_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## Inference Usage

```python
from parallel_decoder import ParallelDiffusionDecoder
from diffusion_core import LuminaModel, LuminaConfig

config = LuminaConfig(
    vocab_size=32000,
    hidden_dim=2048,
    num_layers=24,
    num_heads=16,
    head_dim=128,
    num_experts=8,
    top_k=2,
    inter_dim=5632,
    mask_token_id=31999,
    pad_token_id=0,
    model_size_params=5e9,
    label_smoothing=0.1,
    reweight_factor=1.0,
    z_loss_coeff=0.001,
    load_coeff=0.01
)
model = LuminaModel(config)
model.eval()

decoder = ParallelDiffusionDecoder(
    model=model,
    mask_token_id=config.mask_token_id,
    vocab_size=config.vocab_size,
    eot_token_id=None
)

prompt_tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda")  # Example prompt

generated = decoder.generate(
    prompt_tokens=prompt_tokens,
    max_new_tokens=256,
    iterations=15,
    base_threshold=0.85,
    decay_factor=0.15,
    entropy_limit=0.5,
    temperature=1.0,
    top_p=0.90,
    eval_noise=0.05
)

```

## 3. Troubleshooting

- **CUDA Out of Memory**: Reduce `hidden_dim`, `num_layers`, `max_new_tokens`, or batch size. Use `torch.cuda.empty_cache()` before generation.
- **NaN Loss**: Check that `pad_token_id` and `mask_token_id` are correctly set and not overlapping with valid tokens. Lower learning rate if gradients explode.
- **Slow Inference**: Increase `iterations` gradually. Ensure `KVCacheManager` max_seq_len matches total sequence length.
- **MoE Routing Issues**: Adjust `capacity_factor` (default 1.25), `load_coeff`, or `z_loss_coeff` if load balancing loss is high.
- **Attention Mask Errors**: Verify `is_committed` and `padding_mask` tensors have matching shapes.
- **Rotary Embedding Cache Issues**: Ensure consistent device and dtype across model components.

For custom tokenizers, always align `vocab_size`, `mask_token_id`, and `pad_token_id` with your tokenizer vocabulary.