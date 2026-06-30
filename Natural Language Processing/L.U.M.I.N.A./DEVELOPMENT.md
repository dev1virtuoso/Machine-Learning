# Development and Integration Guide

## 1. Overview

This guide provides high-level instructions for developers integrating L.U.M.I.N.A. components into existing training or inference pipelines.

## Integration Tutorial

### Step 1: Model Setup

```python
from diffusion_core import LuminaConfig, LuminaModel, MaskedDiffusionProcess
from parallel_decoder import ParallelDiffusionDecoder

config = LuminaConfig(
    vocab_size=your_vocab_size,
    hidden_dim=2048,
    num_layers=24,
    num_heads=16,
    head_dim=128,
    num_experts=8,
    top_k=2,
    inter_dim=5632,
    mask_token_id=your_mask_token_id,
    pad_token_id=your_pad_token_id,
    model_size_params=5e9
)

model = LuminaModel(config)
diffusion_process = MaskedDiffusionProcess(config)
```

### Step 2: Training Integration

Integrate into your training loop:

```python
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        x_0 = batch["input_ids"].to(device)
        t = diffusion_process.schedule.sample_timesteps(x_0.shape[0], device)
        
        x_t, mask_condition, padding_mask, p_keep = diffusion_process.forward_process(x_0, t)
        
        logits, aux_loss = model(x_t, t)
        
        loss = diffusion_process.compute_loss(logits, x_0, mask_condition, p_keep)
        total_loss = loss + aux_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

### Step 3: Inference Integration

```python
model.eval()
decoder = ParallelDiffusionDecoder(
    model=model,
    mask_token_id=config.mask_token_id,
    vocab_size=config.vocab_size,
    eot_token_id=your_eot_token_id
)

def generate_text(prompt_tokens, max_new_tokens=512):
    return decoder.generate(
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        iterations=20,
        base_threshold=0.82,
        decay_factor=0.12,
        temperature=0.95,
        top_p=0.92
    )
```

### Step 4: Custom Extensions

- Extend `LuminaDecoderLayer` to add new modules.
- Override `AttentionMaskFactory.create_masks` for custom masking logic.
- Modify `SparseLoRAMoE` in `lora_moe.py` to adjust routing or expert architecture.
- Implement custom timestep scheduling by subclassing `DiffusionSchedule`.

## High Level Troubleshooting

- **Scaling Issues**: For larger models, enable gradient checkpointing by ensuring `model_size_params > 1e9`. Monitor auxiliary losses for MoE stability.
- **Memory Management**: Use `KVCacheManager.reset()` between generations. Consider model parallelism for layers when scaling beyond single GPU.
- **Convergence Problems**: Tune `label_smoothing`, `reweight_factor`, `z_loss_coeff`, and `load_coeff` based on validation VLB bound from `compute_vlb_bound`.
- **Inference Quality**: Adjust `base_threshold`, `decay_factor`, and `entropy_limit` according to task-specific confidence requirements. Increase `iterations` for higher quality at the cost of speed.
- **Compatibility**: Ensure all tensors share the same device and dtype. When integrating with HF tokenizers, map special tokens correctly.
- **Performance Bottlenecks**: Profile `SparseLoRAMoE` routing and `LuminaAttention` with cache enabled for long sequences.

Test integration incrementally: start with small `num_layers` and `hidden_dim`, then scale up while monitoring loss curves and generation coherence.