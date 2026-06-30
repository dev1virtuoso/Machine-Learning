import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from diffusion_core import AttentionMaskFactory

class KVCacheManager:
    def __init__(self, batch_size: int, max_seq_len: int, num_layers: int, 
                 num_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.k_cache = torch.zeros(self.num_layers, self.batch_size, self.num_heads, self.max_seq_len, self.head_dim, device=self.device, dtype=self.dtype)
        self.v_cache = torch.zeros(self.num_layers, self.batch_size, self.num_heads, self.max_seq_len, self.head_dim, device=self.device, dtype=self.dtype)

    def update_layer(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        seq_len = new_k.size(2)
        max_pos = min(seq_len, self.max_seq_len)
        self.k_cache[layer_idx, :, :, :max_pos, :] = new_k[:, :, :max_pos, :]
        self.v_cache[layer_idx, :, :, :max_pos, :] = new_v[:, :, :max_pos, :]

class ParallelDiffusionDecoder:
    def __init__(self, model: nn.Module, mask_token_id: int, vocab_size: int, eot_token_id: Optional[int] = None):
        self.model = model
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.eot_token_id = eot_token_id
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, 
                 iterations: int = 15, base_threshold: float = 0.85, 
                 decay_factor: float = 0.15, entropy_limit: float = 0.5,
                 temperature: float = 1.0, top_p: float = 0.90, eval_noise: float = 0.05) -> torch.Tensor:
        
        device = prompt_tokens.device
        batch_size, prompt_len = prompt_tokens.shape
        total_len = prompt_len + max_new_tokens
        
        config = getattr(self.model, "config", None)
        num_layers = getattr(config, "num_layers", 12)
        num_heads = getattr(config, "num_heads", 12)
        head_dim = getattr(config, "head_dim", 64)
        dtype = next(self.model.parameters()).dtype
        
        cache_manager = KVCacheManager(
            batch_size=batch_size, max_seq_len=total_len, 
            num_layers=num_layers, num_heads=num_heads, head_dim=head_dim, 
            device=device, dtype=dtype
        )
        
        generated_block = torch.full((batch_size, max_new_tokens), self.mask_token_id, dtype=torch.long, device=device)
        committed_sequence = torch.cat([prompt_tokens, generated_block], dim=1)
        draft_sequence = committed_sequence.clone()
        
        is_committed = torch.zeros((batch_size, total_len), dtype=torch.bool, device=device)
        is_committed[:, :prompt_len] = True
        
        is_prompt = is_committed.clone() 
        
        if hasattr(self.model, "schedule"):
            time_steps = self.model.schedule.get_inference_timesteps(iterations, device)
        else:
            time_steps = torch.linspace(1.0, 0.0, iterations + 1, device=device)
        
        for step in range(iterations):
            current_input = torch.where(is_committed, committed_sequence, draft_sequence)
            padding_mask = (current_input == (config.pad_token_id if config else 0))
            
            t_tensor = torch.full((batch_size,), time_steps[step], device=device, dtype=torch.float32)
            attn_mask = AttentionMaskFactory.create_masks(is_committed, padding_mask)

            logits, _ = self.model(
                current_input, t_tensor, is_prompt=is_prompt, padding_mask=padding_mask, 
                attn_mask=attn_mask, cache_manager=cache_manager
            )
            
            probs = torch.softmax(logits, dim=-1)
            max_probs, greedy_tokens = torch.max(probs, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            
            progress_ratio = step / float(iterations)
            current_threshold = base_threshold - (decay_factor * progress_ratio)
            dynamic_entropy_limit = entropy_limit * (1.0 + progress_ratio * 0.5)
            
            confident_mask = (max_probs > current_threshold) & (entropy < dynamic_entropy_limit)
            force_commit = (step == iterations - 1)
            newly_committed = (confident_mask | force_commit) & ~is_committed
            
            committed_sequence = torch.where(newly_committed, greedy_tokens, committed_sequence)
            is_committed = is_committed | newly_committed
            
            if self.eot_token_id is not None:
                has_eot = ((committed_sequence == self.eot_token_id) & is_committed)[:, prompt_len:]
                if has_eot.any(dim=-1).all():
                    break
                    
            if (is_committed[:, prompt_len:] | padding_mask[:, prompt_len:]).all(dim=-1).all():
                break
                
            if not force_commit:
                noisy_logits = (logits + torch.randn_like(logits) * eval_noise) / max(temperature, 1e-5)
                noisy_logits = noisy_logits.masked_fill(is_committed.unsqueeze(-1), float('-inf'))
                
                sorted_logits, sorted_indices = torch.sort(noisy_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                remove_mask = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                filtered_logits = noisy_logits.masked_fill(remove_mask, float('-inf'))
                
                flat_filtered_probs = torch.softmax(filtered_logits.view(-1, self.vocab_size), dim=-1)
                sampled_tokens = torch.multinomial(flat_filtered_probs, num_samples=1).view(batch_size, total_len)
                
                draft_sequence = torch.where(~is_committed, sampled_tokens, committed_sequence)
                
        return committed_sequence