import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import lora_moe

class LuminaConfig:
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 2048, 
                 num_layers: int = 24, num_heads: int = 16, head_dim: int = 128,
                 num_experts: int = 8, top_k: int = 2, inter_dim: int = 5632,
                 mask_token_id: int = 31999, pad_token_id: int = 0,
                 model_size_params: float = 5e9, label_smoothing: float = 0.1,
                 reweight_factor: float = 1.0, z_loss_coeff: float = 0.001, 
                 load_coeff: float = 0.01):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.inter_dim = inter_dim
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.model_size_params = model_size_params
        self.label_smoothing = label_smoothing
        self.reweight_factor = reweight_factor
        self.z_loss_coeff = z_loss_coeff
        self.load_coeff = load_coeff

class DiffusionSchedule:
    def __init__(self, schedule_s: float = 0.008, schedule_mode: str = "importance"):
        super().__init__()
        assert schedule_mode in ["importance", "uniform"]
        self.schedule_s = schedule_s
        self.schedule_mode = schedule_mode
        self.alpha_0 = math.cos((schedule_s / (1.0 + schedule_s)) * (math.pi / 2)) ** 2

    def get_p_keep(self, t: torch.Tensor) -> torch.Tensor:
        s = self.schedule_s
        alpha_t = torch.cos(((t + s) / (1.0 + s)) * (math.pi / 2)) ** 2
        p_keep = alpha_t / self.alpha_0
        return torch.clamp(p_keep, min=0.0, max=1.0)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        u = torch.rand((batch_size,), device=device)
        if self.schedule_mode == "importance":
            return torch.clamp(torch.sqrt(u), min=0.0, max=1.0)
        return u

    def get_inference_timesteps(self, iterations: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(1.0, 0.0, iterations + 1, device=device)

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim > 1: t = t.view(-1)
        half_dim = self.hidden_dim // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.hidden_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return self.mlp(emb)

class MaskedDiffusionProcess(nn.Module):
    def __init__(self, config: LuminaConfig, schedule_mode: str = "importance"):
        super().__init__()
        self.config = config
        self.schedule = DiffusionSchedule(schedule_mode=schedule_mode)

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x_0.shape
        t_reshaped = t.unsqueeze(-1).expand(-1, seq_len)
        p_keep_per_token = self.schedule.get_p_keep(t_reshaped)
        
        t_zero_mask = (t_reshaped < 1e-6)
        p_keep_per_token = torch.where(t_zero_mask, torch.ones_like(p_keep_per_token), p_keep_per_token)
            
        rand_probs = torch.rand_like(x_0, dtype=torch.float32)
        mask_condition = (rand_probs > p_keep_per_token)
        
        padding_mask = (x_0 == self.config.pad_token_id)
        mask_condition = mask_condition & ~padding_mask
        
        x_t = x_0.clone()
        x_t[mask_condition] = self.config.mask_token_id
        return x_t, mask_condition, padding_mask, p_keep_per_token

    def compute_loss(self, logits: torch.Tensor, x_0: torch.Tensor, mask_condition: torch.Tensor, p_keep_per_token: torch.Tensor) -> torch.Tensor:
        flat_logits = logits.view(-1, self.config.vocab_size)
        flat_targets = x_0.view(-1)
        
        ce_loss = F.cross_entropy(
            flat_logits, flat_targets, 
            reduction='none', 
            label_smoothing=self.config.label_smoothing,
            ignore_index=self.config.pad_token_id
        ).view(x_0.shape)
        
        masked_ce_loss = ce_loss * mask_condition.float()
        
        snr_weights = 1.0 / torch.clamp(1.0 - p_keep_per_token, min=1e-4)
        weighted_loss = masked_ce_loss * snr_weights * self.config.reweight_factor
        
        num_masked_tokens = torch.clamp(mask_condition.float().sum(dim=-1), min=1.0)
        return (weighted_loss.sum(dim=-1) / num_masked_tokens).mean()

    @torch.no_grad()
    def compute_vlb_bound(self, logits: torch.Tensor, x_0: torch.Tensor, mask_condition: torch.Tensor, p_keep_per_token: torch.Tensor) -> torch.Tensor:
        flat_logits = logits.view(-1, self.config.vocab_size)
        flat_targets = x_0.view(-1)
        
        ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none', ignore_index=self.config.pad_token_id).view(x_0.shape)
        masked_ce_loss = ce_loss * mask_condition.float()
        
        vlb_weights = 1.0 / torch.clamp(1.0 - p_keep_per_token, min=1e-4)
        vlb_loss = masked_ce_loss * vlb_weights
        
        return (vlb_loss.sum(dim=-1) / torch.clamp(mask_condition.float().sum(dim=-1), min=1.0)).mean()

class AttentionMaskFactory:
    @staticmethod
    def create_masks(is_committed: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = is_committed.shape
        device = is_committed.device
        
        valid_mask = (~padding_mask).unsqueeze(1)
        base_bidirectional = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)
        
        is_comm = is_committed.unsqueeze(1)
        
        committed_attn = is_comm.unsqueeze(-1) & is_comm.unsqueeze(-2)
        
        draft_attn = (~is_comm).unsqueeze(-1) & valid_mask.unsqueeze(-2)
        
        final_allowed_mask = torch.where(
            is_comm.unsqueeze(-1), 
            committed_attn, 
            draft_attn
        ) & base_bidirectional
        
        attn_mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=device, dtype=torch.float32)
        return attn_mask.masked_fill(~final_allowed_mask, float('-inf'))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=torch.device("cpu"), dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if (seq_len > self.max_seq_len_cached or 
            self.cos_cached.device != x.device or 
            self.cos_cached.dtype != x.dtype):
            self._set_cos_sin_cache(max(seq_len, 8192), device=x.device, dtype=x.dtype)
            
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class LuminaAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None, cache_manager: Optional[Any] = None, 
                layer_idx: Optional[int] = None) -> torch.Tensor:
        b, s, _ = h.shape
        q = self.q_proj(h).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if cache_manager is not None and layer_idx is not None:
            cache_manager.update_layer(layer_idx, k, v)
            k = cache_manager.k_cache[layer_idx, :, :, :s, :]
            v = cache_manager.v_cache[layer_idx, :, :, :s, :]
            
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask[..., :s, :s]
            
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(b, s, self.num_heads * self.head_dim)
        return self.out_proj(output)

class LuminaDecoderLayer(nn.Module):
    def __init__(self, config: LuminaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_ln = nn.LayerNorm(config.hidden_dim)
        self.attn = LuminaAttention(config.hidden_dim, config.num_heads, config.head_dim)
        self.moe_ln = nn.LayerNorm(config.hidden_dim)
        
        self.moe = lora_moe.SparseLoRAMoE(
            hidden_dim=config.hidden_dim, inter_dim=config.inter_dim,
            num_experts=config.num_experts, top_k=config.top_k,
            capacity_factor=1.25
        )

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None, cache_manager: Optional[Any] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = h
        h_attn = self.attn(self.attn_ln(h), cos, sin, attn_mask, cache_manager, self.layer_idx)
        h = residual + h_attn
        
        residual = h
        moe_out, moe_aux_loss = self.moe(self.moe_ln(h))
        h = residual + moe_out
        
        return h, moe_aux_loss.mean()

class LuminaModel(nn.Module):
    def __init__(self, config: LuminaConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.type_emb = nn.Embedding(2, config.hidden_dim)
        self.rotary_emb = RotaryEmbedding(config.head_dim)
        self.t_projector = TimestepEmbedding(config.hidden_dim)
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        )
        
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        
        self.layers = nn.ModuleList([LuminaDecoderLayer(config, i) for i in range(config.num_layers)])
        self.final_ln = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor, is_prompt: Optional[torch.Tensor] = None, 
                padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None, 
                cache_manager: Optional[Any] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s = x.shape
        
        if is_prompt is None:
            is_prompt = torch.zeros((b, s), dtype=torch.bool, device=x.device)
            
        type_ids = (~is_prompt).long()
        h = self.token_emb(x) + self.type_emb(type_ids)
        cos, sin = self.rotary_emb(h, s)
        
        t_emb = self.t_projector(t).unsqueeze(1).expand(-1, s, -1) 
        scale, shift = self.gate_mlp(t_emb).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift
        
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=h.dtype)
        for layer in self.layers:
            if self.config.model_size_params > 1e9 and self.training:
                h, aux_loss = torch.utils.checkpoint.checkpoint(
                    layer.__call__, h, cos, sin, attn_mask, cache_manager, 
                    use_reentrant=False, preserve_rng_state=True
                )
            else:
                h, aux_loss = layer(h, cos, sin, attn_mask, cache_manager)
                
            total_aux_loss = total_aux_loss + aux_loss
            
        return self.lm_head(self.final_ln(h)), total_aux_loss