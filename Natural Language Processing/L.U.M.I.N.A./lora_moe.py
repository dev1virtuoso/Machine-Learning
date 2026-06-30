import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.scaling = lora_alpha / r
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(self.dropout(x), self.lora_A)
        return base + F.linear(lora, self.lora_B) * self.scaling

class StackedLoRAExperts(nn.Module):
    def __init__(self, num_experts: int, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.scaling = lora_alpha / r
        
        self.weight = nn.Parameter(torch.empty(num_experts, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_experts, 1, out_features))
        self.lora_A = nn.Parameter(torch.empty(num_experts, in_features, r))
        self.lora_B = nn.Parameter(torch.empty(num_experts, r, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.weight, self.lora_A]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.bmm(x, self.weight) + self.bias
        lora = torch.bmm(torch.bmm(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora

class SparseLoRAMoE(nn.Module):
    def __init__(self, hidden_dim: int, inter_dim: int, num_experts: int = 8, top_k: int = 2, 
                 r: int = 8, lora_alpha: int = 16, capacity_factor: float = 1.25, 
                 z_loss_coeff: float = 0.001, load_coeff: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.inter_dim = inter_dim
        self.z_loss_coeff = z_loss_coeff
        self.load_coeff = load_coeff
        
        self.router = LoRALinear(hidden_dim, num_experts, r=r, lora_alpha=lora_alpha)
        self.expert_gate_up = StackedLoRAExperts(num_experts, hidden_dim, 2 * inter_dim, r=r, lora_alpha=lora_alpha)
        self.expert_down = StackedLoRAExperts(num_experts, inter_dim, hidden_dim, r=r, lora_alpha=lora_alpha)
        self.expert_bias = nn.Parameter(torch.zeros(num_experts, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = x.shape
        flat_x = x.view(-1, hidden_dim)
        total_tokens = flat_x.size(0)
        
        router_logits = self.router(flat_x)
        
        if self.training:
            noise_scale = 0.05 / math.sqrt(self.num_experts)
            router_logits = router_logits + torch.randn_like(router_logits) * noise_scale
            
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
            
        raw_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(raw_probs, self.top_k, dim=-1)
        
        topk_probs = topk_probs / torch.clamp(topk_probs.sum(dim=-1, keepdim=True), min=1e-6)
        
        flat_experts = topk_indices.view(-1)
        flat_probs = topk_probs.view(-1)
        flat_tokens = torch.arange(total_tokens, device=x.device).repeat_interleave(self.top_k)
        
        sorted_experts, sort_idx = torch.sort(flat_experts)
        sorted_tokens = flat_tokens[sort_idx]
        sorted_probs = flat_probs[sort_idx]
        
        expert_counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        start_offsets = F.pad(torch.cumsum(expert_counts, dim=0)[:-1], (1, 0))
        local_positions = torch.arange(len(sorted_experts), device=x.device) - start_offsets[sorted_experts]
        
        capacity = max(int((total_tokens / self.num_experts) * self.capacity_factor * self.top_k), 4)
        valid_mask = local_positions < capacity
        
        valid_experts = sorted_experts[valid_mask]
        valid_tokens = sorted_tokens[valid_mask]
        valid_probs = sorted_probs[valid_mask]
        valid_local_positions = local_positions[valid_mask]
        
        expert_inputs = torch.zeros(self.num_experts, capacity, hidden_dim, device=x.device, dtype=x.dtype)
        expert_inputs[valid_experts, valid_local_positions] = flat_x[valid_tokens]
        
        gate_up_out = self.expert_gate_up(expert_inputs)
        gate, up = gate_up_out.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        expert_outputs = self.expert_down(activated) + self.expert_bias
        
        extracted_results = expert_outputs[valid_experts, valid_local_positions] * valid_probs.unsqueeze(-1)
        moe_output = torch.zeros_like(flat_x).index_add_(0, valid_tokens, extracted_results)
        global_expert_counts = torch.bincount(flat_experts, minlength=self.num_experts)
        frac_tokens = global_expert_counts.float() / max(1.0, float(flat_experts.size(0)))
        mean_routing = raw_probs.mean(dim=0)
        
        load_balancing_loss = self.num_experts * torch.sum(frac_tokens * mean_routing)
        total_aux_loss = (self.load_coeff * load_balancing_loss) + (self.z_loss_coeff * z_loss)
        
        return moe_output.view(batch_size, seq_len, hidden_dim), total_aux_loss
    
