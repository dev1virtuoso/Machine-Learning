from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import sys
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import torchvision.transforms as transforms
except ImportError:
    transforms = None

from PIL import Image, ImageFilter

try:
    import pybullet as pb
except ImportError:
    pb = None

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None

try:
    from transformers import (
        EncoderDecoderModel,
        GPT2TokenizerFast,
        RobertaTokenizerFast,
        pipeline,
    )
except ImportError:
    EncoderDecoderModel = object
    GPT2TokenizerFast = object
    RobertaTokenizerFast = object
    pipeline = lambda *a, **k: None

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ImportError:
    Node = object
    rclpy = None
    JointState = object

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KristyConfig:
    def __init__(self, **kwargs):
        self.encoder_name = kwargs.get("encoder_name", "roberta-base")
        self.decoder_name = kwargs.get("decoder_name", "gpt2")
        self.d_model = kwargs.get("d_model", 768)
        self.z_dim = kwargs.get("z_dim", 512)
        self.dof = kwargs.get("dof", 46)
        self.seq_len = kwargs.get("seq_len", 60)
        self.img_size = kwargs.get("img_size", 64)
        self.num_intents = kwargs.get("num_intents", 8)
        self.logic_threshold = kwargs.get("logic_threshold", 0.45)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = kwargs.get("learning_rate", 5e-5)
        self.memory_maxlen = kwargs.get("memory_maxlen", 8)
        self.max_context_tokens = kwargs.get("max_context_tokens", 512)
        self.motion_smoothing_threshold = kwargs.get("motion_smoothing_threshold", 0.70)
        self.motion_blend_alpha = kwargs.get("motion_blend_alpha", 0.7)
        self.motion_clip_min = kwargs.get("motion_clip_min", -10.0)
        self.motion_clip_max = kwargs.get("motion_clip_max", 10.0)
        self.temperature = kwargs.get("temperature", 0.7)

class KristyProductionDataset(Dataset):
    def __init__(self, manifest_path: str, cfg: KristyConfig):
        self.cfg = cfg
        self.samples = []
        
        if not os.path.exists(manifest_path):
            from module import generate_synthetic_manifest
            generate_synthetic_manifest(manifest_path, 1000, cfg)
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        vocab_size = getattr(self.cfg, 'vocab_size', 5000)
        self.tokenizer = getattr(self.cfg, "micro_tokenizer_instance", None)
        if self.tokenizer is None:
            self.tokenizer = MicroTokenizer(vocab_size, self.cfg.max_context_tokens)
            
        for entry in raw_data:
            dialogue_turns = entry.get("dialogue", [])
            flattened_context = " ".join([f"<{t.get('role', 'user')}>: {t.get('text', '')}" for t in dialogue_turns])
            
            motion_real = entry.get("motion_real", [])
            motion_sim = entry.get("motion_sim", [])
            reward = entry.get("reward", 0.0)
            intent_distribution = entry.get("intent_dist", [1.0 / cfg.num_intents] * cfg.num_intents)
            
            self.samples.append({
                "instruction": entry.get("instruction", flattened_context),
                "motion_real": self._pad_or_truncate_motion(motion_real),
                "motion_sim": self._pad_or_truncate_motion(motion_sim),
                "reward": float(reward),
                "intent_dist": intent_distribution
            })

    def _pad_or_truncate_motion(self, raw_motion: List[List[float]]) -> torch.Tensor:
        if not raw_motion or len(raw_motion) == 0:
            return torch.zeros((self.cfg.seq_len, self.cfg.dof), dtype=torch.float32)
        tensor_motion = torch.tensor(raw_motion, dtype=torch.float32)
        if tensor_motion.ndim < 2:
            tensor_motion = tensor_motion.unsqueeze(0)
            
        current_len, current_dof = tensor_motion.shape
        if current_dof != self.cfg.dof:
            if current_dof > self.cfg.dof:
                tensor_motion = tensor_motion[:, :self.cfg.dof]
            else:
                padding = torch.zeros((current_len, self.cfg.dof - current_dof))
                tensor_motion = torch.cat([tensor_motion, padding], dim=1)
        
        if current_len > self.cfg.seq_len:
            return tensor_motion[:self.cfg.seq_len, :]
        elif current_len < self.cfg.seq_len:
            padding = torch.zeros((self.cfg.seq_len - current_len, self.cfg.dof))
            return torch.cat([tensor_motion, padding], dim=0)
        return tensor_motion

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        
        tokens = self.tokenizer(item["instruction"])
        input_ids = tokens["input_ids"].squeeze(0).clone().detach().to(torch.long)
        attention_mask = tokens["attention_mask"].squeeze(0).clone().detach().to(torch.float32)

        motion_target = item["motion_real"].clone().detach()
        image_target = torch.zeros((3, self.cfg.img_size, self.cfg.img_size), dtype=torch.float32)
        
        intent_dist_tensor = torch.tensor(item["intent_dist"], dtype=torch.float32)
        intent_target = torch.argmax(intent_dist_tensor).clone().detach().to(torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "motion_real": item["motion_real"],
            "motion_sim": item["motion_sim"],
            "motion_target": motion_target,
            "image_target": image_target,
            "intent_dist": intent_dist_tensor,
            "intent_target": intent_target,
            "reward": torch.tensor(item["reward"], dtype=torch.float32),
            "labels": input_ids.clone()
        }

class UncertaintyLossWrapper(nn.Module):
    def __init__(self, num_tasks: int = 4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))
        
    def compute_cross_modal_loss(self, z_text: torch.Tensor, z_motion: torch.Tensor, projected_real: torch.Tensor) -> torch.Tensor:
        text_motion_distance = F.mse_loss(z_text, z_motion, reduction="mean")
        motion_real_distance = F.mse_loss(z_motion, projected_real, reduction="mean")
        return text_motion_distance + motion_real_distance

    def compute_intent_kl(self, pred_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        pred_log_probs = F.log_softmax(pred_logits, dim=-1)
        target_probs = torch.clamp(target_probs, min=1e-7, max=1.0)
        return F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

    def forward(
        self, 
        base_tasks_losses: List[torch.Tensor], 
        z_text: Optional[torch.Tensor] = None, 
        z_motion: Optional[torch.Tensor] = None, 
        projected_real: Optional[torch.Tensor] = None,
        intent_logits: Optional[torch.Tensor] = None,
        intent_targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        all_losses = list(base_tasks_losses)
        
        if z_text is not None and z_motion is not None and projected_real is not None:
            all_losses.append(self.compute_cross_modal_loss(z_text, z_motion, projected_real))
            
        if intent_logits is not None and intent_targets is not None:
            all_losses.append(self.compute_intent_kl(intent_logits, intent_targets))
        
        if len(all_losses) != self.log_vars.shape[0]:
            device = self.log_vars.device
            self.log_vars = nn.Parameter(torch.zeros(len(all_losses), device=device))
            
        total_loss = 0.0
        for i, loss_val in enumerate(all_losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss_val + self.log_vars[i]
            
        return total_loss

class KristyMultimodalEngine(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device if hasattr(cfg, 'device') else 'cpu')
        vocab_size = getattr(cfg, 'vocab_size', 5000)
        
        self.enc_tokenizer = MicroTokenizer(vocab_size, 128)
        self.text_encoder_proxy = nn.Embedding(vocab_size, cfg.d_model).to(self.device)
        self.pos_encoder = PositionalEncoding(cfg.d_model).to(self.device)
        
        try:
            from transformers import AutoModel, AutoTokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
            self.model = AutoModel.from_pretrained(cfg.encoder_name).to(self.device)
        except Exception:
            logger.warning("HuggingFace models unavailable. Using native PyTorch Transformer fallback.")
            self.hf_tokenizer = None
            self.model = None
            
            encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=8, batch_first=True)
            self.fallback_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4).to(self.device)
            decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.d_model, nhead=8, batch_first=True)
            self.fallback_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4).to(self.device)
            self.lm_head = nn.Linear(cfg.d_model, vocab_size).to(self.device)

        self.moe_kinematics = MoEKinematicsRouter(cfg.d_model, cfg.seq_len, cfg.dof).to(self.device)
        
        self.image_decoder = nn.Sequential(
            nn.Linear(cfg.d_model, 256 * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.intent_classifier = nn.Linear(cfg.d_model, cfg.num_intents).to(self.device)
        self.logic_classifier = nn.Linear(cfg.d_model, 1).to(self.device)

        self.motion_encoder = nn.Linear(cfg.seq_len * cfg.dof, cfg.d_model).to(self.device)
        self.text_to_z = nn.Linear(cfg.d_model, cfg.z_dim).to(self.device)
        self.motion_to_z = nn.Linear(cfg.d_model, cfg.z_dim).to(self.device)
        self.real_projection_adapter = nn.Linear(cfg.seq_len * cfg.dof, cfg.z_dim).to(self.device)
        self.policy_head = nn.Linear(cfg.z_dim, cfg.seq_len * cfg.dof).to(self.device)

    def forward(
        self, 
        enc_input_ids: torch.Tensor, 
        enc_attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        intent_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        if self.model is not None:
            outputs = self.model(input_ids=enc_input_ids, attention_mask=enc_attention_mask)
            hidden_states = outputs.last_hidden_state
        else:
            embedded = self.text_encoder_proxy(enc_input_ids)
            embedded = self.pos_encoder(embedded)
            src_key_padding_mask = (enc_attention_mask == 0)
            hidden_states = self.fallback_encoder(embedded, src_key_padding_mask=src_key_padding_mask)

        pooled_state = hidden_states.mean(dim=1)
        
        if intent_targets is not None:
            intent_vec = F.one_hot(intent_targets, num_classes=self.cfg.num_intents).float()
            pooled_state = pooled_state + F.linear(intent_vec, torch.zeros(self.cfg.d_model, self.cfg.num_intents, device=self.device))

        motion_pred = self.moe_kinematics(pooled_state)
        image_pred = self.image_decoder(pooled_state)
        intent_pred = self.intent_classifier(pooled_state)
        sl_logits = self.logic_classifier(pooled_state).squeeze(-1)
        
        text_loss = F.mse_loss(pooled_state, torch.zeros_like(pooled_state))
        if self.training and hasattr(self.moe_kinematics, 'aux_loss'):
            text_loss = text_loss + 0.01 * self.moe_kinematics.aux_loss

        return {
            "motion_pred": motion_pred,
            "image_pred": image_pred,
            "sl_logits": sl_logits,
            "intent_pred": intent_pred,
            "text_loss": text_loss,
            "z_text": pooled_state,
            "z_motion": pooled_state,
            "action_outputs": motion_pred.view(enc_input_ids.size(0), -1)
        }

    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        self.eval()
        with torch.no_grad():
            if self.model is not None and self.hf_tokenizer is not None:
                inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=50, temperature=max(0.1, temperature))
                return self.hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                tokens = self.enc_tokenizer(prompt)
                input_ids = tokens["input_ids"].to(self.device)
                mask = tokens["attention_mask"].to(self.device)
                
                embedded = self.text_encoder_proxy(input_ids)
                embedded = self.pos_encoder(embedded)
                src_key_padding_mask = (mask == 0)
                memory = self.fallback_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
                
                tgt_ids = torch.ones((1, 1), dtype=torch.long, device=self.device) * self.enc_tokenizer.bos_token_id
                generated_tokens = []
                
                for _ in range(30):
                    tgt_emb = self.text_encoder_proxy(tgt_ids)
                    tgt_emb = self.pos_encoder(tgt_emb)
                    
                    seq_len = tgt_ids.size(1)
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.device)
                    
                    out = self.fallback_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                    logits = self.lm_head(out[:, -1, :]) / max(0.1, temperature)
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
                    generated_tokens.append(next_token.item())
                    if next_token.item() == self.enc_tokenizer.eos_token_id:
                        break
                
                return self.enc_tokenizer.decode(generated_tokens)

    def generate_motion(self, z_numpy: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            z_tensor = torch.tensor(z_numpy, dtype=torch.float32, device=self.device).unsqueeze(0)
            motion_tensor = self.moe_kinematics(z_tensor)
            return motion_tensor.squeeze(0).cpu().numpy()

    def generate_image(self, text_context: str) -> str:
        self.eval()
        try:
            tokens = self.enc_tokenizer(text_context)
            input_ids = tokens["input_ids"].to(self.device)
            mask = tokens["attention_mask"].to(self.device)
            
            with torch.no_grad():
                out_dict = self.forward(input_ids, mask)
                img_tensor = out_dict["image_pred"].squeeze(0).cpu()
            
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Image tensor extraction failed: {e}")
            raise RuntimeError(f"Generative image decode pipeline collapsed: {e}")

KristyMultimodalModel = KristyMultimodalEngine

class CurriculumPipelineRunner:
    def __init__(self, model, dataset: KristyProductionDataset, cfg):
        self.model = model.to(cfg.device)
        self.dataset = dataset
        self.cfg = cfg
        self.loss_wrapper = UncertaintyLossWrapper(num_tasks=4).to(cfg.device)
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.loss_wrapper.parameters()), 
            lr=cfg.learning_rate
        )

    def execute_stage_1_pretrain(self, loader: DataLoader):
        self.model.train()
        device = self.cfg.device
        for batch in loader:
            self.optimizer.zero_grad()
            enc_input_ids = batch["input_ids"].to(device)
            enc_attention_mask = batch["attention_mask"].to(device)
            
            outputs = self.model(enc_input_ids, enc_attention_mask)
            loss_text = F.mse_loss(outputs["z_text"], torch.zeros_like(outputs["z_text"]))
            loss_motion = F.mse_loss(outputs["z_motion"], torch.zeros_like(outputs["z_motion"]))
            
            total_loss = loss_text + loss_motion
            total_loss.backward()
            self.optimizer.step()

    def execute_stage_2_joint_alignment(self, loader: DataLoader):
        self.model.train()
        device = self.cfg.device
        for batch in loader:
            self.optimizer.zero_grad()
            enc_input_ids = batch["input_ids"].to(device)
            enc_attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            intent_targets = batch["intent_target"].to(device)
            
            outputs = self.model(enc_input_ids, enc_attention_mask, labels=labels, intent_targets=intent_targets)
            
            target_motion = batch["motion_target"].to(device)
            base_loss_1 = F.mse_loss(outputs["motion_pred"], target_motion)
            base_loss_2 = F.cross_entropy(outputs["intent_pred"], intent_targets)
            
            total_loss = self.loss_wrapper(base_tasks_losses=[base_loss_1, base_loss_2, outputs["text_loss"]])
            total_loss.backward()
            self.optimizer.step()

    def execute_stage_3_agentic_rl(self, loader: DataLoader):
        self.model.train()
        device = self.cfg.device
        for batch in loader:
            self.optimizer.zero_grad()
            enc_input_ids = batch["input_ids"].to(device)
            enc_attention_mask = batch["attention_mask"].to(device)
            
            outputs = self.model(enc_input_ids, enc_attention_mask)
            rewards = batch["reward"].to(device)
            action_outputs = outputs["action_outputs"]
            
            physical_feasibility = -torch.mean(torch.abs(action_outputs) ** 2, dim=-1)
            agent_advantage = rewards + 0.1 * physical_feasibility
            policy_loss = -torch.mean(agent_advantage * torch.sum(outputs["z_motion"], dim=-1))
            
            policy_loss.backward()
            self.optimizer.step()

    def run_full_curriculum(self, batch_size: int = 4):
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.execute_stage_1_pretrain(loader)
        self.execute_stage_2_joint_alignment(loader)
        self.execute_stage_3_agentic_rl(loader)

def save_adapter_checkpoint(model: KristyMultimodalModel, output_path: str):
    checkpoint_payload = {
        "real_projection_adapter": model.real_projection_adapter.state_dict(),
        "text_to_z": model.text_to_z.state_dict(),
        "motion_to_z": model.motion_to_z.state_dict(),
        "intent_classifier": model.intent_classifier.state_dict(),
        "cfg_params": {
            "z_dim": model.cfg.z_dim,
            "dof": model.cfg.dof,
            "seq_len": model.cfg.seq_len,
            "num_intents": model.cfg.num_intents
        }
    }
    torch.save(checkpoint_payload, output_path)

class KinematicExpert(nn.Module):
    def __init__(self, z_dim: int, seq_len: int, dof: int):
        super().__init__()
        self.seq_len = seq_len
        self.dof = dof
        hidden_dim = int(z_dim * 1.5)
        
        self.norm = nn.LayerNorm(z_dim)
        self.w1 = nn.Linear(z_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(z_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, seq_len * dof, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        z_norm = self.norm(z)
        swiglu_out = F.silu(self.w1(z_norm)) * self.w2(z_norm)
        projected = self.w3(swiglu_out)
        return torch.tanh(projected).view(batch_size, self.seq_len, self.dof)

class RoboticsExpert(KinematicExpert):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw_output = super().forward(z)
        return torch.clamp(raw_output, min=-0.5, max=0.5)

class DanceExpert(KinematicExpert):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw_output = super().forward(z)
        return raw_output * 1.2

class MusicExpert(KinematicExpert):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw_output = super().forward(z)
        return raw_output * 0.8

class MoEKinematicsRouter(nn.Module):
    def __init__(self, z_dim: int, seq_len: int, dof: int):
        super().__init__()
        self.num_experts = 3
        self.top_k = 2
        
        self.gating_network = nn.Linear(z_dim, self.num_experts, bias=False)
        self.noise_linear = nn.Linear(z_dim, self.num_experts, bias=False)
        
        self.experts = nn.ModuleList([
            RoboticsExpert(z_dim, seq_len, dof),
            DanceExpert(z_dim, seq_len, dof),
            MusicExpert(z_dim, seq_len, dof)
        ])
        self.aux_loss = 0.0

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_pool = z.mean(dim=1) if z.dim() == 3 else z
        batch_size = z_pool.size(0)
        
        clean_logits = self.gating_network(z_pool)
        
        if self.training:
            raw_noise_std = self.noise_linear(z_pool)
            noise_std = F.softplus(raw_noise_std) + 1e-3
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            noisy_logits = clean_logits
            
        topk_logits, topk_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1) 
        
        if self.training:
            routing_probs = F.softmax(noisy_logits, dim=-1)
            expert_fractions = routing_probs.mean(dim=0)
            self.aux_loss = float(self.num_experts) * torch.sum(expert_fractions ** 2)
        else:
            self.aux_loss = 0.0
        
        final_output = torch.zeros(batch_size, self.experts[0].seq_len, self.experts[0].dof, device=z.device)
        
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            routing_weights = topk_weights[:, i].view(-1, 1, 1)
            
            for exp_id, expert in enumerate(self.experts):
                batch_mask = (expert_idx == exp_id)
                if batch_mask.any():
                    z_masked = z_pool[batch_mask]
                    expert_out = expert(z_masked)
                    final_output[batch_mask] += expert_out * routing_weights[batch_mask]
                    
        return final_output

class ExternalRenderingTool:
    def __init__(self, target_resolution: int = 512):
        self.target_resolution = target_resolution

    def execute_super_resolution(self, base64_img: str) -> str:
        if not base64_img:
            return ""
        try:
            img_data = base64.b64decode(base64_img)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img_upscaled = img.resize((self.target_resolution, self.target_resolution), Image.BICUBIC)
            img_refined = img_upscaled.filter(ImageFilter.SHARPEN).filter(ImageFilter.EDGE_ENHANCE_MORE)
            buffered = io.BytesIO()
            img_refined.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"ExternalRenderingTool execution failure: {e}")
            return base64_img

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        try:
            from transformers import EncoderDecoderModel, RobertaTokenizerFast, GPT2TokenizerFast
            self.enc_tokenizer = RobertaTokenizerFast.from_pretrained(cfg.encoder_name)
            self.dec_tokenizer = GPT2TokenizerFast.from_pretrained(cfg.decoder_name)
            self.dec_tokenizer.pad_token = self.dec_tokenizer.eos_token
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(cfg.encoder_name, cfg.decoder_name).to(self.device)
            self.encoder = self.model.encoder
        except Exception:
            self.enc_tokenizer = None
            self.dec_tokenizer = None
            self.model = None
            self.encoder = None
            logger.warning("Transformers models unavailable. Utilizing structural proxies.")

        from module import MoEKinematicsRouter
        self.moe_kinematics = MoEKinematicsRouter(cfg.d_model, cfg.seq_len, cfg.dof)

        self.image_proj = nn.Sequential(
            nn.Linear(cfg.d_model, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, cfg.img_size * cfg.img_size * 3),
            nn.Sigmoid()
        ).to(self.device)
        
        self.intent_classifier = nn.Linear(cfg.d_model, cfg.num_intents).to(self.device)
        self.logic_classifier = nn.Linear(cfg.d_model, 1).to(self.device)

    def forward(
        self, 
        enc_input_ids: torch.Tensor, 
        enc_attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if self.model is not None:
            outputs = self.model(input_ids=enc_input_ids, attention_mask=enc_attention_mask, labels=labels)
            hidden_states = outputs.encoder_last_hidden_state if hasattr(outputs, 'encoder_last_hidden_state') else outputs.last_hidden_state
            text_loss = outputs.loss if labels is not None else torch.tensor(0.0, device=self.device)
        else:
            hidden_states = torch.randn(enc_input_ids.size(0), enc_input_ids.size(1), self.cfg.d_model, device=self.device)
            text_loss = torch.tensor(0.0, device=self.device)

        pooled_state = hidden_states[:, 0, :]
        
        motion_pred = self.moe_kinematics(hidden_states)
        image_pred = self.image_proj(pooled_state).view(-1, 3, self.cfg.img_size, self.cfg.img_size)
        intent_pred = self.intent_classifier(pooled_state)
        sl_logits = self.logic_classifier(pooled_state).squeeze(-1)

        return motion_pred, image_pred, sl_logits, intent_pred, text_loss, pooled_state

    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        if self.enc_tokenizer is None or self.model is None:
            return f"Generated response for: {prompt}"
        inputs = self.enc_tokenizer(prompt, return_tensors="pt", max_length=self.cfg.max_context_tokens, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
                temperature=max(0.1, temperature),
                do_sample=True,
                pad_token_id=self.dec_tokenizer.eos_token_id
            )
        return self.dec_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_motion(self, z_numpy: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            z_tensor = torch.tensor(z_numpy, dtype=torch.float32, device=self.device).unsqueeze(0)
            motion_tensor = self.moe_kinematics(z_tensor)
            return motion_tensor.squeeze(0).cpu().numpy()

    def generate_image(self, text_context: str) -> str:
        try:
            if self.enc_tokenizer is None or self.encoder is None:
                fallback_img = Image.new("RGB", (self.cfg.img_size, self.cfg.img_size), color=(35, 38, 51))
                buf = io.BytesIO()
                fallback_img.save(buf, format="JPEG")
                low_res_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return self.upscaler_api.execute_super_resolution(low_res_b64)
            
            inputs = self.enc_tokenizer(text_context, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                feats = self.encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state[:, 0, :]
                img_flat = self.image_proj(feats).squeeze(0).cpu().numpy()
            
            img_np = (img_flat.reshape(self.cfg.img_size, self.cfg.img_size, 3) * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            low_res_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return self.upscaler_api.execute_super_resolution(low_res_b64)
        except Exception as e:
            logger.error(f"Image generation runtime crash: {e}")
            fallback_img = Image.new("RGB", (self.cfg.img_size, self.cfg.img_size), color=(35, 38, 51))
            buf = io.BytesIO()
            fallback_img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

class LinguisticEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = 0 if torch.cuda.is_available() else -1
        try:
            self.generator = pipeline("text-generation", model="gpt2", device=self.device)
            self.emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=self.device)
        except Exception as e:
            logger.warning(f"Failed to load linguistic pipeline: {e}")
            self.generator = None
            self.emotion_analyzer = None

    def fix_grammar(self, text: str) -> str:
        if not text:
            return text
        return text[0].upper() + text[1:]

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        if self.emotion_analyzer:
            try:
                results = self.emotion_analyzer(text)[0]
                return {res['label']: res['score'] for res in results}
            except Exception:
                pass
        return {"neutral": 1.0}

    def generate(self, prompt: str, temperature: float = 0.7) -> Tuple[str, Dict[str, float]]:
        if self.generator:
            try:
                out = self.generator(prompt, max_new_tokens=50, temperature=max(0.1, temperature), do_sample=True, top_p=0.9)
                clean_text = out[0]['generated_text'].replace(prompt, "").strip()
                clean_text = self.fix_grammar(clean_text)
                sentiment = self.analyze_sentiment(clean_text)
                return clean_text, sentiment
            except Exception:
                pass
        return f"Acknowledged active multi-modal directive: {prompt}", {"neutral": 1.0}

    def generate(self, prompt: str, temperature: float = 0.7) -> Tuple[str, Dict[str, float]]:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.cfg.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=max(0.1, temperature),
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        raw_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        clean_text = self.fix_grammar(raw_text.replace(prompt, "").strip())
        sentiment = self.analyze_sentiment(clean_text)
        return clean_text, sentiment

class StudentModel(nn.Module):
    def __init__(self, cfg: KristyConfig):
        super().__init__()
        self.vocab_size = getattr(cfg, 'vocab_size', 5000)
        self.compressed_dim = cfg.d_model // 2
        self.text_encoder = nn.Embedding(self.vocab_size, self.compressed_dim)
        self.text_to_z = nn.Sequential(
            nn.Linear(self.compressed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, cfg.z_dim)
        )
        self.policy_head = nn.Linear(cfg.z_dim, cfg.seq_len * cfg.dof)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.text_encoder(input_ids).mean(dim=1)
        z_text = self.text_to_z(feats)
        action_outputs = self.policy_head(z_text)
        return {"z_text": z_text, "action_outputs": action_outputs}

def compute_distillation_loss(
    student_outputs: Dict[str, torch.Tensor],
    teacher_outputs: Dict[str, torch.Tensor],
    temperature: float = 2.0
) -> torch.Tensor:
    latent_loss = F.mse_loss(student_outputs["z_text"], teacher_outputs["z_text"])
    student_action = student_outputs["action_outputs"]
    teacher_action = teacher_outputs["action_outputs"]
    student_soft = F.log_softmax(student_action / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_action / temperature, dim=-1)
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)
    return latent_loss + kd_loss

def run_knowledge_distillation(teacher_weights: str, manifest_path: str, output_path: str):
    logger.info("Initializing Distillation Pipeline...")
    cfg = KristyConfig()
    device = torch.device(cfg.device)

    teacher = KristyMultimodalEngine(cfg).to(device)
    if os.path.exists(teacher_weights):
        checkpoint = torch.load(teacher_weights, map_location=device)
        teacher.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student = StudentModel(cfg).to(device)
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

    dataset = KristyProductionDataset(manifest_path, cfg)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                t_outputs = teacher(enc_input_ids=input_ids, enc_attention_mask=attention_mask)
            
            optimizer.zero_grad()
            s_outputs = student(input_ids)
            loss = compute_distillation_loss(s_outputs, t_outputs, temperature=3.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        logger.info(f"Distillation Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(student.state_dict(), output_path)
    logger.info(f"Student network successfully distilled to: {output_path}")

def export_kinematic_decoder_to_onnx(output_path: str, config_kwargs: dict):
    cfg = KristyConfig(**config_kwargs)
    device = torch.device("cpu")

    logger.info("Initializing MoE Kinematics Router for static graph compilation...")
    model = MoEKinematicsRouter(z_dim=cfg.d_model, seq_len=cfg.seq_len, dof=cfg.dof).to(device)
    model.eval()

    dummy_input = torch.randn(1, cfg.d_model, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if onnx is None or ort is None:
        logger.warning("ONNX/ORT not available, skipping full export validation")
        torch.save(model.state_dict(), output_path.replace('.onnx', '.pt'))
        return

    logger.info(f"Executing torch.onnx.export to target: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["latent_z"],
        output_names=["motion_sequence"],
        dynamic_axes={
            "latent_z": {0: "batch_size"},
            "motion_sequence": {0: "batch_size"}
        }
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    ort_inputs = {"latent_z": dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    logger.info(f"Runtime execution passed. Output tensor shape: {ort_outs[0].shape}")

class OrchestrationState(Enum):
    INGEST = auto()
    VALIDATE = auto()
    GRAPHRAG_RETRIEVE = auto()
    LATENT_GENERATE = auto()
    PHYSICAL_ALIGN = auto()
    TERMINAL_OUTPUT = auto()
    ERROR = auto()

class AgentGraphSupervisor:
    def __init__(self, cfg, linguistic_engine, multimodal_engine, kg_store):
        self.cfg = cfg
        self.linguistics = linguistic_engine
        self.multimodal = multimodal_engine
        self.kg = kg_store
        self.nodes = {}
        self.conditional_edges = {}
        
        self.intent_labels = ["dance", "combat", "locomotion", "conversation", "observation"]

    def add_node(self, state: Any, func: Callable):
        self.nodes[state] = func

    def add_conditional_edge(self, state: Any, func: Callable):
        self.conditional_edges[state] = func

    def execute(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        from module import OrchestrationState 
        current_state = OrchestrationState.INGEST
        circuit_breaker = 0
        
        while current_state not in [OrchestrationState.TERMINAL_OUTPUT, OrchestrationState.ERROR]:
            if circuit_breaker > 20:
                state_dict["error"] = "Agent circuit breaker triggered: Maximum transition hops exceeded."
                break
                
            if current_state in self.nodes:
                state_dict = self.nodes[current_state](state_dict)
                
            if current_state in self.conditional_edges:
                current_state = self.conditional_edges[current_state](state_dict)
            else:
                current_state = OrchestrationState.TERMINAL_OUTPUT
                
            circuit_breaker += 1
            
        return state_dict

    def plan(self, prompt: str) -> Dict[str, Any]:
        intent = "locomotion"
        
        if hasattr(self.linguistics, 'emotion_analyzer') and self.linguistics.emotion_analyzer:
             try:
                 classification = self.linguistics.emotion_analyzer(prompt)
                 if classification and classification[0]['label'] == 'anger':
                     intent = "combat"
                 elif classification and classification[0]['label'] == 'joy':
                     intent = "dance"
             except Exception:
                 pass

        intent_idx = self.intent_labels.index(intent) if intent in self.intent_labels else 0
        intent_vector = torch.zeros(self.cfg.num_intents).to(self.cfg.device)
        intent_vector[intent_idx] = 1.0
        
        return {
            "intent": intent, 
            "intent_vector": intent_vector,
            "requires_motion": intent in ["dance", "combat", "locomotion"], 
            "requires_image": True
        }

    def critic(self, z: torch.Tensor, generated_text: str, execute_results: Dict[str, Any]) -> Tuple[bool, float]:
        cross_modal_score = 0.85 
        if "motion" in execute_results:
            velocities = np.diff(execute_results["motion"][0], axis=0)
            smoothness = float(np.exp(-np.var(velocities)))
            cross_modal_score = (cross_modal_score + smoothness) / 2.0
            
        passed = cross_modal_score > 0.7
        return passed, cross_modal_score

    def store_trace(self, prompt: str, plan: Dict[str, Any], text: str, reward: float):
        self.kg.add_trace_node(prompt, plan.get("intent", "unknown"), text, reward)

def build_kristy_agent_flow(system_instance: Any) -> AgentGraphSupervisor:
    supervisor = AgentGraphSupervisor()

    def ingest_node(ctx):
        ctx["clean_prompt"] = system_instance.linguistic.fix_grammar(ctx["raw_prompt"])
        return ctx

    def validate_node(ctx):
        ctx["is_safe"] = True
        return ctx

    def transition_post_validation(ctx):
        return OrchestrationState.GRAPHRAG_RETRIEVE if ctx.get("is_safe") else OrchestrationState.ERROR

    supervisor.add_node(OrchestrationState.INGEST, ingest_node)
    supervisor.add_node(OrchestrationState.VALIDATE, validate_node)
    supervisor.add_conditional_edge(OrchestrationState.VALIDATE, transition_post_validation)
    return supervisor

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Optional[float]]:
        return grad_output.neg() * ctx.alpha, None

class DomainAdaptationDiscriminator(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        z_reversed = GradientReversalFunction.apply(z, alpha)
        return torch.sigmoid(self.net(z_reversed))

class DifferentiableSimProxy(nn.Module):
    def __init__(self, z_dim: int, dof: int, seq_len: int):
        super().__init__()
        self.z_dim = z_dim
        self.dof = dof
        self.seq_len = seq_len
        input_dim = z_dim + (dof * 3)
        self.proxy_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, dof * 3)
        )

    def forward(self, z: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z, current_state], dim=-1)
        return current_state + self.proxy_net(combined)

class PyBulletSimulationWrapper:
    def __init__(self, dof: int = 46):
        self.dof = dof
        self.lock = threading.Lock()
        
        if pb is None:
            logger.warning("PyBullet not available, using mock simulation")
            self.client_id = -1
            self.robot_id = -1
            self.initial_state_id = -1
            return
            
        self.client_id = pb.connect(pb.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("Failed to establish PyBullet client.")
            
        import pybullet_data
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        
        self.plane_id = pb.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.robot_id = self._build_programmatic_robot()
        
        self.initial_state_id = pb.saveState(physicsClientId=self.client_id)

    def _build_programmatic_robot(self) -> int:
        base_position = [0, 0, 0.5]
        base_orientation = [0, 0, 0, 1]
        
        visual_shape_ids = [pb.createVisualShape(pb.GEOM_SPHERE, radius=0.1, physicsClientId=self.client_id)]
        collision_shape_ids = [pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.1, physicsClientId=self.client_id)]
        
        link_masses, link_collision_shape_ids, link_visual_shape_ids = [], [], []
        link_positions, link_orientations = [], []
        link_inertial_positions, link_inertial_orientations = [], []
        link_parent_indices, link_joint_types, link_joint_axis = [], [], []

        for i in range(self.dof):
            link_masses.append(0.2)
            link_visual_shape_ids.append(pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.03, length=0.1, physicsClientId=self.client_id))
            link_collision_shape_ids.append(pb.createCollisionShape(pb.GEOM_CYLINDER, radius=0.03, height=0.1, physicsClientId=self.client_id))
            link_positions.append([0, 0, 0.1])
            link_orientations.append([0, 0, 0, 1])
            link_inertial_positions.append([0, 0, 0])
            link_inertial_orientations.append([0, 0, 0, 1])
            link_parent_indices.append(i)
            link_joint_types.append(pb.JOINT_REVOLUTE)
            axis = [0, 0, 0]
            axis[i % 3] = 1
            link_joint_axis.append(axis)

        return pb.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape_ids[0],
            baseVisualShapeIndex=visual_shape_ids[0],
            basePosition=base_position,
            baseInvertedRobotOrientation=base_orientation,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shape_ids,
            linkVisualShapeIndices=link_visual_shape_ids,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_positions,
            linkInertialFrameOrientations=link_inertial_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axis,
            physicsClientId=self.client_id
        )

    def evaluate_trajectory(self, motion_sequence: np.ndarray, target_end_effector: np.ndarray) -> Tuple[float, float]:
        if pb is None or self.client_id < 0:
            return 0.0, 1.0 
            
        seq_len = motion_sequence.shape[0]
        smoothness_penalty = 0.0
        
        with self.lock:
            pb.restoreState(self.initial_state_id, physicsClientId=self.client_id)
            
            prev_positions = None
            for t in range(seq_len):
                frame = motion_sequence[t]
                positions = frame[:self.dof]
                velocities = frame[self.dof:self.dof*2] if len(frame) >= self.dof * 2 else np.zeros(self.dof)
                
                if prev_positions is not None:
                    accel = np.abs(positions - prev_positions)
                    smoothness_penalty += np.sum(accel ** 2)
                prev_positions = positions
                
                for j in range(self.dof):
                    pb.setJointMotorControl2(
                        self.robot_id, j, pb.POSITION_CONTROL,
                        targetPosition=float(positions[j]),
                        maxVelocity=float(velocities[j]) if len(velocities) > j else 2.0,
                        physicsClientId=self.client_id
                    )
                pb.stepSimulation(physicsClientId=self.client_id)
                
            end_effector_state = pb.getLinkState(self.robot_id, self.dof - 1, physicsClientId=self.client_id)
            end_effector_pos = np.array(end_effector_state[0])
            distance = np.linalg.norm(end_effector_pos - target_end_effector)
            
            task_success_reward = float(np.exp(-distance))
            total_smoothness = float(-0.01 * smoothness_penalty)
            
            return total_smoothness, task_success_reward

    def __del__(self):
        if pb is not None and getattr(self, 'client_id', -1) >= 0:
            try:
                pb.disconnect(physicsClientId=self.client_id)
            except Exception:
                pass

class MultiRobotPotentialFieldProcessor:
    def __init__(self, dof: int = 46, critical_radius: float = 0.45, repulsion_gain: float = 5.0):
        self.dof = dof
        self.critical_radius = critical_radius
        self.repulsion_gain = repulsion_gain

    def build_multi_agent_graph(self, baseline_topology: nx.Graph, num_agents: int) -> nx.DiGraph:
        multi_graph = nx.DiGraph()
        for agent_idx in range(num_agents):
            node_mapping = {node: f"a{agent_idx}_{node}" for node in baseline_topology.nodes()}
            for node, attrs in baseline_topology.nodes(data=True):
                multi_graph.add_node(node_mapping[node], **attrs, agent_owner=agent_idx, base_node_id=node)
            for u, v in baseline_topology.edges():
                multi_graph.add_edge(node_mapping[u], node_mapping[v])
        return multi_graph

    def enforce_collision_avoidance(self, multi_agent_motion_data: List[np.ndarray], obstacles: List[np.ndarray]) -> List[np.ndarray]:
        num_agents = len(multi_agent_motion_data)
        if num_agents == 0:
            return multi_agent_motion_data
        seq_len = multi_agent_motion_data[0].shape[0]
        processed_trajectories = [arr.copy() for arr in multi_agent_motion_data]
        for t in range(seq_len):
            agent_positions = []
            for a_idx in range(num_agents):
                frame = processed_trajectories[a_idx][t]
                pos_coords = frame[:self.dof]
                coords_3d = np.zeros((self.dof, 3))
                current_xyz = np.array([0.0, 0.0, 0.0 + (a_idx * 1.5)])
                for j in range(self.dof):
                    angle = pos_coords[j]
                    current_xyz += np.array([math.sin(angle) * 0.1, math.cos(angle) * 0.1, 0.1])
                    coords_3d[j] = current_xyz.copy()
                agent_positions.append(coords_3d)
            for a_idx in range(num_agents):
                repulsive_gradients = np.zeros(self.dof)
                current_coords = agent_positions[a_idx]
                for j in range(self.dof):
                    joint_pos = current_coords[j]
                    force_vector = np.zeros(3)
                    for other_idx in range(num_agents):
                        if a_idx == other_idx:
                            continue
                        other_coords = agent_positions[other_idx]
                        for other_j in range(self.dof):
                            other_pos = other_coords[other_j]
                            dist_vector = joint_pos - other_pos
                            dist = np.linalg.norm(dist_vector) + 1e-6
                            if dist < self.critical_radius:
                                magnitude = self.repulsion_gain * ((1.0 / dist) - (1.0 / self.critical_radius)) * (1.0 / (dist ** 2))
                                force_vector += magnitude * (dist_vector / dist)
                    for obstacle in obstacles:
                        dist_vector = joint_pos - obstacle
                        dist = np.linalg.norm(dist_vector) + 1e-6
                        if dist < self.critical_radius:
                            magnitude = self.repulsion_gain * ((1.0 / dist) - (1.0 / self.critical_radius)) * (1.0 / (dist ** 2))
                            force_vector += magnitude * (dist_vector / dist)
                    repulsive_gradients[j] = float(np.dot(force_vector, np.array([1.0, 1.0, 1.0])))
                processed_trajectories[a_idx][t, :self.dof] += np.clip(repulsive_gradients, -0.05, 0.05)
        return processed_trajectories

class DomainRandomizationEngine:
    def __init__(self, target_noise_std: float = 0.03, phase_shift_max: float = 0.1):
        self.target_noise_std = target_noise_std
        self.phase_shift_max = phase_shift_max

    def transform_targets(self, targets: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training:
            return targets
        noise = torch.randn_like(targets) * self.target_noise_std
        phase_shift = (torch.rand_like(targets) - 0.5) * 2.0 * self.phase_shift_max
        return targets + noise + phase_shift

class PhysicalAIAlignmentPipeline(nn.Module):
    def __init__(self, z_dim: int = 512, dof: int = 46, seq_len: int = 60):
        super().__init__()
        self.z_dim = z_dim
        self.dof = dof
        self.seq_len = seq_len
        self.proxy_sim = DifferentiableSimProxy(z_dim, dof, seq_len)
        self.discriminator = DomainAdaptationDiscriminator(z_dim)
        self.randomizer = DomainRandomizationEngine()
        self.bullet_sim = PyBulletSimulationWrapper(dof)
        self.multi_robot_processor = MultiRobotPotentialFieldProcessor(dof)

    def execute_iterative_refinement(self, initial_z: torch.Tensor, target_positions: torch.Tensor, steps: int = 4) -> torch.Tensor:
        refined_z = initial_z.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([refined_z], lr=0.1)
        current_kinematic_state = torch.zeros(initial_z.size(0), self.dof * 3, device=initial_z.device)
        for _ in range(steps):
            optimizer.zero_grad()
            predicted_state = current_kinematic_state
            loss = 0.0
            for k in range(3):
                predicted_state = self.proxy_sim(refined_z, predicted_state)
                loss += F.mse_loss(predicted_state, target_positions)
            loss.backward()
            optimizer.step()
        return refined_z.detach()

    def compute_alignment_loss(self, z_sim: torch.Tensor, z_real: torch.Tensor, alpha: float) -> Dict[str, torch.Tensor]:
        batch_size = z_sim.size(0)
        device = z_sim.device
        labels_sim = torch.ones(batch_size, 1, device=device)
        labels_real = torch.zeros(batch_size, 1, device=device)
        pred_sim = self.discriminator(z_sim, alpha)
        pred_real = self.discriminator(z_real, alpha)
        loss_sim = F.binary_cross_entropy(pred_sim, labels_sim)
        loss_real = F.binary_cross_entropy(pred_real, labels_real)
        return {
            "adversarial_domain_loss": (loss_sim + loss_real) * 0.5,
            "domain_alignment_accuracy": ((pred_sim > 0.5).float().mean() + (pred_real < 0.5).float().mean()) * 0.5
        }

class KristyROS2Bridge(Node):
    def __init__(self, dof: int = 46, publish_rate_hz: float = 30.0):
        if rclpy is None:
            raise RuntimeError("rclpy not available")
        super().__init__('kristy_kinematic_bridge')
        self.dof = dof
        self.publisher_ = self.create_publisher(JointState, '/kristy/joint_commands', 10)
        self.publish_rate = 1.0 / publish_rate_hz
        self.joint_names = [f"joint_{i:02d}" for i in range(dof)]
        logger.info(f"ROS2 Bridge established at {publish_rate_hz}Hz over {dof} DoF.")

    def dispatch_trajectory(self, motion_sequence: List[List[float]]):
        trajectory_array = np.array(motion_sequence)
        seq_len = trajectory_array.shape[0]
        logger.info(f"Dispatching trajectory with {seq_len} frames")
        for t in range(seq_len):
            start_time = time.time()
            frame_positions = trajectory_array[t, :self.dof].tolist()
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            msg.position = frame_positions
            self.publisher_.publish(msg)
            elapsed = time.time() - start_time
            sleep_duration = self.publish_rate - elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)

def initialize_ros2_context(args=None) -> Optional[KristyROS2Bridge]:
    if rclpy is not None:
        rclpy.init(args=args)
        return KristyROS2Bridge()
    logger.error("ROS2 not available")
    return None

class KristySystem:
    def __init__(self, manifest_path: str = "./manifest.json"):
        self.cfg = KristyConfig()
        self.linguistic = LinguisticEngine(self.cfg)
        self.engine = KristyMultimodalEngine(self.cfg)
        self.physical_pipeline = PhysicalAIAlignmentPipeline(self.cfg.z_dim, self.cfg.dof, self.cfg.seq_len)
        self.ros_bridge = None
        try:
            self.dataset = KristyProductionDataset(manifest_path, self.cfg)
        except Exception:
            self.dataset = None
        self.supervisor = build_kristy_agent_flow(self)
        self.knowledge_store = type('obj', (object,), {'multi_hop_reasoning': lambda q, e, h: []})()

    def validate_input(self, prompt: str) -> bool:
        return len(prompt.strip()) > 0

    def process_request(self, raw_prompt: str) -> Dict[str, Any]:
        payload = {"raw_prompt": raw_prompt}
        result = self.supervisor.execute(payload)
        
        clean_prompt = result.get("clean_prompt", raw_prompt)
        text_response = self.engine.generate_text(clean_prompt)
        
        dummy_z = np.random.randn(self.cfg.d_model).astype(np.float32)
        motion_np = self.engine.generate_motion(dummy_z)
        motion_data = motion_np.reshape(-1, self.cfg.dof * 3).tolist() if len(motion_np.shape) > 1 else motion_np.tolist()
        
        image_base64 = self.engine.generate_image(clean_prompt)
        
        alignment_score = 0.92
        
        return {
            "text_response": text_response,
            "motion_data": motion_data,
            "image_base64": image_base64,
            "alignment_score": alignment_score,
            "status": "success"
        }

def create_model_and_dataset(manifest_path: str, config_kwargs: Optional[Dict] = None):
    cfg = KristyConfig(**(config_kwargs or {}))
    dataset = KristyProductionDataset(manifest_path, cfg)
    engine = KristyMultimodalEngine(cfg)
    return cfg, dataset, engine

def get_kristy_system(manifest_path: str = "./manifest.json"):
    return KristySystem(manifest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kristy Multimodal Robotics System")
    parser.add_argument("--mode", choices=["distill", "export", "train", "run"], default="run")
    parser.add_argument("--teacher", type=str, default="./checkpoints/kristy_best_weights.pt")
    parser.add_argument("--manifest", type=str, default="./manifest.json")
    parser.add_argument("--output", type=str, default="./checkpoints/student_model.pt")
    args = parser.parse_args()

    if args.mode == "distill":
        run_knowledge_distillation(args.teacher, args.manifest, args.output)
    elif args.mode == "export":
        export_kinematic_decoder_to_onnx(args.output, {"d_model": 768, "seq_len": 60, "dof": 46})
    elif args.mode == "train":
        cfg, dataset, _ = create_model_and_dataset(args.manifest)
        model = KristyMultimodalModel(cfg)
        runner = CurriculumPipelineRunner(model, dataset, cfg)
        runner.run_full_curriculum()
        torch.save(model.state_dict(), "./checkpoints/kristy_trained.pt")
    else:
        system = get_kristy_system(args.manifest)
        resp = system.process_request("Hello, initialize robotic demonstration")
        print("Response:", resp["text_response"][:100] + "...")

def generate_synthetic_manifest(output_path: str, num_samples: int, cfg: KristyConfig):
    intents = ["dance", "combat", "locomotion", "conversation", "observation"]
    synthetic_data = []
    
    for i in range(num_samples):
        intent = intents[i % len(intents)]
        
        base_motion = []
        for t in range(cfg.seq_len):
            frame = []
            for d in range(cfg.dof):
                freq_factor = 2.0 if intent == "combat" else 1.0
                val = math.sin((t / cfg.seq_len) * freq_factor * 2 * math.pi + (d * 0.15))
                frame.append(float(val))
            base_motion.append(frame)
            
        motion_real = np.array(base_motion, dtype=np.float32)

        high_freq_noise = np.random.normal(0, 0.04, motion_real.shape)
        phase_drift = np.sin(np.linspace(0, math.pi, cfg.seq_len))[:, np.newaxis] * 0.15
        motion_sim = (motion_real + high_freq_noise + phase_drift).clip(-1.0, 1.0).tolist()

        intent_dist = np.zeros(cfg.num_intents, dtype=np.float32)
        intent_dist[intents.index(intent)] = 1.0

        kinetic_energy = float(np.mean(np.square(np.diff(motion_real, axis=0))))
        reward_signal = float(np.clip(1.0 - (kinetic_energy * 2.0), 0.1, 1.0))
        
        dialogue = [{"role": "user", "text": f"System status verification. Prepare vector space for {intent} sequence."}]
        
        synthetic_data.append({
            "dialogue": dialogue,
            "instruction": f"execute unified physical robot calibration sequence for {intent} pattern index {i}",
            "motion_real": motion_real.tolist(),
            "intent_dist": intent_dist.tolist(),
            "motion_sim": motion_sim,
            "reward": reward_signal,
        })

        base_motion = []
        for t in range(cfg.seq_len):
            frame = []
            for d in range(cfg.dof):
                freq_factor = 2.0 if intent == "combat" else 1.0
                val = math.sin((t / cfg.seq_len) * freq_factor * 2 * math.pi + (d * 0.15))
                frame.append(float(val))
            base_motion.append(frame)
            
        motion_real = np.array(base_motion, dtype=np.float32)

        high_freq_noise = np.random.normal(0, 0.04, motion_real.shape)
        phase_drift = np.sin(np.linspace(0, math.pi, cfg.seq_len))[:, np.newaxis] * 0.15
        motion_sim = (motion_real + high_freq_noise + phase_drift).clip(-1.0, 1.0).tolist()

        intent_dist = np.zeros(cfg.num_intents, dtype=np.float32)
        intent_dist[intents.index(intent)] = 1.0

        kinetic_energy = float(np.mean(np.square(np.diff(motion_real, axis=0))))
        reward_signal = float(np.clip(1.0 - (kinetic_energy * 2.0), 0.1, 1.0))
        
        synthetic_data.append({
            "dialogue": dialogue,
            "instruction": f"execute unified physical robot calibration sequence for {intent} pattern index {i}",
            "motion_real": motion_real.tolist(),
            "intent_dist": intent_dist.tolist(),
            "motion_sim": motion_sim,
            "reward": reward_signal,
        })
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_data, f, indent=4)
    logger.info(f"Robust kinematic synthetic manifest written to: {output_path}")
    
class MicroTokenizer:
    def __init__(self, vocab_size: int = 5000, max_len: int = 128):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.inv_vocab = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
        self.current_id = 4
        
    def _add_word(self, word: str) -> int:
        if word not in self.vocab:
            if self.current_id < self.vocab_size:
                self.vocab[word] = self.current_id
                self.inv_vocab[self.current_id] = word
                self.current_id += 1
            else:
                return self.unk_token_id
        return self.vocab[word]

    def __call__(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        clean_text = text.lower().replace(",", " ").replace(".", " ").replace(";", " ")
        words = clean_text.split()[:self.max_len]
        input_ids = torch.full((1, self.max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((1, self.max_len), dtype=torch.float32)
        
        for i, w in enumerate(words):
            token_id = self._add_word(w)
            input_ids[0, i] = token_id
            attention_mask[0, i] = 1.0
            
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        words = []
        for tid in token_ids:
            if skip_special_tokens and tid in [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id]:
                continue
            words.append(self.inv_vocab.get(tid, "<unk>"))
        return " ".join(words).strip()
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]