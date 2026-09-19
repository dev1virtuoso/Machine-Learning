from __future__ import annotations

import json
import math
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import networkx as nx

from transformers import (
    EncoderDecoderModel,
    GPT2TokenizerFast,
    RobertaTokenizerFast,
    pipeline,
)

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
        self.logic_threshold = kwargs.get("logic_threshold", 0.45)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = kwargs.get("learning_rate", 5e-5)
        self.memory_maxlen = kwargs.get("memory_maxlen", 15)
        self.max_context_tokens = kwargs.get("max_context_tokens", 512)
        self.num_intents = kwargs.get("num_intents", 4)

class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(torch.matmul(adj, x)))

class KGModule(nn.Module):
    """Enriched Knowledge Graph Module driven by Watts-Strogatz structural priors."""
    def __init__(self, num_nodes: int = 100, embedding_dim: int = 64, hidden_dim: int = 128, z_dim: int = 512):
        super().__init__()
        self.num_nodes = num_nodes
        
        g = nx.watts_strogatz_graph(n=num_nodes, k=6, p=0.1, seed=42)
        adj = nx.adjacency_matrix(g).toarray()
        adj = adj + np.eye(num_nodes)
        
        deg = np.sum(adj, axis=1)
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        deg_inv_sqrt[deg == 0] = 0.0
        D_inv = np.diag(deg_inv_sqrt)
        adj_norm = D_inv @ adj @ D_inv
        
        self.register_buffer("adj", torch.tensor(adj_norm, dtype=torch.float32))
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        
        self.gcn1 = GCNLayer(embedding_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, z_dim)
        self.res_proj = nn.Linear(embedding_dim, z_dim)
        
        self.routing_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        x = self.gcn1(self.node_embeddings, self.adj)
        x = self.gcn2(x, self.adj)
        x = self.gcn3(x, self.adj) + self.res_proj(self.node_embeddings)
        
        scores = torch.matmul(text_features, x.t()) / (torch.clamp(self.routing_temperature, min=1e-3))
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, x)

class FiLM(nn.Module):
    def __init__(self, z_dim: int, features: int):
        super().__init__()
        self.scale_linear = nn.Linear(z_dim, features)
        self.shift_linear = nn.Linear(z_dim, features)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        scale = self.scale_linear(z)
        shift = self.shift_linear(z)
        if x.dim() == 4:
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + scale) + shift

class DanceDecoder(nn.Module):
    def __init__(self, z_dim: int, seq_len: int = 60, dof_dim: int = 138, hidden_dim: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.project = nn.Linear(z_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, dof_dim)
        self.film = FiLM(z_dim, hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.project(z))
        h_mod = self.film(h, z)
        x = h_mod.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.rnn(x)
        return self.fc(out)

class ImageDecoder(nn.Module):
    def __init__(self, z_dim: int, img_size: int = 64, channels: int = 3):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(z_dim, 256 * 4 * 4)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.film1 = FiLM(z_dim, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.film2 = FiLM(z_dim, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.film3 = FiLM(z_dim, 32)
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.film4 = FiLM(z_dim, 16)
        
        self.to_rgb = nn.Conv2d(16, channels, kernel_size=3, padding=1)
        self.skip_proj = nn.Linear(z_dim, channels * img_size * img_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        x = self.fc(z).view(batch_size, 256, 4, 4)
        
        x = self.film1(F.relu(self.up1(x)), z)
        x = self.film2(F.relu(self.up2(x)), z)
        x = self.film3(F.relu(self.up3(x)), z)
        x = self.film4(F.relu(self.up4(x)), z)
        
        rgb = torch.sigmoid(self.to_rgb(x))
        skip = torch.sigmoid(self.skip_proj(z)).view(batch_size, 3, self.img_size, self.img_size)
        return (rgb + skip) / 2.0

class KristyMultimodalEngine(nn.Module):
    def __init__(self, cfg: KristyConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        self.enc_tokenizer = RobertaTokenizerFast.from_pretrained(cfg.encoder_name)
        self.dec_tokenizer = GPT2TokenizerFast.from_pretrained(cfg.decoder_name)
        if self.dec_tokenizer.pad_token is None:
            self.dec_tokenizer.pad_token = self.dec_tokenizer.eos_token
            
        self.text_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            cfg.encoder_name, cfg.decoder_name
        )
        self.text_model.config.decoder_start_token_id = self.dec_tokenizer.bos_token_id
        self.text_model.config.pad_token_id = self.dec_tokenizer.pad_token_id
        
        self.kg_module = KGModule(num_nodes=100, embedding_dim=768, hidden_dim=256, z_dim=cfg.z_dim)
        
        self.motion_decoder = DanceDecoder(z_dim=cfg.z_dim, seq_len=cfg.seq_len, dof_dim=cfg.dof * 3)
        self.image_decoder = ImageDecoder(z_dim=cfg.z_dim, img_size=cfg.img_size)
        
        self.intent_classifier = nn.Linear(cfg.z_dim, cfg.num_intents)
        self.logic_gate = nn.Linear(cfg.z_dim, 1)
        
        self.motion_latent_proj = nn.Linear(cfg.dof * 3, cfg.z_dim)
        self.image_latent_proj = nn.Linear(3 * cfg.img_size * cfg.img_size, cfg.z_dim)

    def forward(
        self,
        enc_input_ids: torch.Tensor,
        enc_attention_mask: torch.Tensor,
        dec_input_ids: Optional[torch.Tensor] = None,
        dec_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        encoder_outputs = self.text_model.encoder(
            input_ids=enc_input_ids,
            attention_mask=enc_attention_mask
        )
        pooled_text = encoder_outputs[0][:, 0, :]
        
        z_shared = self.kg_module(pooled_text)
        
        motion_pred = self.motion_decoder(z_shared)
        image_pred = self.image_decoder(z_shared)
        intent_pred = self.intent_classifier(z_shared)
        sl_logits = self.logic_gate(z_shared).squeeze(-1)
        
        if labels is not None:
            text_outputs = self.text_model(
                input_ids=enc_input_ids,
                attention_mask=enc_attention_mask,
                decoder_input_ids=dec_input_ids,
                decoder_attention_mask=dec_attention_mask,
                labels=labels
            )
            text_loss = text_outputs.loss
            text_logits = text_outputs.logits
        else:
            text_loss = torch.tensor(0.0, device=self.device)
            text_logits = None
            
        return motion_pred, image_pred, sl_logits, intent_pred, text_logits, text_loss

    def generate_text(self, enc_input_ids: torch.Tensor, enc_attention_mask: torch.Tensor, max_length: int = 128, temperature: float = 0.7) -> torch.Tensor:
        return self.text_model.generate(
            input_ids=enc_input_ids,
            attention_mask=enc_attention_mask,
            max_length=max_length,
            temperature=temperature,
            do_sample=True if temperature > 0.0 else False,
            decoder_start_token_id=self.dec_tokenizer.bos_token_id,
            pad_token_id=self.dec_tokenizer.pad_token_id,
            eos_token_id=self.dec_tokenizer.eos_token_id
        )

    def load_model(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "state_dict" in checkpoint:
            self.load_state_dict(checkpoint["state_dict"])
        else:
            self.load_state_dict(checkpoint)

class UncertaintyLossWrapper(nn.Module):
    def __init__(self, num_tasks: int = 5):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss

class KristyProductionDataset(Dataset):
    def __init__(self, manifest_path: str, cfg: KristyConfig):
        self.cfg = cfg
        self.samples = []
        
        if not os.path.exists(manifest_path):
            logger.warning(f"Manifest missing at {manifest_path}. Generating fully working stub array structure.")
            stub = [
                {
                    "text_input": "initialize core console structure",
                    "text_output": "Console active. Multimodal data gates ready.",
                    "intent": 0,
                    "logic": 1.0,
                    "motion_data": np.zeros((cfg.seq_len, cfg.dof * 3)).tolist(),
                    "image_data": np.ones((3, cfg.img_size, cfg.img_size)).tolist()
                }
            ] * 10
            with open(manifest_path, "w", encoding="utf-8") as out_f:
                json.dump(stub, out_f)

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        enc_tokenizer = RobertaTokenizerFast.from_pretrained(cfg.encoder_name)
        dec_tokenizer = GPT2TokenizerFast.from_pretrained(cfg.decoder_name)
        if dec_tokenizer.pad_token is None:
            dec_tokenizer.pad_token = dec_tokenizer.eos_token

        valid_count = 0
        for item in data:
            if "text_input" not in item or "text_output" not in item:
                continue
                
            motion_valid = True
            image_valid = True
            
            if "motion_path" in item and not os.path.exists(item["motion_path"]):
                motion_valid = False
            if "image_path" in item and not os.path.exists(item["image_path"]):
                image_valid = False
                
            if motion_valid and image_valid:
                valid_count += 1
                
                if "motion_path" in item:
                    motion = np.load(item["motion_path"])
                else:
                    motion = np.array(item["motion_data"], dtype=np.float32)
                    
                if "image_path" in item:
                    img = Image.open(item["image_path"]).convert("RGB")
                    transform = transforms.Compose([
                        transforms.Resize((cfg.img_size, cfg.img_size)),
                        transforms.ToTensor()
                    ])
                    image_tensor = transform(img)
                else:
                    image_tensor = torch.tensor(item["image_data"], dtype=torch.float32)

                enc_in = enc_tokenizer(item["text_input"], truncation=True, max_length=512, padding="max_length")
                dec_out = dec_tokenizer(item["text_output"], truncation=True, max_length=128, padding="max_length")
                
                labels = list(dec_out["input_ids"])
                labels = [label if label != dec_tokenizer.pad_token_id else -100 for label in labels]
                
                self.samples.append({
                    "enc_input_ids": torch.tensor(enc_in["input_ids"], dtype=torch.long),
                    "enc_attention_mask": torch.tensor(enc_in["attention_mask"], dtype=torch.long),
                    "dec_input_ids": torch.tensor(dec_out["input_ids"], dtype=torch.long),
                    "dec_attention_mask": torch.tensor(dec_out["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "target_motion": torch.tensor(motion, dtype=torch.float32).view(cfg.seq_len, cfg.dof * 3),
                    "target_image": image_tensor,
                    "target_intent": torch.tensor(item.get("intent", 0), dtype=torch.long),
                    "target_logic": torch.tensor(item.get("logic", 1.0), dtype=torch.float32)
                })

        pct = (valid_count / len(data)) * 100 if data else 0.0
        logger.info(f"Dataset Data Validation Step completed: {valid_count}/{len(data)} verified samples ({pct:.2f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]

class LinguisticEngine:
    def __init__(self, cfg: KristyConfig):
        device_id = 0 if torch.cuda.is_available() and "cuda" in cfg.device else -1
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="bhadresh-savani/distilbert-base-uncased-emotion",
                device=device_id
            )
        except Exception:
            self.sentiment_analyzer = None

        try:
            self.gec = pipeline(
                "text2text-generation",
                model="t5-base",
                device=device_id
            )
        except Exception:
            self.gec = None

    def fix_grammar(self, text: str) -> str:
        if not text or not self.gec:
            return text
        try:
            res = self.gec(f"grammar: {text.strip()}", max_length=128, early_stopping=True)
            return res[0]["generated_text"]
        except Exception:
            return text

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        if self.sentiment_analyzer:
            try:
                res = self.sentiment_analyzer(text[:512])[0]
                return {"label": res["label"].upper(), "score": float(res["score"])}
            except Exception:
                pass
        return {"label": "NEUTRAL", "score": 1.0}