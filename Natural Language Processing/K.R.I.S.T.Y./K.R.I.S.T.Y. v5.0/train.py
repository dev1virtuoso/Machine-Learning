from __future__ import annotations

import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from module import KristyConfig, KristyMultimodalEngine, KristyProductionDataset, generate_synthetic_manifest
from main import PhysicalAIAlignmentPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CurriculumTrainer:
    def __init__(self, cfg: KristyConfig, dataset_path: str, output_dir: str):
        self.cfg = cfg
        self.output_dir = output_dir
        self.device = torch.device(cfg.device)
        
        if not os.path.exists(dataset_path):
            logger.info("Target dataset absent. Instantiating valid structural sequences...")
            generate_synthetic_manifest(dataset_path, 40, cfg)
            
        self.model = KristyMultimodalEngine(cfg).to(self.device)
        self.dataset = KristyProductionDataset(dataset_path, cfg)
        self.train_loader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
        self.pipeline = PhysicalAIAlignmentPipeline(cfg)

    def execute_stage_1_pretrain(self, epochs: int = 5):
        logger.info("Commencing Stage 1: Continuous Multimodal Loss Minimization...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                target_motion = batch["motion_real"].to(self.device)
                target_intent = batch["intent_target"].to(self.device)
                
                outputs = self.model(ids, mask, intent_targets=target_intent)
                
                loss_m = F.mse_loss(outputs["motion_pred"], target_motion)
                loss_i = F.cross_entropy(outputs["intent_pred"], target_intent)
                
                loss = loss_m + 0.4 * loss_i
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            logger.info(f"[Stage 1 - Epoch {epoch+1}/{epochs}] Continuous Joint Loss: {total_loss / len(self.train_loader):.4f}")

    def execute_stage_2_alignment(self, epochs: int = 3):
        logger.info("Commencing Stage 2: Dynamic PyBullet Alignment Optimization...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(ids, mask)
                motion_pred = outputs["motion_pred"]
                
                motion_np = motion_pred.detach().cpu().numpy()
                aligned_batch = []
                for b in range(self.cfg.batch_size):
                    _, refined_trajectory = self.pipeline.evaluate_trajectory(motion_np[b])
                    aligned_batch.append(refined_trajectory)
                    
                target_aligned = torch.tensor(np.array(aligned_batch), dtype=torch.float32, device=self.device)
                loss_align = F.mse_loss(motion_pred, target_aligned)
                
                loss_align.backward()
                self.optimizer.step()
                total_loss += loss_align.item()
                
            logger.info(f"[Stage 2 - Epoch {epoch+1}/{epochs}] Kinematic Fit Loss: {total_loss / len(self.train_loader):.4f}")
            
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "kristy_best_weights.pt"))
        logger.info("Best architectural parameters compiled successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, default="./manifest.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    config = KristyConfig()
    trainer = CurriculumTrainer(config, args.manifest_path, args.output_dir)
    trainer.execute_stage_1_pretrain(epochs=2)
    trainer.execute_stage_2_alignment(epochs=1)