from __future__ import annotations

import argparse
import logging
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup

from module import (
    KristyConfig,
    KristyMultimodalEngine,
    KristyProductionDataset,
    UncertaintyLossWrapper,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_kristy(
    manifest_path: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 5e-5,
    gradient_accumulation_steps: int = 2,
    early_stopping_patience: int = 3
):
    os.makedirs(output_dir, exist_ok=True)
    cfg = KristyConfig()
    cfg.learning_rate = lr
    device = torch.device(cfg.device)

    logger.info("Commencing production dataset file audit validation pass...")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest data not located: {manifest_path}")

    full_dataset = KristyProductionDataset(manifest_path, cfg)
    
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    if val_size == 0:
        val_size = 1
        train_size = max(1, len(full_dataset) - 1)

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], torch.Generator().manual_seed(42)
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = KristyMultimodalEngine(cfg).to(device)
    loss_balancer = UncertaintyLossWrapper(num_tasks=5).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_balancer.parameters()),
        lr=cfg.learning_rate,
        weight_decay=0.01,
    )

    total_steps = (len(train_dataloader) // gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_combined_score = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        loss_balancer.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            enc_input_ids = batch["input_ids"].to(device)
            enc_attention_mask = batch["attention_mask"].to(device)
            target_motion = batch["motion_target"].to(device)
            target_image = batch["image_target"].to(device)
            target_intent = batch["intent_target"].to(device)
            target_logic = batch["logic_target"].to(device)
            labels = batch["labels"].to(device)
            
            if model.training:
                target_motion = target_motion + torch.randn_like(target_motion) * 0.01
                if torch.rand(1).item() > 0.5:
                    target_image = torch.flip(target_image, dims=[-1])

            motion_pred, image_pred, sl_logits, intent_pred, text_loss, *_ = model(
                enc_input_ids=enc_input_ids,
                enc_attention_mask=enc_attention_mask,
                labels=labels
            )

            loss_m = mse_loss(motion_pred, target_motion)
            loss_i = mse_loss(image_pred, target_image)
            loss_intent = ce_loss(intent_pred, target_intent)
            loss_logic = bce_loss(sl_logits, target_logic)

            total_loss = loss_balancer([loss_m, loss_i, loss_intent, loss_logic, text_loss])
            total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += total_loss.item() * gradient_accumulation_steps
            if step % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1} | Step {step} | Loss: {total_loss.item() * gradient_accumulation_steps:.4f} | "
                    f"Text Loss: {text_loss.item():.4f} | Motion MSE: {loss_m.item():.4f}"
                )

        model.eval()
        val_text_loss = 0.0
        val_loss_m = 0.0
        val_loss_i = 0.0
        val_intent_correct = 0
        val_brier = 0.0
        val_total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                v_input_ids = batch["input_ids"].to(device)
                v_attention_mask = batch["attention_mask"].to(device)
                v_t_motion = batch["motion_target"].to(device)
                v_t_image = batch["image_target"].to(device)
                v_t_intent = batch["intent_target"].to(device)
                v_t_logic = batch["logic_target"].to(device)
                v_labels = batch["labels"].to(device)

                m_p, i_p, s_l, int_p, t_l, *_ = model(
                    enc_input_ids=v_input_ids,
                    enc_attention_mask=v_attention_mask,
                    labels=v_labels
                )

                bs = v_input_ids.size(0)
                val_text_loss += t_l.item() * bs
                val_loss_m += mse_loss(m_p, v_t_motion).item() * bs
                val_loss_i += mse_loss(i_p, v_t_image).item() * bs
                val_intent_correct += (torch.argmax(int_p, dim=-1) == v_t_intent).sum().item()
                val_brier += torch.sum((torch.sigmoid(s_l) - v_t_logic) ** 2).item()
                val_total += bs

        if val_total > 0:
            avg_text_l = val_text_loss / val_total
            perplexity = math.exp(min(avg_text_l, 20.0))
            avg_motion_mse = val_loss_m / val_total
            avg_image_mse = val_loss_i / val_total
            avg_brier = val_brier / val_total
            intent_acc = (val_intent_correct / val_total) * 100

            logger.info(f"--- Metric Epoch Verification Bounds {epoch+1} ---")
            logger.info(f"Val Motion MSE: {avg_motion_mse:.5f} | Val Image MSE: {avg_image_mse:.5f}")
            logger.info(f"Text Perplexity: {perplexity:.4f} | Alignment Accuracy: {intent_acc:.2f}%")
            logger.info(f"Logic Alignment Calibration (Brier Score): {avg_brier:.5f}")

            combined_score = avg_motion_mse + avg_image_mse + (avg_text_l * 0.1)

            if combined_score < best_combined_score:
                best_combined_score = combined_score
                patience_counter = 0
                ckpt_path = os.path.join(output_dir, "kristy_best_weights.pt")
                torch.save({"state_dict": model.state_dict(), "config": cfg}, ckpt_path)
                logger.info(f"New optimal best checkpoint saved: {ckpt_path}")
            else:
                patience_counter += 1
                logger.info(f"No validation improvement. Early stopping patience: {patience_counter}/{early_stopping_patience}")

        epoch_path = os.path.join(output_dir, f"kristy_weights_ep{epoch+1}.pt")
        torch.save({"state_dict": model.state_dict(), "config": cfg}, epoch_path)
        logger.info(f"Epoch operational run saved safely: {epoch_path}")

        if patience_counter >= early_stopping_patience:
            logger.info("Early stopping condition triggered. Training complete.")
            break

def run_hyperparameter_sweep(manifest_path: str, output_dir: str):
    logger.info("Starting manual grid-search hyperparameter sweep...")
    learning_rates = [3e-5, 5e-5]
    batch_sizes = [4, 8]
    
    for lr in learning_rates:
        for bs in batch_sizes:
            sweep_dir = os.path.join(output_dir, f"sweep_lr{lr}_bs{bs}")
            logger.info(f"Running sweep configurations: LR={lr}, Batch Size={bs}")
            try:
                train_kristy(manifest_path, sweep_dir, epochs=2, batch_size=bs, lr=lr)
            except Exception as e:
                logger.error(f"Sweep failed for LR={lr}, BS={bs}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, default="./manifest.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    args = parser.parse_args()

    if args.sweep:
        run_hyperparameter_sweep(args.manifest_path, args.output_dir)
    else:
        train_kristy(args.manifest_path, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)