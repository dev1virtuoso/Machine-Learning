from __future__ import annotations

import json
import logging
import threading
import queue
import time
import uuid
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from PIL import Image
from pytorch_metric_learning import losses
from transformers import AutoProcessor, AutoModel, pipeline
import timm

from pydantic import Field
from pydantic_settings import BaseSettings

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.INFO)
logger = logging.getLogger("LEPAUTE.Core")

class LepauteConfig(BaseSettings):
    """Unified Industrial Configuration Layer with Strict Validation."""
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    data_store: str = "lepaute_data.json"
    frames_dir: str = "frames"
    max_disk_files: int = 2000
    object_names: List[str] = ["Turbine Blade", "PCB Assembly", "Calibration Marker", "Industrial Tool"]
    
    # Model Configuration
    backbone_name: str = "mobilenetv3_large_100"
    dl_inference_freq: int = 5
    
    # Intrinsic Camera Parameters (640x480 standard fallback)
    fx: float = 640.0
    fy: float = 640.0
    cx: float = 320.0
    cy: float = 240.0
    orig_w: int = 640
    orig_h: int = 480

    class Config:
        env_prefix = "LEPAUTE_"

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

class CameraIOStream:
    def __init__(self, config: LepauteConfig, video_source: int | str = 0, buffer_size: int = 3, mock_mode: bool = False):
        self.config = config
        self.mock_mode = mock_mode
        self.video_source = video_source
        self.frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap = None
        self.reconnect_attempts = 0
        
        if not self.mock_mode:
            self._init_hardware()

    def _init_hardware(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.orig_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.orig_h)
        else:
            logger.warning(f"Hardware {self.video_source} unavailable. Falling back to synthetic mock_mode.")
            self.mock_mode = True

    def start(self) -> CameraIOStream:
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        phi = 0.0
        while self.running:
            if self.mock_mode:
                # Advanced Industrial Mock Generator
                frame = np.zeros((self.config.orig_h, self.config.orig_w, 3), dtype=np.float32)
                gradient = np.linspace(30, 90 + 30 * np.sin(phi * 0.3), self.config.orig_w)
                frame[:] = gradient
                
                cx, cy = int(self.config.cx + 120 * np.cos(phi)), int(self.config.cy + 60 * np.sin(phi * 1.5))
                cv2.rectangle(frame, (cx-40, cy-40), (cx+40, cy+40), (50, 150, 200), -1)
                
                noise = np.random.normal(0, 8, frame.shape)
                frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
                
                phi += 0.1
                time.sleep(0.033)
            else:
                if self.cap is None or not self.cap.isOpened():
                    self.reconnect_attempts += 1
                    backoff = min(60.0, (2 ** self.reconnect_attempts))
                    logger.error(f"Camera feed lost. Reconnecting in {backoff}s...")
                    time.sleep(backoff)
                    try:
                        self._init_hardware()
                    except Exception:
                        continue
                else:
                    self.reconnect_attempts = 0
                    
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                if frame.shape[:2] != (self.config.orig_h, self.config.orig_w):
                    frame = cv2.resize(frame, (self.config.orig_w, self.config.orig_h))

            if self.frame_queue.full():
                try: self.frame_queue.get_nowait()
                except queue.Empty: pass
            self.frame_queue.put(frame)

    def get_latest_frame(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        try: return self.frame_queue.get(timeout=timeout)
        except queue.Empty: return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()

class DiskManager:
    def __init__(self, config: LepauteConfig):
        self.config = config
        self.directory = Path(config.frames_dir)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_frame(self, frame: np.ndarray) -> str:
        frame_uuid = f"frame_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(self.directory / frame_uuid), frame)
        self._cleanup()
        return frame_uuid
        
    def _cleanup(self):
        files = sorted(self.directory.glob("frame_*.jpg"), key=os.path.getmtime)
        if len(files) > self.config.max_disk_files:
            for f in files[:-self.config.max_disk_files]:
                try: f.unlink()
                except OSError: pass

class DenseSE3Tracker:
    def __init__(self, config: LepauteConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.flow_weights = Raft_Small_Weights.DEFAULT
        self.flow_model = raft_small(weights=self.flow_weights, progress=False).to(self.device).eval()
        self.flow_transform = self.flow_weights.transforms()
        self.depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=self.device)
        
        self.depth_ema: Optional[np.ndarray] = None
        self.last_pose = np.zeros(6, dtype=np.float32)
        self.velocity = np.zeros(6, dtype=np.float32)

    @torch.no_grad()
    def estimate_pose(self, prev_frame: np.ndarray, curr_frame: np.ndarray, run_depth: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
        H, W = prev_frame.shape[:2]
        align_H, align_W = (H // 8) * 8, (W // 8) * 8
        
        rgb1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
        
        img1_t = torch.from_numpy(cv2.resize(rgb1, (align_W, align_H))).permute(2, 0, 1).float()
        img2_t = torch.from_numpy(cv2.resize(rgb2, (align_W, align_H))).permute(2, 0, 1).float()
        b1, b2 = self.flow_transform(img1_t.unsqueeze(0), img2_t.unsqueeze(0))
        
        flow_aligned = self.flow_model(b1.to(self.device), b2.to(self.device))[-1][0]
        flow = F.interpolate(flow_aligned.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
        flow[0, :, :] *= (W / align_W)
        flow[1, :, :] *= (H / align_H)
        flow_np = flow.cpu().numpy()
        
        if run_depth or self.depth_ema is None:
            depth_res = self.depth_pipe(Image.fromarray(rgb1))
            depth_raw = np.array(depth_res["depth"]).astype(np.float32)
            
            depth_map = 1000.0 / (depth_raw + 1e-5) 
            
            if self.depth_ema is None:
                self.depth_ema = depth_map
            else:
                self.depth_ema = 0.8 * self.depth_ema + 0.2 * depth_map
                
        active_depth = self.depth_ema
        
        y, x = np.mgrid[0:H:15, 0:W:15]
        pts1_2d = np.vstack((x.flatten(), y.flatten())).T
        
        pts1_3d, pts2_2d = [], []
        K = self.config.K
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        for pt in pts1_2d:
            px, py = int(pt[0]), int(pt[1])
            dx, dy = flow_np[:, py, px]
            if np.linalg.norm([dx, dy]) > 100: 
                continue
                
            Z = active_depth[py, px]
            if Z <= 0.05: continue
            X = (px - cx) * Z / fx
            Y = (py - cy) * Z / fy
            
            pts1_3d.append([X, Y, Z])
            pts2_2d.append([px + dx, py + dy])

        if len(pts1_3d) < 15:
            pred_pose = self.last_pose + self.velocity
            return pred_pose, 2.0, active_depth

        pts1_3d, pts2_2d = np.array(pts1_3d, dtype=np.float32), np.array(pts2_2d, dtype=np.float32)
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(pts1_3d, pts2_2d, K, None, reprojectionError=3.0)
        except Exception as e:
            logger.error(f"PnP Solvers failed: {e}")
            return self.last_pose + self.velocity, 2.0, active_depth
        
        if not success or inliers is None or len(inliers) < 8:
            return self.last_pose + self.velocity, 2.0, active_depth

        xi = np.concatenate([tvec.flatten(), rvec.flatten()]).astype(np.float32)
        
        inlier_idx = inliers.flatten()
        rmse = float(np.mean(np.linalg.norm(pts1_3d[inlier_idx][:, :2] - pts2_2d[inlier_idx], axis=1)))
        
        alpha = max(0.05, min(0.9, 1.0 - (rmse / 5.0)))
        self.velocity = (1 - alpha) * self.velocity + alpha * (xi - self.last_pose)
        self.last_pose = xi
        
        return xi, rmse, active_depth

class SigLIPClassifier:
    def __init__(self, config: LepauteConfig):
        self.config = config
        self.device = torch.device(config.device)
        model_name = "google/siglip-base-patch16-224"
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.target_labels = self.config.object_names
        inputs = self.processor(text=self.target_labels, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.text_embeds = self.model.get_text_features(**inputs)
            self.text_embeds = F.normalize(self.text_embeds, p=2, dim=-1)

    @torch.no_grad()
    def classify(self, frame: np.ndarray) -> Tuple[str, float]:
        if not self.target_labels: 
            return "Unknown", 0.0
            
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        image_embeds = self.model.get_image_features(**inputs)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        
        temperature = self.model.temperature.exp()
        logits = torch.matmul(image_embeds, self.text_embeds.t()) * temperature
        probs = F.softmax(logits, dim=-1)
        val, idx = probs.topk(1)
        
        return self.target_labels[idx[0].item()], float(val[0].item())

class DepthAwareSE3Warping(nn.Module):
    def __init__(self, config: LepauteConfig):
        super().__init__()
        self.config = config
        self.register_buffer("base_K", torch.tensor(config.K, dtype=torch.float32))

    def _rodrigues(self, r: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        theta = torch.norm(r, dim=1, keepdim=True)
        safe_theta = torch.clamp(theta, min=eps)
        r_norm = r / safe_theta
        
        B = r.size(0)
        K_mat = torch.zeros(B, 3, 3, device=r.device)
        K_mat[:, 0, 1], K_mat[:, 0, 2] = -r_norm[:, 2], r_norm[:, 1]
        K_mat[:, 1, 0], K_mat[:, 1, 2] = r_norm[:, 2], -r_norm[:, 0]
        K_mat[:, 2, 0], K_mat[:, 2, 1] = -r_norm[:, 1], r_norm[:, 0]
        
        I = torch.eye(3, device=r.device).unsqueeze(0)
        safe_theta_unsq = safe_theta.unsqueeze(-1)
        return I + torch.sin(safe_theta_unsq) * K_mat + (1 - torch.cos(safe_theta_unsq)) * torch.bmm(K_mat, K_mat)

    def forward(self, feat: torch.Tensor, depth: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.size()
        
        K_scaled = self.base_K.clone()
        K_scaled[0, 0] *= (W / self.config.orig_w)
        K_scaled[1, 1] *= (H / self.config.orig_h)
        K_scaled[0, 2] *= (W / self.config.orig_w)
        K_scaled[1, 2] *= (H / self.config.orig_h)
        
        K_inv = torch.inverse(K_scaled)
        
        tvec = xi[:, 0:3].unsqueeze(-1)
        R = self._rodrigues(xi[:, 3:6])
        R_inv = R.transpose(1, 2)
        t_inv = -torch.bmm(R_inv, tvec)
        
        y, x = torch.meshgrid(torch.arange(H, device=feat.device), torch.arange(W, device=feat.device), indexing='ij')
        x_flat = x.flatten().unsqueeze(0).expand(B, -1)
        y_flat = y.flatten().unsqueeze(0).expand(B, -1)
        ones = torch.ones_like(x_flat)
        coords_2d = torch.stack([x_flat, y_flat, ones], dim=1).to(dtype=torch.float32)
        
        safe_depth = torch.clamp(depth.view(B, 1, H * W), min=1e-2, max=1e4)
        safe_depth = torch.nan_to_num(safe_depth, nan=1.0, posinf=1e4, neginf=1e-2)
        
        K_inv_b = K_inv.unsqueeze(0).expand(B, -1, -1)
        coords_3d = torch.bmm(K_inv_b, coords_2d) * safe_depth
        coords_3d_warped = torch.bmm(R_inv, coords_3d) + t_inv
        
        K_b = K_scaled.unsqueeze(0).expand(B, -1, -1)
        coords_2d_warped = torch.bmm(K_b, coords_3d_warped)
        
        Z_warped = torch.clamp(coords_2d_warped[:, 2:3, :], min=1e-2)
        x_warped = coords_2d_warped[:, 0, :] / Z_warped.squeeze(1)
        y_warped = coords_2d_warped[:, 1, :] / Z_warped.squeeze(1)
        
        x_norm = (x_warped / (W - 1)) * 2 - 1
        y_norm = (y_warped / (H - 1)) * 2 - 1
        
        x_norm = torch.clamp(torch.nan_to_num(x_norm, 0.0), -2.0, 2.0)
        y_norm = torch.clamp(torch.nan_to_num(y_norm, 0.0), -2.0, 2.0)
        
        grid = torch.stack([x_norm, y_norm], dim=-1).view(B, H, W, 2)
        return F.grid_sample(feat, grid, align_corners=True, padding_mode='zeros')

class TransformerModel(nn.Module):
    def __init__(self, config: LepauteConfig, out_features: int = 128):
        super().__init__()
        self.config = config
        self.backbone = timm.create_model(self.config.backbone_name, pretrained=True, num_classes=0, features_only=True)
        self.se3_layer = DepthAwareSE3Warping(config)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.orig_h, config.orig_w)
            feats = self.backbone(dummy)
            out_channels = feats[-1].shape[1] 
            
        self.fc = nn.Linear(out_channels, out_features)
        self.pose_regressor = nn.Sequential(
            nn.Linear(out_features, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
        
    def forward(self, img: torch.Tensor, depth: torch.Tensor, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(img)[-1]
        B, C, H, W = features.size()
        
        depth_resized = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        trans_feat = self.se3_layer(features, depth_resized, xi)
        
        embedded = self.fc(self.pooling(trans_feat).view(B, -1))
        pose_pred = self.pose_regressor(embedded)
        
        return embedded, pose_pred

class EquivariantDataset(Dataset):
    def __init__(self, data_list: List[Dict], config: LepauteConfig):
        self.config = config
        self.frames_dir = Path(config.frames_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.orig_h, config.orig_w), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.valid_samples = [i for i in data_list if (self.frames_dir / i["frame_a"]).exists() and (self.frames_dir / i["frame_b"]).exists()]

    @lru_cache(maxsize=500)
    def _load_img(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            return np.zeros((self.config.orig_h, self.config.orig_w, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        item = self.valid_samples[idx]
        img_a = self.transform(self._load_img(self.frames_dir / item["frame_a"]))
        img_b = self.transform(self._load_img(self.frames_dir / item["frame_b"]))
        
        xi = torch.tensor(item["lie_params"], dtype=torch.float32)
        
        if "depth_mean" in item:
            depth = torch.ones((self.config.orig_h, self.config.orig_w), dtype=torch.float32) * item["depth_mean"]
        else:
            depth = torch.ones((self.config.orig_h, self.config.orig_w), dtype=torch.float32)
            
        return img_a, img_b, depth, xi, idx

def train_industrial_loop(model: nn.Module, dataset: Dataset, config: LepauteConfig, epochs: int = 1) -> float:
    if len(dataset) < 2: return 0.0
    model.train()
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    contrastive_loss_fn = losses.NTXentLoss(temperature=0.1)
    pose_loss_fn = nn.HuberLoss(delta=1.0)
    
    last_loss = 0.0
    device = torch.device(config.device)
    for epoch in range(epochs):
        for img1, img2, depth, xi, labels in loader:
            img1, img2 = img1.to(device, non_blocking=True), img2.to(device, non_blocking=True)
            depth, xi, labels = depth.to(device, non_blocking=True), xi.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            emb1, pose_pred1 = model(img1, depth, torch.zeros_like(xi))
            emb2, pose_pred2 = model(img2, depth, xi)
            
            embeddings = torch.cat([emb1, emb2], dim=0)
            target_labels = torch.cat([labels, labels], dim=0)
            c_loss = contrastive_loss_fn(embeddings, target_labels)
            
            p_loss = pose_loss_fn(pose_pred2, xi)
            
            total_loss = c_loss + (0.7 * p_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            last_loss = total_loss.item()
    return last_loss

def save_to_disk(data: List[Dict], config: LepauteConfig):
    try:
        with open(config.data_store, "w") as f: 
            json.dump(data[-2000:], f, indent=4)
    except Exception as e: 
        logger.error(f"Save error tracking: {e}")

def load_data(config: LepauteConfig) -> List[Dict]:
    try:
        if os.path.exists(config.data_store):
            with open(config.data_store, "r") as f: 
                return json.load(f)
    except Exception: 
        pass
    return []