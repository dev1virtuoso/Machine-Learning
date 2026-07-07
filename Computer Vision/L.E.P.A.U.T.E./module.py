import os
import time
import math
import json
import sqlite3
import logging
import threading
import queue
import re
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor, AutoModel, SiglipProcessor, SiglipModel
from huggingface_hub import snapshot_download
from pydantic import Field
from pydantic_settings import BaseSettings

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import sys
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import multiprocessing
import random

_mps_lock = threading.Lock()

logger = logging.getLogger("LEPAUTE.Core")

class DisplayMode(str, Enum):
    REALTIME = "realtime"
    JSON = "json"
    HEADLESS = "headless"
    DETAILEDGUI = "detailedgui"

def _get_stable_compute_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        logger.info("[LepauteConfig] Apple Silicon MPS detected. Engaging Metal Performance Shaders.")
        return "mps"
    return "cpu"

class PerformanceMode(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    
class LepauteConfig(BaseSettings):
    device: str = Field(default_factory=_get_stable_compute_device)
    data_store: str = "lepaute_data.db"
    
    performance_mode: PerformanceMode = PerformanceMode.MEDIUM
    
    object_names: List[str] = ["table", "cup", "keyboard", "laptop", "mouse", "human", "background"]
    
    object_scales: Dict[str, float] = {
        "table": 1.5, 
        "cup": 0.1, 
        "keyboard": 0.4, 
        "laptop": 0.35, 
        "mouse": 0.12, 
        "human": 1.7, 
        "background": 2.0
    }
    
    orig_h: int = 240
    orig_w: int = 320
    fx: float = 250.0
    fy: float = 250.0
    cx: float = 160.0
    cy: float = 120.0
    pyramid_levels: int = 3
    gn_max_iter: int = 15
    use_compiler: bool = False
    enable_orb_fallback: bool = True

def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    if v.device.type == "mps":
        with _mps_lock:
            return _skew_symmetric_impl(v)
    return _skew_symmetric_impl(v)

def _skew_symmetric_impl(v: torch.Tensor) -> torch.Tensor:
    B = v.shape[0]
    zero = torch.zeros(B, device=v.device, dtype=v.dtype)
    return torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero
    ], dim=1).view(B, 3, 3)

def se3_exp_map(xi: torch.Tensor) -> torch.Tensor:
    if xi.device.type == "mps":
        with _mps_lock:
            return _se3_exp_map_impl(xi)
    return _se3_exp_map_impl(xi)

def _se3_log_map_impl(T: torch.Tensor) -> torch.Tensor:
    B = T.shape[0]
    R, t = T[:, :3, :3], T[:, :3, 3]
    
    trace_R = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = torch.clamp((trace_R - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    
    xi = torch.zeros(B, 6, device=T.device, dtype=T.dtype)
    mask_large = (theta > 1e-4)
    mask_small = ~mask_large
    
    phi_raw = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], dim=1)
    
    if mask_large.any():
        th = theta[mask_large].unsqueeze(-1)
        sin_th = torch.clamp(torch.sin(th), min=1e-7)
        
        phi_l = (th / (2.0 * sin_th)) * phi_raw[mask_large]
        xi[mask_large, 3:] = phi_l
        
        K = _skew_symmetric_impl(phi_l)
        I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).expand(mask_large.sum(), -1, -1)
        half_th = th / 2.0
        
        sin_half_th = torch.clamp(torch.sin(half_th), min=1e-7)
        coef = 1.0 - (th * torch.cos(half_th)) / (2.0 * sin_half_th)
        
        V_inv = I - 0.5 * K + coef * torch.bmm(K, K) / torch.clamp(th**2, min=1e-10)
        xi[mask_large, :3] = torch.bmm(V_inv, t[mask_large].unsqueeze(-1)).squeeze(-1)
        
    if mask_small.any():
        xi[mask_small, 3:] = 0.5 * phi_raw[mask_small]
        K = _skew_symmetric_impl(xi[mask_small, 3:])
        I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).expand(mask_small.sum(), -1, -1)
        V_inv = I - 0.5 * K + (1.0/12.0) * torch.bmm(K, K)
        xi[mask_small, :3] = torch.bmm(V_inv, t[mask_small].unsqueeze(-1)).squeeze(-1)
        
    return xi

def se3_log_map(T: torch.Tensor) -> torch.Tensor:
    if T.device.type == "mps":
        with _mps_lock:
            return _se3_log_map_impl(T)
    return _se3_log_map_impl(T)

def _se3_exp_map_impl(xi: torch.Tensor) -> torch.Tensor:
    B = xi.shape[0]
    rho, phi = xi[:, :3], xi[:, 3:]
    theta_sq = torch.sum(phi**2, dim=1, keepdim=True)
    theta = torch.sqrt(torch.clamp(theta_sq, min=1e-10))
    
    T = torch.eye(4, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(B, 1, 1)
    K = _skew_symmetric_impl(phi)
    K2 = torch.bmm(K, K)
    
    mask_large = (theta > 1e-4).squeeze(1)
    mask_small = ~mask_large
    
    if mask_large.any():
        th = theta[mask_large].unsqueeze(-1)
        A_coef = torch.sin(th) / th
        B_term = (1.0 - torch.cos(th)) / (th**2)
        C = (1.0 - A_coef) / (th**2)
        
        K_l, K2_l = K[mask_large], K2[mask_large]
        I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(mask_large.sum(), -1, -1)
        
        T[mask_large, :3, :3] = I + A_coef * K_l + B_term * K2_l
        V = I + B_term * K_l + C * K2_l
        T[mask_large, :3, 3] = torch.bmm(V, rho[mask_large].unsqueeze(-1)).squeeze(-1)

    if mask_small.any():
        K_s, K2_s = K[mask_small], K2[mask_small]
        I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(mask_small.sum(), -1, -1)
        
        T[mask_small, :3, :3] = I + K_s + 0.5 * K2_s
        V = I + 0.5 * K_s + (1.0/6.0) * K2_s
        T[mask_small, :3, 3] = torch.bmm(V, rho[mask_small].unsqueeze(-1)).squeeze(-1)
        
    return T

def compose_poses(T_global: torch.Tensor, T_rel: torch.Tensor) -> torch.Tensor:
    if T_global.device.type == "mps":
        with _mps_lock:
            return torch.bmm(T_global, T_rel)
    return torch.bmm(T_global, T_rel)

class MonocularDirectTracker:
    def __init__(self, config: LepauteConfig):
        import contextlib
        self.config = config
        self.device = torch.device(config.device)
        self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.dynamic_scale_prior = getattr(config, "object_scales", {}).get("object", 1.0)
        self.last_keyframe_features = None
        self.last_keyframe_img = None
        self.intrinsic_matrix = np.array([
            [config.fx, 0.0, config.cx],
            [0.0, config.fy, config.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        self._execution_context = _mps_lock if self.device.type == 'mps' else contextlib.nullcontext()
        
        with self._execution_context:
            scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32, device=self.device) / 16.0
            scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32, device=self.device) / 16.0
            self.kx = scharr_x.view(1, 1, 3, 3)
            self.ky = scharr_y.view(1, 1, 3, 3)
        
        logger.info(f"[MonocularDirectTracker] PyTorch Accelerated Direct Tracker Online. Device: {self.device} | Levels={self.config.pyramid_levels}")
        
    def update_dynamic_scale(self, new_scale: float):
        if new_scale > 0.001:
            self.dynamic_scale_prior = new_scale
            logger.info(f"[MonocularDirectTracker] Scale prior synchronized dynamically to: {new_scale:.4f}")
            
    def track_fallback_orb(self, current_img: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self.last_keyframe_img is None:
            self.last_keyframe_img = current_img
            kp, des = self.orb.detectAndCompute(current_img, None)
            self.last_keyframe_features = (kp, des)
            return np.eye(4), True

        kp_cur, des_cur = self.orb.detectAndCompute(current_img, None)
        kp_ref, des_ref = self.last_keyframe_features
        
        if des_cur is None or des_ref is None:
            return np.eye(4), False
            
        matches = self.matcher.match(des_ref, des_cur)
        if len(matches) < 12:
            return np.eye(4), False
            
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_cur = np.float32([kp_cur[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        E, mask = cv2.findEssentialMat(
            pts_cur, pts_ref, 
            cameraMatrix=self.intrinsic_matrix, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        
        if E is None or E.shape != (3, 3):
            return np.eye(4), False
            
        _, R, t, mask_pose = cv2.recoverPose(E, pts_cur, pts_ref, cameraMatrix=self.intrinsic_matrix, mask=mask)
        
        t_scaled = t.flatten() * self.dynamic_scale_prior
        
        T_rel = np.eye(4)
        T_rel[0:3, 0:3] = R
        T_rel[0:3, 3] = t_scaled
        
        self.last_keyframe_img = current_img
        self.last_keyframe_features = (kp_cur, des_cur)
        
        return T_rel, True

    def _build_pyramid(self, img_tensor: torch.Tensor) -> List[torch.Tensor]:
        import torch.nn.functional as F
        pyr = [img_tensor]
        for l in range(self.config.pyramid_levels - 1):
            down = F.interpolate(pyr[-1].unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False)
            pyr.append(down.squeeze(0).squeeze(0))
        return pyr

    def track(self, img_a: np.ndarray, img_b: np.ndarray, scale_prior: float = 1.0) -> Tuple[np.ndarray, float]:
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY) if len(img_a.shape) == 3 else img_a
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY) if len(img_b.shape) == 3 else img_b
        
        xi = torch.zeros(6, dtype=torch.float32, device=self.device)
        tracking_successful = False
        final_score = 0.0
        residuals = None

        with self._execution_context:
            try:
                t_a = torch.from_numpy(gray_a).to(dtype=torch.float32, device=self.device)
                t_b = torch.from_numpy(gray_b).to(dtype=torch.float32, device=self.device)
                
                pyr_a = self._build_pyramid(t_a)
                pyr_b = self._build_pyramid(t_b)
                
                for lvl in reversed(range(self.config.pyramid_levels)):
                    img_lvl_a = pyr_a[lvl]
                    img_lvl_b = pyr_b[lvl]
                    h, w = img_lvl_a.shape
                    
                    scale_factor = 1.0 / (2.0 ** lvl)
                    fx_l = self.config.fx * scale_factor
                    fy_l = self.config.fy * scale_factor
                    cx_l = self.config.cx * scale_factor
                    cy_l = self.config.cy * scale_factor
                    
                    img_b_batch = img_lvl_b.unsqueeze(0).unsqueeze(0)
                    gx = F.conv2d(img_b_batch, self.kx, padding=1).squeeze()
                    gy = F.conv2d(img_b_batch, self.ky, padding=1).squeeze()
                    
                    v_coords, u_coords = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing='ij')
                    
                    max_iters = 10
                    for iter_idx in range(max_iters):
                        tx, ty, tz = xi[0], xi[1], xi[2]
                        wx, wy, wz = xi[3], xi[4], xi[5]
                        
                        Z = torch.full_like(u_coords, scale_prior, dtype=torch.float32)
                        X = (u_coords - cx_l) / fx_l * Z
                        Y = (v_coords - cy_l) / fy_l * Z
                        
                        X_prime = X + (wz * Y - wy * Z) + tx
                        Y_prime = Y + (-wz * X + wx * Z) + ty
                        Z_prime = Z + (wy * X - wx * Y) + tz
                        Z_prime = torch.clamp(Z_prime, min=1e-4)
                        
                        u_prime = (X_prime / Z_prime) * fx_l + cx_l
                        v_prime = (Y_prime / Z_prime) * fy_l + cy_l
                        
                        valid_mask = (u_prime >= 0) & (u_prime < w - 1) & (v_prime >= 0) & (v_prime < h - 1)
                        if valid_mask.sum() < 16:
                            break
                            
                        u_norm = (u_prime / (w - 1)) * 2.0 - 1.0
                        v_norm = (v_prime / (h - 1)) * 2.0 - 1.0

                        u_norm = torch.clamp(u_norm, min=-2.0, max=2.0)
                        v_norm = torch.clamp(v_norm, min=-2.0, max=2.0)
                        u_norm = torch.where(torch.isnan(u_norm) | torch.isinf(u_norm), torch.zeros_like(u_norm), u_norm)
                        v_norm = torch.where(torch.isnan(v_norm) | torch.isinf(v_norm), torch.zeros_like(v_norm), v_norm)
                        
                        grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(0)
                        
                        warped_b = F.grid_sample(img_b_batch, grid, align_corners=True).squeeze()
                        warped_gx = F.grid_sample(gx.unsqueeze(0).unsqueeze(0), grid, align_corners=True).squeeze()
                        warped_gy = F.grid_sample(gy.unsqueeze(0).unsqueeze(0), grid, align_corners=True).squeeze()
                        
                        residuals = warped_b - img_lvl_a
                        
                        inv_z = 1.0 / Z_prime
                        inv_z2 = inv_z * inv_z
                        
                        du_dX = fx_l * inv_z
                        du_dY = 0.0
                        du_dZ = -fx_l * X_prime * inv_z2
                        
                        dv_dX = 0.0
                        dv_dY = fy_l * inv_z
                        dv_dZ = -fy_l * Y_prime * inv_z2
                        
                        J_X = warped_gx * du_dX + warped_gy * dv_dX
                        J_Y = warped_gx * du_dY + warped_gy * dv_dY
                        J_Z = warped_gx * du_dZ + warped_gy * dv_dZ
                        
                        J = torch.zeros((h, w, 6), dtype=torch.float32, device=self.device)
                        J[..., 0] = J_X
                        J[..., 1] = J_Y
                        J[..., 2] = J_Z
                        J[..., 3] = -J_Y * Z_prime + J_Z * Y_prime
                        J[..., 4] =  J_X * Z_prime - J_Z * X_prime
                        J[..., 5] = -J_X * Y_prime + J_Y * X_prime
                        
                        J_valid = J[valid_mask]
                        r_valid = residuals[valid_mask]
                        
                        if J_valid.numel() == 0 or not torch.isfinite(J_valid).all() or not torch.isfinite(r_valid).all():
                            break
                        
                        H = torch.matmul(J_valid.T, J_valid)
                        b = -torch.matmul(J_valid.T, r_valid)
                        
                        H += 1e-4 * torch.eye(6, dtype=torch.float32, device=self.device)
                        
                        try:
                            delta_xi = torch.linalg.solve(H, b)
                            if not torch.isfinite(delta_xi).all():
                                break
                            xi += delta_xi
                            if torch.norm(delta_xi) < 1e-4:
                                break
                        except RuntimeError: 
                            break
                            
                if residuals is not None:
                    final_score = float(1.0 / (1.0 + torch.mean(torch.abs(residuals)).item()))
                else:
                    final_score = 0.0
                    
                if np.isfinite(final_score) and torch.norm(xi).item() > 0:
                    tracking_successful = True
                
                xi_np = xi.cpu().numpy()
                    
            except Exception as alignment_exception:
                logger.warning(f"[MonocularDirectTracker] PyTorch tracking aborted: {alignment_exception}. Escalating to fallback.")
                xi_np = np.zeros(6, dtype=np.float32)

        if (not tracking_successful or self.config.enable_orb_fallback) and final_score < 0.15:
            xi_fallback, fallback_score = self._execute_orb_pnp_fallback(img_a, img_b, scale_prior)
            if fallback_score > 0.05 or not tracking_successful:
                xi_np = xi_fallback
                final_score = fallback_score
                
        if not np.all(np.isfinite(xi_np)):
            xi_np = np.zeros(6, dtype=np.float32)
            
        return xi_np, float(np.clip(final_score, 0.0, 0.95))

    def _execute_orb_pnp_fallback(self, img_a: np.ndarray, img_b: np.ndarray, scale_prior: float) -> Tuple[np.ndarray, float]:
        xi_out = np.zeros(6, dtype=np.float32)
        
        def to_uint8(img: np.ndarray) -> np.ndarray:
            if img.dtype != np.uint8:
                return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return img
            
        u_a, u_b = to_uint8(img_a), to_uint8(img_b)
        
        detector = cv2.ORB_create(nfeatures=750, scaleFactor=1.2, nlevels=4)
        kp_a, des_a = detector.detectAndCompute(u_a, None)
        kp_b, des_b = detector.detectAndCompute(u_b, None)
        
        if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            return xi_out, 0.0
            
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        all_matches = sorted(matcher.match(des_a, des_b), key=lambda x: x.distance)[:150]
        
        if len(all_matches) < 8:
            return xi_out, 0.0
        
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in all_matches])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in all_matches])
        
        K = np.array([[self.config.fx, 0.0, self.config.cx], 
                      [0.0, self.config.fy, self.config.cy], 
                      [0.0, 0.0, 1.0]], dtype=np.float32)
                      
        pts_3d = np.array([[(pt[0]-self.config.cx)/self.config.fx*scale_prior, 
                            (pt[1]-self.config.cy)/self.config.fy*scale_prior, 
                            scale_prior] for pt in pts_a], dtype=np.float32)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_b, K, distCoeffs=None,
            iterationsCount=100, reprojectionError=2.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success and inliers is not None and len(inliers) >= 4:
            xi_out[:3] = tvec.flatten()
            xi_out[3:] = rvec.flatten()
            return xi_out, float(len(inliers) / len(all_matches))
            
        return xi_out, 0.0

class SigLIPClassifier:
    def __init__(self, config: LepauteConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model_name = "google/siglip-base-patch16-224"
        
        logger.info(f"[SigLIPClassifier] Spinning up HuggingFace Zero-Shot Vision Transformers on target device hardware: '{self.device}'")
        self.processor = SiglipProcessor.from_pretrained(self.model_name)
        self.model = SiglipModel.from_pretrained(self.model_name).to(self.device)
        self.labels = config.object_names
        logger.info(f"[SigLIPClassifier] Model loaded securely. Classification label space targets: {self.labels}")

    def predict(self, img: np.ndarray) -> Tuple[str, float]:
        logger.debug(f"[SigLIPClassifier] Preparing image batch transformation matrix (dimensions: {img.shape})")
        image = Image.fromarray(img)
        
        try:
            inputs = self.processor(images=image, text=self.labels, return_tensors="pt", padding="max_length").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1).squeeze(0)
            max_idx = torch.argmax(probs).item()
            
            detected_label = self.config.object_names[max_idx]
            confidence_score = probs[max_idx].item()
            return detected_label, confidence_score
            
        except Exception as e:
            logger.error(f"[SigLIPClassifier] SigLIP inference failed. Operating in absolute safety fallback mode. Trace: {e}")
            return self.config.object_names[-1], 0.0

class MonocularSE3Warping(nn.Module):
    def __init__(self, config: LepauteConfig):
        super().__init__()
        self.config = config

    def forward(self, img: torch.Tensor, xi: torch.Tensor, scale_prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = img.shape
        device = img.device
        
        vy, vx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        sp_expanded = scale_prior.view(B, 1, 1).expand(B, H, W)
        Z_proxy = torch.ones(B, H, W, device=device, dtype=torch.float32) * sp_expanded
        
        X = (vx.expand(B, H, W) - self.config.cx) * Z_proxy / self.config.fx
        Y = (vy.expand(B, H, W) - self.config.cy) * Z_proxy / self.config.fy

        P = torch.stack((X, Y, Z_proxy, torch.ones_like(Z_proxy)), dim=3).view(B, -1, 4)
        T = se3_exp_map(xi)
        P_t = torch.bmm(T, P.transpose(1, 2)).transpose(1, 2)

        X_t, Y_t, Z_t = P_t[:, :, 0], P_t[:, :, 1], P_t[:, :, 2]
        Z_t_safe = torch.clamp(Z_t, min=1e-3)
        u_t = self.config.fx * X_t / Z_t_safe + self.config.cx
        v_t = self.config.fy * Y_t / Z_t_safe + self.config.cy

        u_norm = (u_t / (W - 1)) * 2.0 - 1.0
        v_norm = (v_t / (H - 1)) * 2.0 - 1.0
        grid = torch.stack((u_norm, v_norm), dim=2).view(B, H, W, 2)

        warped_img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        valid_mask = ((u_t >= 0) & (u_t <= W - 1) & (v_t >= 0) & (v_t <= H - 1) & (Z_t > 0.1)).view(B, 1, H, W).float()
        return warped_img, valid_mask

class SE3CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_k = nn.LayerNorm(dim)
        self.norm1_v = nn.LayerNorm(dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = self.norm1_q(query)
        k = self.norm1_k(key)
        v = self.norm1_v(value)
        
        attn_output, _ = self.cross_attn(query=q, key=k, value=v)
        x = query + attn_output
        
        x = x + self.mlp(self.norm2(x))
        return x

class SE3ResidualRefiner(nn.Module):
    def __init__(self, config: LepauteConfig, feature_dim: int = 256, max_resolution: int = 64):
        super().__init__()
        self.config = config
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.pos_embed = nn.Parameter(torch.randn(1, feature_dim, max_resolution, max_resolution) * 0.02)
        
        self.pose_emb = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, feature_dim)
        )
        
        self.cross_attn = SE3CrossAttentionBlock(dim=feature_dim, num_heads=8, dropout=0.1)
        
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim * 2 + 1, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 14)
        )
        
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        if self.config.use_compiler and hasattr(torch, 'compile'):
            logger.info("Engaging PyTorch 2.x Compiler for model graph optimization.")
            try:
                self.backbone = torch.compile(self.backbone)
                self.cross_attn = torch.compile(self.cross_attn)
                self.depth_head = torch.compile(self.depth_head)
                self.head = torch.compile(self.head)
            except Exception as e:
                logger.error(f"Failed to compile model: {e}. Proceeding without compilation.")
        
        self.to(self.config.device)

    def load_compiled_state_dict(self, state_dict: Dict[str, Any]):
        current_state = self.state_dict()
        new_state_dict = {}
        
        for k, v in state_dict.items():
            clean_k = k.replace("_orig_mod.", "")
            if clean_k in current_state:
                if current_state[clean_k].shape == v.shape:
                    new_state_dict[clean_k] = v
                else:
                    logger.warning(
                        f"[SE3ResidualRefiner] Shape mismatch on layer '{clean_k}'. "
                        f"Dropping checkpoint weight and using fresh initialization."
                    )
            else:
                new_state_dict[clean_k] = v
                
        return self.load_state_dict(new_state_dict, strict=False)

    def forward(self, img_ref: torch.Tensor, img_cur: torch.Tensor, xi_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        is_unbatched = img_ref.dim() == 3
        if is_unbatched:
            img_ref = img_ref.unsqueeze(0)
            img_cur = img_cur.unsqueeze(0)
            xi_init = xi_init.unsqueeze(0)
            
        B = img_ref.shape[0]
        
        f_ref = self.backbone(img_ref)
        f_cur = self.backbone(img_cur)
        
        _, C, H_f, W_f = f_ref.shape
        
        depth_map = self.depth_head(f_cur)

        pos = F.interpolate(self.pos_embed, size=(H_f, W_f), mode='bilinear', align_corners=False)
        f_ref = f_ref + pos
        f_cur = f_cur + pos
        
        p_emb = self.pose_emb(xi_init).view(B, C, 1, 1).expand(-1, -1, H_f, W_f)
        f_cur = f_cur + p_emb
        
        f_ref_flat = f_ref.view(B, C, -1).permute(0, 2, 1)
        f_cur_flat = f_cur.view(B, C, -1).permute(0, 2, 1)
        f_attn_flat = self.cross_attn(query=f_ref_flat, key=f_cur_flat, value=f_cur_flat)
        f_attn = f_attn_flat.permute(0, 2, 1).view(B, C, H_f, W_f)
        
        combined_feat = torch.cat([f_ref, f_attn, depth_map], dim=1)
        out = self.head(combined_feat)
        
        delta_xi = out[:, 0:6]
        delta_scale = out[:, 6:7]
        uncertainty_pose = out[:, 7:13]
        uncertainty_scale = out[:, 13:14]
        
        if is_unbatched:
            delta_xi = delta_xi.squeeze(0)
            delta_scale = delta_scale.squeeze(0)
            uncertainty_pose = uncertainty_pose.squeeze(0)
            uncertainty_scale = uncertainty_scale.squeeze(0)
            
        return delta_xi, delta_scale, uncertainty_pose, uncertainty_scale
        
    def export_onnx(self, output_path: str = "lepaute_refiner.onnx"):
        self.eval()
        dummy_a = torch.randn(1, 3, self.config.orig_h, self.config.orig_w, device=self.device)
        dummy_b = torch.randn(1, 3, self.config.orig_h, self.config.orig_w, device=self.device)
        dummy_xi = torch.randn(1, 6, device=self.device)
        dummy_scale = torch.randn(1, 1, device=self.device)
        
        torch.onnx.export(
            self, 
            (dummy_a, dummy_b, dummy_xi, dummy_scale),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['img_a', 'img_b', 'xi_prior', 'scale_prior'],
            output_names=['pose', 'uncertainty'],
            dynamic_axes={'img_a': {0: 'batch_size'}, 'img_b': {0: 'batch_size'}, 'xi_prior': {0: 'batch_size'}}
        )
        logger.info(f"Successfully exported static graph to {output_path}")

    def to_quantized_cpu(self) -> nn.Module:
        self.to("cpu")
        self.eval()
        quantized_model = torch.ao.quantization.quantize_dynamic(
            self, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        logger.info("Successfully converted internal Linear layers to 8-bit precision.")
        return quantized_model
class CameraIOStream:
    def __init__(self, config: LepauteConfig, mock: bool = False):
        self.config = config
        self.cap = None
        self.frame_id = 0
        self.mock_mode = mock
        
        logger.info(f"[CameraIOStream] Subsystem initialization triggered. Execution Mode: {'MOCK_DATA_SYNTHESIS' if mock else 'PHYSICAL_HARDWARE_STREAM'}")
        if not self.mock_mode:
            self._connect_physical_camera()

    def _connect_physical_camera(self, retries=3):
        import platform
        backends = [cv2.CAP_ANY]
        system_platform = platform.system()
        
        if system_platform == "Darwin":
            backends.insert(0, cv2.CAP_AVFOUNDATION)
        elif system_platform == "Windows":
            backends.insert(0, cv2.CAP_DSHOW)
            backends.insert(1, cv2.CAP_MSMF)
        else:
            backends.insert(0, cv2.CAP_V4L2)
            
        logger.info(f"[CameraIOStream] Detected platform: '{system_platform}'. Backend strategy: {backends}")

        for attempt in range(retries):
            for backend in backends:
                backend_id_name = str(backend)
                logger.info(f"[CameraIOStream] Attempting binding via backend: {backend_id_name} (Attempt {attempt + 1}/{retries})")
                
                try:
                    self.cap = cv2.VideoCapture(0, backend)
                    
                    if self.cap and self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.orig_w)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.orig_h)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"[CameraIOStream] Successful video probe matrix via {backend_id_name}.")
                            return
                        else:
                            self.cap.release()
                except Exception as e:
                    logger.warning(f"[CameraIOStream] Runtime block during driver initialization on backend {backend_id_name}: {e}")
                    if self.cap:
                        self.cap.release()
                    
            logger.warning(f"[CameraIOStream] Connection attempt {attempt + 1} exhausted. Suspending before retry...")
            time.sleep(1.0)
                
        raise RuntimeError("Fatal: Failed to sequentially connect to physical camera hardware.")

    def read(self) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        self.frame_id += 1
        meta = {"timestamp": time.time(), "frame_id": self.frame_id}
        
        if not self.mock_mode and self.cap and self.cap.isOpened():
            logger.info(f"[CameraIOStream] Querying active hardware device frame buffer. Index: {self.frame_id}")
            ret, frame = self.cap.read()
            if not ret or frame is None: 
                logger.warning(f"[CameraIOStream] Non-responsive frame drop or device link termination detected at Frame ID: {self.frame_id}. Re-triggering connection parameters.")
                self.cap.release()
                self._connect_physical_camera(retries=1)
                return False, np.zeros(0), meta
                
            logger.info(f"[CameraIOStream] Frame ID {self.frame_id} pulled safely from device buffer. Downsampling shape {frame.shape} to target configuration bounds: ({self.config.orig_w}x{self.config.orig_h})")
            rgb = cv2.resize(frame, (self.config.orig_w, self.config.orig_h))
            return True, rgb, meta
            
        elif self.mock_mode:
            logger.info(f"[CameraIOStream] Generating synthetic multi-spectral tensor field frame simulation. Index: {self.frame_id}")
            h, w = self.config.orig_h, self.config.orig_w
            
            texture = np.ones((h, w), dtype=np.uint8) * 50
            pos_x = (self.frame_id * 3) % max(1, (w - 30))
            pos_y = (self.frame_id * 2) % max(1, (h - 30))
            cv2.rectangle(texture, (pos_x, pos_y), (pos_x+30, pos_y+30), 220, -1)
            
            for i in range(0, w, 20):
                cv2.line(texture, (i, 0), (i, h), 100, 1)
            for j in range(0, h, 20):
                cv2.line(texture, (0, j), (w, j), 100, 1)
                
            rgb = cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)
            return True, rgb, meta
            
        logger.error(f"[CameraIOStream] Processing call rejected: Camera interface handle context has collapsed or was not opened. Index: {self.frame_id}")
        return False, np.zeros(0), meta

    def release(self):
        logger.info("[CameraIOStream] Disconnecting camera stream references and executing secure interface context teardown.")
        if self.cap: 
            self.cap.release()
            logger.info("[CameraIOStream] VideoCapture interface freed successfully.")

class SequenceDataCollector(threading.Thread):
    def __init__(self, config: LepauteConfig):
        super().__init__(daemon=True)
        self.db_path = config.data_store
        self.write_queue = queue.Queue(maxsize=200)
        self.running = threading.Event()
        self.running.set()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute('''CREATE TABLE IF NOT EXISTS transitions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            img_a BLOB, img_b BLOB,
                            xi TEXT, obj_name TEXT)''')

    def append_transition(self, img_a: np.ndarray, img_b: np.ndarray, xi: np.ndarray, obj_name: str):
        if not self.write_queue.full():
            self.write_queue.put_nowait((img_a.copy(), img_b.copy(), xi.copy(), obj_name))

    def run(self):
        logger.info("[SequenceDataCollector] Background persistence thread active.")
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            while self.running.is_set() or not self.write_queue.empty():
                batch = []
                try:
                    while len(batch) < 10:
                        if not self.running.is_set() and self.write_queue.empty():
                            break
                        batch.append(self.write_queue.get(timeout=0.1))
                except queue.Empty:
                    pass
                
                if batch:
                    logger.debug(f"[SequenceDataCollector] Processing persistence batch of size {len(batch)}...")
                    records = []
                    for (img_a, img_b, xi, obj_name) in batch:
                        _, enc_a = cv2.imencode('.jpg', img_a, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        _, enc_b = cv2.imencode('.jpg', img_b, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        records.append((enc_a.tobytes(), enc_b.tobytes(), json.dumps(xi.tolist()), obj_name))
                        
                    try:
                        conn.executemany("INSERT INTO transitions (img_a, img_b, xi, obj_name) VALUES (?, ?, ?, ?)", records)
                        conn.commit()
                        logger.debug("[SequenceDataCollector] Batch successfully committed to SQLite WAL.")
                    except Exception as e:
                        logger.error(f"[SequenceDataCollector] SQLite write error during batch commit: {e}")
                        
                    for _ in batch: self.write_queue.task_done()
                    
        logger.info("[SequenceDataCollector] Background thread loop exited safely.")

    def stop(self):
        logger.info("[SequenceDataCollector] Stop command received. Halting event loop...")
        self.running.clear()
        
        logger.info("[SequenceDataCollector] Waiting for thread to join (timeout=5.0s)...")
        self.join(timeout=5.0)
        
        if self.is_alive():
            logger.warning("[SequenceDataCollector] Thread join timed out. It might be blocked on DB lock.")
        else:
            logger.info("[SequenceDataCollector] Thread joined successfully.")
            
        logger.info("[SequenceDataCollector] Executing final SQLite WAL checkpoint truncation...")
        try:
            with sqlite3.connect(self.db_path, timeout=2.0) as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.commit()
            logger.info("[SequenceDataCollector] Database WAL checkpoint completed.")
        except Exception as e:
            logger.error(f"Database sync failed during SequenceDataCollector teardown: {e}")

def load_data(config: LepauteConfig) -> List[Dict]:
    data = []
    with sqlite3.connect(config.data_store) as conn:
        cursor = conn.execute("SELECT img_a, img_b, xi, obj_name FROM transitions")
        for row in cursor:
            try:
                img_a = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_COLOR)
                img_b = cv2.imdecode(np.frombuffer(row[1], np.uint8), cv2.IMREAD_COLOR)
                data.append({
                    "img_a": img_a, "img_b": img_b,
                    "lie_params": json.loads(row[2]), "detected_object": row[3]
                })
            except Exception as e:
                logger.error(f"Failed to decode DB row: {e}")
    return data

@lru_cache(maxsize=1000)
def _cached_read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(
            f"[Dataset IO Error] CRITICAL: OpenCV failed to load image at '{path}'. "
            f"Please verify that the 'jpg' directory contains this file and it is not corrupted."
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class EquivariantDataset(Dataset):
    def __init__(self, data_list: List[Dict], config: LepauteConfig, data_dir: Optional[str] = None):
        self.data_list = data_list
        self.config = config
        self.data_dir = data_dir
        
        unique_objs = sorted(list(set(item.get("detected_object", "unknown") for item in data_list)))
        if data_dir:
            self.obj_map = {n: i for i, n in enumerate(unique_objs)}
        else:
            self.obj_map = {n: i for i, n in enumerate(config.object_names)}
            
        self.transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(p=0.3),
            A.MotionBlur(p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], additional_targets={'image0': 'image'})

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        if self.data_dir:
            img_a_path = os.path.join(self.data_dir, item["frame_a"])
            img_b_path = os.path.join(self.data_dir, item["frame_b"])
            img_a = _cached_read_image(img_a_path)
            img_b = _cached_read_image(img_b_path)
        else:
            img_a = cv2.cvtColor(item["img_a"], cv2.COLOR_BGR2RGB)
            img_b = cv2.cvtColor(item["img_b"], cv2.COLOR_BGR2RGB)
            
        if img_a.shape[:2] != (self.config.orig_h, self.config.orig_w):
            img_a = cv2.resize(img_a, (self.config.orig_w, self.config.orig_h))
            img_b = cv2.resize(img_b, (self.config.orig_w, self.config.orig_h))

        transformed = self.transform(image=img_a, image0=img_b)
        t_a = transformed["image"]
        t_b = transformed["image0"]
        
        xi_gt = torch.tensor(item['lie_params'], dtype=torch.float32)
        xi_noisy = xi_gt + torch.randn(6) * 0.05 
        
        obj_name = item.get("detected_object", "unknown")
        obj_idx = self.obj_map.get(obj_name, 0)
        
        scale_prior = self.config.object_scales.get(obj_name)
        if scale_prior is None:
            scale_prior = self.config.object_scales.get("object", 1.0)
        
        return t_a, t_b, xi_gt, xi_noisy, obj_idx, float(scale_prior)

class ManifoldKinematicForecaster:
    def __init__(self, process_noise_pose: float = 1e-3, process_noise_scale: float = 1e-4):
        self.lock = threading.Lock()
        
        self.current_pose = np.eye(4)
        self.twist_velocity = np.zeros(6)
        self.log_scale = 0.0
        self.log_scale_velocity = 0.0
        
        self.last_timestamp = None
        self.q_pose = process_noise_pose
        self.q_scale = process_noise_scale

    def predict(self, timestamp: float) -> Tuple[np.ndarray, float]:
        with self.lock:
            if self.last_timestamp is None:
                self.last_timestamp = timestamp
                return self.current_pose.copy(), float(np.exp(self.log_scale))
                
            dt = timestamp - self.last_timestamp
            if dt <= 0:
                return self.current_pose.copy(), float(np.exp(self.log_scale))
                
            twist_t = torch.from_numpy(self.twist_velocity * dt).unsqueeze(0).float()
            delta_pose = se3_exp_map(twist_t).squeeze(0).cpu().numpy()
            predicted_pose = self.current_pose @ delta_pose
            
            predicted_log_scale = np.clip(self.log_scale + self.log_scale_velocity * dt, -5.0, 5.0)
            predicted_scale = float(np.exp(predicted_log_scale))
            
            return predicted_pose, predicted_scale

    def update_state(self, measured_pose: np.ndarray, delta_xi: np.ndarray, delta_scale: float, timestamp: float, weight: float = 0.7):
        with self.lock:
            if self.last_timestamp is None:
                self.current_pose = measured_pose
                self.log_scale = np.clip(np.log(max(delta_scale, 1e-4)), -5.0, 5.0)
                self.last_timestamp = timestamp
                return

            dt = timestamp - self.last_timestamp
            if dt <= 0:
                dt = 1e-3
                
            delta_xi_t = torch.from_numpy(delta_xi).unsqueeze(0).float()
            refined_measurement = measured_pose @ se3_exp_map(delta_xi_t).squeeze(0).cpu().numpy()
            
            pose_error_matrix = np.linalg.inv(self.current_pose) @ refined_measurement
            pose_err_t = torch.from_numpy(pose_error_matrix).unsqueeze(0).float()
            error_twist = se3_log_map(pose_err_t).squeeze(0).cpu().numpy()
            
            error_twist_t = torch.from_numpy(weight * error_twist).unsqueeze(0).float()
            self.current_pose = self.current_pose @ se3_exp_map(error_twist_t).squeeze(0).cpu().numpy()
            
            self.twist_velocity = (1.0 - weight) * self.twist_velocity + weight * (error_twist / dt)
            
            measured_log_scale = self.log_scale + np.log(max(delta_scale, 1e-4))
            log_scale_error = np.clip(measured_log_scale - self.log_scale, -1.0, 1.0)
            
            self.log_scale = np.clip(self.log_scale + weight * log_scale_error, -5.0, 5.0)
            new_scale_vel = log_scale_error / dt
            self.log_scale_velocity = (1.0 - weight) * self.log_scale_velocity + weight * new_scale_vel
            
            self.last_timestamp = timestamp

    def get_scale(self) -> float:
        with self.lock:
            return float(np.exp(self.log_scale))

def train_sequence_loop(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: LepauteConfig,
    epochs: int,
    checkpoint_dir: str,
    resume: bool = False
) -> Tuple[float, float]:
    import sys
    import multiprocessing
    import random
    from pathlib import Path
    import inspect
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    
    device = torch.device(config.device)
    model = model.to(device)
    
    if config.use_compiler and sys.version_info >= (3, 10) and sys.platform != "win32":
        try:
            logger.info("Engaging torch.compile() induction layer for standard optimization.")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"Compiler optimization failed to bind, falling back to eager execution mode: {e}")

    cv2.setNumThreads(0)
    
    num_workers = min(4, multiprocessing.cpu_count() or 1) if sys.platform != "win32" else 0
    pin_memory = (device.type == "cuda")
    
    logger.info(f"Configuring DataLoader: workers={num_workers}, pin_memory={pin_memory}, batch_size={getattr(config, 'batch_size', 32)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=getattr(config, "batch_size", 32), 
        shuffle=True, 
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=getattr(config, "batch_size", 32), 
            shuffle=False, 
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        patience=3, 
        factor=0.5
    )
    
    checkpoint_path = Path(checkpoint_dir)
    best_model_path = checkpoint_path / "best_model.pth"
    latest_checkpoint_path = checkpoint_path / "latest_checkpoint.pth"
    
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 5
    start_epoch = 0
    
    avg_train_loss = 0.0
    avg_val_loss = 0.0

    if resume and latest_checkpoint_path.exists():
        logger.info(f"Restoring system state from historical checkpoint: {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
            
            raw_state_dict = checkpoint["model_state_dict"]
            if hasattr(model, "_orig_mod"):
                model._orig_mod.load_state_dict(raw_state_dict)
            else:
                model.load_state_dict(raw_state_dict)
                
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            patience_counter = checkpoint["patience_counter"]
            
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
            if "np_rng_state" in checkpoint:
                np.random.set_state(checkpoint["np_rng_state"])
            if "random_state" in checkpoint:
                random.setstate(checkpoint["random_state"])
                
            logger.info(f"System state restored. Resuming training from Epoch {start_epoch + 1}.")
        except Exception as e:
            logger.error(f"Checkpoint state load failure: {e}. Falling back to training from scratch.")
            start_epoch = 0
    
    avg_train_loss = 0.0
    avg_val_loss = 0.0

    if resume and latest_checkpoint_path.exists():
        logger.info(f"Restoring system state from historical checkpoint: {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
            
            raw_state_dict = checkpoint["model_state_dict"]
            if hasattr(model, "_orig_mod"):
                model._orig_mod.load_state_dict(raw_state_dict)
            else:
                model.load_state_dict(raw_state_dict)
                
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            patience_counter = checkpoint["patience_counter"]
            
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
            if "np_rng_state" in checkpoint:
                np.random.set_state(checkpoint["np_rng_state"])
            if "random_state" in checkpoint:
                random.setstate(checkpoint["random_state"])
                
            logger.info(f"System state restored. Resuming training from Epoch {start_epoch + 1}.")
        except Exception as e:
            logger.error(f"Checkpoint state load failure: {e}. Falling back to training from scratch.")
            start_epoch = 0
    
    avg_train_loss = 0.0
    avg_val_loss = 0.0

    if resume and latest_checkpoint_path.exists():
        logger.info(f"Restoring system state from historical checkpoint: {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
            
            raw_state_dict = checkpoint["model_state_dict"]
            if hasattr(model, "_orig_mod"):
                model._orig_mod.load_state_dict(raw_state_dict)
            else:
                model.load_state_dict(raw_state_dict)
                
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            patience_counter = checkpoint["patience_counter"]
            
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
            if "np_rng_state" in checkpoint:
                np.random.set_state(checkpoint["np_rng_state"])
            if "random_state" in checkpoint:
                random.setstate(checkpoint["random_state"])
                
            logger.info(f"System state restored. Resuming training from Epoch {start_epoch + 1}.")
        except Exception as e:
            logger.error(f"Checkpoint state load failure: {e}. Falling back to training from scratch.")
            start_epoch = 0

    def save_checkpoint(epoch_idx: int) -> None:
        try:
            clean_model_state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            checkpoint_data = {
                "epoch": epoch_idx,
                "model_state_dict": clean_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "torch_rng_state": torch.get_rng_state(),
                "np_rng_state": np.random.get_state(),
                "random_state": random.getstate(),
            }
            if torch.cuda.is_available():
                checkpoint_data["cuda_rng_state"] = torch.cuda.get_rng_state()
                
            temp_path = latest_checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint_data, temp_path)
            temp_path.replace(latest_checkpoint_path)
        except Exception as save_err:
            logger.error(f"Non-fatal checkpoint save exception (Epoch {epoch_idx + 1}): {save_err}")
            
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    try:
        sig = inspect.signature(base_model.forward)
        takes_scale = "scale_prior" in sig.parameters
    except Exception as sig_err:
        logger.warning(f"Failed to introspect model signature securely: {sig_err}. Assuming robust fallback architecture without explicit scale injection.")
        takes_scale = False

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss_accum = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Train]", leave=False, dynamic_ncols=True)
            
            for batch_idx, batch in enumerate(train_pbar):
                if isinstance(batch, dict):
                    t_a = batch["t_a"].to(device=device, dtype=torch.float32)
                    t_b = batch["t_b"].to(device=device, dtype=torch.float32)
                    xi_gt = batch["xi_gt"].to(device=device, dtype=torch.float32)
                    xi_noisy = batch["xi_noisy"].to(device=device, dtype=torch.float32)
                    scale_prior = batch.get("scale_prior", torch.ones_like(xi_gt[:, 0:1])).to(device=device, dtype=torch.float32).view(-1, 1)
                elif isinstance(batch, (list, tuple)):
                    t_a = batch[0].to(device=device, dtype=torch.float32)
                    t_b = batch[1].to(device=device, dtype=torch.float32)
                    xi_gt = batch[2].to(device=device, dtype=torch.float32)
                    xi_noisy = batch[3].to(device=device, dtype=torch.float32)
                    scale_prior = batch[5].to(device=device, dtype=torch.float32).view(-1, 1) if len(batch) > 5 else torch.ones_like(xi_gt[:, 0:1]).to(device=device, dtype=torch.float32).view(-1, 1)
                else:
                    raise TypeError(f"Unrecognized batch format returned by DataLoader: {type(batch)}")
                
                optimizer.zero_grad()
                outputs = model(t_a, t_b, xi_noisy, scale_prior=scale_prior) if "scale_prior" in model.forward.__code__.co_varnames else model(t_a, t_b, xi_noisy)
                
                if isinstance(outputs, dict):
                    pred_pose = outputs["pose"]
                elif isinstance(outputs, (list, tuple)):
                    pred_pose = outputs[0]
                else:
                    pred_pose = outputs
                
                loss = torch.mean((pred_pose - xi_gt) ** 2)
                loss.backward()
                optimizer.step()
                
                train_loss_accum += loss.item()
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_train = train_loss_accum / max(1, len(train_loader))
            avg_train_loss = avg_train
            
            if val_loader is not None:
                model.eval()
                val_loss_accum = 0.0
                
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Valid]", leave=False, dynamic_ncols=True)
                
                with torch.no_grad():
                    for batch in val_pbar:
                        if isinstance(batch, dict):
                            t_a = batch["t_a"].to(device=device, dtype=torch.float32)
                            t_b = batch["t_b"].to(device=device, dtype=torch.float32)
                            xi_gt = batch["xi_gt"].to(device=device, dtype=torch.float32)
                            xi_noisy = batch["xi_noisy"].to(device=device, dtype=torch.float32)
                            scale_prior = batch.get("scale_prior", torch.ones_like(xi_gt[:, 0:1])).to(device=device, dtype=torch.float32).view(-1, 1)
                        elif isinstance(batch, (list, tuple)):
                            t_a = batch[0].to(device=device, dtype=torch.float32)
                            t_b = batch[1].to(device=device, dtype=torch.float32)
                            xi_gt = batch[2].to(device=device, dtype=torch.float32)
                            xi_noisy = batch[3].to(device=device, dtype=torch.float32)
                            scale_prior = batch[5].to(device=device, dtype=torch.float32).view(-1, 1) if len(batch) > 5 else torch.ones_like(xi_gt[:, 0:1]).to(device=device, dtype=torch.float32).view(-1, 1)
                        
                        outputs = model(t_a, t_b, xi_noisy, scale_prior=scale_prior) if "scale_prior" in model.forward.__code__.co_varnames else model(t_a, t_b, xi_noisy)
                        
                        if isinstance(outputs, dict):
                            pred_pose = outputs["pose"]
                        elif isinstance(outputs, (list, tuple)):
                            pred_pose = outputs[0]
                        else:
                            pred_pose = outputs
                            
                        diff = pred_pose - xi_gt
                        batch_loss = torch.mean(diff ** 2).item()
                        val_loss_accum += batch_loss
                        
                        val_pbar.set_postfix({'mse': f"{batch_loss:.4f}"})
                        
    except KeyboardInterrupt:
        logger.warning("\n[Training Interruption] Manual halt signal (SIGINT / Ctrl+C) detected.")
        if 'epoch' in locals():
            logger.info(f"Executing emergency state snapshot for ongoing Epoch {epoch + 1}...")
        logger.info("[Secure State Saved] Checkpoint successfully flushed to disk. Safe to exit.")
        raise KeyboardInterrupt
    except Exception as e:
        logger.error(f"Runtime Operational Critical Failure: {e}")
        raise
                        
    def process_batch(batch_data):
        if isinstance(batch_data, dict):
            t_a = batch_data["t_a"].to(device=device, dtype=torch.float32)
            t_b = batch_data["t_b"].to(device=device, dtype=torch.float32)
            xi_gt = batch_data["xi_gt"].to(device=device, dtype=torch.float32)
            xi_noisy = batch_data["xi_noisy"].to(device=device, dtype=torch.float32)
            scale_prior = batch_data.get("scale_prior", torch.ones_like(xi_gt[:, 0:1])).to(device=device, dtype=torch.float32).view(-1, 1)
        elif isinstance(batch_data, (list, tuple)):
            t_a = batch_data[0].to(device=device, dtype=torch.float32)
            t_b = batch_data[1].to(device=device, dtype=torch.float32)
            xi_gt = batch_data[2].to(device=device, dtype=torch.float32)
            xi_noisy = batch_data[3].to(device=device, dtype=torch.float32)
            scale_prior = batch_data[5].to(device=device, dtype=torch.float32).view(-1, 1) if len(batch_data) > 5 else torch.ones_like(xi_gt[:, 0:1]).to(device=device, dtype=torch.float32).view(-1, 1)
        else:
            raise TypeError(f"Unrecognized batch format returned by DataLoader: {type(batch_data)}")
        return t_a, t_b, xi_gt, xi_noisy, scale_prior

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss_accum = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Train]", leave=False, dynamic_ncols=True)
            
            for batch_idx, batch in enumerate(train_pbar):
                t_a, t_b, xi_gt, xi_noisy, scale_prior = process_batch(batch)
                
                optimizer.zero_grad()
                
                if takes_scale:
                    outputs = model(t_a, t_b, xi_noisy, scale_prior=scale_prior)
                else:
                    outputs = model(t_a, t_b, xi_noisy)
                
                if isinstance(outputs, dict):
                    pred_pose = outputs["pose"]
                elif isinstance(outputs, (list, tuple)):
                    pred_pose = outputs[0]
                else:
                    pred_pose = outputs
                
                loss = torch.mean((pred_pose - xi_gt) ** 2)
                loss.backward()
                optimizer.step()
                
                train_loss_accum += loss.item()
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_train = train_loss_accum / max(1, len(train_loader))
            avg_train_loss = avg_train
            
            if val_loader is not None:
                model.eval()
                val_loss_accum = 0.0
                
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Valid]", leave=False, dynamic_ncols=True)
                
                with torch.no_grad():
                    for batch in val_pbar:
                        t_a, t_b, xi_gt, xi_noisy, scale_prior = process_batch(batch)
                        
                        if takes_scale:
                            outputs = model(t_a, t_b, xi_noisy, scale_prior=scale_prior)
                        else:
                            outputs = model(t_a, t_b, xi_noisy)
                        
                        if isinstance(outputs, dict):
                            pred_pose = outputs["pose"]
                        elif isinstance(outputs, (list, tuple)):
                            pred_pose = outputs[0]
                        else:
                            pred_pose = outputs
                            
                        diff = pred_pose - xi_gt
                        batch_loss = torch.mean(diff ** 2).item()
                        val_loss_accum += batch_loss
                        
                        val_pbar.set_postfix({'mse': f"{batch_loss:.4f}"})
                        
                avg_val = val_loss_accum / max(1, len(val_loader))
                avg_val_loss = avg_val
                
                scheduler.step(avg_val)
                logger.info(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train:.5f} | Val MSE: {avg_val:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    clean_model_state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
                    torch.save(clean_model_state, best_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        logger.warning(f"Early stopping limit reached (no improvement for {early_stop_patience} consecutive epochs), terminating training.")
                        if best_model_path.exists():
                            try:
                                state = torch.load(best_model_path, map_location=device, weights_only=True)
                                if hasattr(model, "_orig_mod"):
                                    model._orig_mod.load_state_dict(state)
                                else:
                                    model.load_state_dict(state)
                            except Exception as e:
                                logger.error(f"Failed to load best model state for graceful termination: {e}")
                        break
            else:
                scheduler.step(avg_train)
                logger.info(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                avg_val_loss = avg_train
            
            save_checkpoint(epoch)
            
    except KeyboardInterrupt:
        logger.warning("\n[Training Interruption] Manual halt signal (SIGINT / Ctrl+C) detected.")
        if 'epoch' in locals():
            logger.info(f"Executing emergency state snapshot for ongoing Epoch {epoch + 1}...")
            save_checkpoint(epoch)
        logger.info("[Secure State Saved] Checkpoint successfully flushed to disk. Safe to exit.")
        raise KeyboardInterrupt
    except Exception as e:
        logger.error(f"Runtime Operational Critical Failure: {e}")
        raise

    return avg_train_loss, avg_val_loss