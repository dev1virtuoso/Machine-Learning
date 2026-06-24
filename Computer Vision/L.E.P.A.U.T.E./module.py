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



_mps_lock = threading.Lock()

logger = logging.getLogger("LEPAUTE.Core")

class DisplayMode(str, Enum):
    REALTIME = "realtime"
    JSON = "json"
    HEADLESS = "headless"

def _get_stable_compute_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        logger.warning("[LepauteConfig] Apple Silicon MPS bypassed. Forcing 'cpu' to prevent fatal Metal command buffer collisions in multi-threaded inference.")
    return "cpu"


class LepauteConfig(BaseSettings):
    device: str = Field(default_factory=_get_stable_compute_device)
    data_store: str = "lepaute_data.db"
    object_names: List[str] = ["table", "cup", "keyboard", "laptop", "mouse", "background"]
    object_scales: Dict[str, float] = {
        "table": 1.5, "cup": 0.1, "keyboard": 0.4, 
        "laptop": 0.35, "mouse": 0.12, "background": 2.0
    }
    orig_h: int = 240
    orig_w: int = 320
    fx: float = 250.0
    fy: float = 250.0
    cx: float = 160.0
    cy: float = 120.0
    pyramid_levels: int = 3
    gn_max_iter: int = 15
    use_compiler: bool = True
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

def _se3_exp_map_impl(xi: torch.Tensor) -> torch.Tensor:
    B = xi.shape[0]
    rho, phi = xi[:, :3], xi[:, 3:]
    theta_sq = torch.sum(phi**2, dim=1, keepdim=True)
    theta = torch.sqrt(theta_sq + 1e-10)
    
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

def se3_log_map(T: torch.Tensor) -> torch.Tensor:
    if T.device.type == "mps":
        with _mps_lock:
            return _se3_log_map_impl(T)
    return _se3_log_map_impl(T)

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
        phi_l = (th / (2.0 * torch.sin(th))) * phi_raw[mask_large]
        xi[mask_large, 3:] = phi_l
        
        K = _skew_symmetric_impl(phi_l)
        I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).expand(mask_large.sum(), -1, -1)
        half_th = th / 2.0
        V_inv = I - 0.5 * K + (1.0 - (th * torch.cos(half_th)) / (2.0 * torch.sin(half_th))) * torch.bmm(K, K) / (th**2)
        xi[mask_large, :3] = torch.bmm(V_inv, t[mask_large].unsqueeze(-1)).squeeze(-1)
        
    if mask_small.any():
        xi[mask_small, 3:] = 0.5 * phi_raw[mask_small]
        K = _skew_symmetric_impl(xi[mask_small, 3:])
        I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0).expand(mask_small.sum(), -1, -1)
        V_inv = I - 0.5 * K + (1.0/12.0) * torch.bmm(K, K)
        xi[mask_small, :3] = torch.bmm(V_inv, t[mask_small].unsqueeze(-1)).squeeze(-1)
        
    return xi

def compose_poses(T_global: torch.Tensor, T_rel: torch.Tensor) -> torch.Tensor:
    if T_global.device.type == "mps":
        with _mps_lock:
            return torch.bmm(T_global, T_rel)
    return torch.bmm(T_global, T_rel)

class MonocularDirectTracker:
    """
    Subsystem for direct photometric alignment on SE(3) manifolds across multi-level image pyramids.
    Includes a robust ORB feature-based fallback mechanism to guarantee tracking continuity under
    extreme dynamic conditions, low-texture environments, or initialization noise.
    """
    def __init__(self, config: Any):
        self.config = config
        self.fx = getattr(config, "fx", 250.0)
        self.fy = getattr(config, "fy", 250.0)
        self.cx = getattr(config, "cx", 160.0)
        self.cy = getattr(config, "cy", 120.0)
        self.pyramid_levels = getattr(config, "pyramid_levels", 3)
        self.enable_orb_fallback = getattr(config, "enable_orb_fallback", True)
        logger.info(f"[MonocularDirectTracker] Initialized with camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy} | Levels={self.pyramid_levels}")

    def track(self, img_a: np.ndarray, img_b: np.ndarray, scale_prior: float = 1.0) -> Tuple[np.ndarray, float]:
        logger.info(f"[MonocularDirectTracker] Starting tracking optimization cycle. Prior Scale: {scale_prior:.4f}m | Target Shapes: {img_a.shape} -> {img_b.shape}")

        if len(img_a.shape) == 3:
            gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY if img_a.shape[2] == 3 else cv2.COLOR_BGR2GRAY)
        else:
            gray_a = img_a.copy()
            
        if len(img_b.shape) == 3:
            gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY if img_b.shape[2] == 3 else cv2.COLOR_BGR2GRAY)
        else:
            gray_b = img_b.copy()

        gray_a = gray_a.astype(np.float32)
        gray_b = gray_b.astype(np.float32)

        pyr_a = [gray_a]
        pyr_b = [gray_b]
        for l in range(self.pyramid_levels - 1):
            pyr_a.append(cv2.pyrDown(pyr_a[-1]))
            pyr_b.append(cv2.pyrDown(pyr_b[-1]))

        xi = np.zeros(6, dtype=np.float32)
        tracking_successful = False
        final_score = 0.0
        residuals = np.array([0.0], dtype=np.float32)

        try:
            for lvl in reversed(range(self.pyramid_levels)):
                img_lvl_a = pyr_a[lvl]
                img_lvl_b = pyr_b[lvl]
                h, w = img_lvl_a.shape
                
                scale_factor = 1.0 / (2.0 ** lvl)
                fx_l = self.fx * scale_factor
                fy_l = self.fy * scale_factor
                cx_l = self.cx * scale_factor
                cy_l = self.cy * scale_factor
                
                scharr_x = cv2.Scharr(img_lvl_b, cv2.CV_32F, 1, 0)
                scharr_y = cv2.Scharr(img_lvl_b, cv2.CV_32F, 0, 1)
                
                u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
                
                max_iters = 10
                for iter_idx in range(max_iters):
                    tx, ty, tz = xi[0], xi[1], xi[2]
                    wx, wy, wz = xi[3], xi[4], xi[5]
                    
                    Z = scale_prior
                    X = (u_coords - cx_l) / fx_l * Z
                    Y = (v_coords - cy_l) / fy_l * Z
                    
                    X_prime = X + (wz * Y - wy * Z) + tx
                    Y_prime = Y + (-wz * X + wx * Z) + ty
                    Z_prime = Z + (wy * X - wx * Y) + tz
                    
                    Z_prime = np.maximum(Z_prime, 1e-4)
                    
                    u_prime = (X_prime / Z_prime) * fx_l + cx_l
                    v_prime = (Y_prime / Z_prime) * fy_l + cy_l
                    
                    valid_mask = (u_prime >= 0) & (u_prime < w - 1) & (v_prime >= 0) & (v_prime < h - 1)
                    if np.sum(valid_mask) < 16:
                        break
                        
                    warped_b = cv2.remap(img_lvl_b, u_prime.astype(np.float32), v_prime.astype(np.float32), cv2.INTER_LINEAR)
                    warped_gx = cv2.remap(scharr_x, u_prime.astype(np.float32), v_prime.astype(np.float32), cv2.INTER_LINEAR)
                    warped_gy = cv2.remap(scharr_y, u_prime.astype(np.float32), v_prime.astype(np.float32), cv2.INTER_LINEAR)
                    
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

                    J = np.zeros((h, w, 6), dtype=np.float32)
                    J[..., 0] = J_X
                    J[..., 1] = J_Y
                    J[..., 2] = J_Z
                    J[..., 3] = -J_Y * Z_prime + J_Z * Y_prime
                    J[..., 4] =  J_X * Z_prime - J_Z * X_prime
                    J[..., 5] = -J_X * Y_prime + J_Y * X_prime
                    
                    J_valid = J[valid_mask]
                    r_valid = residuals[valid_mask]
                    
                    H = np.dot(J_valid.T, J_valid)
                    b = -np.dot(J_valid.T, r_valid)
                    
                    H += 1e-4 * np.eye(6, dtype=np.float32)
                    
                    try:
                        delta_xi = np.linalg.solve(H, b)
                        xi += delta_xi
                        if np.linalg.norm(delta_xi) < 1e-4:
                            break
                    except np.linalg.LinAlgError:
                        break
                        
            final_score = float(1.0 / (1.0 + np.mean(np.abs(residuals))))
            if np.isfinite(final_score) and np.linalg.norm(xi) > 0:
                tracking_successful = True
                
        except Exception as alignment_exception:
            logger.warning(f"[MonocularDirectTracker] Direct alignment error encounter: {alignment_exception}. Evaluating robust fallback context track.")

        if (not tracking_successful or self.enable_orb_fallback) and final_score < 0.15:
            xi_fallback, fallback_score = self._execute_orb_pnp_fallback(img_a, img_b, scale_prior)
            if fallback_score > 0.05 or not tracking_successful:
                xi = xi_fallback
                final_score = fallback_score

        if not np.all(np.isfinite(xi)):
            xi = np.zeros(6, dtype=np.float32)
        if not (0.0 <= final_score <= 1.0):
            final_score = float(np.clip(final_score, 0.0, 0.95))

        return xi, final_score

    def _execute_orb_pnp_fallback(self, img_a: np.ndarray, img_b: np.ndarray, scale_prior: float) -> Tuple[np.ndarray, float]:
        xi_out = np.zeros(6, dtype=np.float32)
        
        def to_uint8(img: np.ndarray) -> np.ndarray:
            if img.dtype != np.uint8:
                img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                return img_normalized.astype(np.uint8)
            return img
            
        u_a = to_uint8(img_a)
        u_b = to_uint8(img_b)
        
        detector = cv2.ORB_create(nfeatures=750, scaleFactor=1.2, nlevels=4)
        kp_a, des_a = detector.detectAndCompute(u_a, None)
        kp_b, des_b = detector.detectAndCompute(u_b, None)
        
        if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            return xi_out, 0.0
            
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        all_matches = matcher.match(des_a, des_b)
        if len(all_matches) < 8:
            return xi_out, 0.0
            
        all_matches = sorted(all_matches, key=lambda x: x.distance)[:150]
        
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in all_matches])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in all_matches])
        
        K = np.array([[self.fx, 0.0, self.cx],
                      [0.0, self.fy, self.cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

        pts_3d = []
        pts_2d_b = []
        for i, pt in enumerate(pts_a):
            Z = scale_prior
            X = (pt[0] - self.cx) / self.fx * Z
            Y = (pt[1] - self.cy) / self.fy * Z
            pts_3d.append([X, Y, Z])
            pts_2d_b.append(pts_b[i])
            
        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d_b = np.array(pts_2d_b, dtype=np.float32)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d_b, K, distCoeffs=None,
            iterationsCount=100, reprojectionError=2.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success and inliers is not None and len(inliers) >= 4:
            xi_out[:3] = tvec.flatten()
            xi_out[3:] = rvec.flatten()
            score = float(len(inliers) / len(all_matches))
            return xi_out, score
            
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
        logger.info(f"[SigLIPClassifier] Preparing image batch transformation matrix (dimensions: {img.shape})")
        image = Image.fromarray(img)
        
        try:
            inputs = self.processor(images=image, text=self.labels, return_tensors="pt", padding="max_length").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            logger.info("[SigLIPClassifier] Parsing output distribution probability tensors...")
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1).squeeze(0)
            max_idx = torch.argmax(probs).item()
            
            detected_label = self.config.object_names[max_idx]
            confidence_score = probs[max_idx].item()
            logger.info(f"[SigLIPClassifier] Classification complete! Best Match: '{detected_label}' with confidence {confidence_score:.4f}")
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
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.xi_proj = nn.Sequential(nn.Linear(6, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, embed_dim))

    def forward(self, visual_feat: torch.Tensor, xi_prior: torch.Tensor) -> torch.Tensor:
        B, C, H, W = visual_feat.shape
        v_flat = visual_feat.view(B, C, -1).permute(0, 2, 1) 
        q_xi = self.xi_proj(xi_prior).unsqueeze(1) 
        attn_out, _ = self.mha(query=q_xi, key=v_flat, value=v_flat)
        attn_spatial = attn_out.permute(0, 2, 1).unsqueeze(-1).expand(B, C, H, W)
        return self.norm(visual_feat + attn_spatial)
    
class SE3ResidualRefiner(nn.Module):
    def __init__(self, config: LepauteConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.ln_conv = nn.LayerNorm(64)
        
        self.flat_dim = 64 * (config.orig_h // 4) * (config.orig_w // 4)
        
        self.ln_xi = nn.LayerNorm(6)
        
        self.fc1 = nn.Linear(self.flat_dim + 6 + 1, 128)
        self.fc2 = nn.Linear(128, 6)
        self.fc_unc = nn.Linear(128, 6)

    def load_compiled_state_dict(self, state_dict: Dict[str, Any]):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        
        return self.load_state_dict(new_state_dict, strict=True)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor, xi_noisy: torch.Tensor, scale_prior: torch.Tensor) -> Dict[str, torch.Tensor]:
        img_a = img_a.to(dtype=torch.float32)
        img_b = img_b.to(dtype=torch.float32)
        xi_noisy = xi_noisy.to(dtype=torch.float32).view(xi_noisy.size(0), -1)
        scale_prior = scale_prior.to(dtype=torch.float32).view(scale_prior.size(0), 1)

        x = torch.cat([img_a, img_b], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.permute(0, 2, 3, 1)
        x = self.ln_conv(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), -1)
        
        xi_normalized = self.ln_xi(xi_noisy)
        xi_clamped = torch.clamp(xi_normalized, min=-10.0, max=10.0)
        
        feat = torch.cat([x, xi_clamped, scale_prior], dim=1)
        feat = F.relu(self.fc1(feat))
        
        pose_output = self.fc2(feat)
        unc_output = self.fc_unc(feat)
        
        unc_clamped = torch.clamp(unc_output, min=-6.0, max=6.0)
        
        return {
            "pose": pose_output, 
            "uncertainty": unc_clamped
        }

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
        
        if sys.platform == "darwin":
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        elif sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
            
        logger.info(f"[CameraIOStream] Detected system environment: '{sys.platform}'. Restricted backend lookup vector: {backends}")

        for attempt in range(retries):
            for backend in backends:
                backend_id_name = str(backend)
                logger.info(f"[CameraIOStream] Attempting device binding to index 0 using backend framework: {backend_id_name} (Attempt {attempt + 1}/{retries})...")
                
                try:
                    self.cap = cv2.VideoCapture(0, backend)
                    
                    if self.cap.isOpened():
                        logger.info(f"[CameraIOStream] Port opened successfully via backend {backend_id_name}. Syncing dimensional matrices...")
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.orig_w)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.orig_h)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) 
                        
                        logger.info("[CameraIOStream] Verifying frame readability to assert matrix stream validity...")
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"[CameraIOStream] Success! Validation frame matrix received. Engaged device safely using backend: {backend_id_name}")
                            return
                        else:
                            logger.warning(f"[CameraIOStream] Backend {backend_id_name} reported open handle but read query failed. Releasing device interface context.")
                            self.cap.release()
                    else:
                        logger.warning(f"[CameraIOStream] Hardware address index 0 rejected connection using backend driver code: {backend_id_name}")
                except Exception as e:
                    logger.error(f"[CameraIOStream] Intercepted runtime crash during driver initialization on backend {backend_id_name}: {e}")
                    if self.cap:
                        self.cap.release()
                    continue
                    
            logger.warning(f"[CameraIOStream] Operational connection attempt {attempt + 1} exhausted. Re-trying frame sequencing loops...")
            time.sleep(0.5 * (attempt + 1))
                
        raise RuntimeError("Fatal: Failed to connect to physical camera hardware. Running in production without mock flag requires real data.")

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
            texture = np.random.randint(0, 255, (h, w), dtype=np.uint8)
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
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            while self.running.is_set() or not self.write_queue.empty():
                batch = []
                try:
                    while len(batch) < 10:
                        batch.append(self.write_queue.get(timeout=0.1))
                except queue.Empty:
                    pass
                
                if batch:
                    records = []
                    for (img_a, img_b, xi, obj_name) in batch:
                        _, enc_a = cv2.imencode('.jpg', img_a, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        _, enc_b = cv2.imencode('.jpg', img_b, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        records.append((enc_a.tobytes(), enc_b.tobytes(), json.dumps(xi.tolist()), obj_name))
                        
                    conn.executemany("INSERT INTO transitions (img_a, img_b, xi, obj_name) VALUES (?, ?, ?, ?)", records)
                    conn.commit()
                    for _ in batch: self.write_queue.task_done()

    def stop(self):
        self.running.clear()
        self.join(timeout=5.0)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

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
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

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
            img_a_path = os.path.join(self.data_dir, "frames", item["frame_a"])
            img_b_path = os.path.join(self.data_dir, "frames", item["frame_b"])
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
        match = re.search(r"obj_(\d+)", obj_name)
        if match:
            obj_name = "object"
            
        obj_idx = self.obj_map.get(obj_name, 0)
        scale_prior = self.config.object_scales.get(obj_name, 1.0)
        
        return t_a, t_b, xi_gt, xi_noisy, obj_idx, scale_prior

class ManifoldKinematicForecaster:
    def __init__(self):
        self.vel_history = deque(maxlen=10)
        self.velocity = np.zeros(6, dtype=np.float32)
        self.last_xi = None
        self.last_time = None

    def update(self, current_xi: np.ndarray, timestamp: float):
        if self.last_xi is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt > 1e-4:
                new_vel = (current_xi - self.last_xi) / dt
                self.vel_history.append(new_vel)
                self.velocity = np.median(np.array(self.vel_history), axis=0)
        self.last_xi = current_xi.copy()
        self.last_time = timestamp

    def predict_next(self, current_xi: np.ndarray, dt: float) -> np.ndarray:
        vel_tensor = torch.from_numpy(self.velocity * dt).float().unsqueeze(0)
        T_vel = se3_exp_map(vel_tensor)
        xi_tensor = torch.from_numpy(current_xi).float().unsqueeze(0)
        T_curr = se3_exp_map(xi_tensor)
        T_next = torch.bmm(T_vel, T_curr)
        return se3_log_map(T_next).squeeze(0).numpy()

def train_sequence_loop(
    model: nn.Module, 
    train_dataset: Dataset, 
    val_dataset: Optional[Dataset], 
    config: Any, 
    epochs: int,
    checkpoint_dir: Optional[str] = None
) -> Tuple[float, float]:
    
    device = torch.device(config.device)
    is_cuda = (device.type == "cuda")
    
    if config.use_compiler and hasattr(torch, 'compile'):
        try:
            logger.info("Enabling PyTorch 2.x Torch Compile for hardware acceleration...")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"Current environment does not support AOT compilation, automatically switched back to native mode: {e}")
        
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, 
        pin_memory=is_cuda, num_workers=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, 
        pin_memory=is_cuda, num_workers=2
    ) if val_dataset else None
    
    best_val_loss = float('inf')
    final_train_loss = 0.0
    patience_counter = 0
    early_stop_patience = 5
    
    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        best_model_path = ckpt_path / "best_model.pth"
    else:
        best_model_path = Path("best_model.pth")
    
    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0.0
        
        for t_a, t_b, xi_gt, xi_noisy, _, scale_prior in train_loader:
            t_a = t_a.to(device=device, dtype=torch.float32, non_blocking=is_cuda)
            t_b = t_b.to(device=device, dtype=torch.float32, non_blocking=is_cuda)
            xi_gt = xi_gt.to(device=device, dtype=torch.float32, non_blocking=is_cuda)
            xi_noisy = xi_noisy.to(device=device, dtype=torch.float32, non_blocking=is_cuda)
            scale_prior = scale_prior.to(device=device, dtype=torch.float32, non_blocking=is_cuda)
            
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(t_a, t_b, xi_noisy, scale_prior=scale_prior)
            diff = outputs["pose"] - xi_gt
            
            unc = outputs["uncertainty"]
            loss_variance_term = 0.5 * torch.exp(-unc) * (diff ** 2)
            loss_penalty_term = 0.5 * unc
            loss = torch.mean(loss_variance_term + loss_penalty_term) + 1e-7

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Detected critical loss anomaly (NaN/Inf), skipping current batch. "
                           f"Unc Range: [{unc.min().item():.4f}, {unc.max().item():.4f}]")
                continue
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_accum += loss.item()
            
        avg_train = train_loss_accum / max(1, len(train_loader))
        final_train_loss = avg_train
        
        if checkpoint_dir:
            epoch_ckpt = ckpt_path / f"checkpoint_epoch_{epoch+1:03d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train,
            }, epoch_ckpt)

        if val_loader:
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for t_a, t_b, xi_gt, xi_noisy, _, scale_prior in val_loader:
                    t_a = t_a.to(device=device, dtype=torch.float32)
                    t_b = t_b.to(device=device, dtype=torch.float32)
                    xi_gt = xi_gt.to(device=device, dtype=torch.float32)
                    xi_noisy = xi_noisy.to(device=device, dtype=torch.float32)
                    scale_prior = scale_prior.to(device=device, dtype=torch.float32)
                    
                    outputs = model(t_a, t_b, xi_noisy, scale_prior=scale_prior)
                    diff = outputs["pose"] - xi_gt
                    val_loss_accum += torch.mean(diff ** 2).item()
                    
            avg_val = val_loss_accum / max(1, len(val_loader))
            scheduler.step(avg_val)
            logger.info(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train:.5f} | Val MSE: {avg_val:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logger.warning(f"Early stopping limit reached (no improvement for {early_stop_patience} consecutive epochs), terminating training.")
                    if best_model_path.exists():
                        model.load_state_dict(torch.load(best_model_path, map_location=device))
                    break
        else:
            scheduler.step(avg_train)
            logger.info(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train:.5f}")

    return final_train_loss, best_val_loss if val_loader else 0.0