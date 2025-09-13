import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Tuple, List
import kornia as K
import asyncio
import platform
import json
import os
from collections import defaultdict
import multiprocessing
import time

DATA_STORE = defaultdict(list)

def get_device():
    """Prioritize CUDA, then MPS (Apple Mac), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
DTYPE = torch.float32

# Object name mapping for detection legend
OBJECT_NAMES = {
    0: "Book",
    1: "Pen",
    2: "Phone",
    3: "Mug",
    4: "Keyboard",
    5: "Mouse",
    6: "Notebook",
    7: "Bottle",
    8: "Glasses",
    9: "Wallet"
}

class DynamicResourceManager:
    """Dynamically adjust CPU threads and processing delay to maintain target FPS."""
    def __init__(self, target_fps: float = 10.0, max_threads: int = multiprocessing.cpu_count()):
        self.target_fps = target_fps
        self.max_threads = max(max_threads, 1)
        self.min_threads = 1
        self.current_threads = max(1, self.max_threads // 2)
        self.base_delay = 0.01  # Base delay for GPU load control
        self.dynamic_delay = self.base_delay
        self.fps_history = []
        self.max_history = 10  # Number of frames to average FPS
        torch.set_num_threads(self.current_threads)

    def update(self, frame_time: float):
        """Update resources based on measured frame time."""
        actual_fps = 1.0 / frame_time if frame_time > 0 else float('inf')
        self.fps_history.append(actual_fps)
        if len(self.fps_history) > self.max_history:
            self.fps_history.pop(0)
        
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else actual_fps
        
        # Adjust threads if FPS deviates significantly
        if avg_fps < self.target_fps * 0.9:  # FPS too low, reduce load
            if self.current_threads > self.min_threads:
                self.current_threads -= 1
                torch.set_num_threads(self.current_threads)
            if self.dynamic_delay > 0.001:
                self.dynamic_delay *= 0.8  # Reduce delay to process faster
        elif avg_fps > self.target_fps * 1.1:  # FPS too high, increase load
            if self.current_threads < self.max_threads:
                self.current_threads += 1
                torch.set_num_threads(self.current_threads)
            self.dynamic_delay = min(self.dynamic_delay * 1.2, 0.1)  # Cap max delay
        
        return self.dynamic_delay

def get_collected_data() -> List[dict]:
    return DATA_STORE["pipeline_data"]

def create_meshgrid(height: int, width: int, device: torch.device = DEVICE) -> torch.Tensor:
    x = torch.linspace(0, width - 1, width, device=device, dtype=DTYPE)
    y = torch.linspace(0, height - 1, height, device=device, dtype=DTYPE)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid

def custom_match_nn(descriptors1: torch.Tensor, descriptors2: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    descriptors1 = descriptors1.cpu()
    descriptors2 = descriptors2.cpu()
    dists = torch.cdist(descriptors1[0], descriptors2[0], p=2)
    dists1, idx1 = torch.min(dists, dim=1)
    dists2, idx2 = torch.min(dists, dim=0)
    matches = []
    for i in range(dists1.shape[0]):
        if idx2[idx1[i]] == i and dists1[i] < threshold:
            matches.append([i, idx1[i]])
    return torch.tensor(matches, dtype=torch.long) if matches else torch.empty((0, 2), dtype=torch.long)

class DataPreprocessing:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))

    def normalize_2d(self, image: np.ndarray) -> torch.Tensor:
        if image.shape[-1] != 3:
            raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = self.clahe.apply(image_gray)
        image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)
        image_gray = cv2.Laplacian(image_gray, cv2.CV_8U, ksize=3)
        image_gray = cv2.convertScaleAbs(image_gray)
        image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, self.target_size)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.tensor(image, dtype=DTYPE, device=DEVICE).permute(2, 0, 1)
        return tensor

class DataAugmentation:
    def __init__(self):
        self.transform = K.augmentation.AugmentationSequential(
            K.augmentation.RandomRotation(degrees=30.0),
            K.augmentation.RandomAffine(degrees=15.0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            K.augmentation.RandomHorizontalFlip(p=0.5),
            data_keys=["input"]
        ).to(DEVICE)

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image.unsqueeze(0))[0].squeeze(0)

class GeometricTransformationExtraction:
    def __init__(self):
        self.local_feature = K.feature.LocalFeature(
            detector=K.feature.DISK().to(DEVICE),
            descriptor=K.feature.SIFTDescriptor(patch_size=32).to(DEVICE)
        ).to(DEVICE)
        self.orb = cv2.ORB_create(nfeatures=20000, scaleFactor=1.2, nlevels=8, edgeThreshold=5)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.01)

    def extract_2d_transform(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        image1 = image1.to(device=DEVICE, dtype=DTYPE)
        image2 = image2.to(device=DEVICE, dtype=DTYPE)
        rgb1 = image1.unsqueeze(0)
        rgb2 = image2.unsqueeze(0)
        
        img1_np = image1.detach().permute(1, 2, 0).cpu().numpy()
        img2_np = image2.detach().permute(1, 2, 0).cpu().numpy()
        img1_var, img2_var = img1_np.std(), img2_np.std()
        if img1_var < 0.015 or img2_var < 0.015:
            return torch.zeros(3, device=DEVICE, dtype=DTYPE)
        
        try:
            feats1 = self.local_feature(rgb1.to("cpu")).to(DEVICE)
            feats2 = self.local_feature(rgb2.to("cpu")).to(DEVICE)
            if not (isinstance(feats1, dict) and isinstance(feats2, dict) and
                    "keypoints" in feats1 and "descriptors" in feats1 and
                    "keypoints" in feats2 and "descriptors" in feats2):
                src_pts, dst_pts = self._orb_fallback(img1_np, img2_np)
            else:
                keypoints1, descriptors1 = feats1["keypoints"], feats1["descriptors"]
                keypoints2, descriptors2 = feats2["keypoints"], feats2["descriptors"]
        except:
            src_pts, dst_pts = self._orb_fallback(img1_np, img2_np)
            if len(src_pts) == 0:
                return torch.zeros(3, device=DEVICE, dtype=DTYPE)
            return self._process_keypoints(src_pts, dst_pts)
        
        if (descriptors1 is None or descriptors2 is None or
            descriptors1.shape[0] != 1 or descriptors2.shape[0] != 1 or
            descriptors1.shape[2] != 128 or descriptors2.shape[2] != 128 or
            descriptors1.shape[1] == 0 or descriptors2.shape[1] == 0):
            src_pts, dst_pts = self._orb_fallback(img1_np, img2_np)
            if len(src_pts) == 0:
                return torch.zeros(3, device=DEVICE, dtype=DTYPE)
        else:
            max_kpts = 1000
            if keypoints1.shape[1] > max_kpts:
                indices = torch.randperm(keypoints1.shape[1], device=DEVICE)[:max_kpts]
                keypoints1 = keypoints1[:, indices]
                descriptors1 = descriptors1[:, indices]
            if keypoints2.shape[1] > max_kpts:
                indices = torch.randperm(keypoints2.shape[1], device=DEVICE)[:max_kpts]
                keypoints2 = keypoints2[:, indices]
                descriptors2 = descriptors2[:, indices]
            
            matches = custom_match_nn(descriptors1, descriptors2, threshold=0.9)
            if matches.shape[0] == 0:
                src_pts, dst_pts = self._orb_fallback(img1_np, img2_np)
                if len(src_pts) == 0:
                    return torch.zeros(3, device=DEVICE, dtype=DTYPE)
            else:
                src_idx, dst_idx = matches[:, 0], matches[:, 1]
                src_pts = keypoints1[0][src_idx].to(DEVICE)
                dst_pts = keypoints2[0][dst_idx].to(DEVICE)
        
        return self._process_keypoints(src_pts, dst_pts)

    def _process_keypoints(self, src_pts: torch.Tensor, dst_pts: torch.Tensor) -> torch.Tensor:
        if len(src_pts) < 4:
            return torch.zeros(3, device=DEVICE, dtype=DTYPE)
        
        homography_result = cv2.findHomography(src_pts.cpu().numpy(), dst_pts.cpu().numpy(), cv2.RANSAC, 5.0)
        if homography_result is None:
            return torch.zeros(3, device=DEVICE, dtype=DTYPE)
        
        if isinstance(homography_result, np.ndarray):
            H = homography_result
        elif isinstance(homography_result, tuple) and len(homography_result) >= 1:
            H = homography_result[0]
            if H is None:
                return torch.zeros(3, device=DEVICE, dtype=DTYPE)
        else:
            return torch.zeros(3, device=DEVICE, dtype=DTYPE)
        
        H = torch.tensor(H, dtype=DTYPE, device=DEVICE)
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if abs(det) < 1e-8:
            return torch.zeros(3, device=DEVICE, dtype=DTYPE)
        theta = torch.atan2(H[1, 0], H[0, 0] + 1e-8)
        tx, ty = H[0, 2], H[1, 2]
        return torch.tensor([theta, tx, ty], dtype=DTYPE, device=DEVICE)

    def _orb_fallback(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        img1_gray = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        img1_gray = clahe.apply(img1_gray)
        img2_gray = clahe.apply(img2_gray)
        kp1, des1 = self.orb.detectAndCompute(img1_gray, None)
        kp2, des2 = self.orb.detectAndCompute(img2_gray, None)
        if des1 is None or des2 is None:
            return self._sift_fallback(img1, img2)
        matches = self.bf_matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:500]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        return (torch.tensor(src_pts, device=DEVICE, dtype=DTYPE),
                torch.tensor(dst_pts, device=DEVICE, dtype=DTYPE))

    def _sift_fallback(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        img1_gray = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        img1_gray = clahe.apply(img1_gray)
        img2_gray = clahe.apply(img2_gray)
        kp1, des1 = self.sift.detectAndCompute(img1_gray, None)
        kp2, des2 = self.sift.detectAndCompute(img2_gray, None)
        if des1 is None or des2 is None:
            return torch.tensor([], device=DEVICE), torch.tensor([], device=DEVICE)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:500]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        return (torch.tensor(src_pts, device=DEVICE, dtype=DTYPE),
                torch.tensor(dst_pts, device=DEVICE, dtype=DTYPE))

class LieGroupRepresentation:
    @staticmethod
    def se2_to_matrix(params: torch.Tensor) -> torch.Tensor:
        params = params.to(dtype=DTYPE, device=DEVICE)
        theta, tx, ty = params.unbind(-1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        batch_size = theta.shape[0] if theta.dim() > 0 else 1
        matrix = torch.zeros(batch_size, 3, 3, dtype=DTYPE, device=DEVICE)
        matrix[:, 0, 0] = cos_theta
        matrix[:, 0, 1] = -sin_theta
        matrix[:, 1, 0] = sin_theta
        matrix[:, 1, 1] = cos_theta
        matrix[:, 0, 2] = tx
        matrix[:, 1, 2] = ty
        matrix[:, 2, 2] = 1
        return matrix

    @staticmethod
    def lie_algebra_to_params(lie_alg: torch.Tensor) -> torch.Tensor:
        return lie_alg.to(dtype=DTYPE, device=DEVICE)

class LieGroupConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2).to(DEVICE)
        self.lie_basis = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=DTYPE, device=DEVICE)

    def forward(self, x: torch.Tensor, lie_params: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=DTYPE, device=DEVICE)
        lie_params = lie_params.to(dtype=DTYPE, device=DEVICE)
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]
        grid = create_meshgrid(height, width, device=DEVICE)
        transform_matrices = LieGroupRepresentation.se2_to_matrix(lie_params)
        grid = grid.expand(batch_size, -1, -1, -1)
        grid_transformed = K.geometry.transform_points(transform_matrices, grid.view(batch_size, -1, 2))
        grid_transformed = grid_transformed.view(batch_size, height, width, 2)
        x_transformed = K.geometry.warp_perspective(x, transform_matrices, (height, width))
        return self.conv(x_transformed)

class LieGroupAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads).to(DEVICE)
        self.lie_projection = nn.Linear(3, dim).to(DEVICE)

    def forward(self, x: torch.Tensor, lie_params: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=DTYPE, device=DEVICE)
        lie_params = lie_params.to(dtype=DTYPE, device=DEVICE)
        lie_embed = self.lie_projection(lie_params)
        x = x + lie_embed.unsqueeze(1)
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int):
        super().__init__()
        self.lie_attention = LieGroupAttention(dim, heads).to(DEVICE)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        ).to(DEVICE)
        self.norm1 = nn.LayerNorm(dim).to(DEVICE)
        self.norm2 = nn.LayerNorm(dim).to(DEVICE)

    def forward(self, x: torch.Tensor, lie_params: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=DTYPE, device=DEVICE)
        lie_params = lie_params.to(dtype=DTYPE, device=DEVICE)
        x = self.norm1(x + self.lie_attention(x, lie_params))
        x = self.norm2(x + self.ffn(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 128, heads: int = 4, ff_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.lie_conv = LieGroupConvLayer(in_channels, dim, kernel_size=3).to(DEVICE)
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(dim, heads, ff_dim) for _ in range(num_layers)
        ]).to(DEVICE)
        self.to_embedding = nn.Linear(dim * 224 * 224, dim).to(DEVICE)
        self.fc = nn.Linear(dim, 10).to(DEVICE)

    def forward(self, x: torch.Tensor, lie_params: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=DTYPE, device=DEVICE)
        lie_params = lie_params.to(dtype=DTYPE, device=DEVICE)
        x = self.lie_conv(x, lie_params)
        x = x.flatten(1)
        x = self.to_embedding(x).unsqueeze(1)
        for layer in self.encoder:
            x = layer(x, lie_params)
        output = self.fc(x.squeeze(1))
        return output

class PreTraining:
    def __init__(self, model: nn.Module, train_dataset: List[torch.Tensor], val_dataset: List[torch.Tensor], lr: float = 1e-3):
        self.model = model.to(DEVICE)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.transform_extract = GeometricTransformationExtraction()
        self.lie_rep = LieGroupRepresentation()
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    async def train_step(self, image1: torch.Tensor, image2: torch.Tensor, label: torch.Tensor):
        image1 = image1.to(dtype=DTYPE, device=DEVICE)
        image2 = image2.to(dtype=DTYPE, device=DEVICE)
        label = label.to(device=DEVICE)
        lie_params = torch.zeros(3, device=DEVICE, dtype=DTYPE)
        if self.transform_extract is not None:
            lie_params = self.transform_extract.extract_2d_transform(image1, image2)
        lie_params = self.lie_rep.lie_algebra_to_params(lie_params.unsqueeze(0))
        output = self.model(image1.unsqueeze(0), lie_params)
        loss = self.criterion(output, label.unsqueeze(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        _, predicted = torch.max(output, 1)
        detected_object = OBJECT_NAMES.get(predicted.item(), "Unknown")
        data = {
            "image1": image1.detach().cpu().numpy(),
            "image2": image2.detach().cpu().numpy(),
            "lie_params": lie_params.detach().cpu().numpy().tolist(),
            "output": output.detach().cpu().numpy().tolist(),
            "loss": float(loss.item()),
            "label": label.detach().cpu().numpy().tolist(),
            "detected_object": detected_object
        }
        return loss.item(), data

class Training:
    def __init__(self, pre_training: PreTraining):
        self.pre_training = pre_training
        self.optimizer = pre_training.optimizer
        self.criterion = pre_training.criterion

    async def train(self, num_epochs: int):
        total_loss = 0
        for epoch in range(num_epochs):
            for image1, image2, label in self.pre_training.train_dataset:
                loss, _ = await self.pre_training.train_step(image1, image2, label)
                total_loss += loss
                await asyncio.sleep(0.01)
            if len(self.pre_training.train_dataset) > 0:
                total_loss = 0

class Optimization:
    def __init__(self, training: Training):
        self.training = training

    def adjust_parameters(self):
        pass

class Evaluation:
    def __init__(self, model: nn.Module, val_dataset: List[torch.Tensor]):
        self.model = model.to(DEVICE)
        self.val_dataset = val_dataset
        self.transform_extract = GeometricTransformationExtraction()
        self.lie_rep = LieGroupRepresentation()

    def evaluate(self) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for image1, image2, label in self.val_dataset:
                image1 = image1.to(dtype=DTYPE, device=DEVICE)
                image2 = image2.to(dtype=DTYPE, device=DEVICE)
                label = label.to(device=DEVICE)
                lie_params = torch.zeros(3, device=DEVICE, dtype=DTYPE)
                if self.transform_extract is not None:
                    lie_params = self.transform_extract.extract_2d_transform(image1, image2)
                lie_params = self.lie_rep.lie_algebra_to_params(lie_params.unsqueeze(0))
                output = self.model(image1.unsqueeze(0), lie_params)
                _, predicted = torch.max(output, 1)
                detected_object = OBJECT_NAMES.get(predicted.item(), "Unknown")
                total += 1
                correct += (predicted == label).item()
                DATA_STORE["pipeline_data"].append({
                    "image1": image1.detach().cpu().numpy().tolist(),
                    "image2": image2.detach().cpu().numpy().tolist(),
                    "lie_params": lie_params.detach().cpu().numpy().tolist(),
                    "output": output.detach().cpu().numpy().tolist(),
                    "predicted": predicted.detach().cpu().numpy().tolist(),
                    "label": label.detach().cpu().numpy().tolist(),
                    "detected_object": detected_object
                })
        return correct / total if total > 0 else 0.0

class Hyperparameter:
    def __init__(self, training: Training):
        self.training = training

    def tune(self, lr_values: List[float]):
        best_lr = lr_values[0]
        best_acc = 0
        eval_module = Evaluation(self.training.pre_training.model, self.training.pre_training.val_dataset)
        for lr in lr_values:
            self.training.pre_training.optimizer = torch.optim.Adam(
                self.training.pre_training.model.parameters(), lr=lr
            )
            asyncio.run(self.training.train(1))
            acc = eval_module.evaluate()
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        return best_lr

def display_raw_frame(frame: np.ndarray, frame_count: int, data: dict = None):
    display_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 0)  # Green text
    thickness = 1
    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 20), font, font_scale, color, thickness)
    
    if data:
        y = 40
        line_spacing = 20
        cv2.putText(display_frame, f"SO(2) theta: {data['lie_params'][0][0]:.4f}",
                    (10, y), font, font_scale, color, thickness)
        cv2.putText(display_frame, f"SE(2) params (tx, ty): ({data['lie_params'][0][1]:.2f}, {data['lie_params'][0][2]:.2f})",
                    (10, y + line_spacing), font, font_scale, color, thickness)
        cv2.putText(display_frame, f"Model output: {[f'{x:.2f}' for x in data['output'][0][:3]]}...",
                    (10, y + 2 * line_spacing), font, font_scale, color, thickness)
        cv2.putText(display_frame, f"Loss: {data['loss']:.4f}",
                    (10, y + 3 * line_spacing), font, font_scale, color, thickness)
        cv2.putText(display_frame, f"Label: {data['label']}",
                    (10, y + 4 * line_spacing), font, font_scale, color, thickness)
        cv2.putText(display_frame, f"Detected Object: {data['detected_object']}",
                    (10, y + 5 * line_spacing), font, font_scale, color, thickness)
    
    cv2.imshow("Raw Frame", display_frame)
    cv2.waitKey(1)

def display_or_save_data(data: dict, mode: str = "json", save_json: bool = False):
    print(f"\nData Entry:")
    print(f"  SO(2) theta: {data['lie_params'][0][0]:.4f}")
    print(f"  SE(2) params (tx, ty): ({data['lie_params'][0][1]:.2f}, {data['lie_params'][0][2]:.2f})")
    print(f"  Model output: {data['output']}")
    print(f"  Loss: {data['loss']}")
    print(f"  Label: {data['label']}")
    print(f"  Detected Object: {data['detected_object']}")
    
    if mode == "json" and save_json:
        DATA_STORE["pipeline_data"].append({
            "image1": data["image1"].tolist(),
            "image2": data["image2"].tolist(),
            "lie_params": data["lie_params"],
            "output": data["output"],
            "loss": data["loss"],
            "label": data["label"],
            "detected_object": data["detected_object"]
        })

def display_processed_images(image1: torch.Tensor, image2: torch.Tensor, data: dict):
    img1 = image1.detach().cpu().numpy().transpose(1, 2, 0)
    img2 = image2.detach().cpu().numpy().transpose(1, 2, 0)
    img1 = (img1 * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.0
    img2 = (img2 * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.0
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    img1 = np.ascontiguousarray(img1)
    img2 = np.ascontiguousarray(img2)
    cv2.putText(img1, f"Preprocessed Lie: {data['lie_params'][0]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img1, f"Loss: {data['loss']:.4f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img1, f"Output: {[f'{x:.2f}' for x in data['output'][0][:3]]}...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img2, f"Label: {data['label']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img2, f"Detected Object: {data['detected_object']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Preprocessed Image 1", img1)
    cv2.imshow("Preprocessed Image 2", img2)
    cv2.waitKey(1)

async def main(display_mode: str = "json", frames_dir: str = "frames", unlimited: bool = False, save_json: bool = False, save_image: bool = False):
    if display_mode not in ["json", "gui", "realtime"]:
        return
    
    if save_image:
        os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    FPS = 10
    resource_manager = DynamicResourceManager(target_fps=FPS)
    frame_count = 0
    frame_paths = []
    prev_frame = None
    
    preprocess = DataPreprocessing()
    augment = DataAugmentation()
    model = TransformerModel().to(DEVICE)
    train_dataset = []
    val_dataset = []
    pre_training = PreTraining(model, train_dataset, val_dataset)
    training = Training(pre_training)
    optimization = Optimization(training)
    hyperparam = Hyperparameter(training)

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            if display_mode == "realtime" and prev_frame is not None:
                frame1 = prev_frame
                frame2 = frame
                frame1 = preprocess.normalize_2d(frame1)
                frame2 = preprocess.normalize_2d(frame2)
                frame1_aug = augment.apply(frame1)
                frame2_aug = augment.apply(frame2)
                
                frame1_var = frame1.detach().cpu().numpy().std()
                frame2_var = frame2.detach().cpu().numpy().std()
                if frame1_var < 0.015 or frame2_var < 0.015:
                    display_raw_frame(frame, frame_count)
                    frame_count += 1
                    prev_frame = frame
                    frame_time = time.time() - start_time
                    dynamic_delay = resource_manager.update(frame_time)
                    await asyncio.sleep(max(1.0 / FPS - frame_time, 0) + dynamic_delay)
                    continue
                
                label = torch.tensor(0, dtype=torch.long, device=DEVICE)
                train_dataset.append((frame1_aug, frame2_aug, label))
                loss, data = await pre_training.train_step(frame1_aug, frame2_aug, label)
                display_raw_frame(frame, frame_count, data)
                display_or_save_data(data, mode=display_mode, save_json=save_json)
            else:
                display_raw_frame(frame, frame_count)
            
            if save_image:
                frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            
            if prev_frame is not None:
                frame1 = prev_frame
                frame2 = frame
                frame1 = preprocess.normalize_2d(frame1)
                frame2 = preprocess.normalize_2d(frame2)
                frame1_aug = augment.apply(frame1)
                frame2_aug = augment.apply(frame2)
                
                frame1_var = frame1.detach().cpu().numpy().std()
                frame2_var = frame2.detach().cpu().numpy().std()
                if frame1_var < 0.015 or frame2_var < 0.015:
                    frame_count += 1
                    prev_frame = frame
                    frame_time = time.time() - start_time
                    dynamic_delay = resource_manager.update(frame_time)
                    await asyncio.sleep(max(1.0 / FPS - frame_time, 0) + dynamic_delay)
                    continue
                
                label = torch.tensor(0, dtype=torch.long, device=DEVICE)
                train_dataset.append((frame1_aug, frame2_aug, label))
                loss, data = await pre_training.train_step(frame1_aug, frame2_aug, label)
                display_or_save_data(data, mode=display_mode, save_json=save_json)
                if display_mode == "gui":
                    display_processed_images(frame1_aug, frame2_aug, data)
            
            frame_count += 1
            prev_frame = frame
            frame_time = time.time() - start_time
            dynamic_delay = resource_manager.update(frame_time)
            await asyncio.sleep(max(1.0 / FPS - frame_time, 0) + dynamic_delay)
            
            if not unlimited and frame_count >= FPS:
                break
    
    finally:
        if 'cap' in locals():
            cap.release()
        if display_mode == "json" and save_json and platform.system() != "Emscripten":
            with open("lepaute_data.json", "w") as f:
                json.dump(DATA_STORE["pipeline_data"], f)
        if display_mode in ["gui", "realtime"]:
            cv2.destroyAllWindows()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
