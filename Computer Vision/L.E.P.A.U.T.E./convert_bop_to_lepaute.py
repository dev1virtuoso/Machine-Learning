import argparse
import json
import logging
import shutil
import uuid
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] BOP_CONVERTER: %(message)s"
)
logger = logging.getLogger("BOP_CONVERTER")

def calculate_relative_pose(R_A: np.ndarray, t_A: np.ndarray, R_B: np.ndarray, t_B: np.ndarray) -> List[float]:
    R_A_T = R_A.T
    R_rel = R_B @ R_A_T
    t_rel = t_B - (R_rel @ t_A)
    
    trace_R = np.trace(R_rel)
    cos_theta = np.clip((trace_R - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(cos_theta)
    
    phi_raw = np.array([
        R_rel[2, 1] - R_rel[1, 2],
        R_rel[0, 2] - R_rel[2, 0],
        R_rel[1, 0] - R_rel[0, 1]
    ])
    
    if theta > 1e-4:
        phi = (theta / (2.0 * np.sin(theta))) * phi_raw
        K = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
        I = np.eye(3)
        half_theta = theta / 2.0
        V_inv = I - 0.5 * K + (1.0 - (theta * np.cos(half_theta)) / (2.0 * np.sin(half_theta))) * (K @ K) / (theta ** 2)
        rho = V_inv @ t_rel.flatten()
    else:
        phi = 0.5 * phi_raw
        K = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
        I = np.eye(3)
        V_inv = I - 0.5 * K + (1.0 / 12.0) * (K @ K)
        rho = V_inv @ t_rel.flatten()

    return np.concatenate([rho, phi]).astype(float).tolist()

def process_scene_pair(
    frame_a_id: str, 
    frame_b_id: str, 
    scene_path: Path, 
    frames_dir: Path, 
    scene_gt: Dict, 
    target_obj_ids: Optional[List[int]],
    translation_scale: float
) -> List[Dict[str, Any]]:
    
    gt_a = scene_gt.get(frame_a_id)
    gt_b = scene_gt.get(frame_b_id)
    
    if gt_a is None or gt_b is None:
        return []

    obj_dict_a = {obj["obj_id"]: obj for obj in gt_a}
    obj_dict_b = {obj["obj_id"]: obj for obj in gt_b}
    common_objs = set(obj_dict_a.keys()).intersection(set(obj_dict_b.keys()))

    if target_obj_ids:
        common_objs = common_objs.intersection(set(target_obj_ids))

    if not common_objs:
        return []

    rgb_dir = scene_path / "rgb"
    rgb_a_path = next(rgb_dir.glob(f"{int(frame_a_id):06d}.*"), None)
    rgb_b_path = next(rgb_dir.glob(f"{int(frame_b_id):06d}.*"), None)

    if not rgb_a_path or not rgb_b_path:
        return []

    uuid_a, uuid_b = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
    out_a, out_b = f"frame_{uuid_a}{rgb_a_path.suffix}", f"frame_{uuid_b}{rgb_b_path.suffix}"
    
    shutil.copy2(rgb_a_path, frames_dir / out_a)
    shutil.copy2(rgb_b_path, frames_dir / out_b)

    records = []
    for obj_id in common_objs:
        R_A = np.array(obj_dict_a[obj_id]["cam_R_m2c"]).reshape(3, 3)
        t_A = np.array(obj_dict_a[obj_id]["cam_t_m2c"]).reshape(3, 1) / translation_scale
        R_B = np.array(obj_dict_b[obj_id]["cam_R_m2c"]).reshape(3, 3)
        t_B = np.array(obj_dict_b[obj_id]["cam_t_m2c"]).reshape(3, 1) / translation_scale

        records.append({
            "timestamp": 0.0,
            "lie_params": calculate_relative_pose(R_A, t_A, R_B, t_B),
            "loss": 0.0,
            "detected_object": f"obj_{obj_id:06d}",
            "confidence": 1.0,
            "frame_a": out_a,
            "frame_b": out_b
        })
    return records

def process_bop_scene(scene_path: Path, frames_dir: Path, stride: int, target_obj_ids: Optional[List[int]], translation_scale: float) -> List[Dict]:
    scene_gt_path = scene_path / "scene_gt.json"
    if not scene_gt_path.exists():
        return []

    with open(scene_gt_path, "r") as f:
        scene_gt = json.load(f)

    frame_ids = sorted(scene_gt.keys(), key=int)
    scene_records = []
    for i in range(0, len(frame_ids) - stride, stride):
        scene_records.extend(process_scene_pair(frame_ids[i], frame_ids[i + stride], scene_path, frames_dir, scene_gt, target_obj_ids, translation_scale))
    return scene_records

def run_conversion(bop_dir: str, output_dir: str, stride: int, target_obj_ids: Optional[List[int]], workers: int, translation_scale: float):
    bop_path, out_path = Path(bop_dir), Path(output_dir)
    frames_dir = out_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = [d for d in bop_path.iterdir() if d.is_dir() and d.name.isdigit()]
    all_records = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_bop_scene, s, frames_dir, stride, target_obj_ids, translation_scale): s for s in scene_dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Scenes"):
            all_records.extend(future.result())

    with open(out_path / "lepaute_data.json", "w") as f:
        json.dump(all_records, f, indent=4)
    logger.info(f"Complete. Generated {len(all_records)} records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bop_dir", required=True)
    parser.add_argument("--output_dir", default="./lepaute_dataset")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--obj_ids", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1000.0)
    args = parser.parse_args()
    run_conversion(args.bop_dir, args.output_dir, args.stride, args.obj_ids, args.workers, args.scale)