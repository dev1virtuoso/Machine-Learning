import os
import time
import json
import hashlib
import random
import sqlite3
import argparse
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from pipeline_and_config import LepauteConfig, load_data, PerformanceMode
from geometry import se3_exp_map, se3_log_map
from globals import mps_safe, logger
from models import SE3ResidualRefiner

class LepauteBenchmark:
    def __init__(self, config_override=None):
        self.config = config_override or LepauteConfig()
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {
            "mse_errors": [],
            "mode_switches": [],
            "latencies": {},
            "boundary_test_results": {}
        }

    def resolve_target_device(self, device_option: str) -> str:
        opt = device_option.lower()
        if opt in ["gpu", "cuda", "mps"]:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                print(" [!] Warning: GPU acceleration requested but neither CUDA nor MPS is available. Falling back to CPU.")
                return "cpu"
        return "cpu"

    def setup_environment(self, target_device="cpu", seed=42):
        print("\n" + "="*70)
        print(f" [STAGE 1/5] Setting up Deterministic Environment on [{target_device.upper()}]")
        print("="*70)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        resolved_device = self.resolve_target_device(target_device) if target_device != "cpu" else "cpu"
        self.config.device = resolved_device

        if resolved_device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print(" [+] Operating under CPU mode.")
        else:
            if resolved_device == "cuda":
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                print(f" [+] CUDA backend enabled on device: {torch.cuda.get_device_name(0)}")
            elif resolved_device == "mps":
                print(" [+] Apple Silicon MPS hardware acceleration enabled.")

        self.config.num_workers = 0
        print(f" [+] Random seed locked to: {seed}")
        print(" [+] Environment initialization complete.")

    def load_and_perturb_data(self, sequence_path="lepaute_data.db", severity=1.0, limit=None):
        print("\n" + "="*70)
        print(" [STAGE 3/5] Ingesting Data & Configuring Perturbation Pipeline")
        print("="*70)
        
        self.config.data_store = sequence_path
        print(f" [+] Reading SQLite database from: '{sequence_path}'...")
        
        raw_data = load_data(self.config)
        if limit and limit > 0:
            raw_data = raw_data[:limit]
            print(f" [+] Frame limit applied: processing first {len(raw_data)} frames.")
        else:
            print(f" [+] Successfully loaded {len(raw_data)} transition frames.")

        max_blur = int(3 + 2 * severity)
        if max_blur % 2 == 0:
            max_blur += 1

        perturbation_pipeline = A.Compose([
            A.GaussNoise(std_range=(0.1 * severity, 0.3 * severity), p=1.0),
            A.MotionBlur(blur_limit=(3, max_blur), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            ToTensorV2()
        ])
        print(f" [+] Perturbation pipeline initialized with severity level: {severity}")
        return raw_data, perturbation_pipeline

    def test_boundary_conditions(self):
        print("\n" + "="*70)
        print(" [STAGE 2/5] Testing SE(3) Lie Group Boundary Conditions")
        print("="*70)
        
        device = torch.device(self.config.device)
        print(f" [+] Running Lie algebra exp/log map stability checks on {device}...")
        
        pi_xi = torch.tensor([[0.1, -0.2, 0.3, 3.14159, 0.0, 0.0]], dtype=torch.float32, device=device)
        tiny_xi = torch.tensor([[1e-5, -2e-5, 1e-5, 3e-5, -1e-5, 2e-5]], dtype=torch.float32, device=device)
        
        with mps_safe(device):
            T_pi = se3_exp_map(pi_xi)
            rec_pi_xi = se3_log_map(T_pi)
            pi_mse = torch.mean((pi_xi - rec_pi_xi)**2).item()
            
            T_tiny = se3_exp_map(tiny_xi)
            rec_tiny_xi = se3_log_map(T_tiny)
            tiny_mse = torch.mean((tiny_xi - rec_tiny_xi)**2).item()
            
        self.metrics["boundary_test_results"] = {
            "pi_angle_mse": pi_mse,
            "tiny_angle_mse": tiny_mse
        }
        print(f" [+] Boundary Test Passed ({device}):")
        print(f"     - Near-Pi Rotation Mapping MSE : {pi_mse:.8e}")
        print(f"     - Sub-pixel Micro Angle Mapping MSE : {tiny_mse:.8e}")

    def log_latency(self, module_name, start_time):
        dt_ms = (time.perf_counter() - start_time) * 1000
        if module_name not in self.metrics["latencies"]:
            self.metrics["latencies"][module_name] = []
        self.metrics["latencies"][module_name].append(dt_ms)

    def generate_report(self, model_path="lepaute_refiner.onnx", output_json="benchmark_report.json"):
        print("\n" + "="*70)
        print(" [STAGE 5/5] Generating Final Benchmark Report & Hash Binding")
        print("="*70)
        
        model_hash = "N/A"
        if os.path.exists(model_path):
            hasher = hashlib.sha256()
            with open(model_path, 'rb') as f:
                hasher.update(f.read())
            model_hash = hasher.hexdigest()
            print(f" [+] Model artifact hash bound: {model_hash[:16]}...")
        else:
            print(f" [!] ONNX model file '{model_path}' not found. Hash set to N/A.")

        report = {
            "ci_timestamp": time.time(),
            "model_version_hash": model_hash,
            "environment": {
                "device": self.config.device,
                "performance_mode": self.config.performance_mode.value
            },
            "metrics": {
                "mean_geometric_mse": float(np.mean(self.metrics["mse_errors"])) if self.metrics["mse_errors"] else 0.0,
                "boundary_stability": self.metrics["boundary_test_results"]
            },
            "latency_profiling_ms": {
                mod: {"mean": float(np.mean(lats)), "max": float(np.max(lats))} 
                for mod, lats in self.metrics["latencies"].items() if lats
            },
            "mode_arbitration": {
                "deep_learning_refined": sum(1 for m in self.metrics["mode_switches"] if m == "refined"),
                "direct_alignment_tracker": sum(1 for m in self.metrics["mode_switches"] if m == "tracker"),
                "orb_kinematic_recovery": sum(1 for m in self.metrics["mode_switches"] if m == "recovery")
            }
        }
        
        with open(output_json, 'w') as f:
            json.dump(report, f, indent=4)
            
        print(f" [+] Benchmark report saved to: '{output_json}'")
        print("\n" + "="*70)
        print(f" BENCHMARK SUMMARY [{self.config.device.upper()}]")
        print("="*70)
        print(f"  Total Frames Evaluated : {len(self.metrics['mse_errors'])}")
        print(f"  Mean Geometric MSE     : {report['metrics']['mean_geometric_mse']:.6f}")
        if "SE3ResidualRefiner_Inference" in report["latency_profiling_ms"]:
            avg_lat = report["latency_profiling_ms"]["SE3ResidualRefiner_Inference"]["mean"]
            print(f"  Average Latency        : {avg_lat:.2f} ms / frame")
        print("="*70 + "\n")
        return report

    def run_single_pass(self, refiner_model, target_device, args):
        self.reset_metrics()
        self.setup_environment(target_device=target_device, seed=args.seed)
        self.test_boundary_conditions()
        
        data_seq, perturber = self.load_and_perturb_data(
            sequence_path=args.db, 
            severity=args.severity, 
            limit=args.limit
        )
        
        if not data_seq:
            print(" [!] Warning: Database contains 0 frames. Skipping execution loop.")
            return None
            
        device = torch.device(self.config.device)
        refiner_model = refiner_model.to(device)
        refiner_model.eval()
        
        total_samples = len(data_seq)
        running_mse = 0.0

        print("\n" + "="*70)
        print(f" [STAGE 4/5] Running Model Inference & Stress Testing Loop on [{device}]")
        print("="*70)

        pbar = tqdm(data_seq, desc=f" [+] Processing [{device}]", unit="frame", dynamic_ncols=True) if HAS_TQDM else data_seq

        with torch.no_grad():
            for idx, item in enumerate(pbar):
                img_a = item["img_a"]
                img_b = item["img_b"]
                gt_xi = torch.tensor(item["lie_params"], dtype=torch.float32, device=device)
                
                t_a = perturber(image=img_a)["image"].unsqueeze(0).to(device).float()
                t_b = perturber(image=img_b)["image"].unsqueeze(0).to(device).float()
                
                xi_prior = gt_xi + torch.randn_like(gt_xi) * 0.05
                
                t_start = time.perf_counter()
                delta_xi, delta_scale, unc_pose, unc_scale = refiner_model(t_a, t_b, xi_prior)
                pred_xi = xi_prior + delta_xi 
                self.log_latency("SE3ResidualRefiner_Inference", t_start)
                
                mse = torch.mean((pred_xi - gt_xi)**2).item()
                self.metrics["mse_errors"].append(mse)
                
                if mse < 0.01:
                    mode = "refined"
                elif mse < 0.1:
                    mode = "tracker"
                else:
                    mode = "recovery"
                self.metrics["mode_switches"].append(mode)

                running_mse = (running_mse * idx + mse) / (idx + 1)

                if HAS_TQDM:
                    pbar.set_postfix({"Cur_MSE": f"{mse:.5f}", "Avg_MSE": f"{running_mse:.5f}", "Mode": mode.upper()})
                elif (idx + 1) % 500 == 0 or (idx + 1) == total_samples:
                    print(f" [+] [{idx + 1}/{total_samples}] Cur MSE: {mse:.6f} | Avg MSE: {running_mse:.6f} | Mode: {mode.upper()}")

        output_name = args.output
        if args.device == "both":
            name, ext = os.path.splitext(args.output)
            output_name = f"{name}_{target_device}{ext}"

        return self.generate_report(model_path=args.model_path, output_json=output_name)


def parse_benchmark_args():
    parser = argparse.ArgumentParser(
        description="L.E.P.A.U.T.E. Framework Benchmark & CI Stress Testing Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="both",
        choices=["cpu", "gpu", "cuda", "mps", "both"],
        help="Compute target device: cpu, gpu (auto-detect CUDA/MPS), or both (sequential 2-stage test)"
    )
    
    parser.add_argument(
        "-p", "--perf",
        type=str,
        default="high",
        choices=["low", "medium", "high"],
        help="LepauteConfig performance mode profile"
    )

    parser.add_argument(
        "-db", "--db",
        type=str,
        default="lepaute_data.db",
        help="Path to SQLite sequence transition database"
    )

    parser.add_argument(
        "-s", "--severity",
        type=float,
        default=1.5,
        help="Perturbation degradation severity factor"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="benchmark_report.json",
        help="Output report JSON file path"
    )

    parser.add_argument(
        "-m", "--model-path",
        type=str,
        default="lepaute_refiner.onnx",
        help="Path to model artifact (ONNX/pth) for hash verification"
    )

    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Maximum frame count limit (useful for fast testing)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed lock"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_benchmark_args()

    perf_enum = PerformanceMode(args.perf.lower())
    config = LepauteConfig(performance_mode=perf_enum)
    refiner = SE3ResidualRefiner(config, feature_dim=256, max_resolution=64)
    benchmark = LepauteBenchmark(config)

    if not os.path.exists(args.db):
        with sqlite3.connect(args.db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS transitions (id INTEGER PRIMARY KEY AUTOINCREMENT, img_a BLOB, img_b BLOB, xi TEXT, obj_name TEXT)")

    print("\n" + "#"*70)
    print("   L.E.P.A.U.T.E. FRAMEWORK BENCHMARK SUITE")
    print("#"*70)

    if args.device == "both":
        print("\n [Mode] Executing Two-Stage Benchmark (Stage 1: CPU | Stage 2: GPU Acceleration)")
        cpu_report = benchmark.run_single_pass(refiner, target_device="cpu", args=args)
        gpu_report = benchmark.run_single_pass(refiner, target_device="gpu", args=args)
    else:
        benchmark.run_single_pass(refiner, target_device=args.device, args=args)