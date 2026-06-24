import os
import time
import logging
import threading
import queue
import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional

from module import (
    LepauteConfig, DisplayMode, CameraIOStream, MonocularDirectTracker,
    SigLIPClassifier, SE3ResidualRefiner, SequenceDataCollector, ManifoldKinematicForecaster,
    se3_exp_map, se3_log_map, compose_poses
)

import argparse
import sys

logger = logging.getLogger("LEPAUTE.Pipeline")

class InferenceWorker(threading.Thread):
    def __init__(self, config: LepauteConfig, classifier: SigLIPClassifier, refiner: SE3ResidualRefiner):
        super().__init__(daemon=True)
        self.config = config
        self.classifier = classifier
        self.refiner = refiner
        self.device = torch.device(config.device)
        
        self.job_queue = queue.Queue(maxsize=5) 
        self.state_lock = threading.Lock()
        
        self.history_buffer: Dict[int, Tuple[Tuple[str, float], np.ndarray, float]] = {}
        self.running = threading.Event()
        self.running.set()

    def enqueue_job(self, frame_id: int, img_ref: np.ndarray, img_cur: np.ndarray, tracker_xi_rel: np.ndarray):
        if not self.job_queue.full():
            self.job_queue.put_nowait((frame_id, img_ref.copy(), img_cur.copy(), tracker_xi_rel.copy()))

    def get_latest_resolved_state(self, current_time: float) -> Optional[Tuple[Tuple[str, float], np.ndarray]]:
        with self.state_lock:
            expired_keys = [k for k, v in self.history_buffer.items() if current_time - v[2] > 5.0]
            for k in expired_keys:
                del self.history_buffer[k]
                
            if not self.history_buffer:
                return None
                
            latest_resolved_id = max(self.history_buffer.keys())
            state = self.history_buffer.pop(latest_resolved_id)
            
            obsolete_keys = [k for k in list(self.history_buffer.keys()) if k < latest_resolved_id]
            for k in obsolete_keys:
                del self.history_buffer[k]
                
            return state[0], state[1]

    def run(self):
        while self.running.is_set():
            try:
                task = self.job_queue.get(timeout=0.1)
                frame_id, img_ref, img_cur, tracker_xi_rel = task
                
                obj_name, conf = self.classifier.predict(img_cur)
                scale_prior_val = self.config.object_scales.get(obj_name, 1.0)
                scale_prior = torch.tensor([scale_prior_val], dtype=torch.float32, device=self.device)
                
                t_ref = torch.from_numpy(img_ref).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                t_cur = torch.from_numpy(img_cur).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                t_xi = torch.from_numpy(tracker_xi_rel).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.refiner(t_ref, t_cur, t_xi, scale_prior)
                    refined_xi_rel = outputs["pose"].squeeze(0).cpu().numpy()
                    
                completion_time = time.time()
                
                with self.state_lock:
                    self.history_buffer[frame_id] = ((obj_name, conf), refined_xi_rel, completion_time)
                        
                self.job_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Inference Subsystem Crash: {e}")
                continue

    def stop(self):
        self.running.clear()
        while not self.job_queue.empty():
            try:
                self.job_queue.get_nowait()
                self.job_queue.task_done()
            except queue.Empty:
                break
        self.join(timeout=2.0)

def run_pipeline(
    config: LepauteConfig,
    display_mode: DisplayMode = DisplayMode.HEADLESS,
    unlimited: bool = False,
    save_json: bool = True,
    mock: bool = False
) -> List[Dict[str, Any]]:
    
    logger.info("Initializing Hardened Monocular LEPAUTE Subsystem.")
    
    stream = CameraIOStream(config=config, mock=mock)
    tracker = MonocularDirectTracker(config=config)
    collector = SequenceDataCollector(config=config)
    if save_json: collector.start()
        
    forecaster = ManifoldKinematicForecaster()
    classifier = SigLIPClassifier(config=config)
    refiner = SE3ResidualRefiner(config=config).to(config.device)

    model_path = "./checkpoints/best_model.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=config.device)
        if 'model_state_dict' in checkpoint:
            refiner.load_compiled_state_dict(checkpoint['model_state_dict'])
        else:
            refiner.load_compiled_state_dict(checkpoint)
            print(f"Loaded trained model: {model_path}")
    else:
        print(f"Model {model_path} not found, using randomly initialized model")
        
    refiner.eval()
    
    worker = InferenceWorker(config=config, classifier=classifier, refiner=refiner)
    worker.start()
        
    success, prev_rgb, prev_meta = stream.read()
    if not success: 
        logger.error("Camera acquisition failed permanently. Terminating.")
        return []

    processed_payloads = []
    max_frames = 50 if not unlimited else 99999
    
    T_global = torch.eye(4, dtype=torch.float32, device=config.device).unsqueeze(0)
    prev_xi = np.zeros(6, dtype=np.float32)
    last_stamp = prev_meta["timestamp"]
    
    current_obj_name = "background"
    
    try:
        for frame_idx in range(max_frames):
            success, frame_rgb, metadata = stream.read()
            if not success: continue
                
            current_stamp = metadata["timestamp"]
            current_id = metadata["frame_id"]
            dt = current_stamp - last_stamp
            
            async_state = worker.get_latest_resolved_state(current_stamp)
            if async_state is not None:
                (obj_name, conf), refined_xi_rel = async_state
                current_obj_name = obj_name 
                has_async = True
            else:
                has_async = False
                
            scale_prior = config.object_scales.get(current_obj_name, 1.0)
            
            tracker_xi_rel, track_score = tracker.track(prev_rgb, frame_rgb, scale_prior)
            
            if track_score > 0.1:
                worker.enqueue_job(current_id, prev_rgb, frame_rgb, tracker_xi_rel)
            
            if has_async:
                best_rel_xi = refined_xi_rel
                mode = "Refined"
            elif track_score >= 0.1:
                best_rel_xi = tracker_xi_rel
                mode = "Tracker"
            else:
                best_rel_xi = forecaster.predict_next(prev_xi, dt)
                mode = "Recovery"
                
            t_rel_tensor = torch.from_numpy(best_rel_xi).float().unsqueeze(0).to(config.device)
            T_rel = se3_exp_map(t_rel_tensor)
            
            T_global = compose_poses(T_global.clone(), T_rel)
            xi_global = se3_log_map(T_global).squeeze(0).cpu().numpy()
            
            forecaster.update(xi_global, current_stamp)
            
            summary = {
                "frame_id": current_id,
                "category": current_obj_name, 
                "xi": xi_global.tolist(),
                "tracking_score": track_score, 
                "fusion_mode": mode
            }
            processed_payloads.append(summary)
            
            if save_json:
                collector.append_transition(prev_rgb, frame_rgb, best_rel_xi, current_obj_name)
                
            if display_mode == DisplayMode.JSON:
                print(f"[METRIC] Frame={summary['frame_id']:03d} | Target={summary['category']:<12} | Mode={summary['fusion_mode']:<10} | Score={track_score:.2f} | Prior={scale_prior:.2f}m")

            elif display_mode == DisplayMode.REALTIME:
                gui_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                cv2.putText(gui_frame, f"Frame ID: {current_id:03d}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(gui_frame, f"Target: {current_obj_name}", (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(gui_frame, f"Fusion Mode: {mode}", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(gui_frame, f"Track Score: {track_score:.2f}", (20, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow("LEPAUTE SE(3) Subsystem - Realtime GUI", gui_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Pipeline terminated by user inside GUI window.")
                    break
                    
            prev_rgb = frame_rgb
            prev_xi = best_rel_xi
            last_stamp = current_stamp
            
    except KeyboardInterrupt:
        logger.info("Pipeline Interrupted by Host.")
    finally:
        worker.stop()
        if save_json: collector.stop()
        stream.release()
        if display_mode == DisplayMode.REALTIME:
            cv2.destroyAllWindows()
        logger.info("Pipeline subsystem torn down securely.")
        
    return processed_payloads

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="LEPAUTE SE(3) Monocular Subsystem Execution Pipeline Control"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="gui",
        choices=["headless", "gui", "json"],
        help="Select running mode: headless, gui (real-time window mode), json (data output mode)"
    )
    
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Specify the SQLite database path (optional)"
    )
    
    parser.add_argument(
        "--limit",
        action="store_true",
        help="Limit execution to 50 frames for testing purposes."
    )

    args = parser.parse_args()

    mode_mapping = {
        "headless": DisplayMode.HEADLESS,
        "gui": DisplayMode.REALTIME,
        "json": DisplayMode.JSON
    }
    selected_mode = mode_mapping[args.mode]

    config_kwargs = {}
    if args.db:
        config_kwargs["data_store"] = args.db
        
    config = LepauteConfig(**config_kwargs)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"System booting in mode: {selected_mode.name}")

    run_unlimited = not args.limit
    
    run_pipeline(
        config, 
        display_mode=selected_mode, 
        unlimited=run_unlimited, 
        mock=False
    )