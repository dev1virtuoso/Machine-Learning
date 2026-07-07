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
    LepauteConfig, DisplayMode, PerformanceMode, CameraIOStream, MonocularDirectTracker,
    SigLIPClassifier, SE3ResidualRefiner, SequenceDataCollector, ManifoldKinematicForecaster,
    se3_exp_map, se3_log_map, compose_poses, _mps_lock
)

import argparse
import sys
import signal
import contextlib

logger = logging.getLogger("LEPAUTE.Pipeline")

class GracefulShutdownHandler:
    def __init__(self):
        self.shutdown_requested = False
        self.sigint_count = 0
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            pass

    def _signal_handler(self, sig, frame):
        self.sigint_count += 1
        
        if self.sigint_count == 1:
            logger.info(f"Received termination signal ({sig}). Initiating graceful shutdown sequence...")
            self.shutdown_requested = True
            raise KeyboardInterrupt
            
        elif self.sigint_count == 2:
            logger.warning(f"Graceful shutdown already in progress. The process might be blocked inside a C++ extension (e.g., PyTorch Compilation). Press Ctrl+C again to force abort.")
            
        else:
            logger.critical("Multiple termination signals received. Process is deadlocked. Hard aborting via os._exit(130).")
            import os
            os._exit(130)

class InferenceWorker(threading.Thread):
    def __init__(self, config: LepauteConfig, classifier: SigLIPClassifier, refiner: SE3ResidualRefiner):
        super().__init__(daemon=True)
        self.config = config
        self.classifier = classifier
        self.refiner = refiner
        self.device = torch.device(config.device)
        
        self.job_queue = queue.Queue(maxsize=5) 
        self.state_lock = threading.Lock()
        
        self.health_lock = threading.Lock()
        self.last_heartbeat = time.time()
        
        self.history_buffer: Dict[int, Tuple[Tuple[str, float], np.ndarray, np.ndarray, float]] = {}
        self.running = threading.Event()
        self.running.set()

    def is_healthy(self, timeout_sec: float = 15.0) -> bool:
        if not self.is_alive():
            return False
        with self.health_lock:
            return (time.time() - self.last_heartbeat) < timeout_sec

    def enqueue_job(self, frame_id: int, img_ref: np.ndarray, img_cur: np.ndarray, tracker_xi_rel: np.ndarray):
        if not self.job_queue.full():
            self.job_queue.put_nowait((frame_id, img_ref.copy(), img_cur.copy(), tracker_xi_rel.copy()))

    def get_latest_resolved_state(self, current_time: float) -> Optional[Tuple[Tuple[str, float], np.ndarray, np.ndarray]]:
        with self.state_lock:
            current_keys = list(self.history_buffer.keys())
            for k in current_keys:
                if current_time - self.history_buffer[k][3] > 5.0:
                    del self.history_buffer[k]
                    
            if not self.history_buffer:
                return None
                
            latest_resolved_id = max(self.history_buffer.keys())
            state = self.history_buffer.pop(latest_resolved_id)
            
            obsolete_keys = [k for k in list(self.history_buffer.keys()) if k <= latest_resolved_id]
            for k in obsolete_keys:
                self.history_buffer.pop(k, None)
                
            return state[0], state[1], state[2]

    def stop(self):
        logger.info("[InferenceWorker] Stop command received. Halting event loop...")
        self.running.clear()
        
        logger.debug("[InferenceWorker] Flushing internal job queue...")
        while not self.job_queue.empty():
            try:
                self.job_queue.get_nowait()
                self.job_queue.task_done()
            except queue.Empty:
                break
                
        logger.info("[InferenceWorker] Waiting for thread to join (timeout=2.0s)...")
        self.join(timeout=2.0)
        
        if self.is_alive():
            logger.warning("[InferenceWorker] Thread join timed out! The worker is likely stuck in a blocking C++ Native call (e.g., PyTorch Compilation). It will be safely orphaned.")
        else:
            logger.info("[InferenceWorker] Thread successfully joined and terminated.")

    def run(self):
        lock_context = _mps_lock if self.device.type == "mps" else contextlib.nullcontext()

        while self.running.is_set():
            with self.health_lock:
                self.last_heartbeat = time.time()
                
            try:
                task = self.job_queue.get(timeout=0.1)
                frame_id, img_ref, img_cur, tracker_xi_rel = task
                
                with lock_context:
                    obj_name, conf = self.classifier.predict(img_cur)
                    
                    t_ref = torch.from_numpy(img_ref).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                    t_cur = torch.from_numpy(img_cur).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
                    t_xi = torch.from_numpy(tracker_xi_rel).float().unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        delta_xi, delta_scale, unc_pose, unc_scale = self.refiner(t_ref, t_cur, t_xi)
                        refined_xi_rel = delta_xi.squeeze(0).cpu().numpy()
                        uncertainty = unc_pose.squeeze(0).cpu().numpy()
                    
                completion_time = time.time()
                
                with self.state_lock:
                    self.history_buffer[frame_id] = ((obj_name, conf), refined_xi_rel, uncertainty, completion_time)
                        
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
    config, 
    display_mode: DisplayMode = DisplayMode.HEADLESS,
    unlimited: bool = False,
    save_json: bool = True,
    mock: bool = False
) -> List[Dict[str, Any]]:
    
    logger.info(f"Initializing Hardened Monocular LEPAUTE Subsystem under [{config.performance_mode.value.upper()}] profile.")
    shutdown_handler = GracefulShutdownHandler()
    
    min_frame_time = 0.0
    
    if config.performance_mode == PerformanceMode.LOW:
        config.pyramid_levels = max(1, config.pyramid_levels - 1)
        config.gn_max_iter = max(4, config.gn_max_iter // 2)
        min_frame_time = 0.066  
    elif config.performance_mode == PerformanceMode.HIGH:
        config.pyramid_levels = config.pyramid_levels + 1
        config.gn_max_iter = int(config.gn_max_iter * 1.5)
        min_frame_time = 0.0    

    stream = CameraIOStream(config=config, mock=mock)
    tracker = MonocularDirectTracker(config=config)
    collector = SequenceDataCollector(config=config)
    if save_json: collector.start()
        
    forecaster = ManifoldKinematicForecaster()
    classifier = SigLIPClassifier(config=config)
    
    refiner = SE3ResidualRefiner(config=config, feature_dim=256, max_resolution=64).to(config.device)

    model_path = "./checkpoints/best_model.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=config.device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            refiner.load_compiled_state_dict(checkpoint['model_state_dict'])
        else:
            refiner.load_compiled_state_dict(checkpoint)
            logger.info(f"Loaded trained model: {model_path}")
    else:
        logger.warning(f"Model {model_path} not found, using randomly initialized model")
        
    refiner.eval()
    
    worker = InferenceWorker(config=config, classifier=classifier, refiner=refiner)
    worker.start()
        
    success, prev_rgb, prev_meta = stream.read()
    if not success: 
        logger.error("Camera acquisition failed permanently. Terminating.")
        return []

    processed_payloads = []
    max_frames = 50 if not unlimited else 999999
    
    T_global = torch.eye(4, dtype=torch.float32, device=config.device).unsqueeze(0)
    prev_xi = np.zeros(6, dtype=np.float32)
    last_stamp = prev_meta["timestamp"]
    last_heartbeat_check = time.time()
    
    current_obj_name = "background"
    trajectory_2d = []  
    latest_conf = 0.0
    latest_unc = np.zeros(6, dtype=np.float32)
    
    if display_mode in (DisplayMode.REALTIME, DisplayMode.DETAILEDGUI):
        cv2.namedWindow("LEPAUTE SE(3) Subsystem", cv2.WINDOW_AUTOSIZE)
        try:
            cv2.startWindowThread()
        except Exception:
            pass
    
    try:
        for frame_idx in range(max_frames):
            if shutdown_handler.shutdown_requested:
                logger.info("Main loop broken by graceful shutdown protocol.")
                break
                
            current_time = time.time()
            if current_time - last_heartbeat_check > 5.0:
                if not worker.is_healthy():
                    logger.error("SYSTEM HALT: InferenceWorker failed health check. Preventing zombie lock.")
                    break
                last_heartbeat_check = current_time

            loop_start_wall = time.time()
            success, frame_rgb, metadata = stream.read()
            if not success: continue

            current_stamp = metadata["timestamp"]
            current_id = metadata["frame_id"]
            dt = current_stamp - last_stamp
            
            logger.info(f"[Pipeline] Frame ID {current_id} routed to main thread. Evaluating asynchronous inference state...")
            
            async_state = worker.get_latest_resolved_state(current_stamp)
            if async_state is not None:
                (obj_name, conf), refined_xi_rel, unc = async_state
                current_obj_name = obj_name 
                latest_conf = conf
                latest_unc = unc
                has_async = True
                logger.info(f"[Pipeline] Async state resolved for Frame ID {current_id}: Target='{obj_name}' (Confidence: {conf:.2f}).")
            else:
                has_async = False
                
            scale_prior = config.object_scales.get(current_obj_name, 1.0)
            logger.info(f"[Pipeline] Dispatching Dense Direct Tracker for Frame ID {current_id} | Scale Prior: {scale_prior:.3f}m")
            
            tracker_xi_rel, track_score = tracker.track(prev_rgb, frame_rgb, scale_prior)
            logger.info(f"[Pipeline] Direct Tracker completed. Alignment Score: {track_score:.4f}")
            
            if track_score > 0.1:
                worker.enqueue_job(current_id, prev_rgb, frame_rgb, tracker_xi_rel)
            
            if has_async:
                best_rel_xi = refined_xi_rel
                mode = "Refined"
                applied_scale = scale_prior
            elif track_score >= 0.1:
                best_rel_xi = tracker_xi_rel
                mode = "Tracker"
                applied_scale = scale_prior
            else:
                logger.warning(f"[Pipeline] Tracking alignment failed for Frame ID {current_id} (Score < 0.1). Engaging Manifold Kinematic Forecaster.")
                pred_pose, pred_scale = forecaster.predict(current_stamp)
                T_curr_global = torch.from_numpy(pred_pose).float().to(config.device).unsqueeze(0)
                T_rel_mat = torch.inverse(T_global) @ T_curr_global
                best_rel_xi = se3_log_map(T_rel_mat).squeeze(0).cpu().numpy()
                mode = "Recovery"
                applied_scale = pred_scale
                
            logger.info(f"[Pipeline] Fusing state (Fusion Mode: {mode}). Updating SE(3) global trajectory...")
                
            t_rel_tensor = torch.from_numpy(best_rel_xi).float().unsqueeze(0).to(config.device)
            T_rel = se3_exp_map(t_rel_tensor)
            
            T_global = compose_poses(T_global.clone(), T_rel)
            xi_global = se3_log_map(T_global).squeeze(0).cpu().numpy()
            
            forecaster.update_state(T_global.squeeze(0).cpu().numpy(), np.zeros(6), applied_scale, current_stamp, weight=0.5)
            
            euclidean_trans = T_global[0, :3, 3].cpu().numpy()
            trajectory_2d.append((euclidean_trans[0], euclidean_trans[2]))
            
            if len(trajectory_2d) > 2000:
                trajectory_2d.pop(0)
            
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
                print(f"[METRIC] Frame={summary['frame_id']:03d} | Perf={config.performance_mode.value.upper()} | Target={summary['category']:<12} | Mode={summary['fusion_mode']:<10} | Score={track_score:.2f}")
            
            elif display_mode in (DisplayMode.REALTIME, DisplayMode.DETAILEDGUI):
                gui_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                fps = 1.0 / dt if dt > 0 else 0.0
                
                cv2.putText(gui_frame, f"Frame ID: {current_id:03d} | FPS: {fps:.1f} | Mode: {mode}", (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if display_mode == DisplayMode.DETAILEDGUI:
                    cv2.putText(gui_frame, f"Obj: {current_obj_name} (Conf: {latest_conf:.2f})", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                    cv2.putText(gui_frame, f"Active Scale: {applied_scale:.3f}m", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                    cv2.putText(gui_frame, f"Tracking Score: {track_score:.2f}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if track_score > 0.1 else (0, 0, 255), 1)
                    
                    map_size = 120
                    map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
                    cv2.rectangle(map_img, (0, 0), (map_size-1, map_size-1), (255, 255, 255), 1)
                    
                    if len(trajectory_2d) > 1:
                        pts = np.array(trajectory_2d, dtype=np.float32)
                        pt_min, pt_max = pts.min(axis=0), pts.max(axis=0)
                        rng = np.maximum(pt_max - pt_min, 1e-4)
                        norm_pts = ((pts - pt_min) / rng * (map_size - 20) + 10).astype(np.int32)
                        
                        for i in range(1, len(norm_pts)):
                            cv2.line(map_img, tuple(norm_pts[i-1]), tuple(norm_pts[i]), (0, 255, 0), 1)
                        cv2.circle(map_img, tuple(norm_pts[-1]), 4, (0, 0, 255), -1)
                        
                    h, w = gui_frame.shape[:2]
                    if h >= map_size + 20 and w >= map_size + 20:
                        gui_frame[10:10+map_size, w-map_size-10:w-10] = map_img
                
                cv2.imshow("LEPAUTE SE(3) Subsystem", gui_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Pipeline terminated by user inside GUI window.")
                    break
                    
            prev_rgb = frame_rgb
            prev_xi = best_rel_xi
            last_stamp = current_stamp
            
            if min_frame_time > 0.0:
                loop_elapsed = time.time() - loop_start_wall
                if loop_elapsed < min_frame_time:
                    time.sleep(min_frame_time - loop_elapsed)
                    
    except KeyboardInterrupt:
        logger.warning("Pipeline Interrupted by Host (KeyboardInterrupt caught natively in main loop).")
    except Exception as e:
        logger.error(f"Pipeline crashed due to unexpected runtime fault: {e}")
    finally:
        logger.info("=== STARTING TEARDOWN SEQUENCE ===")
        logger.info("[Teardown] Halting InferenceWorker...")
        worker.stop()
        
        if save_json:
            logger.info("[Teardown] Halting SequenceDataCollector...")
            collector.stop()
            
        logger.info("[Teardown] Releasing CameraIOStream hardware locks...")
        stream.release()
        
        if display_mode in (DisplayMode.REALTIME, DisplayMode.DETAILEDGUI):
            logger.info("[Teardown] Destroying OpenCV windows...")
            cv2.destroyAllWindows()
            
        logger.info("=== PIPELINE SUBSYSTEM TORN DOWN SECURELY ===")
        
    return processed_payloads

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEPAUTE SE(3) Monocular Subsystem Execution Pipeline Control"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="gui",
        choices=["headless", "gui", "json", "detailedgui"],
        help="Select running mode: headless, gui (real-time window mode), json (data output mode), detailedgui (HUD + Trajectory map)"
    )
    
    parser.add_argument(
        "--perf",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Select performance profile: low (slower update frequency/conserves resource), medium (default standard parameters), high (maximum update frequency/unthrottled accuracy)"
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
        "json": DisplayMode.JSON,
        "detailedgui": DisplayMode.DETAILEDGUI
    }
    selected_mode = mode_mapping[args.mode]

    config_kwargs = {}
    if args.db:
        config_kwargs["data_store"] = args.db
        
    config = LepauteConfig(**config_kwargs)
    
    config.performance_mode = PerformanceMode(args.perf)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"System booting in mode: {selected_mode.name} with performance profile: {config.performance_mode.name}")

    run_unlimited = not args.limit
    
    run_pipeline(
        config, 
        display_mode=selected_mode, 
        unlimited=run_unlimited, 
        mock=False
    )