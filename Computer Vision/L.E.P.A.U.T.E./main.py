import os
import time
import logging
import threading
import queue
import cv2
import numpy as np
import torch
from typing import Dict, List, Optional
from torchvision import transforms

from module import (
    LepauteConfig, CameraIOStream, DiskManager, DenseSE3Tracker, 
    SigLIPClassifier, TransformerModel, save_to_disk, load_data
)

logger = logging.getLogger("LEPAUTE.Main")

class InferenceWorker(threading.Thread):
    def __init__(self, config: LepauteConfig, tracker: DenseSE3Tracker, classifier: SigLIPClassifier, model: TransformerModel, disk_mgr: DiskManager, transform: transforms.Compose):
        super().__init__(daemon=True)
        self.config = config
        self.tracker = tracker
        self.classifier = classifier
        self.model = model
        self.disk_mgr = disk_mgr
        self.transform = transform
        
        self.input_queue = queue.Queue(maxsize=3)
        self._output_data = []
        self._latest_result = {"category": "Initializing...", "conf": 0.0, "xi": np.zeros(6), "rmse": 0.0}
        self.lock = threading.Lock()
        
        self.device = torch.device(config.device)
        self.cuda_stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        self.frame_counter = 0

    def get_output_data(self) -> List[Dict]:
        with self.lock: return list(self._output_data)

    def get_latest_result(self) -> Dict:
        with self.lock: return dict(self._latest_result)

    def run(self):
        while True:
            try:
                task = self.input_queue.get(timeout=3.0)
                if task is None: break 
                prev_frame, curr_frame = task
                self.frame_counter += 1
                
                run_heavy = (self.frame_counter % self.config.dl_inference_freq == 0)
                
                # Fast CV Tracking
                xi, rmse, depth_map = self.tracker.estimate_pose(prev_frame, curr_frame, run_depth=run_heavy)
                
                cat, conf = self._latest_result["category"], self._latest_result["conf"]
                
                # Asynchronous Heavy Deep Learning Block
                if run_heavy:
                    cat, conf = self.classifier.classify(curr_frame)
                    rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                    img_t = self.transform(rgb).unsqueeze(0).to(self.device, non_blocking=True)
                    xi_t = torch.from_numpy(xi).unsqueeze(0).to(self.device, non_blocking=True)
                    depth_t = torch.from_numpy(depth_map).unsqueeze(0).to(self.device, non_blocking=True)
                    
                    with torch.no_grad():
                        if self.cuda_stream:
                            with torch.cuda.stream(self.cuda_stream):
                                emb, pose_pred = self.model(img_t, depth_t, xi_t)
                        else:
                            emb, pose_pred = self.model(img_t, depth_t, xi_t)
                    
                    # Blend neural spatial inference directly back into CV estimation
                    pose_pred_np = pose_pred.squeeze(0).cpu().numpy()
                    confidence_factor = min(1.0, max(0.0, 1.0 - (rmse / 10.0)))
                    xi = (1.0 - (0.3 * confidence_factor)) * xi + (0.3 * confidence_factor) * pose_pred_np

                uuid_a = self.disk_mgr.save_frame(prev_frame)
                uuid_b = self.disk_mgr.save_frame(curr_frame)

                record = {
                    "timestamp": time.time(), 
                    "lie_params": xi.tolist(), 
                    "loss": float(rmse),
                    "detected_object": cat, 
                    "confidence": float(conf), 
                    "frame_a": uuid_a, 
                    "frame_b": uuid_b, 
                    "depth_mean": float(np.mean(depth_map)),
                    "depth_std": float(np.std(depth_map))
                }
                
                with self.lock:
                    self._output_data.append(record)
                    self._latest_result = {"category": cat, "conf": conf, "xi": xi, "rmse": rmse}
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Inference Thread Worker Failure: {e}")

def run_pipeline(display_mode: str = "realtime", config: Optional[LepauteConfig] = None, unlimited: bool = True, save_json: bool = True, mock: bool = False) -> List[Dict]:
    cfg = config or LepauteConfig()
    device = torch.device(cfg.device)
    
    try:
        cam = CameraIOStream(config=cfg, mock_mode=mock).start()
    except RuntimeError as e:
        logger.error(str(e))
        return []

    disk_mgr = DiskManager(config=cfg)
    tensor_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((cfg.orig_h, cfg.orig_w), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tracker = DenseSE3Tracker(config=cfg)
    classifier = SigLIPClassifier(config=cfg)
    
    # Ready for TorchScript inference optimization
    raw_model = TransformerModel(config=cfg).to(device).eval()
    model = raw_model
    
    worker = InferenceWorker(cfg, tracker, classifier, model, disk_mgr, tensor_transform)
    worker.start()

    logger.info("Synchronizing hardware buffer streams...")
    prev_frame = None
    for _ in range(30):
        prev_frame = cam.get_latest_frame()
        if prev_frame is not None: break
        time.sleep(0.1)

    fps_time = time.time()
    frames_rendered = 0
    current_fps = 0.0

    try:
        while True:
            frame = cam.get_latest_frame()
            if frame is None: 
                time.sleep(0.01)
                continue

            if worker.input_queue.full():
                try: worker.input_queue.get_nowait()
                except queue.Empty: pass
            
            worker.input_queue.put((prev_frame.copy(), frame.copy()))
            prev_frame = frame.copy()

            res = worker.get_latest_result()
            frames_rendered += 1

            if "realtime" in display_mode:
                vis = frame.copy()
                cv2.putText(vis, f"ID: {res['category']} ({res['conf']:.2f})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                disp = float(np.linalg.norm(res['xi'][:3]))
                cv2.putText(vis, f"SE(3) D: {disp:.3f}m | RMSE: {res['rmse']:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                
                if frames_rendered % 10 == 0:
                    current_fps = 10 / max((time.time() - fps_time), 1e-5)
                    fps_time = time.time()
                cv2.putText(vis, f"UI FPS: {current_fps:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("LEPAUTE Production Engine", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break

            if not unlimited and len(worker.get_output_data()) >= 30: 
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        worker.input_queue.put(None)
        worker.join()

    final_data = worker.get_output_data()
    if save_json and final_data:
        save_to_disk(final_data, config=cfg)
        
    return final_data

def run_main(**kwargs) -> None:
    run_pipeline(display_mode="realtime", unlimited=True, save_json=True, **kwargs)

if __name__ == "__main__":
    run_main()