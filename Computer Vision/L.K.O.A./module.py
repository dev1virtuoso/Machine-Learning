import cv2
import numpy as np
import onnxruntime as ort
import threading
import time
import os
import struct
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any

class VehicleState(IntEnum):
    FORWARD_CRUISE = 0
    CRITICAL_AVOIDANCE = 1
    EMERGENCY_STOP = 2

@dataclass(frozen=True)
class CameraHardwareProfile:
    height_m: float = 0.290
    focal_y_px: float = 714.2
    center_y_px: float = 241.8
    min_valid_distance: float = 0.25
    max_valid_distance: float = 8.00

@dataclass(frozen=True)
class ThermalProfile:
    nominal_temp: float = 68.0
    warning_temp: float = 73.0
    critical_temp: float = 78.0
    recovery_hysteresis: float = 3.0

@dataclass
class ProductionSystemConfig:
    production_mode: bool = True
    target_fps: int = 12
    uart_port: str = "/dev/ttyAMA0"
    uart_baud: int = 115200
    yolo_onnx_path: str = "yolov8n.onnx"
    midas_onnx_path: str = "midas_small.onnx"
    camera_idx: int = 0
    frame_w: int = 640
    frame_h: int = 480
    hardware: CameraHardwareProfile = CameraHardwareProfile()
    thermals: ThermalProfile = ThermalProfile()
    class_priors: Dict[int, List[float]] = field(default_factory=lambda: {
        0: [0.45, 0.92],
        56: [0.50, 0.85],
        11: [0.35, 0.70]
    })

class ThreadSafeCameraReader:
    def __init__(self, config: ProductionSystemConfig):
        self.cfg = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.buffers: List[np.ndarray] = [
            np.zeros((self.cfg.frame_h, self.cfg.frame_w, 3), dtype=np.uint8),
            np.zeros((self.cfg.frame_h, self.cfg.frame_w, 3), dtype=np.uint8)
        ]
        self.front_idx = 0
        self.back_idx = 1
        self.lock = threading.Lock()
        self.is_running = False
        self.frame_ready = False
        self._initialize_hardware_socket()

    def _initialize_hardware_socket(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.cfg.camera_idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_h)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if not self.cap.isOpened():
            return False
        return True

    def start(self) -> 'ThreadSafeCameraReader':
        self.is_running = True
        worker = threading.Thread(target=self._capture_worker_loop, daemon=True)
        worker.name = "AsyncCameraDriver"
        worker.start()
        return self

    def _capture_worker_loop(self):
        backoff = 0.1
        while self.is_running:
            status, frame = self.cap.read() if self.cap else (False, None)
            if not status or frame is None:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 5.0)
                self._initialize_hardware_socket()
                continue
            
            backoff = 0.1
            np.copyto(self.buffers[self.back_idx], frame)
            with self.lock:
                self.front_idx, self.back_idx = self.back_idx, self.front_idx
                self.frame_ready = True

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.frame_ready:
                return False, None
            return True, self.buffers[self.front_idx].copy()

    def terminate(self):
        self.is_running = False
        if self.cap:
            self.cap.release()


class RobustUARTCommandPublisher:
    def __init__(self, config: ProductionSystemConfig):
        self.cfg = config
        self.lock = threading.Lock()
        self.last_transmission_time = time.time()
        self.is_active = True
        self.serial_handle: Any = None
        self._connect_serial()
        
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self.heartbeat_thread.name = "UARTHeartbeat"
        self.heartbeat_thread.start()

    def _connect_serial(self):
        try:
            import serial
            self.serial_handle = serial.Serial(
                port=self.cfg.uart_port,
                baudrate=self.cfg.uart_baud,
                timeout=0.05,
                write_timeout=0.05
            )
        except Exception:
            self.serial_handle = None

    def transmit(self, state: VehicleState, linear_ratio: float, steer_rate: float):
        with self.lock:
            v_byte = int(np.clip(linear_ratio * 255.0, 0.0, 255.0))
            w_byte = int(np.clip((steer_rate + 1.0) * 127.0, 0.0, 254.0))
            state_byte = int(state.value)

            payload = bytes([state_byte, v_byte, w_byte])
            checksum = sum(payload) % 256
            packet = struct.pack(">BB3sBBB", 0xAA, 0x55, payload, checksum, 0x0D, 0x0A)

            attempts = 3
            while attempts > 0:
                try:
                    if self.serial_handle is not None:
                        self.serial_handle.write(packet)
                        self.serial_handle.flush()
                    self.last_transmission_time = time.time()
                    break
                except Exception:
                    attempts -= 1
                    self._connect_serial()
                    time.sleep(0.005)

    def _heartbeat_worker(self):
        while self.is_active:
            time.sleep(0.05)
            if time.time() - self.last_transmission_time >= 0.200:
                self.transmit(VehicleState.EMERGENCY_STOP, 0.0, 0.0)

    def close(self):
        self.is_active = False
        if self.serial_handle:
            try:
                self.serial_handle.close()
            except Exception:
                pass

class LightweightObstacleFilter:
    def __init__(self, alpha: float = 0.5, spatial_gate_px: float = 90.0):
        self.alpha = alpha
        self.spatial_gate_px = spatial_gate_px
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 0

    def process_predictions(self, active_detections: List[Dict[str, Any]], delta_t: float):
        updated_tracks = {}
        
        for det in active_detections:
            x1, y1, x2, y2 = det['box']
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            cls_id = det['cls']
            measured_dist = det['dist']
            
            matched_id = None
            min_distance = self.spatial_gate_px
            
            for t_id, t_data in self.tracks.items():
                if t_data['class_id'] != cls_id:
                    continue
                tx1, ty1, tx2, ty2 = t_data['bbox']
                tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
                dist = np.hypot(cx - tcx, cy - tcy)
                if dist < min_distance:
                    min_distance = dist
                    matched_id = t_id
                    
            if matched_id is not None:
                prior_dist = self.tracks[matched_id]['fused_dist']
                inst_velocity = (measured_dist - prior_dist) / (delta_t + 1e-5)
                smoothed_dist = float(self.alpha * measured_dist + (1.0 - self.alpha) * prior_dist)
                smoothed_vel = float(self.alpha * inst_velocity + (1.0 - self.alpha) * self.tracks[matched_id]['velocity'])
                
                updated_tracks[matched_id] = {
                    'bbox': det['box'],
                    'fused_dist': smoothed_dist,
                    'velocity': smoothed_vel,
                    'class_id': cls_id,
                    'stale_count': 0
                }
            else:
                updated_tracks[self.next_track_id] = {
                    'bbox': det['box'],
                    'fused_dist': measured_dist,
                    'velocity': 0.0,
                    'class_id': cls_id,
                    'stale_count': 0
                }
                self.next_track_id += 1
                
        for t_id, t_data in self.tracks.items():
            if t_id not in updated_tracks:
                if t_data['stale_count'] < 6:
                    extrapolated_dist = t_data['fused_dist'] + (t_data['velocity'] * delta_t)
                    extrapolated_dist = float(np.clip(extrapolated_dist, 0.2, 8.0))
                    
                    updated_tracks[t_id] = {
                        'bbox': t_data['bbox'], 
                        'fused_dist': extrapolated_dist,
                        'velocity': t_data['velocity'],
                        'class_id': t_data['class_id'],
                        'stale_count': t_data['stale_count'] + 1
                    }
                    
        self.tracks = updated_tracks

    def get_extrapolated_detections(self) -> List[Dict[str, Any]]:
        output = []
        for t_id, t_data in self.tracks.items():
            output.append({
                'box': t_data['bbox'],
                'cls': t_data['class_id'],
                'dist': t_data['fused_dist'],
                'stale_count': t_data['stale_count']
            })
        return output


class MultiBandRANSACSolver:
    def __init__(self, hw_config: CameraHardwareProfile):
        self.hw = hw_config
        self.running_pitch_deg = 14.5
        self.running_scale_factor = 1.0
        self.is_calibrated = False
        self.consecutive_failures = 0
        self.ema_alpha = 0.12

    def solve(self, depth_map: np.ndarray) -> Tuple[float, float]:
        th, tw = depth_map.shape
        
        bands = [(0.65, 0.75), (0.75, 0.85), (0.85, 0.95)]
        sample_v = []
        sample_disp = []
        
        for b_start, b_end in bands:
            v_start, v_end = int(th * b_start), int(th * b_end)
            for v in range(v_start, v_end, 4):
                row_stripe = depth_map[v, int(tw * 0.25):int(tw * 0.75)]
                if row_stripe.size > 0:
                    sample_v.append((v / th) * 480.0)
                    sample_disp.append(np.percentile(row_stripe, 30))
                    
        v_arr = np.array(sample_v, dtype=np.float32)
        d_arr = np.array(sample_disp, dtype=np.float32)
        
        n_samples = len(v_arr)
        if n_samples < 8:
            return self.running_pitch_deg, self.running_scale_factor

        best_slope, best_intercept = 0.0, 0.0
        max_inliers = 0
        iterations = 70
        mad_noise = np.median(np.abs(d_arr - np.median(d_arr))) + 1e-4
        inlier_threshold = mad_noise * 0.65

        for _ in range(iterations):
            idx = np.random.choice(n_samples, 2, replace=False)
            v1, v2 = v_arr[idx[0]], v_arr[idx[1]]
            d1, d2 = d_arr[idx[0]], d_arr[idx[1]]
            if np.abs(v1 - v2) < 1e-3: 
                continue
                
            slope = (d2 - d1) / (v2 - v1)
            intercept = d1 - slope * v1
            
            hypotheses = slope * v_arr + intercept
            inliers = np.sum(np.abs(d_arr - hypotheses) < inlier_threshold)
            
            if inliers > max_inliers:
                max_inliers = inliers
                best_slope = slope
                best_intercept = intercept
                if inliers / n_samples > 0.75:
                    break

        inlier_ratio = max_inliers / n_samples
        
        if inlier_ratio < 0.45:
            self.consecutive_failures += 1
            if self.consecutive_failures > 8:
                self.running_pitch_deg = 14.5
                self.running_scale_factor = 1.15
            return self.running_pitch_deg, self.running_scale_factor

        self.consecutive_failures = 0

        estimated_pitch = 14.5 + (best_slope * 12.2)
        estimated_pitch = np.clip(estimated_pitch, 7.5, 22.5)
        
        if self.is_calibrated and (np.abs(estimated_pitch - self.running_pitch_deg) / self.running_pitch_deg) > 0.15:
            return self.running_pitch_deg, self.running_scale_factor

        self.running_pitch_deg = float(self.ema_alpha * estimated_pitch + (1.0 - self.ema_alpha) * self.running_pitch_deg)
        
        calc_scales = []
        for v_px, disp in zip(v_arr, d_arr):
            alpha_rad = np.arctan((v_px - self.hw.center_y_px) / self.hw.focal_y_px)
            total_ang = np.radians(self.running_pitch_deg) + alpha_rad
            if total_ang > 0.05:
                true_dist = self.hw.height_m / np.tan(total_ang)
                calc_scales.append(true_dist * disp)
                
        if calc_scales:
            self.running_scale_factor = float(self.ema_alpha * np.median(calc_scales) + (1.0 - self.ema_alpha) * self.running_scale_factor)
            self.is_calibrated = True

        return self.running_pitch_deg, self.running_scale_factor


class MultiCueBottomRefiner:
    @staticmethod
    def refine(frame: np.ndarray, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        img_h, img_w = frame.shape[:2]
        dm_h, dm_w = depth_map.shape
        
        search_pad = int(img_h * 0.08)
        roi_y1 = max(0, y2 - search_pad)
        roi_y2 = min(img_h - 1, y2 + 6)
        
        if (x2 - x1) <= 0 or (roi_y2 - roi_y1) <= 0:
            return float(y2), 0.3

        roi_bgr = frame[roi_y1:roi_y2, x1:x2]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        sob_y = np.max(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)), axis=1)
        canny_prof = np.sum(cv2.Canny(gray, 35, 95) > 0, axis=1).astype(np.float64)
        
        dm_x1 = int((x1 / img_w) * dm_w)
        dm_x2 = int((x2 / img_w) * dm_w)
        dm_y1 = int((roi_y1 / img_h) * dm_h)
        dm_y2 = int((roi_y2 / img_h) * dm_h)
        
        depth_roi = depth_map[dm_y1:dm_y2, dm_x1:dm_x2]
        if depth_roi.size > 0:
            depth_prof = np.max(np.abs(cv2.Sobel(depth_roi, cv2.CV_64F, 0, 1, ksize=3)), axis=1)
            depth_prof_rescaled = cv2.resize(depth_prof.reshape(-1, 1), (1, roi_y2 - roi_y1)).flatten()
        else:
            depth_prof_rescaled = np.zeros_like(sob_y)

        s_norm = sob_y / (np.max(sob_y) + 1e-5)
        c_norm = canny_prof / (np.max(canny_prof) + 1e-5)
        d_norm = depth_prof_rescaled / (np.max(depth_prof_rescaled) + 1e-5)

        voting_profile = s_norm * 0.25 + c_norm * 0.25 + d_norm * 0.50
        best_local_idx = int(np.argmax(voting_profile))
        
        refined_y2 = float(roi_y1 + best_local_idx)
        
        mean_score = np.mean(voting_profile) + 1e-5
        peak_ratio = voting_profile[best_local_idx] / mean_score
        confidence = float(np.clip(peak_ratio / 5.0, 0.1, 1.0))

        return refined_y2, confidence

class ProductionVisionEngine:
    def __init__(self, config: ProductionSystemConfig):
        self.cfg = config
        self.frame_id = 0
        self.last_frame_time = time.time()
        
        self.yolo_interval = 2
        self.midas_interval = 4
        self.thermal_recovery_latch = False
        
        self.cached_depth_map = np.zeros((256, 256), dtype=np.float32)
        self.active_detections_list: List[Dict[str, Any]] = []
        
        self.current_state = VehicleState.FORWARD_CRUISE
        self.last_state_change = time.time()
        self.state_consensus_history: List[VehicleState] = []
        
        self.ransac = MultiBandRANSACSolver(self.cfg.hardware)
        self.tracker = LightweightObstacleFilter()
        
        self._initialize_onnx_runtimes()
        self._execute_hardware_warmup()

    def _initialize_onnx_runtimes(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.yolo_sess = ort.InferenceSession(self.cfg.yolo_onnx_path, opts, providers=['CPUExecutionProvider'])
        self.midas_sess = ort.InferenceSession(self.cfg.midas_onnx_path, opts, providers=['CPUExecutionProvider'])
        
        self.yolo_in_name = self.yolo_sess.get_inputs()[0].name
        self.midas_in_name = self.midas_sess.get_inputs()[0].name
        
        self.y_h, self.y_w = self.yolo_sess.get_inputs()[0].shape[2:4]
        self.m_h, self.m_w = self.midas_sess.get_inputs()[0].shape[2:4]

    def _execute_hardware_warmup(self):
        dummy_yolo_in = np.zeros((1, 3, self.y_h, self.y_w), dtype=np.float32)
        dummy_midas_in = np.zeros((1, 3, self.m_h, self.m_w), dtype=np.float32)
        for _ in range(3):
            self.yolo_sess.run(None, {self.yolo_in_name: dummy_yolo_in})
            self.midas_sess.run(None, {self.midas_in_name: dummy_midas_in})

    def _get_cpu_temperature(self) -> float:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            return 45.0

    def _manage_thermal_scheduling(self, current_temp: float):
        if self.thermal_recovery_latch:
            if current_temp < (self.cfg.thermals.nominal_temp - self.cfg.thermals.recovery_hysteresis):
                self.thermal_recovery_latch = False
        else:
            if current_temp >= self.cfg.thermals.critical_temp:
                self.thermal_recovery_latch = True

        if self.thermal_recovery_latch or current_temp >= self.cfg.thermals.critical_temp:
            self.yolo_interval = 5
            self.midas_interval = 10
        elif current_temp >= self.cfg.thermals.warning_temp:
            self.yolo_interval = 3
            self.midas_interval = 6
        elif current_temp >= self.cfg.thermals.nominal_temp:
            self.yolo_interval = 2
            self.midas_interval = 4
        else:
            self.yolo_interval = 1
            self.midas_interval = 2

    def _variance_weighted_fusion(self, d_geom: float, b_conf: float, d_prior: float, 
                                  p_conf: float, d_depth: float) -> float:
        v_geom = (0.025 * (d_geom ** 4)) / (b_conf + 1e-5)
        v_prior = (0.045 * (d_prior ** 2)) / (p_conf + 1e-5)
        v_depth = 0.075 * (d_depth ** 4)

        w_geom = 1.0 / max(v_geom, 1e-6)
        w_prior = 1.0 / max(v_prior, 1e-6)
        w_depth = 1.0 / max(v_depth, 1e-6)
        
        sum_weights = w_geom + w_prior + w_depth
        fused = ((d_geom * w_geom) + (d_prior * w_prior) + (d_depth * w_depth)) / sum_weights
        return float(np.clip(fused, self.cfg.hardware.min_valid_distance, self.cfg.hardware.max_valid_distance))

    def _parse_yolo_output(self, output_tensors: List[np.ndarray]) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        predictions = []
        out = output_tensors[0]
        if len(out.shape) == 3:
            out = out[0]
            
        boxes_data = out[:4, :].T 
        scores_data = out[4:, :].T 
        
        img_h, img_w = self.cfg.frame_h, self.cfg.frame_w
        
        for i in range(scores_data.shape[0]):
            row_scores = scores_data[i]
            cls_id = int(np.argmax(row_scores))
            conf = float(row_scores[cls_id])
            
            if conf >= 0.52 and cls_id in self.cfg.class_priors:
                cx, cy, w, h = boxes_data[i]
                x1 = int(np.clip((cx - w / 2.0) * (img_w / self.y_w), 0, img_w - 1))
                y1 = int(np.clip((cy - h / 2.0) * (img_h / self.y_h), 0, img_h - 1))
                x2 = int(np.clip((cx + w / 2.0) * (img_w / self.y_w), 0, img_w - 1))
                y2 = int(np.clip((cy + h / 2.0) * (img_h / self.y_h), 0, img_h - 1))
                predictions.append(((x1, y1, x2, y2), conf, cls_id))
                
        return self._apply_custom_nms(predictions, iou_threshold=0.4)

    def _apply_custom_nms(self, preds: List[Tuple[Tuple[int, int, int, int], float, int]], 
                          iou_threshold: float) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        if not preds:
            return []
        sorted_preds = sorted(preds, key=lambda x: x[1], reverse=True)
        keep = []
        while sorted_preds:
            curr = sorted_preds.pop(0)
            keep.append(curr)
            cx1, cy1, cx2, cy2 = curr[0]
            curr_area = (cx2 - cx1) * (cy2 - cy1)
            
            remains = []
            for item in sorted_preds:
                ix1, iy1, ix2, iy2 = item[0]
                xx1 = max(cx1, ix1)
                yy1 = max(cy1, iy1)
                xx2 = min(cx2, ix2)
                yy2 = min(cy2, iy2)
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                union = curr_area + ((ix2 - ix1) * (iy2 - iy1)) - inter
                if (inter / (union + 1e-6)) < iou_threshold:
                    remains.append(item)
            sorted_preds = remains
        return keep

    def process_frame(self, frame: np.ndarray) -> Tuple[VehicleState, float, float]:
        self.frame_id += 1
        now = time.time()
        delta_t = now - self.last_frame_time
        self.last_frame_time = now
        
        cpu_temp = self._get_cpu_temperature()
        self._manage_thermal_scheduling(cpu_temp)

        if self.frame_id % self.midas_interval == 0 or self.frame_id == 1:
            m_inp = cv2.resize(frame, (self.m_w, self.m_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            m_inp = ((m_inp - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])).transpose(2, 0, 1)[np.newaxis, ...]
            raw_depth = self.midas_sess.run(None, {self.midas_in_name: m_inp.astype(np.float32)})[0].squeeze()
            self.cached_depth_map = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
            self.ransac.solve(self.cached_depth_map)

        if self.frame_id % self.yolo_interval == 0 or self.frame_id == 1:
            y_inp = cv2.resize(frame, (self.y_w, self.y_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            y_inp = y_inp.transpose(2, 0, 1)[np.newaxis, ...]
            yolo_raw = self.yolo_sess.run(None, {self.yolo_in_name: y_inp})
            parsed_boxes = self._parse_yolo_output(yolo_raw)
            
            frame_detections = []
            img_h, img_w = self.cfg.frame_h, self.cfg.frame_w
            pitch_deg, scale_factor = self.ransac.running_pitch_deg, self.ransac.running_scale_factor
            
            for box, conf, cls_id in parsed_boxes:
                bx1, by1, bx2, by2 = box
                bw_px = bx2 - bx1
                
                ref_y2, bottom_conf = MultiCueBottomRefiner.refine(frame, self.cached_depth_map, box)
                
                alpha = np.arctan((self.cfg.hardware.center_y_px - ref_y2) / self.cfg.hardware.focal_y_px)
                tot_angle = np.radians(pitch_deg) + alpha
                d_geom = self.cfg.hardware.height_m / np.tan(tot_angle) if tot_angle > 0.05 else self.cfg.hardware.max_valid_distance
                
                std_w, prior_w = self.cfg.class_priors[cls_id]
                d_prior = (std_w * self.cfg.hardware.focal_y_px) / max(bw_px, 1)
                
                dm_h, dm_w = self.cached_depth_map.shape
                look_x = int(np.clip(((bx1 + bx2) / 2.0 / img_w) * dm_w, 0, dm_w - 1))
                look_y = int(np.clip((ref_y2 / img_h) * dm_h, 0, dm_h - 1))
                disp_val = self.cached_depth_map[look_y, look_x]
                d_depth = scale_factor / max(disp_val, 1e-4)

                fused = self._variance_weighted_fusion(d_geom, bottom_conf, d_prior, prior_w, d_depth)
                frame_detections.append({'box': box, 'cls': cls_id, 'dist': fused})

            self.tracker.process_predictions(frame_detections, delta_t)
            self.active_detections_list = self.tracker.get_extrapolated_detections()
        else:
            self.tracker.process_predictions([], delta_t)
            self.active_detections_list = self.tracker.get_extrapolated_detections()

        dm_h, dm_w = self.cached_depth_map.shape
        r_top, r_bottom = int(dm_h * 0.58), int(dm_h * 0.94)
        split_w = int(dm_w / 5)
        
        zone_c = self.cached_depth_map[r_top:r_bottom, split_w*2:split_w*3]
        zone_l = self.cached_depth_map[r_top:r_bottom, split_w:split_w*2]
        zone_r = self.cached_depth_map[r_top:r_bottom, split_w*3:split_w*4]
        
        base_median = np.median(self.cached_depth_map)
        base_mad = np.median(np.abs(self.cached_depth_map - base_median)) + 1e-5
        
        score_c = (np.percentile(zone_c, 95) - base_median) / base_mad if zone_c.size > 0 else 0.0
        score_l = (np.percentile(zone_l, 95) - base_median) / base_mad if zone_l.size > 0 else 0.0
        score_r = (np.percentile(zone_r, 95) - base_median) / base_mad if zone_r.size > 0 else 0.0

        BLOCK_SIGMA = 3.65
        c_blocked = score_c > BLOCK_SIGMA
        l_blocked = score_l > BLOCK_SIGMA
        r_blocked = score_r > BLOCK_SIGMA

        estop_limit = 0.95
        avoid_limit = 1.95
        
        if cpu_temp >= self.cfg.thermals.critical_temp:
            estop_limit = 1.20
            avoid_limit = 2.30

        estop_condition = False
        avoid_condition = False
        target_obstacle_offset = 0.0
        closest_obstacle_dist = 99.0

        for target in self.active_detections_list:
            dist = target['dist']
            if target['stale_count'] > 0:
                dist *= (1.0 - 0.05 * target['stale_count'])

            bx1, _, bx2, _ = target['box']
            obj_center_x = (bx1 + bx2) / 2.0
            norm_offset = (obj_center_x - (self.cfg.frame_w / 2.0)) / (self.cfg.frame_w / 2.0)

            if dist < estop_limit:
                if abs(norm_offset) < 0.45:
                    estop_condition = True
            elif dist < avoid_limit:
                if abs(norm_offset) < 0.55:
                    c_blocked = True
                avoid_condition = True
                if dist < closest_obstacle_dist:
                    closest_obstacle_dist = dist
                    target_obstacle_offset = norm_offset

        instant_state_vote = VehicleState.FORWARD_CRUISE
        if estop_condition or (c_blocked and l_blocked and r_blocked):
            instant_state_vote = VehicleState.EMERGENCY_STOP
        elif c_blocked or avoid_condition:
            instant_state_vote = VehicleState.CRITICAL_AVOIDANCE

        self.state_consensus_history.append(instant_state_vote)
        if len(self.state_consensus_history) > 3:
            self.state_consensus_history.pop(0)

        if instant_state_vote == VehicleState.EMERGENCY_STOP:
            self.current_state = VehicleState.EMERGENCY_STOP
            self.last_state_change = now
        else:
            if len(self.state_consensus_history) == 3 and all(s == instant_state_vote for s in self.state_consensus_history):
                if (now - self.last_state_change) >= 0.25:
                    self.current_state = instant_state_vote
                    self.last_state_change = now

        if self.current_state == VehicleState.EMERGENCY_STOP:
            return VehicleState.EMERGENCY_STOP, 0.0, 0.0
            
        elif self.current_state == VehicleState.CRITICAL_AVOIDANCE:
            kp_steering = 0.65
            if avoid_condition and closest_obstacle_dist < avoid_limit:
                steer_command = -np.sign(target_obstacle_offset) * (kp_steering * (avoid_limit - closest_obstacle_dist))
            else:
                steer_command = -0.60 if score_l < score_r else 0.60
                
            steer_command = float(np.clip(steer_command, -1.0, 1.0))
            velocity_scale = 0.30 if cpu_temp >= self.cfg.thermals.critical_temp else 0.45
            return VehicleState.CRITICAL_AVOIDANCE, velocity_scale, steer_command
            
        else:
            steer_command = -0.25 if l_blocked else (0.25 if r_blocked else 0.0)
            velocity_scale = 0.50 if cpu_temp >= self.cfg.thermals.critical_temp else 0.85
            if steer_command != 0.0:
                velocity_scale *= 0.70
            return VehicleState.FORWARD_CRUISE, velocity_scale, steer_command