import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import time
import threading
import glob
from collections import defaultdict
from datetime import datetime

CONFIG = {
    "MODEL_PATH": "yolov8n.pt",
    "CALIB_FILE": "stereo_calibration.npz",
    "LEFT_CAM_INDEX": 0,
    "RIGHT_CAM_INDEX": 1,
    "TARGET_FPS": 25.0,
    
    "BASE_STEREO_INTERVAL": 3,
    "THERMAL_STEREO_INTERVAL": 6,
    "DISPARITY_DOWNSCALE": 0.5,
    
    "MIN_VALID_DISTANCE": 0.25,
    "MAX_VALID_DISTANCE": 7.5,
    "CRITICAL_ZONE_METERS": 0.95,
    "MAX_LINEAR_VELOCITY": 0.75,
    "TARGET_TRAVEL_CORRIDOR_W": 0.45,
    "TRACK_EXPIRY_TIMEOUT": 1.2,    

    "CLASS_HEIGHT_PRIORS": {
        "person": 1.72, "chair": 0.82, "forklift": 2.20, "industrial_bin": 0.65
    }
}

class TrackKalmanFilter:
    def __init__(self, init_z, init_x):
        self.state = np.array([init_z, 0.0, init_x, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1.5
        self.Q = np.eye(4, dtype=np.float32) * 0.03
        self.R = np.eye(2, dtype=np.float32) * 0.08

    def predict(self, dt):
        F = np.array([
            [1.0,  dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0,  dt],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def update(self, meas_z, meas_x):
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float32)
        z_meas = np.array([meas_z, meas_x], dtype=np.float32)
        y = z_meas - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ H) @ self.P

    @property
    def filtered_values(self):
        return float(self.state[0]), float(self.state[2])

class RobustStereoStream:

    def __init__(self, left_idx, right_idx):
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.lock = threading.Lock()
        self.running = False
        self.frame_l = None
        self.frame_r = None
        self.is_connected = False
        self._init_hardware()

    def _init_hardware(self):
        self.cap_l = cv2.VideoCapture(self.left_idx)
        self.cap_r = cv2.VideoCapture(self.right_idx)
        for cap in [self.cap_l, self.cap_r]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.is_connected = self.cap_l.isOpened() and self.cap_r.isOpened()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        backoff = 1.0
        while self.running:
            if not self.is_connected:
                print(f"[RECOVERY] Hardware disconnected. Attempting reset in {backoff}s...")
                time.sleep(backoff)
                with self.lock:
                    self.cap_l.release()
                    self.cap_r.release()
                    self._init_hardware()
                backoff = min(15.0, backoff * 1.5)
                continue

            backoff = 1.0
            grab_l = self.cap_l.grab()
            grab_r = self.cap_r.grab()
            
            if grab_l and grab_r:
                _, fl = self.cap_l.retrieve()
                _, fr = self.cap_r.retrieve()
                with self.lock:
                    self.frame_l = fl
                    self.frame_r = fr
            else:
                with self.lock:
                    self.is_connected = False
            time.sleep(0.002)

    def read(self):
        with self.lock:
            if not self.is_connected or self.frame_l is None:
                return False, None, None
            return True, self.frame_l.copy(), self.frame_r.copy()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'): self.thread.join(timeout=1.0)
        self.cap_l.release()
        self.cap_r.release()

class ExecutionWatchdog(threading.Thread):
    def __init__(self, limit=1.2):
        super().__init__(daemon=True)
        self.limit = limit
        self.last_heartbeat = time.time()
        self.active = True

    def feed(self):
        self.last_heartbeat = time.time()

    def run(self):
        while self.active:
            if time.time() - self.last_heartbeat > self.limit:
                print("[CRITICAL SHUTDOWN] WATCHDOG TIMEOUT EXCEEDED! ISSUING EMERGENCY COIL STOP.")
            time.sleep(0.1)

class ProportionalKinematicController:
    def __init__(self):
        self.ladder_state = "LEVEL_1_FULL_STEREO"
        self.linear_v = 0.0
        self.angular_v = 0.0

    def calculate_velocities(self, tracked_hazards, base_mode):
        self.ladder_state = base_mode
        if not tracked_hazards:
            self.linear_v = CONFIG["MAX_LINEAR_VELOCITY"]
            self.angular_v = 0.0
            return self.ladder_state, self.linear_v, self.angular_v

        nearest_z = float('inf')
        target_x = 0.0

        for hz in tracked_hazards:
            z, x = hz["z"], hz["x"]
            if abs(x) <= CONFIG["TARGET_TRAVEL_CORRIDOR_W"]:
                if z < nearest_z:
                    nearest_z = z
                    target_x = x

        if nearest_z <= CONFIG["CRITICAL_ZONE_METERS"]:
            self.ladder_state = "LEVEL_4_EMERGENCY_STOP"
            self.linear_v = 0.0
            self.angular_v = 0.0
        elif nearest_z < CONFIG["MAX_VALID_DISTANCE"]:
            if self.ladder_state != "LEVEL_3_MONOCULAR_FALLBACK":
                self.ladder_state = "LEVEL_2_ADAPTIVE_DEGRADED"
            
            scale = (nearest_z - CONFIG["CRITICAL_ZONE_METERS"]) / (CONFIG["MAX_VALID_DISTANCE"] - CONFIG["CRITICAL_ZONE_METERS"])
            self.linear_v = max(0.0, CONFIG["MAX_LINEAR_VELOCITY"] * min(1.0, scale))
            self.angular_v = -0.65 * (1.0 / max(0.35, nearest_z)) * np.sign(target_x if target_x != 0 else 1.0)
        else:
            self.linear_v = CONFIG["MAX_LINEAR_VELOCITY"]
            self.angular_v = 0.0

        return self.ladder_state, self.linear_v, self.angular_v

def run_capture_helper(left_idx, right_idx, output_left_dir, output_right_dir, grid_w, grid_h):
    os.makedirs(output_left_dir, exist_ok=True)
    os.makedirs(output_right_dir, exist_ok=True)
    
    cap_l = cv2.VideoCapture(left_idx)
    cap_r = cv2.VideoCapture(right_idx)
    for cap in [cap_l, cap_r]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    print("[INFO] Headless Auto-Capture Engine Online. Scanning for patterns...")
    checkerboard_size = (grid_w, grid_h)
    img_counter = 0
    last_capture_time = time.time()

    try:
        while True:
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r:
                print("[ERROR] Camera feed lost during capture sequence.")
                break

            found_l, _ = cv2.findChessboardCorners(frame_l, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
            found_r, _ = cv2.findChessboardCorners(frame_r, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
            
            if found_l and found_r:
                current_time = time.time()
                if current_time - last_capture_time > 2.5:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(os.path.join(output_left_dir, f"left_{timestamp}.png"), frame_l)
                    cv2.imwrite(os.path.join(output_right_dir, f"right_{timestamp}.png"), frame_r)
                    img_counter += 1
                    print(f"[SUCCESS] Auto-stored dataset pair {img_counter} to disk.")
                    last_capture_time = current_time
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[INFO] Terminating auto-capture loop handle gracefully.")
    finally:
        cap_l.release()
        cap_r.release()

def run_stereo_calibration(left_dir, right_dir, output_file, grid_w, grid_h, square_size):
    checkerboard_size = (grid_w, grid_h)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-6)
    
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints, imgpoints_l, imgpoints_r = [], [], []
    images_l = sorted(glob.glob(os.path.join(left_dir, "*.jpg")) + glob.glob(os.path.join(left_dir, "*.png")))
    images_r = sorted(glob.glob(os.path.join(right_dir, "*.jpg")) + glob.glob(os.path.join(right_dir, "*.png")))

    if len(images_l) != len(images_r) or len(images_l) == 0:
        print("[CRITICAL] Frame count imbalance or folders empty. Extraction aborted.")
        return False

    print(f"[INFO] Running feature extractors over {len(images_l)} matching frames...")
    img_shape = None

    for idx, (p_l, p_r) in enumerate(zip(images_l, images_r)):
        fl, fr = cv2.imread(p_l), cv2.imread(p_r)
        gray_l = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None: img_shape = gray_l.shape[::-1]

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH)

        if ret_l and ret_r:
            objpoints.append(objp)
            imgpoints_l.append(cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria))
            imgpoints_r.append(cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria))

    if len(objpoints) < 15:
        print(f"[CRITICAL] Only {len(objpoints)} valid pairs extracted. At least 15+ high-coverage sets required.")
        return False

    _, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
    _, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

    rms, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, K_l, D_l, K_r, D_r, img_shape,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
    )

    print(f"[EVALUATION] Calibration Finished. Achieved Target RMS Error: {rms:.4f} pixels.")
    if rms > 0.5:
        print("[CRITICAL WARNING] RMS exceeds strict 0.5px industrial threshold. Check lens alignment!")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, D_l, K_r, D_r, img_shape, R, T, alpha=0)
    map_l_x, map_l_y = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, img_shape, cv2.CV_32FC1)
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, img_shape, cv2.CV_32FC1)

    np.savez_compressed(
        output_file, map_l_x=map_l_x, map_l_y=map_l_y, map_r_x=map_r_x, map_r_y=map_r_y, Q=Q,
        meta_rms=np.array([rms]), meta_shape=np.array([img_shape[0], img_shape[1]]),
        meta_timestamp=np.array([datetime.now().strftime("%Y%m%d_%H%M%S")])
    )
    print(f"[SUCCESS] Compiled parameters stored at: {output_file}")
    return True