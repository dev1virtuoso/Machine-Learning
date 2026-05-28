import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import time
import argparse
from collections import defaultdict
from module import (
    CONFIG,
    TrackKalmanFilter,
    RobustStereoStream,
    ExecutionWatchdog,
    ProportionalKinematicController,
    run_capture_helper,
    run_stereo_calibration
)

def main():
    parser = argparse.ArgumentParser(description="AGV Headless Production System Engine Orchestrator")
    parser.add_argument("--mode", type=str, choices=["run", "capture", "calibrate"], default="run")
    parser.add_argument("--left-dir", type=str, default="calib_left")
    parser.add_argument("--right-dir", type=str, default="calib_right")
    parser.add_argument("--left-idx", type=int, default=CONFIG["LEFT_CAM_INDEX"])
    parser.add_argument("--right-idx", type=int, default=CONFIG["RIGHT_CAM_INDEX"])
    parser.add_argument("--output", type=str, default=CONFIG["CALIB_FILE"])
    parser.add_argument("--grid-w", type=int, default=9)
    parser.add_argument("--grid-h", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=0.025)
    args = parser.parse_args()

    if args.mode == "capture":
        run_capture_helper(args.left_idx, args.right_idx, args.left_dir, args.right_dir, args.grid_w, args.grid_h)
        return
    elif args.mode == "calibrate":
        run_stereo_calibration(args.left_dir, args.right_dir, args.output, args.grid_w, args.grid_h, args.square_size)
        return

    print("[SYSTEM] Booting AGV Core Stereo Vision Engine in Headless Mode...")
    BASE_OPERATIONAL_MODE = "LEVEL_1_FULL_STEREO"
    map_l_x = map_l_y = map_r_x = map_r_y = Q = None
    
    if os.path.exists(CONFIG["CALIB_FILE"]):
        try:
            with np.load(CONFIG["CALIB_FILE"]) as data:
                map_l_x, map_l_y = data['map_l_x'], data['map_l_y']
                map_r_x, map_r_y = data['map_r_x'], data['map_r_y']
                Q = data['Q']
            FOCAL_LENGTH = abs(Q[2, 3])
            BASELINE = 1.0 / abs(Q[3, 2])
            print(f"[SUCCESS] Metrics Bound. Baseline: {BASELINE*100:.2f}cm, Focal: {FOCAL_LENGTH:.1f}px")
        except Exception:
            BASE_OPERATIONAL_MODE = "LEVEL_3_MONOCULAR_FALLBACK"
    else:
        print("[WARNING] Calibration missing. Forcing LEVEL_3_MONOCULAR_FALLBACK.")
        BASE_OPERATIONAL_MODE = "LEVEL_3_MONOCULAR_FALLBACK"
        FOCAL_LENGTH, BASELINE = 715.0, 0.065

    model = YOLO(CONFIG["MODEL_PATH"])
    cameras = RobustStereoStream(args.left_idx, args.right_idx)
    cameras.start()
    
    watchdog = ExecutionWatchdog()
    watchdog.start()

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=64, blockSize=5,
        P1=8*3*25, P2=32*3*25, disp12MaxDiff=1, uniquenessRatio=11,
        speckleWindowSize=80, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    try:
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(8500)
        wls_filter.setSigma(1.7)
        has_wls = True
    except AttributeError:
        has_wls = False

    controller = ProportionalKinematicController()
    active_filters = defaultdict(lambda: {"filter": None, "last_seen": time.time()})
    
    z_map = x_map = prev_gray_l = None
    frame_idx = 0
    last_loop_timestamp = time.time()

    try:
        while True:
            loop_start = time.time()
            watchdog.feed()

            ret, frame_l, frame_r = cameras.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_idx += 1
            dt = loop_start - last_loop_timestamp
            last_loop_timestamp = loop_start

            if BASE_OPERATIONAL_MODE == "LEVEL_1_FULL_STEREO" and map_l_x is not None:
                frame_l = cv2.remap(frame_l, map_l_x, map_l_y, cv2.INTER_LINEAR)
                frame_r = cv2.remap(frame_r, map_r_x, map_r_y, cv2.INTER_LINEAR)

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            current_fps = 1.0 / max(0.001, dt)
            stereo_interval = CONFIG["BASE_STEREO_INTERVAL"] if current_fps > 18.0 else CONFIG["THERMAL_STEREO_INTERVAL"]

            run_heavy_stereo = (frame_idx % stereo_interval == 0) or z_map is None
            
            if BASE_OPERATIONAL_MODE == "LEVEL_1_FULL_STEREO" and run_heavy_stereo:
                fx = CONFIG["DISPARITY_DOWNSCALE"]
                g_l_s = cv2.resize(gray_l, (0,0), fx=fx, fy=fx, interpolation=cv2.INTER_AREA)
                g_r_s = cv2.resize(gray_r, (0,0), fx=fx, fy=fx, interpolation=cv2.INTER_AREA)
                
                if has_wls:
                    d_l = left_matcher.compute(g_l_s, g_r_s)
                    d_r = right_matcher.compute(g_r_s, g_l_s)
                    disp = wls_filter.filter(d_l, g_l_s, disparity_map_right=d_r)
                else:
                    disp = left_matcher.compute(g_l_s, g_r_s)

                disp = cv2.resize(disp, (gray_l.shape[1], gray_l.shape[0]), interpolation=cv2.INTER_NEAREST)
                true_disparity = (disp.astype(np.float32) / 16.0) * (1.0 / fx)
                
                points_3D = cv2.reprojectImageTo3D(true_disparity, Q)
                z_map = points_3D[:, :, 2]
                x_map = points_3D[:, :, 0]
                
            elif BASE_OPERATIONAL_MODE == "LEVEL_1_FULL_STEREO" and prev_gray_l is not None:
                feat_prev = cv2.goodFeaturesToTrack(prev_gray_l, maxCorners=80, qualityLevel=0.02, minDistance=12)
                if feat_prev is not None and len(feat_prev) > 6:
                    feat_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray_l, gray_l, feat_prev, None)
                    if feat_curr is not None:
                        idx_valid = np.where(status == 1)[0]
                        if len(idx_valid) > 6:
                            H_homog, _ = cv2.findHomography(feat_prev[idx_valid], feat_curr[idx_valid], cv2.RANSAC, 4.0)
                            if H_homog is not None:
                                z_map = cv2.warpPerspective(z_map, H_homog, (gray_l.shape[1], gray_l.shape[0]), flags=cv2.INTER_NEAREST, borderValue=np.inf)
                                x_map = cv2.warpPerspective(x_map, H_homog, (gray_l.shape[1], gray_l.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0.0)

            if z_map is not None:
                z_map[(z_map < CONFIG["MIN_VALID_DISTANCE"]) | (z_map > CONFIG["MAX_VALID_DISTANCE"]) | np.isnan(z_map) | np.isinf(z_map)] = np.inf
            prev_gray_l = gray_l.copy()

            results = model.track(frame_l, persist=True, conf=0.32, verbose=False)
            hazards_in_frame = []

            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    track_id = int(box.id[0].item()) if box.id is not None else None
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_l.shape[1]-1, x2), min(frame_l.shape[0]-1, y2)
                    box_w, box_h = (x2 - x1), (y2 - y1)

                    r_y1, r_y2 = int(y1 + (box_h * 0.65)), int(y2 - (box_h * 0.05))
                    r_x1, r_x2 = int(x1 + (box_w * 0.20)), int(x2 - (box_w * 0.20))

                    raw_z, raw_x = np.inf, 0.0
                    if BASE_OPERATIONAL_MODE == "LEVEL_1_FULL_STEREO" and z_map is not None and (r_y2 > r_y1) and (r_x2 > r_x1):
                        z_roi = z_map[r_y1:r_y2, r_x1:r_x2]
                        valid_zs = z_roi[np.isfinite(z_roi)]
                        if len(valid_zs) > 8:
                            q25, q75 = np.percentile(valid_zs, [25, 75])
                            iqr = q75 - q25
                            clean_zs = valid_zs[(valid_zs >= q25 - 1.5*iqr) & (valid_zs <= q75 + 1.5*iqr)]
                            if len(clean_zs) > 0:
                                raw_z = np.median(clean_zs)
                                raw_x = np.median(x_map[r_y1:r_y2, r_x1:r_x2][np.isfinite(z_map[r_y1:r_y2, r_x1:r_x2])])

                    if np.isinf(raw_z) or raw_z <= 0:
                        prior_h = CONFIG["CLASS_HEIGHT_PRIORS"].get(class_name, 1.1)
                        raw_z = (prior_h * FOCAL_LENGTH) / max(1, box_h)
                        raw_x = (((x1 + (box_w / 2.0)) - (frame_l.shape[1] / 2.0)) * raw_z) / FOCAL_LENGTH

                    if track_id is not None:
                        if active_filters[track_id]["filter"] is None:
                            active_filters[track_id]["filter"] = TrackKalmanFilter(raw_z, raw_x)
                        
                        kf = active_filters[track_id]["filter"]
                        kf.predict(dt)
                        kf.update(raw_z, raw_x)
                        smooth_z, smooth_x = kf.filtered_values
                        active_filters[track_id]["last_seen"] = time.time()
                    else:
                        smooth_z, smooth_x = raw_z, raw_x

                    hazards_in_frame.append({"z": smooth_z, "x": smooth_x})

            now = time.time()
            stale = [k for k, v in active_filters.items() if now - v["last_seen"] > CONFIG["TRACK_EXPIRY_TIMEOUT"]]
            for k in stale: del active_filters[k]

            active_mode, cmd_v, cmd_w = controller.calculate_velocities(hazards_in_frame, BASE_OPERATIONAL_MODE)

            # Performance logging metrics optimized for remote console monitoring over SSH
            if frame_idx % 25 == 0:
                print(f"[STATUS] State: {active_mode} | Active Targets: {len(hazards_in_frame)} | "
                      f"Linear: {cmd_v:.2f} m/s | Angular: {cmd_w:.2f} rad/s | Performance: {current_fps:.1f} FPS")

            loop_duration = time.time() - loop_start
            delay = (1.0 / CONFIG["TARGET_FPS"]) - loop_duration
            if delay > 0: time.sleep(delay)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt caught. System shutting down gracefully...")
    except Exception as runtime_fault:
        print(f"[CRITICAL MALFUNCTION]: {runtime_fault}")
    finally:
        print("[INFO] Releasing physical hardware handles and stopping system threads...")
        watchdog.active = False
        cameras.stop()

if __name__ == "__main__":
    main()