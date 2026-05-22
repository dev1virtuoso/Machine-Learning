from __future__ import annotations

import time
import json
import socket
import sys
import yaml
import struct
import math
import argparse
import os
import threading
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
from typing import Iterable, List, Tuple, Dict, Union, Optional, Deque, Any

import numpy as np
import serial
import serial.tools.list_ports
import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

CONFIG_YAML = robot_config.yaml

@dataclass
class JointConfig:
    port_id: str
    node_id: int
    motor_idx: int
    motor_type: int
    direction_sign: int
    zero_offset: int
    min_limit: int
    max_limit: int


class RobotConfig:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.joints: Dict[str, JointConfig] = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                raw = yaml.safe_load(f)
        else:
            raw = yaml.safe_load(CONFIG_YAML)
            
        for joint_name, info in raw.get("joints", {}).items():
            self.joints[joint_name] = JointConfig(
                port_id=info.get("port_id", "/dev/ttyUSB0"),
                node_id=info["node_id"],
                motor_idx=info["motor_idx"],
                motor_type=info.get("motor_type", 1),
                direction_sign=info.get("direction_sign", 1),
                zero_offset=info.get("zero_offset", 1500),
                min_limit=info.get("min_limit", 1000),
                max_limit=info.get("max_limit", 2000)
            )

    def get_joint(self, name: str) -> JointConfig:
        if name not in self.joints:
            raise KeyError(f"Joint mapping '{name}' not specified in the active robot topology configuration.")
        return self.joints[name]


@dataclass
class MotorCommand:
    node_id: int
    motor_type: int   # 0x01: Joint Motor (CAN), 0x02: PWM Servo
    motor_idx: int
    value: int
    duration: int = 0

    def to_payload(self) -> bytes:
        hi = (self.value >> 8) & 0xFF
        lo = self.value & 0xFF
        return bytes([self.node_id, self.motor_type, self.motor_idx, hi, lo])

def crc8_itu_v1600(data: bytes) -> int:
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc << 1) ^ 0x07 if crc & 0x80 else crc << 1
            crc &= 0xFF
    return crc


def build_frame(payload: bytes) -> bytes:
    return bytes([0xAA]) + payload + bytes([crc8_itu_v1600(payload), 0x55])


def frame_to_hex(frame: bytes, *, include_checksum: bool = True) -> str:
    if include_checksum:
        return " ".join(f"{b:02X}" for b in frame)
    if len(frame) < 2:
        return " ".join(f"{b:02X}" for b in frame)
    return " ".join(f"{b:02X}" for b in frame[:-2])


def autodetect_serial_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "USB" in p.device or "ACM" in p.device or "serial" in p.description.lower():
            return p.device
    return "/dev/ttyUSB0"


def serial_port(
    port: Union[str, Path],
    baudrate: int = 115200,
    timeout: float = 0.1,
    settle: float = 1.0,
) -> serial.Serial:
    try:
        s = serial.Serial(str(port), baudrate, timeout=timeout)
        if settle > 0:
            time.sleep(settle)
        return s
    except serial.SerialException as exc:
        raise RuntimeError(f"[-] Failed to bind serial bus connection on interface {port}: {exc}")


def read_exact(stream: Any, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError(f"Unexpected stream cut while attempting to extract {n} bytes (read {len(buf)}).")
        buf.extend(chunk)
    return bytes(buf)

class HeraclesMaster:
    def __init__(
        self, 
        robot_config: RobotConfig,
        baudrate: int = 115200, 
        timeout: float = 0.1,
        *,
        verbose: bool = False
    ):
        self.config = robot_config
        self.verbose = verbose
        
        self._tx_locks: Dict[str, threading.Lock] = {}
        self._buses: Dict[str, serial.Serial] = {}
        self._states_lock = threading.Lock()
        
        active_ports = {j.port_id for j in self.config.joints.values()}
        if not active_ports:
            active_ports.add(autodetect_serial_port())

        for port in active_ports:
            try:
                self._buses[port] = serial_port(port, baudrate, timeout)
                self._tx_locks[port] = threading.Lock()
            except RuntimeError as err:
                print(f"[!] Target bus {port} initialization failed, spinning mock placeholder: {err}")
                self._buses[port] = None
                self._tx_locks[port] = threading.Lock()

        self.motor_states: Dict[Tuple[str, int, int], dict] = {}
        
        self._stop_rx = threading.Event()
        self._rx_threads: List[threading.Thread] = []
        for port, stream in self._buses.items():
            if stream is not None:
                t = threading.Thread(target=self._rx_loop, args=(port, stream), daemon=True)
                t.start()
                self._rx_threads.append(t)
        
        print(f"[+] Heracles Multi-Bus Matrix active on interfaces: {list(self._buses.keys())}")

    def send_joint_command(self, joint_name: str, algorithmic_offset: float) -> None:
        j = self.config.get_joint(joint_name)
        raw_val = int(j.zero_offset + (j.direction_sign * algorithmic_offset))
        
        raw_val = max(j.min_limit, min(j.max_limit, raw_val))
        cmd = MotorCommand(j.node_id, j.motor_type, j.motor_idx, raw_val)
        
        self.dispatch_raw_command(j.port_id, cmd)

    def dispatch_raw_command(self, port_id: str, cmd: MotorCommand) -> None:
        bus = self._buses.get(port_id)
        if bus is None:
            return
            
        payload = cmd.to_payload()
        frame = build_frame(payload)
        
        lock = self._tx_locks[port_id]
        with lock:
            try:
                bus.setRTS(True)
                bus.write(frame)
                bus.flush()
                bus.setRTS(False)
            except Exception as e:
                if self.verbose:
                    print(f"[!] Write failure on channel {port_id}: {e}")
            
        if self.verbose:
            print(f"[->][{port_id}] Sent: {frame_to_hex(frame)}")

    def get_joint_state(self, joint_name: str) -> Tuple[float, float, float]:
        j = self.config.get_joint(joint_name)
        with self._states_lock:
            state = self.motor_states.get((j.port_id, j.node_id, j.motor_idx))
            if state is not None:
                norm_pos = (state["position"] - j.zero_offset) * j.direction_sign
                norm_vel = state["velocity"] * j.direction_sign
                norm_trq = state["torque"] * j.direction_sign
                return float(norm_pos), float(norm_vel), float(norm_trq)
            return 0.0, 0.0, 0.0

    def _rx_loop(self, port_id: str, stream: serial.Serial) -> None:
        buffer = bytearray()
        expected_len = 10 
        
        while not self._stop_rx.is_set():
            try:
                if stream.in_waiting > 0:
                    data = stream.read(stream.in_waiting)
                    buffer.extend(data)
                    
                    while True:
                        sof_idx = buffer.find(0xAA)
                        if sof_idx == -1:
                            buffer.clear()
                            break
                        if sof_idx > 0:
                            buffer = buffer[sof_idx:]
                        if len(buffer) < expected_len:
                            break
                            
                        if buffer[expected_len - 1] == 0x55:
                            payload = bytes(buffer[1:8])
                            if crc8_itu_v1600(payload) == buffer[8]:
                                self._unpack_telemetry(port_id, payload)
                                buffer = buffer[expected_len:]
                                continue
                        buffer = buffer[1:]
                else:
                    time.sleep(0.001)
            except Exception as exc:
                if self.verbose:
                    print(f"[!] Warning: Error inside serial telemetry reader on {port_id}: {exc}")
                time.sleep(0.01)

    def _unpack_telemetry(self, port_id: str, payload: bytes) -> None:
        node_id = payload[0]
        telemetry_type = payload[1] 
        motor_idx = payload[2]
        
        if telemetry_type == 0xAF:
            pos = int.from_bytes(payload[3:5], byteorder='big', signed=True)
            torque = int.from_bytes(payload[5:7], byteorder='big', signed=True)
            current_time = time.perf_counter()
            vel = 0.0
            
            key = (port_id, node_id, motor_idx)
            with self._states_lock:
                prev = self.motor_states.get(key)
                if prev is not None:
                    dt = current_time - prev["timestamp"]
                    if dt > 0:
                        vel = (pos - prev["position"]) / dt
                        
                self.motor_states[key] = {
                    "position": float(pos),
                    "velocity": float(vel),
                    "torque": float(torque),
                    "timestamp": current_time
                }

    def close(self) -> None:
        self._stop_rx.set()
        for t in self._rx_threads:
            if t.is_alive():
                t.join(timeout=0.5)
        for stream in self._buses.values():
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        print("[-] Heracles Matrix infrastructure collapsed cleanly.")

class HeraclesPureAlgorithmEstimator:
    def __init__(self, alpha: float = 0.85):
        self.alpha = alpha
        self.pressure_bias = 0.0
        self.contact_confidence = 0.0

    def compute_foot_pressure(self, joints_torque: List[float], velocities: List[float]) -> float:
        raw_torque_sum = sum(abs(t) for t in joints_torque)
        dynamic_loss = sum(abs(v) * 0.12 for v in velocities)
        estimated_load = max(0.0, raw_torque_sum - dynamic_loss)
        
        self.pressure_bias = (self.alpha * self.pressure_bias) + ((1.0 - self.alpha) * estimated_load)
        return float(self.pressure_bias)

    def evaluate_ground_contact(self, raw_pressure: float, target_threshold: float = 12.0) -> bool:
        """Determines binary ground contact status based on confidence levels."""
        if raw_pressure > target_threshold:
            self.contact_confidence = min(1.0, self.contact_confidence + 0.15)
        else:
            self.contact_confidence = max(0.0, self.contact_confidence - 0.20)
        return self.contact_confidence > 0.5

class WholeBodyController:
    def __init__(self, robot_config: RobotConfig):
        self.config = robot_config

    def enforce_safety_limits(self, joint_name: str, target_offset: float) -> float:
        j = self.config.get_joint(joint_name)
        raw_target = j.zero_offset + (j.direction_sign * target_offset)
        clamped_raw = max(j.min_limit, min(j.max_limit, raw_target))
        
        return (clamped_raw - j.zero_offset) * j.direction_sign

class ResidualCPGGaitEngine:
    def __init__(self, base_frequency: float = 1.5, amplitude: float = 250.0):
        self.freq = base_frequency
        self.amp = amplitude
        self.time_tracker = 0.0

    def step_gait_cycle(self, dt: float, model_residual: np.ndarray) -> Dict[str, float]:
        self.time_tracker += dt
        phase_offset = math.sin(2.0 * math.pi * self.freq * self.time_tracker)
        
        left_offset = self.amp * phase_offset
        right_offset = - (self.amp * phase_offset)
        
        if model_residual.size >= 2:
            left_offset += float(model_residual[0] * 50.0)
            right_offset += float(model_residual[1] * 50.0)
            
        return {
            "left_hip_pitch": left_offset,
            "left_knee_pitch": left_offset * 0.5,
            "right_hip_pitch": right_offset,
            "right_knee_pitch": right_offset * 0.5
        }

class CameraWorker:
    def __init__(
        self,
        cam_id: int = 0,
        resolution: Tuple[int, int] = (224, 224),
        interval: float = 0.033,
    ) -> None:
        self.cam_id = cam_id
        self.width, self.height = resolution
        self.interval = interval

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_virtual = not OPENCV_AVAILABLE

    def start(self) -> None:
        if self._is_virtual:
            print("[!] OpenCV libraries missing. Running synthetic visual block generation loop.")
            self._thread = threading.Thread(target=self._virtual_reader, daemon=True)
            self._thread.start()
            return

        self._thread = threading.Thread(target=self._hardware_reader, daemon=True)
        self._thread.start()

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _hardware_reader(self) -> None:
        cap = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cam_id)
            
        if not cap.isOpened():
            print("[!] Warning: Camera index unreachable. Falling back to synthetic array loops.")
            self._virtual_reader()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        next_capture = time.time()
        while not self._stop.is_set():
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (self.width, self.height))
                with self._lock:
                    self._frame = resized
            
            next_capture += self.interval
            sleep_dur = next_capture - time.time()
            if sleep_dur > 0:
                time.sleep(sleep_dur)
            else:
                next_capture = time.time()
        cap.release()

    def _virtual_reader(self) -> None:
        next_capture = time.time()
        while not self._stop.is_set():
            mock_block = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
            with self._lock:
                self._frame = mock_block
            
            next_capture += self.interval
            sleep_dur = next_capture - time.time()
            if sleep_dur > 0:
                time.sleep(sleep_dur)
            else:
                next_capture = time.time()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


def capture_camera_frame(cam_id: int = 0, resolution: Tuple[int, int] = (224, 224)) -> np.ndarray:
    if not OPENCV_AVAILABLE:
        return np.random.randint(0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8)
        
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return np.random.randint(0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.random.randint(0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8)

class PolicyWorker:
    def __init__(self, model_path: Union[Path, str], frame_shape: Tuple[int, int, int], device: str = "cpu"):
        self.model_path = Path(model_path)
        self.frame_shape = frame_shape
        self.device = device
        self._model: Optional[PPO] = None
        self._lock = threading.Lock()
        self._queue: deque[np.ndarray] = deque(maxlen=3)

    def push_frame(self, frame: np.ndarray) -> None:
        if frame.shape != self.frame_shape:
            raise ValueError(f"Expected frame shape {self.frame_shape}, received {frame.shape}")
        with self._lock:
            self._queue.append(frame)

    def predict(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._queue:
                return None
            frame = self._queue[-1].copy()

        if self._model is None:
            if not self.model_path.is_file():
                return None
            self._model = PPO.load(str(self.model_path), device=self.device)

        obs = frame.astype(np.float32) / 255.0
        obs = np.transpose(obs, (2, 0, 1))
        obs = np.expand_dims(obs, axis=0)

        action, _ = self._model.predict(obs, deterministic=True)
        return np.asarray(action[0], dtype=np.float32)

    def __enter__(self) -> PolicyWorker:
        if self.model_path.is_file():
            self._model = PPO.load(str(self.model_path), device=self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class HeraclesEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        robot_config: RobotConfig,
        target_joint: str = "left_knee_pitch",
        target_speed: float = 50.0,
        max_steps: int = 200
    ):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-500.0, high=500.0, shape=(1,), dtype=np.float32)
        
        self.config = robot_config
        self.target_joint = target_joint
        self.target_speed = target_speed
        self.max_steps = max_steps
        self.current_step = 0
        
        self.master = HeraclesMaster(self.config)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_step += 1
        target_offset = float(action[0])
        
        self.master.send_joint_command(self.target_joint, target_offset)
        time.sleep(0.02)
        
        norm_pos, _, _ = self.master.get_joint_state(self.target_joint)

        obs = np.array([norm_pos], dtype=np.float32)
        reward = -float(abs(norm_pos - self.target_speed))
        done = self.current_step >= self.max_steps
        
        return obs, reward, done, {"speed_cmd": self.target_speed}

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.master.send_joint_command(self.target_joint, 0.0)
        time.sleep(0.1)
        norm_pos, _, _ = self.master.get_joint_state(self.target_joint)
        return np.array([norm_pos], dtype=np.float32)

    def close(self) -> None:
        self.master.close()


class RLTrainer:
    def __init__(self, env: gym.Env, total_timesteps: int = 2000):
        self.env = env
        self.timesteps = total_timesteps
        self.model = PPO("MlpPolicy", DummyVecEnv([lambda: self.env]), verbose=0)

    def train(self, base_name: str) -> None:
        print(f"[->] Beginning PPO trajectory updates ({self.timesteps} steps)...")
        self.model.learn(total_timesteps=self.timesteps)
        self.model.save(base_name)
        print(f"[+] Model matrix successfully saved as {base_name}.zip")

    def evaluate(self, base_name: str) -> None:
        print(f"[->] Loading policies from {base_name}.zip for inference testing...")
        loaded_model = PPO.load(base_name)
        obs = self.env.reset()
        for i in range(10):
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, _ = self.env.step(action)
            print(f"  Step {i} -> Observation (Normalized Pos): {obs} | Reward: {reward:.2f}")
            if done:
                break

def run_server(master_instance: HeraclesMaster, host: str = "0.0.0.0", port: int = 40000) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(5)
        print(f"[+] Abstract Command forwarding server listening on {host}:{port}")
        
        try:
            while True:
                conn, addr = s.accept()
                print(f"[+] Connected to client session at {addr}")
                try:
                    with conn:
                        reader = conn.makefile("rb")
                        while True:
                            len_bytes = reader.read(4)
                            if not len_bytes or len(len_bytes) < 4:
                                break

                            msg_len = int.from_bytes(len_bytes, "big")
                            payload = read_exact(reader, msg_len)

                            try:
                                cmds = json.loads(payload.decode("utf-8"))
                                for c in cmds:
                                    if "joint_name" in c:
                                        master_instance.send_joint_command(
                                            c["joint_name"], float(c["value"])
                                        )
                                    else:
                                        legacy_cmd = MotorCommand(
                                            node_id=int(c["node_id"]),
                                            motor_type=int(c["motor_type"]),
                                            motor_idx=int(c["motor_idx"]),
                                            value=int(c["value"])
                                        )
                                        master_instance.dispatch_raw_command(
                                            c.get("port_id", "/dev/ttyUSB0"), legacy_cmd
                                        )
                                conn.sendall(b"ok")
                            except Exception as inner_exc:
                                err = f"err:{inner_exc}".encode("utf-8")
                                conn.sendall(err)
                                print(f"[!] Data decoding error: {inner_exc}")
                except Exception as conn_exc:
                    print(f"[!] Session tracking connection exception: {conn_exc}")
        except KeyboardInterrupt:
            print("\n[-] Shutting down network proxy pipeline.")

def demo_hardware_decoupled_motion(master_instance: HeraclesMaster) -> None:
    print("[->] Initiating multi-bus decoupled hardware synchronization sequence...")
    
    targets = ["left_hip_pitch", "left_knee_pitch", "right_hip_pitch", "right_knee_pitch"]
    
    for offset in [0.0, 150.0, -150.0, 0.0]:
        for joint in targets:
            master_instance.send_joint_command(joint, offset)
        time.sleep(0.2)
        
    print("[+] Framework trajectory baseline test execution complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Heracles Humanoid Platform Integrated Runtime Module")
    parser.add_argument("--config", type=str, default=None, help="Path to robot_config.yaml")
    parser.add_argument("--server", action="store_true", help="Launch TCP command server")
    parser.add_argument("--train", action="store_true", help="Execute RL gait alignment optimizations")
    parser.add_argument("--eval", action="store_true", help="Evaluate locally checked-in policy maps")
    parser.add_argument("--timesteps", type=int, default=2000, help="RL training steps allocation limits")
    parser.add_argument("--model", type=str, default="heracles_ppo_policy", help="Model file target path identifier")
    args = parser.parse_args()

    robot_config = RobotConfig(args.config)

    if args.server:
        master = HeraclesMaster(robot_config, verbose=True)
        try:
            run_server(master)
        finally:
            master.close()
    elif args.train or args.eval:
        env = HeraclesEnv(robot_config)
        trainer = RLTrainer(env, total_timesteps=args.timesteps)
        try:
            if args.train:
                trainer.train(args.model)
            if args.eval:
                trainer.evaluate(args.model)
        finally:
            env.close()
    else:
        master = HeraclesMaster(robot_config, verbose=True)
        try:
            demo_hardware_decoupled_motion(master)
        finally:
            master.close()


if __name__ == "__main__":
    main()

import types
def _proxy_submodule(name: str, attr_names: List[str]):
    mod = types.ModuleType(name)
    for attr in attr_names:
        if attr in globals():
            setattr(mod, attr, globals()[attr])
    sys.modules[name] = mod

_proxy_submodule("heracles.utils", ["crc8_itu_v1600", "build_frame", "frame_to_hex", "autodetect_serial_port", "serial_port", "read_exact", "MotorCommand", "RobotConfig", "JointConfig"])
_proxy_submodule("heracles.master", ["HeraclesMaster"])
_proxy_submodule("heracles.camera_worker", ["CameraWorker", "capture_camera_frame"])
_proxy_submodule("heracles.policy_worker", ["PolicyWorker"])
_proxy_submodule("heracles.remote_sender", ["run_server"])
_proxy_submodule("heracles.state_estimator", ["HeraclesPureAlgorithmEstimator"])
_proxy_submodule("heracles.controllers.wbc", ["WholeBodyController"])
_proxy_submodule("heracles.train", ["HeraclesEnv", "RLTrainer"])