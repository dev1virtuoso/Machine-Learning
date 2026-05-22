import threading
import time
import serial
from typing import Optional, Dict, Tuple, List

class MotorCommand:
    def __init__(self, node_id: int, motor_type: int, motor_idx: int, value: int):
        self.node_id = node_id
        self.motor_type = motor_type
        self.motor_idx = motor_idx
        self.value = value

    def to_payload(self) -> bytes:
        hi = (self.value >> 8) & 0xFF
        lo = self.value & 0xFF
        return bytes([self.node_id, self.motor_type, self.motor_idx, hi, lo])


def crc8_itu_v1600(data: bytes) -> int:
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if (crc & 0x80) else (crc << 1)
            crc &= 0xFF
    return crc


def build_frame(payload: bytes) -> bytes:
    return bytes([0xAA]) + payload + bytes([crc8_itu_v1600(payload), 0x55])


class HeraclesMaster:
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 0.05):
        try:
            self._ser = serial.Serial(port or "/dev/ttyUSB0", baudrate, timeout=timeout)
            print(f"[+] Connected to {self._ser.port}")
        except Exception as e:
            print(f"[!] Serial failed: {e}. Running in virtual mode.")
            self._ser = None

        self._tx_lock = threading.Lock()
        self.motor_states: Dict[Tuple[int, int], dict] = {}
        self._states_lock = threading.Lock()
        self._stop_rx = threading.Event()
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

    def _tx_mode(self):
        if self._ser: self._ser.setRTS(True)

    def _rx_mode(self):
        if self._ser: self._ser.setRTS(False)

    def send_motor_command(self, cmd: MotorCommand):
        payload = cmd.to_payload()
        frame = build_frame(payload)
        with self._tx_lock:
            if self._ser:
                self._tx_mode()
                self._ser.write(frame)
                self._ser.flush()
                self._rx_mode()

    def send_commands(self, cmds: List[MotorCommand]):
        for cmd in cmds:
            self.send_motor_command(cmd)

    def _rx_loop(self):
        buffer = bytearray()
        while not self._stop_rx.is_set():
            if self._ser and self._ser.in_waiting:
                buffer.extend(self._ser.read(self._ser.in_waiting))
            while len(buffer) >= 8:
                sof = buffer.find(0xAA)
                if sof == -1 or len(buffer) - sof < 8:
                    break
                if buffer[sof + 7] == 0x55:
                    payload = bytes(buffer[sof+1:sof+6])
                    if crc8_itu_v1600(payload) == buffer[sof+6]:
                        self._unpack_telemetry(payload)
                    buffer = buffer[sof+8:]
                else:
                    buffer = buffer[sof+1:]
            time.sleep(0.001)

    def _unpack_telemetry(self, payload: bytes):
        if len(payload) < 3 or payload[1] != 0xAF:
            return
        node_id = payload[0]
        motor_idx = payload[2]
        pos = int.from_bytes(payload[3:5], 'big', signed=True)
        torque = int.from_bytes(payload[5:7], 'big', signed=True) if len(payload) >= 7 else 0

        with self._states_lock:
            self.motor_states[(node_id, motor_idx)] = {
                "position": pos,
                "torque": torque,
                "timestamp": time.perf_counter()
            }

    def get_motor_state(self, node_id: int, motor_idx: int) -> Tuple[float, float]:
        with self._states_lock:
            state = self.motor_states.get((node_id, motor_idx))
            return (state["position"], state["torque"]) if state else (1500.0, 0.0)

    def close(self):
        self._stop_rx.set()
        if self._ser and self._ser.is_open:
            self._ser.close()