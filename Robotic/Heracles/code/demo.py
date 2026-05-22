from typing import List
from master import HeraclesMaster, MotorCommand

def demo_shoulder_motion(master_instance: HeraclesMaster) -> None:
    ops: List[MotorCommand] = []
    for i in range(10):
        val = 1500 + int(i * 20)
        ops.append(MotorCommand(1, 0x01, 0, val))
    for i in range(10, 0, -1):
        val = 1500 + int(i * 20)
        ops.append(MotorCommand(1, 0x01, 0, val))
    master_instance.send_commands(ops)