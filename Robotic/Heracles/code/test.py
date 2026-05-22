import unittest
import numpy as np
from master import crc8_itu_v1600, build_frame, MotorCommand
from master import HeraclesMaster

class TestHeracles(unittest.TestCase):
    def test_crc(self):
        self.assertEqual(crc8_itu_v1600(b"\x01\x02\x03\x04\x05"), 0xBC)

    def test_frame(self):
        p = b"\x01\x02\x03"
        f = build_frame(p)
        self.assertEqual(f[0], 0xAA)
        self.assertEqual(f[-1], 0x55)

    def test_motor_command(self):
        cmd = MotorCommand(1, 0x01, 0, 1600)
        self.assertEqual(cmd.to_payload()[0], 1)

if __name__ == "__main__":
    unittest.main()