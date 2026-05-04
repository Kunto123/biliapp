import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

import config
from camera_manager import CameraManager


class CameraRotationTests(unittest.TestCase):
    def test_config_rotation_accepts_supported_values(self):
        for value in ("0", "90", "180", "270"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"BILIRUBIN_CAMERA_ROTATION_TEST": value}):
                    self.assertEqual(
                        config._env_rotation("BILIRUBIN_CAMERA_ROTATION_TEST", 180),
                        int(value),
                    )

    def test_config_rotation_falls_back_on_invalid_values(self):
        for value in ("45", "-90", "bad"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"BILIRUBIN_CAMERA_ROTATION_TEST": value}):
                    self.assertEqual(
                        config._env_rotation("BILIRUBIN_CAMERA_ROTATION_TEST", 180),
                        180,
                    )

    def test_rpicam_command_includes_rotation(self):
        source = np.zeros((2, 3, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", source)
        self.assertTrue(ok)

        proc = mock.Mock()
        proc.returncode = 0
        proc.stdout = encoded.tobytes()
        proc.stderr = b""

        camera = CameraManager.__new__(CameraManager)
        camera._rpicam_cmd = "rpicam-still"
        camera.resolution = (3, 2)
        camera.rotation = 180
        camera.timeout_seconds = 1
        camera.error_message = None

        with mock.patch("camera_manager.subprocess.run", return_value=proc) as run:
            frame = camera._capture_libcamera_frame()

        cmd = run.call_args.args[0]
        rotation_index = cmd.index("--rotation")
        self.assertEqual(cmd[rotation_index + 1], "180")
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[:2], (2, 3))

    def test_opencv_rotation_applies_after_capture(self):
        camera = CameraManager.__new__(CameraManager)
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))

        camera.rotation = 90
        rotated_90 = camera._apply_rotation(frame)
        self.assertEqual(rotated_90.shape, (3, 2, 3))
        np.testing.assert_array_equal(rotated_90, cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))

        camera.rotation = 180
        rotated_180 = camera._apply_rotation(frame)
        self.assertEqual(rotated_180.shape, (2, 3, 3))
        np.testing.assert_array_equal(rotated_180, cv2.rotate(frame, cv2.ROTATE_180))


if __name__ == "__main__":
    unittest.main()
