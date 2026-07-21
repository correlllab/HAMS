import math
import unittest

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"NumPy runtime is unavailable: {exc}") from exc

from h1_robocasa.spatial_memory_geometry import (
    backproject_depth,
    camera_intrinsics_from_fovy,
    camera_to_world_from_mujoco,
)


class SpatialMemoryGeometryTests(unittest.TestCase):
    def test_intrinsics_use_vertical_fov_and_square_pixels(self):
        matrix = camera_intrinsics_from_fovy(640, 480, 60.0)
        expected_focal = 240.0 / math.tan(math.radians(30.0))
        self.assertAlmostEqual(matrix[0, 0], expected_focal)
        self.assertAlmostEqual(matrix[1, 1], expected_focal)
        self.assertEqual(matrix[0, 2], 320.0)
        self.assertEqual(matrix[1, 2], 240.0)

    def test_mujoco_pose_uses_right_down_forward_columns(self):
        matrix = camera_to_world_from_mujoco(
            position=np.asarray([1.0, 2.0, 3.0]),
            forward=np.asarray([1.0, 0.0, 0.0]),
            up=np.asarray([0.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(matrix[:3, 0], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(matrix[:3, 1], [0.0, 0.0, -1.0])
        np.testing.assert_allclose(matrix[:3, 2], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(np.linalg.det(matrix[:3, :3]), 1.0)

    def test_backprojection_follows_opencv_axes(self):
        depth = np.asarray([[2.0]], dtype=np.float32)
        intrinsics = np.asarray(
            [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]
        )
        points = backproject_depth(depth, intrinsics)
        np.testing.assert_allclose(points[0, 0], [0.0, 0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
