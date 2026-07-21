"""Camera geometry shared by the RoboCasa spatial-memory benchmark.

VLMaps expects metric depth and OpenCV-style camera coordinates: x points
right, y points down, and z points forward.  MuJoCo exposes the rendered
camera using world-space ``forward`` and ``up`` vectors, so keeping the
conversion here makes the convention explicit and independently testable.
"""

from __future__ import annotations

import math

import numpy as np


CAMERA_FRAME = "opencv_x_right_y_down_z_forward"


def camera_intrinsics_from_fovy(
    width: int,
    height: int,
    fovy_deg: float,
) -> np.ndarray:
    """Return a pinhole intrinsic matrix for square MuJoCo pixels."""
    if width <= 0 or height <= 0:
        raise ValueError("camera dimensions must be positive")
    if not 0.0 < float(fovy_deg) < 180.0:
        raise ValueError("vertical field of view must be between 0 and 180 degrees")
    focal = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return np.asarray(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def camera_to_world_from_mujoco(
    position: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Convert an ``MjvGLCamera`` pose to an OpenCV camera-to-world matrix."""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    forward = np.asarray(forward, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    forward_norm = np.linalg.norm(forward)
    up_norm = np.linalg.norm(up)
    if forward_norm <= 1e-12 or up_norm <= 1e-12:
        raise ValueError("camera forward and up vectors must be non-zero")
    forward = forward / forward_norm
    up = up / up_norm
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-12:
        raise ValueError("camera forward and up vectors must not be parallel")
    right = right / right_norm
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = np.column_stack((right, down, forward))
    camera_to_world[:3, 3] = position
    return camera_to_world


def backproject_depth(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Backproject a metric depth image into OpenCV camera coordinates.

    The returned array is ``H x W x 3``. Pixels whose depth is zero remain at
    the camera origin and should be removed by the caller's validity mask.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    if depth_m.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    height, width = depth_m.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    pixels = np.stack(
        (columns + 0.5, rows + 0.5, np.ones_like(columns)), axis=-1
    )
    rays = pixels @ np.linalg.inv(intrinsics).T
    return rays * depth_m[..., None]
