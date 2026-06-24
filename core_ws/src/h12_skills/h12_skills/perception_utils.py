"""Small, dependency-light perception helpers for the grasp skill.

Self-contained geometry/IO helpers used by the skills:
  - `decode_compressed_depth_image` / `transform_to_matrix` decode a ROS2
    compressedDepth image and convert a geometry_msgs/Transform to a 4x4 matrix.
  - `deproject_mask` is the pinhole back-projection kernel (mask + depth -> XYZ).
Plus the rotation/quaternion helpers and a Gemini-JSON extractor the skill needs.
"""

import json
import math

import numpy as np
import cv2


# --------------------------------------------------------------- depth + cloud
def decode_compressed_depth_image(msg) -> np.ndarray:
    """Decode a ROS2 compressedDepth image (format '16UC1; compressedDepth') to a
    uint16 depth array (millimetres)."""
    if not msg.format.lower().endswith("compresseddepth"):
        raise ValueError(f"Unsupported depth format: {msg.format}")
    header_size = 12  # 12-byte compressedDepth header precedes the PNG payload
    if len(msg.data) <= header_size:
        raise ValueError("compressedDepth data too short to contain a header")
    compressed_data = bytes(msg.data[header_size:])
    np_arr = np.frombuffer(compressed_data, dtype=np.uint8)
    depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError("cv2.imdecode failed on compressed depth image")
    if depth_image.dtype != np.uint16:
        raise TypeError(f"Expected uint16 depth, got {depth_image.dtype}")
    return depth_image


def deproject_mask(mask, depth_m, fx, fy, cx, cy, min_d=0.1, max_d=3.0):
    """Back-project the True pixels of `mask` (bool HxW) with finite depth into a
    (N, 3) float32 XYZ array in the camera optical frame. Standard pinhole
    back-projection; drops invalid/out-of-range depth.
    `depth_m` is HxW depth in metres aligned to the mask's pixel grid."""
    valid = mask & (depth_m > min_d) & (depth_m < max_d)
    vs, us = np.nonzero(valid)              # vs = rows (y), us = cols (x)
    z = depth_m[valid].astype(np.float32)
    x = ((us - cx) * z / fx).astype(np.float32)
    y = ((vs - cy) * z / fy).astype(np.float32)
    return np.stack([x, y, z], axis=-1)


def transform_points(pts, T):
    """Apply a 4x4 homogeneous transform `T` to an (N, 3) point array."""
    return (pts @ T[:3, :3].T) + T[:3, 3]


# ----------------------------------------------------------- rotations / poses
def transform_to_matrix(tf_msg):
    """Convert a geometry_msgs/Transform into a 4x4 numpy array."""
    tx, ty, tz = tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z
    qx, qy, qz, qw = (tf_msg.rotation.x, tf_msg.rotation.y,
                      tf_msg.rotation.z, tf_msg.rotation.w)
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T


def quat_to_rot(qx, qy, qz, qw):
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ])


def mat_to_quat(R):
    """3x3 rotation matrix -> quaternion (x, y, z, w). Shepperd's method."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    return (x, y, z, w)


def pose_to_matrix(pose):
    """geometry_msgs/Pose -> 4x4 homogeneous matrix."""
    p, q = pose.position, pose.orientation
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


# ------------------------------------------------------------------- gemini io
def extract_json(text):
    """Best-effort extraction of a JSON value from a Gemini text response: strips
    ```json fences, else falls back to the outermost [...] / {...} span. Returns
    the parsed object, or None."""
    if not text:
        return None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            text = "\n".join(lines[i + 1:]).split("```")[0]
            break
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        a, b = text.find(opener), text.rfind(closer)
        if 0 <= a < b:
            try:
                return json.loads(text[a:b + 1])
            except (json.JSONDecodeError, ValueError):
                continue
    return None
