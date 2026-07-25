#!/usr/bin/env python3
"""Per-run base-sway / manipulation telemetry recorder for the skills node.

Mirrors the fridge sway study's recorder (``trial_NN_telemetry.csv``) so a
battery-task run and a fridge run are measured the *same way* and are directly
comparable (see ``sway_analysis.py`` / ``GRASP_DATA_DEEPDIVE.md`` on the
``test/grasping`` branch). A background thread samples, at ~10 Hz wall clock:

  * the robot BASE pose  — ``pelvis`` expressed in the odometry world frame,
  * the grasping-arm gripper point — its GraspGenX frame in ``pelvis``,
  * the task target — a fixed world point (the screw / handle),

and writes one CSV row per tick. Each skill run gets its OWN directory under
``<package>/logs/runs/<skill>/<stamp>_<slug>/`` (host-persistent through the
core_ws bind mount, like the model logs):

    telemetry.csv   t_wall,pelvis_x,pelvis_y,pelvis_z,pelvis_yaw_deg,
                    grip_x,grip_y,grip_z,target_x,target_y,target_z
    result.json     the run outcome + knobs (success, message, arm, target, ...)

WHY the base pose comes from odometry (the ``WORLD_FRAME -> pelvis`` TF) and
NOT from a whole-body CoM or a sim ground-truth pose topic: the identical code
then runs in sim and on the real robot, where the base pose is measured from
IMU/odometry (FAST-LIO) and no privileged CoM/pose signal exists. That is the
whole point of using the base as the "CoM" reference — it is what makes the
sway numbers portable. Feeding a different frame here would break the
comparison with the fridge study.

The recorder is best-effort: any sampling or write error is swallowed and the
row is NaN-filled (the analysis drops NaN rows) — telemetry must never take
down a skill.
"""

import csv
import json
import math
import os
import threading
import time
from datetime import datetime

from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time


# Sample rate [Hz] — matches the fridge recorder's 10 Hz grid so the same
# sway_analysis.py preprocessing (FS = 10) applies unchanged.
RATE_HZ = 10.0
# CSV header — same columns as the fridge telemetry, with door_* renamed
# target_* (the GUIDE's own naming for the battery/real-world case).
COLUMNS = ('t_wall', 'pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_yaw_deg',
           'grip_x', 'grip_y', 'grip_z', 'target_x', 'target_y', 'target_z')
_NAN = float('nan')


def _slug(text):
    """Filesystem-safe lowercase slug of `text` (empty -> 'run')."""
    out = ''.join(c if c.isalnum() else '_' for c in (text or '')).strip('_').lower()
    return out or 'run'


def describe_goal(goal):
    """Flatten a skill goal's primitive fields into a JSON-friendly dict for
    result.json (lists/nested messages summarized, not dumped)."""
    out = {}
    if goal is None:
        return out
    try:
        for name in goal.get_fields_and_field_types():
            v = getattr(goal, name)
            if isinstance(v, (bool, int, float, str)):
                out[name] = v
            elif isinstance(v, (list, tuple, bytes)):
                out[name] = f'[{len(v)} items]'
            else:
                out[name] = type(v).__name__
    except Exception:
        pass
    return out


def _yaw_deg(qx, qy, qz, qw):
    """Yaw about world +Z [deg] from a quaternion (same formula the fridge
    recorder used, so pelvis_yaw_deg is defined identically)."""
    return math.degrees(math.atan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz)))


class RunRecorder:
    """One skill run's telemetry sink. Construct, ``start()``, optionally
    ``note_arm()`` / ``note_target()`` as the skill resolves them, then
    ``finish()`` (which stops the sampler and writes ``result.json``).

    All the ROS access goes through `node` (its ``tf_buffer``, logger and
    ``gripper_aperture``); nothing here spins the executor, so the sampler runs
    happily on its own daemon thread under the node's MultiThreadedExecutor —
    exactly like the grip-hold loop in base.py."""

    def __init__(self, node, skill, out_dir, world_frame, grip_frames,
                 arm=None, request=None):
        self._node = node
        self._skill = skill
        self._dir = out_dir
        self._world_frame = world_frame
        self._grip_frames = grip_frames            # {'left': frame, 'right': frame}
        self._request = request or {}
        self._lock = threading.Lock()
        # Mutable run facts, refined by the skill as it goes.
        self._arm = arm if arm in grip_frames else None
        self._grip_frame = grip_frames.get(self._arm) if self._arm else None
        self._target = None                        # (x, y, z) in the world frame
        # Sampler state.
        self._stop = threading.Event()
        self._thread = None
        self._csv_file = None
        self._csv = None
        self._rows = 0
        self._t0 = None

    # ------------------------------------------------------------------ control
    def start(self):
        """Create the run dir, open telemetry.csv, and spawn the sampler."""
        try:
            os.makedirs(self._dir, exist_ok=True)
            self._csv_file = open(os.path.join(self._dir, 'telemetry.csv'),
                                  'w', newline='')
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(COLUMNS)
        except OSError as e:
            self._node.get_logger().warn(
                f'run telemetry: cannot open {self._dir}: {e}; not recording')
            self._csv = self._csv_file = None
            return self
        self._t0 = time.time()
        self._thread = threading.Thread(
            target=self._loop, name=f'run_telemetry_{self._skill}', daemon=True)
        self._thread.start()
        self._node.get_logger().info(
            f'run telemetry: recording {self._skill} at {RATE_HZ:.0f} Hz -> '
            f'{self._dir}')
        return self

    def note_arm(self, arm):
        """Point the gripper-trajectory column at `arm`'s GraspGenX frame once
        the skill has resolved which hand it is using (auto-select)."""
        if arm not in self._grip_frames:
            return
        with self._lock:
            self._arm = arm
            self._grip_frame = self._grip_frames[arm]

    def note_target(self, xyz):
        """Record the task target as a fixed world point (x, y, z). Accepts a
        geometry_msgs/Pose (its .position is used) or an (x, y, z) triple."""
        pos = getattr(xyz, 'position', None)
        if pos is not None:
            xyz = (pos.x, pos.y, pos.z)
        try:
            t = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        except (TypeError, ValueError, IndexError):
            return
        with self._lock:
            self._target = t

    def finish(self, success=None, message=None, extra=None):
        """Stop the sampler, close the CSV, and write result.json. Safe to call
        once; the caller's finally-block owns this."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except OSError:
                pass
        with self._lock:
            arm, grip_frame, target = self._arm, self._grip_frame, self._target
        aperture = None
        if arm is not None:
            try:
                aperture = self._node.gripper_aperture(arm)
            except Exception:
                aperture = None
        record = {
            'skill': self._skill,
            'success': None if success is None else bool(success),
            'message': None if message is None else str(message),
            'arm': arm,
            'target_world': list(target) if target is not None else None,
            'grip_final_mm': aperture,
            'n_rows': self._rows,
            'rate_hz': RATE_HZ,
            'duration_s': (round(time.time() - self._t0, 3)
                           if self._t0 is not None else None),
            't_start_wall': self._t0,
            'world_frame': self._world_frame,
            'grip_frame': grip_frame,
            'request': self._request,
        }
        if extra:
            record.update(extra)
        try:
            with open(os.path.join(self._dir, 'result.json'), 'w') as f:
                json.dump(record, f, indent=2, default=str)
        except OSError as e:
            self._node.get_logger().warn(
                f'run telemetry: cannot write result.json in {self._dir}: {e}')
        if self._csv is not None:
            self._node.get_logger().info(
                f'run telemetry: {self._skill} wrote {self._rows} rows -> '
                f'{self._dir}')

    # ------------------------------------------------------------------ sampler
    def _loop(self):
        """Sample + append one row every 1/RATE_HZ until stopped."""
        period = 1.0 / RATE_HZ
        while not self._stop.is_set():
            self._sample_row()
            if self._stop.wait(period):
                break

    def _sample_row(self):
        px = py = pz = yaw = _NAN
        gx = gy = gz = _NAN
        tx = ty = tz = _NAN
        with self._lock:
            grip_frame, target = self._grip_frame, self._target
        # Base pose: pelvis expressed in the odometry world frame. This IS the
        # base "CoM" reference the sway analysis uses — from odometry so the
        # same code runs in sim and on the real robot.
        try:
            tf = self._node.tf_buffer.lookup_transform(
                self._world_frame, 'pelvis', Time(),
                timeout=RclpyDuration(seconds=0.05))
            t, q = tf.transform.translation, tf.transform.rotation
            px, py, pz = t.x, t.y, t.z
            yaw = _yaw_deg(q.x, q.y, q.z, q.w)
        except Exception:      # TF miss / lag: NaN row, dropped by the analysis
            pass
        # Gripper point in the pelvis frame (matches the fridge recorder's
        # pelvis -> right_graspgenx_frame lookup, for the active arm).
        if grip_frame is not None:
            try:
                tf = self._node.tf_buffer.lookup_transform(
                    'pelvis', grip_frame, Time(),
                    timeout=RclpyDuration(seconds=0.05))
                gt = tf.transform.translation
                gx, gy, gz = gt.x, gt.y, gt.z
            except Exception:
                pass
        if target is not None:
            tx, ty, tz = target
        if self._csv is None:
            return
        try:
            self._csv.writerow([
                f'{time.time() - self._t0:.2f}',
                f'{px:.4f}', f'{py:.4f}', f'{pz:.4f}', f'{yaw:.2f}',
                f'{gx:.4f}', f'{gy:.4f}', f'{gz:.4f}',
                f'{tx:.4f}', f'{ty:.4f}', f'{tz:.4f}'])
            self._csv_file.flush()
            self._rows += 1
        except (OSError, ValueError):
            pass


def run_dir(log_root, skill, target_object):
    """Absolute per-run directory:
    ``<log_root>/runs/<skill>/<YYYYmmdd_HHMMSS_mmm>_<slug>``."""
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    return os.path.join(log_root, 'runs', skill,
                        f'{stamp}_{_slug(target_object)}')
