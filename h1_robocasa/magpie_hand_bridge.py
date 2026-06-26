"""ROS 2 bridge that exposes the simulated magpie hands on the same
topic / service / action surface that the real magpie_control driver
publishes (gripper_node.py).

One MagpieHandBridge instance handles a single hand. h12_mujoco.py spins
two of them (side="left" and side="right") on a background
MultiThreadedExecutor sharing the sim's threading.Lock so that callbacks
can safely touch MjData while mj_step runs on the main thread.

Topics / services / actions exposed (per side), matching the real driver's
fully-qualified names (gripper_node.py relative 'gripper/...' under namespace
'<side>'):
  /<side>/gripper/state                       magpie_msgs/msg/GripperState (10 Hz)
  /<side>/gripper/open                        std_srvs/srv/Trigger
  /<side>/gripper/close                       std_srvs/srv/Trigger
  /<side>/gripper/set_position                magpie_msgs/srv/SetGripperPosition
  /<side>/gripper/set_force                   magpie_msgs/srv/SetGripperForce
  /<side>/gripper/calibrate                   std_srvs/srv/Trigger
  /<side>/gripper/reset_parameters            std_srvs/srv/Trigger
  /<side>/gripper/deligrasp                   magpie_msgs/action/DeliGrasp
"""
import threading
import time

import mujoco
import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

try:
    from magpie_msgs.action import DeliGrasp
    from magpie_msgs.msg import GripperState
    from magpie_msgs.srv import SetGripperForce, SetGripperPosition
except ModuleNotFoundError as exc:
    if exc.name == "magpie_msgs":
        raise ModuleNotFoundError(
            "magpie_msgs is not available in this shell. Source the workspace "
            "before launching the sim, e.g. "
            "`source /opt/ros/humble/setup.bash && "
            "source ~/HAMS/core_ws/install/setup.bash`."
        ) from exc
    raise


# Aperture calibration: linear map between an MJCF hinge_1 angle (rad) and the
# reported aperture (mm). 0 rad ≈ fingers parallel/open; CLOSED_RAD ≈ tips
# touching. Default chosen so the open/closed states roughly match the real
# magpie's ~0–110 mm range. Tune in one place if the geometry changes.
OPEN_MM = 110.0
CLOSED_MM = 0.0
CLOSED_RAD = 2.05  # matches the upper limit of left_hinge_1 / leftg_left_hinge_1

# Force model: MuJoCo reports actuator torque (Nm) at the hinge; convert to an
# approximate fingertip force (N) using a constant lever arm.
FINGER_LEVER_M = 0.05

# set_force → kp mapping. Position actuator force ≈ kp * (q_target - q), so a
# very rough first-order model is kp ≈ N_max * lever / typical_error. The
# constants below give kp ≈ 10 (current MJCF default) at ~5 N, and clip to a
# safe range so a malicious request can't blow up the sim.
KP_PER_N = 2.0
KP_MIN = 1.0
KP_MAX = 200.0


def mm_to_rad(mm: float) -> float:
    """Aperture millimeters → hinge_1 angle (radians)."""
    f = (OPEN_MM - mm) / (OPEN_MM - CLOSED_MM)
    return float(np.clip(f, 0.0, 1.0)) * CLOSED_RAD


def rad_to_mm(rad: float) -> float:
    """Hinge_1 angle (radians) → aperture millimeters."""
    f = float(np.clip(rad, 0.0, CLOSED_RAD)) / CLOSED_RAD
    return OPEN_MM - f * (OPEN_MM - CLOSED_MM)


class MagpieHandBridge(Node):
    """ROS 2 bridge for one simulated magpie hand."""

    # Both grippers share these internal MJCF actuator/sensor names (the gripper
    # MJCFs are identical), disambiguated by the gripper0_<side>_ prefix passed
    # as name_prefix.
    _NAMES = {
        "act_left":  "left_finger_actuator",
        "act_right": "right_finger_actuator",
        "pos_left":  "left_finger_pos",
        "pos_right": "right_finger_pos",
        "vel_left":  "left_finger_vel",
        "vel_right": "right_finger_vel",
        "trq_left":  "left_finger_torque",
        "trq_right": "right_finger_torque",
    }

    def __init__(self, model, data, side: str, sim_lock: threading.Lock,
                 publish_rate_hz: float = 10.0, name_prefix=""):
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        super().__init__(f"magpie_hand_{side}")

        self.model = model
        self.data = data
        self.sim_lock = sim_lock
        self.side = side
        names = {k: name_prefix + v for k, v in self._NAMES.items()}

        self.act_left  = self._actuator_id(names["act_left"])
        self.act_right = self._actuator_id(names["act_right"])
        self.pos_left  = self._sensor_adr(names["pos_left"])
        self.pos_right = self._sensor_adr(names["pos_right"])
        self.vel_left  = self._sensor_adr(names["vel_left"])
        self.vel_right = self._sensor_adr(names["vel_right"])
        self.trq_left  = self._sensor_adr(names["trq_left"])
        self.trq_right = self._sensor_adr(names["trq_right"])

        # Snapshot the default kp so reset_parameters can restore it. Position
        # actuators store kp in actuator_gainprm[:, 0] and the matching PD
        # bias term in actuator_biasprm[:, 1] = -kp.
        self._kp_default_left  = float(model.actuator_gainprm[self.act_left,  0])
        self._kp_default_right = float(model.actuator_gainprm[self.act_right, 0])

        # Initialize commanded ctrl to fully open so the hand sits open at
        # boot (matches the real driver's default-open behaviour).
        with self.sim_lock:
            self._write_aperture_rad_locked(0.0)

        # ReentrantCallbackGroup so the action server's loop (which sleeps
        # between iterations and may call publishers) doesn't block other
        # callbacks on the same executor.
        self._cb_group = ReentrantCallbackGroup()

        # Match the real driver's fully-qualified names exactly. gripper_node.py
        # uses relative 'gripper/...' names launched under a per-side ROS
        # namespace (namespace='left'/'right'), yielding /<side>/gripper/<thing>.
        # This bridge has no node namespace, so it bakes the side into the
        # relative name to land on the identical /<side>/gripper/<thing> surface.
        ns = f"{side}/gripper"
        self.pub_state = self.create_publisher(GripperState, f"{ns}/state", 10)
        self._publish_period = 1.0 / publish_rate_hz
        self.timer = self.create_timer(
            self._publish_period, self._publish_state, callback_group=self._cb_group)

        self.create_service(Trigger, f"{ns}/open", self._on_open,
                            callback_group=self._cb_group)
        self.create_service(Trigger, f"{ns}/close", self._on_close,
                            callback_group=self._cb_group)
        self.create_service(SetGripperPosition, f"{ns}/set_position",
                            self._on_set_position, callback_group=self._cb_group)
        self.create_service(SetGripperForce, f"{ns}/set_force",
                            self._on_set_force, callback_group=self._cb_group)
        self.create_service(Trigger, f"{ns}/calibrate", self._on_calibrate,
                            callback_group=self._cb_group)
        self.create_service(Trigger, f"{ns}/reset_parameters",
                            self._on_reset_parameters, callback_group=self._cb_group)

        self.action_server = ActionServer(
            self, DeliGrasp, f"{ns}/deligrasp",
            execute_callback=self._on_deligrasp,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"MagpieHandBridge[{side}] up: /{ns}/"
            "{state,open,close,set_position,set_force,calibrate,reset_parameters,deligrasp}")

    # ------------------------------------------------------------------ helpers

    def _actuator_id(self, name: str) -> int:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"actuator '{name}' not found in MJCF")
        return int(aid)

    def _sensor_adr(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f"sensor '{name}' not found in MJCF")
        return int(self.model.sensor_adr[sid])

    def _write_aperture_rad_locked(self, target_rad: float) -> None:
        """Caller must hold sim_lock. Mirrors the rad value to both finger
        actuators with the per-side ctrlrange sign convention.

        Both `left_finger_actuator*` ctrlrange is [-0.035, 2.70] (positive
        closes); `right_finger_actuator*` ctrlrange is [-2.70, 0.035]
        (negative closes). Mirroring across the sign keeps the two fingers
        symmetric.
        """
        lo_l, hi_l = self.model.actuator_ctrlrange[self.act_left]
        lo_r, hi_r = self.model.actuator_ctrlrange[self.act_right]
        self.data.ctrl[self.act_left]  = float(np.clip( target_rad, lo_l, hi_l))
        self.data.ctrl[self.act_right] = float(np.clip(-target_rad, lo_r, hi_r))

    def _read_aperture_mm_locked(self) -> tuple[float, float]:
        """Caller must hold sim_lock. Returns (mean_mm, [right_mm, left_mm])."""
        rad_left  = float(self.data.sensordata[self.pos_left])
        rad_right = float(self.data.sensordata[self.pos_right])
        mm_left  = rad_to_mm( rad_left)
        mm_right = rad_to_mm(-rad_right)  # right finger ctrl sign is negated
        # finger_positions order matches gripper_node.py:107-110 — [right, left].
        return 0.5 * (mm_left + mm_right), [mm_right, mm_left]

    def _read_force_n_locked(self) -> float:
        """Caller must hold sim_lock. Mean fingertip force in N."""
        torque_l = float(self.data.sensordata[self.trq_left])
        torque_r = float(self.data.sensordata[self.trq_right])
        # torque on the hinge → force at fingertip via constant lever arm.
        return float(0.5 * (abs(torque_l) + abs(torque_r)) / FINGER_LEVER_M)

    def _read_vel_locked(self) -> float:
        v_l = float(self.data.sensordata[self.vel_left])
        v_r = float(self.data.sensordata[self.vel_right])
        return max(abs(v_l), abs(v_r))

    def _set_kp_locked(self, kp: float) -> None:
        """Caller must hold sim_lock. Updates kp on both finger actuators."""
        kp = float(np.clip(kp, KP_MIN, KP_MAX))
        for act in (self.act_left, self.act_right):
            self.model.actuator_gainprm[act, 0] = kp
            # Position actuators bake the PD-bias as biasprm[1] = -kp.
            self.model.actuator_biasprm[act, 1] = -kp

    # -------------------------------------------------------------- publishers

    def _publish_state(self) -> None:
        try:
            with self.sim_lock:
                aperture_mm, finger_positions = self._read_aperture_mm_locked()
                force_n  = self._read_force_n_locked()
                vel_max  = self._read_vel_locked()
            msg = GripperState()
            msg.position = float(aperture_mm)
            msg.force = float(force_n)
            msg.temperature = 25.0  # no thermal model in sim — report a constant.
            msg.is_moving = bool(vel_max > 0.01)
            msg.contact_detected = bool(force_n > 0.5)
            msg.finger_positions = [float(p) for p in finger_positions]
            self.pub_state.publish(msg)
        except Exception as e:
            self.get_logger().warning(f"state publish failed: {e}")

    # ------------------------------------------------------------ service impls

    def _on_open(self, request, response):
        with self.sim_lock:
            self._write_aperture_rad_locked(0.0)
        response.success = True
        response.message = f"{self.side} hand opening"
        return response

    def _on_close(self, request, response):
        with self.sim_lock:
            self._write_aperture_rad_locked(CLOSED_RAD)
        response.success = True
        response.message = f"{self.side} hand closing"
        return response

    def _on_set_position(self, request, response):
        try:
            target_mm = float(request.position)
            target_rad = mm_to_rad(target_mm)
            with self.sim_lock:
                self._write_aperture_rad_locked(target_rad)
                actual_mm, _ = self._read_aperture_mm_locked()
            response.success = True
            response.actual_position = float(actual_mm)
            response.message = (
                f"{self.side} target {target_mm:.1f} mm "
                f"({target_rad:.3f} rad); actual {actual_mm:.1f} mm "
                f"(speed {request.speed:.2f} ignored in sim)"
            )
        except Exception as e:
            response.success = False
            response.actual_position = 0.0
            response.message = f"set_position failed: {e}"
        return response

    def _on_set_force(self, request, response):
        try:
            kp = max(request.max_force, 0.0) * KP_PER_N
            with self.sim_lock:
                self._set_kp_locked(kp)
            response.success = True
            response.message = (
                f"{self.side} force limit {request.max_force:.2f} N "
                f"(kp={float(np.clip(kp, KP_MIN, KP_MAX)):.2f})"
            )
        except Exception as e:
            response.success = False
            response.message = f"set_force failed: {e}"
        return response

    def _on_calibrate(self, request, response):
        # Sim hand is calibrated by construction — no-op success.
        response.success = True
        response.message = f"{self.side} hand: sim calibration is implicit"
        return response

    def _on_reset_parameters(self, request, response):
        try:
            with self.sim_lock:
                self.model.actuator_gainprm[self.act_left,  0] = self._kp_default_left
                self.model.actuator_gainprm[self.act_right, 0] = self._kp_default_right
                self.model.actuator_biasprm[self.act_left,  1] = -self._kp_default_left
                self.model.actuator_biasprm[self.act_right, 1] = -self._kp_default_right
            response.success = True
            response.message = (
                f"{self.side} kp restored to "
                f"({self._kp_default_left:.1f}, {self._kp_default_right:.1f})"
            )
        except Exception as e:
            response.success = False
            response.message = f"reset_parameters failed: {e}"
        return response

    # ------------------------------------------------------------ deligrasp

    def _on_deligrasp(self, goal_handle):
        """Sim-side deligrasp: approach → close while monitoring force.

        Loops up to MAX_ITERS: if measured force < initial_force after a
        short settle, close by additional_closure mm and bump the force
        ceiling by additional_force N. Stops on contact, cancellation, or
        iteration cap. Mirrors the API of gripper_node.py:222-282.
        """
        params = goal_handle.request.params
        feedback = DeliGrasp.Feedback()
        force_log: list[float] = []

        MAX_ITERS = 8
        SETTLE_S = 0.20

        try:
            current_aperture = float(params.goal_aperture)
            current_force_target = float(params.initial_force)

            # Phase 1: approach to initial goal aperture with initial force.
            feedback.phase = "approach"
            feedback.current_aperture = current_aperture
            feedback.current_force = 0.0
            goal_handle.publish_feedback(feedback)

            with self.sim_lock:
                self._set_kp_locked(current_force_target * KP_PER_N)
                self._write_aperture_rad_locked(mm_to_rad(current_aperture))

            time.sleep(SETTLE_S)

            # Phase 2: iterative force-feedback close.
            feedback.phase = "initial_close"
            goal_handle.publish_feedback(feedback)

            for it in range(MAX_ITERS):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result = DeliGrasp.Result()
                    result.success = False
                    result.message = "cancelled"
                    result.final_aperture, _ = self._read_state_snapshot()
                    result.final_force = 0.0
                    result.force_log = force_log
                    return result

                with self.sim_lock:
                    current_force = self._read_force_n_locked()
                    actual_mm, _ = self._read_aperture_mm_locked()
                force_log.append(current_force)

                feedback.current_aperture = float(actual_mm)
                feedback.current_force = float(current_force)
                goal_handle.publish_feedback(feedback)

                if current_force >= current_force_target:
                    break  # contact achieved.

                # Slipping / no contact yet — close further and bump force.
                current_aperture = max(
                    0.0, current_aperture - float(params.additional_closure))
                current_force_target += float(params.additional_force)
                feedback.phase = (
                    "additional_close" if it > 0 else "force_increase")
                feedback.current_aperture = current_aperture
                goal_handle.publish_feedback(feedback)

                with self.sim_lock:
                    self._set_kp_locked(current_force_target * KP_PER_N)
                    self._write_aperture_rad_locked(mm_to_rad(current_aperture))
                time.sleep(SETTLE_S)

            if params.complete_grasp:
                feedback.phase = "complete"
                goal_handle.publish_feedback(feedback)

            with self.sim_lock:
                final_force = self._read_force_n_locked()
                final_aperture, _ = self._read_aperture_mm_locked()

            result = DeliGrasp.Result()
            result.success = True
            result.message = (
                f"deligrasp done: aperture {final_aperture:.1f} mm, "
                f"force {final_force:.2f} N over {len(force_log)} iters"
            )
            result.final_aperture = float(final_aperture)
            result.final_force = float(final_force)
            result.force_log = force_log
            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f"deligrasp failed: {e}")
            result = DeliGrasp.Result()
            result.success = False
            result.message = f"deligrasp failed: {e}"
            result.final_aperture = 0.0
            result.final_force = 0.0
            result.force_log = force_log
            goal_handle.abort()
            return result

    def _read_state_snapshot(self) -> tuple[float, float]:
        """Convenience: returns (aperture_mm, force_n) under the sim_lock."""
        with self.sim_lock:
            aperture_mm, _ = self._read_aperture_mm_locked()
            force_n = self._read_force_n_locked()
        return aperture_mm, force_n
