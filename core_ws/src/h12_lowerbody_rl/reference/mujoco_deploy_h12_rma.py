"""
RMA deploy for H1-2 with name-based joint/actuator indexing.

Works with either the bare H1-2 model (motor actuators) or the H1-2 + Magpie
gripper model (position actuators + extra gripper hinges). Joints and actuators
are resolved by name, so any model that exposes the 27 canonical H1-2 motor
joints can be used without touching the script.

- Hand-only 3D forces (left/right wrist); no torso. e_t = 15 upper-body + left_xyz(3) + right_xyz(3) = 21.
- Apply forces to left_wrist_roll_link and right_wrist_roll_link; build e_t and run encoder -> z_t.
- Base policy input = [proprio history (3*76), z_t history (3*8)] = 252 dim.
"""
import sys
import os
import time
import collections
import datetime
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

RMA_LATENT_DIM = 8
RMA_ACTOR_Z_DIM = 24   # 3 * 8
RMA_ET_DIM = 21        # 15 upper dof + left_xyz(3) + right_xyz(3), hand-only

# Canonical 27 motor joints the policy was trained on (12 legs + torso + 7 left arm + 7 right arm).
# Order matters: must match the training joint order so policy outputs map to the right joints.
POLICY_JOINT_NAMES = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
NUM_LEG_JOINTS = 12
NUM_POLICY_JOINTS = len(POLICY_JOINT_NAMES)  # 27


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    array_keys = ["kps", "kds", "default_angles", "cmd_scale", "cmd_init"]
    if "kps_arms" in config:
        array_keys.extend(["kps_arms", "kds_arms"])
    if "default_angles_arms" in config:
        array_keys.append("default_angles_arms")
    if "left_hand_force" in config:
        config["left_hand_force"] = np.array(config["left_hand_force"], dtype=np.float32)
    if "right_hand_force" in config:
        config["right_hand_force"] = np.array(config["right_hand_force"], dtype=np.float32)
    for key in array_keys:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)
    return config


def build_joint_index(m):
    """Resolve POLICY_JOINT_NAMES to qpos/qvel addresses and actuator indices.

    Returns:
        qpos_adr: int32[27], indices into d.qpos for each policy joint
        qvel_adr: int32[27], indices into d.qvel for each policy joint
        act_idx:  int32[27], actuator indices for each policy joint (same order)
        extra_act_idx: int32[k], remaining actuators (e.g. Magpie gripper) not driven by the policy
    """
    qpos_adr = np.zeros(NUM_POLICY_JOINTS, dtype=np.int32)
    qvel_adr = np.zeros(NUM_POLICY_JOINTS, dtype=np.int32)
    act_idx = np.zeros(NUM_POLICY_JOINTS, dtype=np.int32)
    for i, name in enumerate(POLICY_JOINT_NAMES):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not in model")
        qpos_adr[i] = m.jnt_qposadr[jid]
        qvel_adr[i] = m.jnt_dofadr[jid]
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(f"Actuator '{name}' not in model (policy expects 1:1 name match)")
        act_idx[i] = aid
    policy_set = set(act_idx.tolist())
    extra_act_idx = np.array([i for i in range(m.nu) if i not in policy_set], dtype=np.int32)
    return qpos_adr, qvel_adr, act_idx, extra_act_idx


def detect_actuator_kind(m, act_idx):
    """Return 'position' or 'motor' for the policy actuators (must all be the same kind)."""
    kinds = set()
    for aid in act_idx:
        # Position actuator: biastype=AFFINE with biasprm[1] != 0 (i.e. -kp on qpos).
        # Motor actuator: biastype=NONE.
        if m.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_biasprm[aid, 1] != 0.0:
            kinds.add("position")
        else:
            kinds.add("motor")
    if len(kinds) > 1:
        raise ValueError(f"Policy actuators have mixed kinds: {kinds}. Make them uniform in the XML.")
    return kinds.pop()


def override_position_gains(m, act_idx, kps, kvs):
    """Overwrite per-actuator position gains (kp/kv) so YAML training gains take effect."""
    for slot, aid in enumerate(act_idx):
        kp = float(kps[slot])
        kv = float(kvs[slot])
        m.actuator_gainprm[aid, 0] = kp
        m.actuator_biasprm[aid, 1] = -kp
        m.actuator_biasprm[aid, 2] = -kv


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_rotate_inverse(q, v):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    q_conj = np.array([w, -x, -y, -z])
    return np.array([
        v[0] * (q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])


def get_gravity_orientation(quat):
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))


def compute_observation(d, config, action, cmd, height_cmd, qpos_adr, qvel_adr):
    """Single observation, 76-dim. Reads joint state by indexed address (gripper-hinge-safe)."""
    qj = d.qpos[qpos_adr].copy()        # 27
    dqj = d.qvel[qvel_adr].copy()       # 27
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()

    n_joints = NUM_POLICY_JOINTS
    padded_defaults = np.zeros(n_joints, dtype=np.float32)
    padded_defaults[:NUM_LEG_JOINTS] = config["default_angles"][:NUM_LEG_JOINTS]
    if "default_angles_arms" in config:
        arms = config["default_angles_arms"]
        padded_defaults[NUM_LEG_JOINTS:NUM_LEG_JOINTS + len(arms)] = arms

    qj_scaled = (qj - padded_defaults) * config["dof_pos_scale"]
    dqj_scaled = dqj * config["dof_vel_scale"]
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config["ang_vel_scale"]

    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + NUM_LEG_JOINTS
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd[:3] * config["cmd_scale"]
    single_obs[3:4] = np.array([height_cmd])
    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation
    single_obs[10 : 10 + n_joints] = qj_scaled
    single_obs[10 + n_joints : 10 + 2 * n_joints] = dqj_scaled
    single_obs[10 + 2 * n_joints : 10 + 2 * n_joints + NUM_LEG_JOINTS] = action
    return single_obs, single_obs_dim


def build_et_mujoco(d, qpos_adr, left_hand_force_xyz, right_hand_force_xyz):
    """e_t = 15 upper-body dof + left_xyz(3) + right_xyz(3) = 21 (hand-only, matches Isaac build_et_from_gym)."""
    upper = d.qpos[qpos_adr[NUM_LEG_JOINTS:NUM_POLICY_JOINTS]].copy()  # 15 = torso + arms
    return np.concatenate(
        [upper, np.asarray(left_hand_force_xyz, dtype=np.float32), np.asarray(right_hand_force_xyz, dtype=np.float32)],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Upper-body perturbation trajectories (right shoulder circle, left arm
# triangle). Indices into arm_target (15-vec = torso + 7L + 7R):
#   left:  1=shoulder_pitch, 2=shoulder_roll, 3=shoulder_yaw, 4=elbow
#   right: 8=shoulder_pitch, 9=shoulder_roll, 10=shoulder_yaw, 11=elbow
# ---------------------------------------------------------------------------

def parse_perturbations(config):
    rac = config.get("right_arm_circle") or {}
    lat = config.get("left_arm_triangle") or {}
    spec = {
        "right_circle": {
            "enabled": bool(rac.get("enabled", False)),
            "freq_hz": float(rac.get("freq_hz", 0.5)),
            "sp_amp": float(rac.get("shoulder_pitch_amp", 0.5)),
            "sp_center": float(rac.get("shoulder_pitch_center", 0.0)),
            "sr_amp": float(rac.get("shoulder_roll_amp", 0.3)),
            "sr_center": float(rac.get("shoulder_roll_center", -0.6)),
            "elbow": float(rac.get("elbow_target", 1.2)),
        },
        "left_triangle": {
            "enabled": bool(lat.get("enabled", False)),
            "period_s": float(lat.get("period_s", 4.5)),
            "vertices": np.asarray(
                lat.get("vertices", [[0.5, 0.3, 1.0], [-0.3, 0.9, 1.0], [0.2, 1.2, 1.0]]),
                dtype=np.float32,
            ),
        },
    }
    return spec


def apply_perturbations(arm_target_base, t, spec):
    out = arm_target_base.copy()
    rc = spec["right_circle"]
    if rc["enabled"]:
        phase = 2.0 * np.pi * rc["freq_hz"] * t
        out[8]  = rc["sp_center"] + rc["sp_amp"] * np.sin(phase)
        out[9]  = rc["sr_center"] + rc["sr_amp"] * np.cos(phase)
        out[11] = rc["elbow"]
    lt = spec["left_triangle"]
    if lt["enabled"]:
        vs = lt["vertices"]
        period = lt["period_s"]
        n = vs.shape[0]
        seg_f = ((t % period) / period) * n
        seg_i = int(seg_f) % n
        seg_t = seg_f - int(seg_f)
        v0, v1 = vs[seg_i], vs[(seg_i + 1) % n]
        sp, sr, elb = v0 + seg_t * (v1 - v0)
        out[1] = sp   # left_shoulder_pitch
        out[2] = sr   # left_shoulder_roll
        out[4] = elb  # left_elbow
    return out


# ---------------------------------------------------------------------------
# Run logging + plotting
# ---------------------------------------------------------------------------

def quat_to_rpy(q):
    """quaternion (w, x, y, z) -> (roll, pitch, yaw) in radians."""
    w, x, y, z = q
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def make_run_dir(name=None):
    sub = name if name else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(_SCRIPT_DIR, "runs", sub)
    os.makedirs(out, exist_ok=True)
    return out


def save_plots(log, run_dir):
    """Save tracking / stability / trajectory plots from collected log dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(log["t"])

    # Base stability
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, log["base_z"], label="z"); axes[0].set_ylabel("base z [m]"); axes[0].grid(True); axes[0].axhline(1.0, color="gray", ls="--", lw=0.5)
    axes[1].plot(t, np.rad2deg(log["base_roll"]), label="roll", color="C1"); axes[1].plot(t, np.rad2deg(log["base_pitch"]), label="pitch", color="C2"); axes[1].set_ylabel("deg"); axes[1].grid(True); axes[1].legend()
    axes[2].plot(t, log["action_max"], color="C3"); axes[2].set_ylabel("|action|max"); axes[2].set_xlabel("t [s]"); axes[2].grid(True)
    fig.suptitle("Base stability under upper-body perturbation")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "base_stability.png"), dpi=120)
    plt.close(fig)

    # Per-arm tracking (target vs actual)
    for side in ("left", "right"):
        keys = (f"{side}_sp", f"{side}_sr", f"{side}_elbow")
        labels = ("shoulder_pitch", "shoulder_roll", "elbow")
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for ax, k, lbl in zip(axes, keys, labels):
            ax.plot(t, log[f"{k}_tgt"], label="target", lw=1.5)
            ax.plot(t, log[f"{k}_act"], label="actual", lw=1.0, alpha=0.8)
            ax.set_ylabel(f"{lbl} [rad]"); ax.grid(True); ax.legend()
        axes[-1].set_xlabel("t [s]")
        fig.suptitle(f"{side.capitalize()} arm tracking")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"{side}_arm_tracking.png"), dpi=120)
        plt.close(fig)

    # Shape in (pitch, roll) plane: left triangle, right circle
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(log["left_sp_tgt"], log["left_sr_tgt"], label="target", lw=2)
    axes[0].plot(log["left_sp_act"], log["left_sr_act"], label="actual", lw=1, alpha=0.7)
    axes[0].set_xlabel("shoulder_pitch [rad]"); axes[0].set_ylabel("shoulder_roll [rad]")
    axes[0].set_title("Left arm (triangle)"); axes[0].grid(True); axes[0].legend(); axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].plot(log["right_sp_tgt"], log["right_sr_tgt"], label="target", lw=2)
    axes[1].plot(log["right_sp_act"], log["right_sr_act"], label="actual", lw=1, alpha=0.7)
    axes[1].set_xlabel("shoulder_pitch [rad]"); axes[1].set_ylabel("shoulder_roll [rad]")
    axes[1].set_title("Right arm (circle)"); axes[1].grid(True); axes[1].legend(); axes[1].set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "arm_shapes.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Offscreen video recording
# ---------------------------------------------------------------------------

class VideoRecorder:
    """Pipes raw RGB frames from mujoco.Renderer into ffmpeg, which encodes H.264.

    cv2.VideoWriter with the 'mp4v' fourcc writes FMP4 (MPEG-4 Part 2) — that
    container is rejected by most modern players (Totem, browsers). Using
    ffmpeg directly with libopenh264 produces standard H.264 in an mp4 that
    plays everywhere.
    """

    def __init__(self, m, path, fps=30, width=1280, height=720, track_body=None, bitrate="3M"):
        import subprocess
        self._subprocess = subprocess
        # Resize the model's offscreen framebuffer if the requested resolution is larger.
        if m.vis.global_.offwidth < width:
            m.vis.global_.offwidth = width
        if m.vis.global_.offheight < height:
            m.vis.global_.offheight = height
        self.renderer = mujoco.Renderer(m, width=width, height=height)
        self.cam = mujoco.MjvCamera()
        # Tracking camera if a body name is supplied; else a fixed orbit view.
        if track_body is not None:
            tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, track_body)
            if tid >= 0:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.cam.trackbodyid = tid
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        else:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 3.0
        self.cam.azimuth = 135.0
        self.cam.elevation = -15.0
        self.cam.lookat[:] = (0.0, 0.0, 0.9)
        self.path = path
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24",
            "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libopenh264", "-b:v", bitrate, "-pix_fmt", "yuv420p",
            path,
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg not on PATH; needed for video recording.") from e

    def capture(self, d):
        self.renderer.update_scene(d, camera=self.cam)
        frame = self.renderer.render()  # HxWx3 uint8 RGB
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            err = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
            raise RuntimeError(f"ffmpeg died during encoding:\n{err}")

    def close(self):
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except self._subprocess.TimeoutExpired:
            self.proc.kill()
        self.renderer.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(_SCRIPT_DIR, "h1_2_rma_magpie.yaml"))
    parser.add_argument(
        "--no_encode",
        action="store_true",
        help="Zero the wrist forces applied to the sim (no force perturbation). The encoder still runs — joint positions are read from the sim and e_t is built with zero forces.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record an mp4 of the run (no live viewer; offscreen renderer + cv2.VideoWriter).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subdirectory name under runs/. Defaults to timestamped (YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--encoder-mode",
        choices=["full", "zero_et", "no_encoder"],
        default="full",
        help="full = z_t = encoder(real e_t); zero_et = z_t = encoder(zeros); no_encoder = z_t = zeros (encoder bypassed).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    no_encode = args.no_encode or config.get("no_encode", False)

    config_dir = os.path.dirname(os.path.abspath(args.config))
    for key in ["policy_path", "xml_path", "encoder_path"]:
        if key in config and config[key] and isinstance(config[key], str) and not os.path.isabs(config[key]):
            config[key] = os.path.normpath(os.path.join(config_dir, config[key]))

    m = mujoco.MjModel.from_xml_path(config["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]

    qpos_adr, qvel_adr, act_idx, extra_act_idx = build_joint_index(m)
    actuator_kind = detect_actuator_kind(m, act_idx)
    print(f"Model DOFs (qpos): {d.qpos.shape[0]}, total joints: {m.njnt - 1}, ctrl size: {d.ctrl.shape[0]}")
    print(f"Policy joints resolved: {NUM_POLICY_JOINTS}, extra (non-policy) actuators: {extra_act_idx.size}")
    print(f"Actuator kind for policy joints: {actuator_kind}")

    # Build combined kp/kv arrays for the 27 policy joints (12 legs + 15 arm/torso).
    kps_all = np.concatenate([config["kps"], config["kps_arms"]]).astype(np.float32)
    kds_all = np.concatenate([config["kds"], config["kds_arms"]]).astype(np.float32)
    if len(kps_all) != NUM_POLICY_JOINTS:
        raise ValueError(f"kps+kps_arms has {len(kps_all)} entries; need {NUM_POLICY_JOINTS}.")
    if actuator_kind == "position":
        override_position_gains(m, act_idx, kps_all, kds_all)
        print("Overrode XML position-actuator gains with YAML kps/kds.")

    # RMA: body ids and forces (hand-only)
    left_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    apply_forces = left_wrist_id >= 0 and right_wrist_id >= 0
    if not apply_forces:
        print("Warning: RMA force bodies (left/right_wrist_roll_link) not found; skipping xfrc_applied.")
    left_hand_force = config["left_hand_force"].copy()
    right_hand_force = config["right_hand_force"].copy()
    if no_encode:
        left_hand_force[:] = 0.0
        right_hand_force[:] = 0.0
        print("--no_encode: wrist forces zeroed in sim; encoder runs from joint positions + zero forces.")

    # RMA: encoder
    encoder = None
    if config.get("encoder_path") and os.path.isfile(config["encoder_path"]):
        from RMA.rma_modules.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
        encoder = EnvFactorEncoder(EnvFactorEncoderCfg())
        encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu", weights_only=True))
        encoder.eval()
        print(f"Loaded encoder from {config['encoder_path']}")
    else:
        print("No encoder_path or file not found; z_t will be zeros.")

    action = np.zeros(config["num_actions"], dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()
    cmd = config["cmd_init"].copy()
    height_cmd = config["height_cmd"]
    arm_target = config.get("default_angles_arms", np.zeros(NUM_POLICY_JOINTS - NUM_LEG_JOINTS, dtype=np.float32)).astype(np.float32)
    arm_target_base = arm_target.copy()
    gripper_cmd = float(config.get("gripper_cmd", 0.0))

    pert_spec = parse_perturbations(config)
    if pert_spec["right_circle"]["enabled"]:
        rc = pert_spec["right_circle"]
        print(f"right_arm_circle: freq={rc['freq_hz']} Hz, sp=({rc['sp_center']}±{rc['sp_amp']}), sr=({rc['sr_center']}±{rc['sr_amp']}), elbow={rc['elbow']}")
    if pert_spec["left_triangle"]["enabled"]:
        lt = pert_spec["left_triangle"]
        print(f"left_arm_triangle: period={lt['period_s']} s, vertices={lt['vertices'].tolist()}")

    log = {k: [] for k in (
        "t", "base_z", "base_roll", "base_pitch", "action_max",
        "left_sp_tgt", "left_sp_act", "left_sr_tgt", "left_sr_act", "left_elbow_tgt", "left_elbow_act",
        "right_sp_tgt", "right_sp_act", "right_sr_tgt", "right_sr_act", "right_elbow_tgt", "right_elbow_act",
    )}

    single_obs, single_obs_dim = compute_observation(d, config, action, cmd, height_cmd, qpos_adr, qvel_adr)
    obs_history = collections.deque(maxlen=config["obs_history_len"])
    for _ in range(config["obs_history_len"]):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))

    z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)

    policy = torch.jit.load(config["policy_path"])
    print(policy)
    counter = 0

    leg_act_idx = act_idx[:NUM_LEG_JOINTS]
    arm_act_idx = act_idx[NUM_LEG_JOINTS:NUM_POLICY_JOINTS]
    leg_qpos_adr = qpos_adr[:NUM_LEG_JOINTS]
    leg_qvel_adr = qvel_adr[:NUM_LEG_JOINTS]
    arm_qpos_adr = qpos_adr[NUM_LEG_JOINTS:NUM_POLICY_JOINTS]
    arm_qvel_adr = qvel_adr[NUM_LEG_JOINTS:NUM_POLICY_JOINTS]

    # Run setup: output dir, optional video recorder, initial perturbation
    record_video = args.record or bool(config.get("record_video", False))
    run_dir = make_run_dir(args.run_name)
    print(f"Run output dir: {run_dir}")
    print(f"Encoder mode: {args.encoder_mode}")

    video = None
    render_every = 0
    if record_video:
        video_path = os.path.join(run_dir, "run.mp4")
        fps = int(config.get("video_fps", 30))
        video = VideoRecorder(m, video_path, fps=fps, track_body="pelvis")
        render_every = max(1, int(round((1.0 / config["simulation_dt"]) / fps)))
        print(f"Recording video -> {video_path} @ {fps} fps (every {render_every} sim steps)")

    arm_target = apply_perturbations(arm_target_base, 0.0, pert_spec)

    duration = config["simulation_duration"]

    def step_once():
        nonlocal arm_target, action, target_dof_pos, counter
        t = counter * config["simulation_dt"]

        d.xfrc_applied[:] = 0
        if apply_forces:
            d.xfrc_applied[left_wrist_id, :3] = left_hand_force
            d.xfrc_applied[right_wrist_id, :3] = right_hand_force

        arm_target = apply_perturbations(arm_target_base, t, pert_spec)

        if actuator_kind == "position":
            d.ctrl[leg_act_idx] = target_dof_pos
            d.ctrl[arm_act_idx] = arm_target
        else:
            leg_tau = pd_control(
                target_dof_pos, d.qpos[leg_qpos_adr], config["kps"],
                np.zeros_like(config["kps"]), d.qvel[leg_qvel_adr], config["kds"],
            )
            arm_tau = pd_control(
                arm_target, d.qpos[arm_qpos_adr], config["kps_arms"],
                np.zeros_like(config["kps_arms"]), d.qvel[arm_qvel_adr], config["kds_arms"],
            )
            leg_tau = np.clip(np.nan_to_num(leg_tau, nan=0.0, posinf=0.0, neginf=0.0), -200.0, 200.0)
            arm_tau = np.clip(np.nan_to_num(arm_tau, nan=0.0, posinf=0.0, neginf=0.0), -200.0, 200.0)
            d.ctrl[leg_act_idx] = leg_tau
            d.ctrl[arm_act_idx] = arm_tau
        if extra_act_idx.size > 0:
            d.ctrl[extra_act_idx] = gripper_cmd

        mujoco.mj_step(m, d)
        counter += 1

        if counter % config["control_decimation"] == 0:
            single_obs, _ = compute_observation(d, config, action, cmd, height_cmd, qpos_adr, qvel_adr)
            obs_history.append(single_obs)
            if args.encoder_mode == "no_encoder" or encoder is None:
                z_t = np.zeros(RMA_LATENT_DIM, dtype=np.float32)
            else:
                if args.encoder_mode == "zero_et":
                    e_t = np.zeros(RMA_ET_DIM, dtype=np.float32)
                else:  # full
                    e_t = build_et_mujoco(d, qpos_adr, left_hand_force, right_hand_force)
                with torch.no_grad():
                    z_t = encoder(torch.from_numpy(e_t).unsqueeze(0).float()).numpy().squeeze()
            z_history[1:, :] = z_history[:-1, :].copy()
            z_history[0, :] = z_t
            z_flat = np.flip(z_history, axis=0).flatten().astype(np.float32)
            proprio = np.concatenate(list(obs_history), axis=0)
            actor_obs = np.concatenate([proprio, z_flat], axis=0).astype(np.float32)
            assert actor_obs.shape[0] == config["num_obs"], (actor_obs.shape[0], config["num_obs"])
            action = policy(torch.from_numpy(actor_obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * config["action_scale"] + config["default_angles"]

            roll, pitch, _ = quat_to_rpy(d.qpos[3:7])
            log["t"].append(t)
            log["base_z"].append(float(d.qpos[2]))
            log["base_roll"].append(float(roll))
            log["base_pitch"].append(float(pitch))
            log["action_max"].append(float(np.max(np.abs(action))))
            log["left_sp_tgt"].append(float(arm_target[1])); log["left_sp_act"].append(float(d.qpos[qpos_adr[NUM_LEG_JOINTS + 1]]))
            log["left_sr_tgt"].append(float(arm_target[2])); log["left_sr_act"].append(float(d.qpos[qpos_adr[NUM_LEG_JOINTS + 2]]))
            log["left_elbow_tgt"].append(float(arm_target[4])); log["left_elbow_act"].append(float(d.qpos[qpos_adr[NUM_LEG_JOINTS + 4]]))
            log["right_sp_tgt"].append(float(arm_target[8])); log["right_sp_act"].append(float(d.qpos[qpos_adr[NUM_LEG_JOINTS + 8]]))
            log["right_sr_tgt"].append(float(arm_target[9])); log["right_sr_act"].append(float(d.qpos[qpos_adr[NUM_LEG_JOINTS + 9]]))
            log["right_elbow_tgt"].append(float(arm_target[11])); log["right_elbow_act"].append(float(d.qpos[qpos_adr[NUM_LEG_JOINTS + 11]]))

            if counter % (config["control_decimation"] * 50) == 0:
                print(f"t={t:.2f}  base_z={d.qpos[2]:.3f}  z_t[0]={z_t[0]:.3f}")

    try:
        if record_video:
            print("Headless recording mode (no live viewer).")
            while (counter * config["simulation_dt"]) < duration:
                step_once()
                if counter % render_every == 0:
                    video.capture(d)
        else:
            with mujoco.viewer.launch_passive(m, d) as viewer:
                start = time.time()
                while viewer.is_running() and (time.time() - start) < duration:
                    step_start = time.time()
                    step_once()
                    viewer.sync()
                    delay = m.opt.timestep - (time.time() - step_start)
                    if delay > 0:
                        time.sleep(delay)
    finally:
        if video is not None:
            video.close()
            print(f"Wrote video: {video.path}")
        if log["t"]:
            save_plots(log, run_dir)
            print(f"Saved plots to {run_dir}")


if __name__ == "__main__":
    main()
