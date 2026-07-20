# ALMI lower-body policy (vendored)

`policy_lstm_12800.pt` is the trained H1-2 lower-body locomotion policy from
**[TeleHuman/ALMI-Open](https://github.com/TeleHuman/ALMI-Open)** ("Adversarial
Locomotion and Motion Imitation for Humanoid Policy Learning",
[arXiv:2504.14305](https://arxiv.org/abs/2504.14305)), licensed
**BSD-3-Clause**. It is the checkpoint that repo ships at
`Data_Collection/mujoco/policy_lstm_12800.pt` and drives with
`save_trajectory_data.py` to collect the ALMI-X dataset (copied unmodified,
2026-07-18).

## Interface (what `AlmiPolicy` in `policy.py` implements)

- TorchScript `PolicyExporterLSTM` (legged_gym export): an LSTM whose
  hidden/cell state live **inside the module** as buffers, updated on every
  `policy(obs)` call, plus an MLP actor head. `policy.reset_memory()` zeros the
  recurrent state — call it whenever the policy (re-)engages.
- Runs at 50 Hz on a 21-DoF view of the H1-2 (no wrists): legs 0-11, torso 12,
  shoulder pitch/roll/yaw + elbow per arm. Same ordering as `/lowstate` with
  the wrist joints (17-19, 24-26) dropped.
- 65-d observation:
  `[gyro*0.25 (3) | projected gravity (3) | cmd*[2,2,0.25] (3) |
    (q21-default) (21) | dq21*0.05 (21) | prev action (12) |
    left_sin_phase, right_sin_phase (2)]`
  Gait clock: 0.8 s period, right foot offset by half a period; when
  `||cmd|| < 0.1` both phase terms are forced to 0 and the policy stands
  still (this is why it covers stand + locomote in one policy).
- 12 leg actions: `target_q = action * 0.25 + default_angles[:12]`.

Upper body: during ALMI-X collection the torso+arms were PD-driven externally
(replayed AMASS motions) — the lower policy only *observes* them. In HAMS the
frame_task IK plays that role through the safety layer.

## Usage

### Real robot (companion desktop)

The desktop bringup defaults to ALMI (`lowerbody:=almi`); every leg path
stays behind the `start_position_verified` interlock. Hoist the robot first —
snug, legs free to bend — and keep a hand on the e-stop. Launching IS the
start of motion:

```bash
ros2 launch h1_bringup h1_real_desktop_bringup.launch.py start_position_verified:=true
```

What happens, in order (`lowerbody_controller_node`):

1. The three policies load, then `active_policy='almi': auto-activating`.
2. **Pre-pose**: the legs ramp (cosine blend over `prepose_ramp_s`, default
   2 s) from the measured pose to the ALMI crouch (hip −0.4, knee 0.8,
   ankle −0.4). The ramp starts at the measured pose so full-gain PD never
   sees a step — stepping straight to nominal tripped the safety layer's
   hip-yaw dq e-stop on the real robot.
3. **Arming gate**: the node holds the crouch and reminds every 5 s until
   the operator confirms:

   ```bash
   ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger
   ```

   If the log says `NOT SETTLED (timeout)`, do **not** confirm — the legs
   never reached the crouch; investigate first. (The sim bringup passes
   `engage_wait_for_confirm: false` and auto-engages for unattended runs.)
4. On confirm the LSTM engages at zero command (stand). The
   `elastic band toggle service unavailable` WARN right after is expected on
   real — the band service is sim-only; the policy re-seeds its memory at
   that moment (clean trained stand-start).
5. Slacken the hoist gradually. There is no post-engage ramp — the hoist is
   the ramp.

### Driving it

One policy covers both modes: `||[vx, vy, wz]|| < 0.1` stands (gait phase
forced to 0), above it walks. Publish `/cmd_vel` (teleop or nav2):

- Crossing 0.1 from below resets the LSTM once — the trained episode-start
  condition; without it the gait never engages from a quiet stand (see the
  `AlmiPolicy` docstring). Dropping below 0.05 re-arms the reset and the
  robot stands again.
- `cmd_gain` `[1.5, 1.0, 2.0]` amplifies commands before the policy sees
  them (the checkpoint undertracks, yaw worst) and `cmd_smooth_alpha` 0.15
  averages out sign-flipping follower corrections. Start with vx 0.2–0.3.
- `almi` never auto-switches — auto_switch only toggles fame<->walk, so the
  active policy stays pinned.

### Sim

```bash
ros2 launch h1_bringup h1_sim_bringup.launch.py lowerbody:=almi
```
