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
