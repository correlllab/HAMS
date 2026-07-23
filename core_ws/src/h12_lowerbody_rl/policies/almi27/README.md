# ALMI27 — wrist-inclusive ALMI lower-body policy (vendored)

`policy_lstm_almi27_7000.pt` is the **in-house-trained** wrist-inclusive ALMI
locomotion/balance policy for the Unitree H1-2. It is an *additional* policy
alongside stock `almi` (which is left untouched).

## What's different from `almi`

| | `almi` (stock) | **`almi27`** |
|---|---|---|
| Robot view | 21 DoF, **wrists dropped** (`/lowstate` minus idx 17-19, 24-26) | **all 27 DoF** (full arms incl. elbow_roll + 2 wrists per arm) |
| Observation | 65-d | **77-d** |
| Weights | upstream `policy_lstm_12800.pt` (TeleHuman/ALMI-Open) | trained here, 2026-07-22 |
| Leg actions | 12 | 12 (same) |

Because ALMI27 **observes the wrist joints**, the legs stay balanced under full
arm **and** wrist motion — instead of treating wrist motion as an unobserved
disturbance the way stock `almi` does.

## Provenance / training

Full ALMI adversarial campaign (TeleHuman/ALMI-Open framework, BSD-3-Clause), run
in-house on a 27-DoF H1-2 with the six wrist DoF added:

1. **lower-1** — robust base: legs vs. arm+wrist motion (AMASS wave gestures +
   injected smooth-random wrist trajectories), ramped to full via curriculum.
2. **upper-1** — a learned adversary driving the whole 15-DoF upper body.
3. **lower-2 (this policy, `model_7000`)** — legs retrained against that adversary.

All stages added **1.5 m/s pushes + ±10% motor-strength randomization**. The final
policy held ~900/1000 survival (20 s episodes) at full perturbation. Exported to
TorchScript via the ALMI-Open `PolicyExporterLSTM` (same stateful `forward(obs)` +
`reset_memory()` interface as the stock ALMI export).

## Interface (what `Almi27Policy` in `policy.py` implements)

- Stateful TorchScript LSTM, **50 Hz**, obs 77 → 12 leg actions. `reset_memory()`
  on (re-)engage; stand<->walk onset resets the LSTM (same quirk as `almi`).
- 77-d obs: `gyro*0.25 (3) | proj gravity (3) | cmd*[2,2,0.25] (3) |
  (q27 − default) (27) | dq27*0.05 (27) | prev action (12) | left/right sin phase (2)`.
  `q27`/`default` are full `/lowstate` order (== training URDF order).
- Gait clock 0.8 s; `||cmd|| < 0.1` forces phase 0 → the policy **stands** at zero
  command (covers stand + locomote in one policy). Legs: `target = a*0.25 + default[:12]`.

## Usage

Sim bringup (leaves `almi` as the default):

```bash
ros2 launch h1_bringup h1_sim_bringup.launch.py lowerbody:=almi27
```

`almi27` is `STAND_CAPABLE`, so under `auto_switch` it stands and hands locomotion
to the walk policy on `/cmd_vel`, exactly like `almi`. Pin with `auto_switch:=false`.

## Tuning note

`cmd_gain`/`cmd_smooth_alpha` in `almi27.yaml` are **neutral** (identity / off) —
this is a freshly-trained checkpoint, so measure its `/cmd_vel` tracking and re-tune
if it undertracks (stock `almi` needed `[1.5,1,2]` / `0.15` for the upstream
checkpoint's undertracking; ALMI27 may differ).
