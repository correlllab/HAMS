#!/usr/bin/env python3
"""FABEL probe (2026-07-07): the AGENT-SERVER half of the one-shot
PlanIteration A/B against the deploy node.

Runs the dist-packages agent_server (same libmjpc.a as the node) on the
CANONICAL frozen stand state (= bench_stand.py key spawn: base z 1.020,
identity quat, legs at the 'stand' keyframe, arms zero, qvel 0), stepping
the planner exactly like the node's free-running planner thread while the
plant is frozen. Run with H12_PDUMP=1 so the planner prints its internals
(the server subprocess inherits the env); diff those PDUMP lines against
the node's from the same frozen bench.

  docker exec -e H12_PDUMP=1 -e H12_PDUMP_EVERY=50 hams_ros \
      python3 /home/code/h12_sim_scripts/probe_pdump.py
"""
import os

import numpy as np
from mujoco_mpc import agent as agent_lib

TASK = os.environ.get("PROBE_TASK", "Stabilize H12 Magpie")
N_ITERS = int(os.environ.get("PROBE_ITERS", "300"))
DT = float(os.environ.get("PROBE_DT", "0.01"))

with agent_lib.Agent(task_id=TASK) as ag:
    st = ag.get_state()
    qpos = np.array(st.qpos)
    qvel = np.array(st.qvel)
    print(f"[probe] nq={len(qpos)} nv={len(qvel)} t0={st.time:.3f}",
          flush=True)
    print(f"[probe] task params: {ag.get_task_parameters()}", flush=True)

    # canonical frozen stand state (object/task slots beyond the robot keep
    # the server's home-keyframe defaults, same as the node's sd)
    qpos[0:3] = [0.0, 0.0, 1.020]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[7:13] = [0.0, -0.25, 0.12, 0.35, -0.21, 0.0]
    qpos[13:19] = [0.0, -0.25, -0.12, 0.35, -0.21, 0.0]
    qpos[19:34] = 0.0
    qvel[:33] = 0.0

    t = 0.0
    for i in range(N_ITERS):
        ag.set_state(time=t, qpos=qpos, qvel=qvel)
        ag.planner_step()
        t += DT
        if i % 50 == 0:
            a = ag.get_action(time=t)
            print(f"[probe] i={i:4d} t={t:6.2f}  act hipP={a[1]:+.4f} "
                  f"knee={a[3]:+.4f} ankP={a[4]:+.4f}", flush=True)

    a = ag.get_action(time=t)
    print(f"[probe] FINAL act hipP={a[1]:+.4f} knee={a[3]:+.4f} "
          f"ankP={a[4]:+.4f}", flush=True)
    w = ag.get_cost_weights()
    print(f"[probe] cost weights ({len(w)}): {w}", flush=True)
