"""gRPC ``mujoco_mpc.Agent`` wrapper — the Python analog of the C++ node's embedded
``mjpc::Agent``.

The C++ node embeds the planner and calls ``PlanIteration`` in-process at full rate;
here each call is a gRPC round-trip to the ``agent_server`` child process, so the
planner runs slower (the C++ header's "~65x slower than realtime" caveat — this is
the "Python gRPC bridge" it refers to). The control logic is identical; only the
planner throughput differs.

C++ embedded-Agent op            ->  this client / gRPC Agent
  SetTaskList + LoadModel + Init ->  Agent(task_id, model=None)   (server loads bundled task XML)
  SetParamByName("residual_*")   ->  set_task_parameters({...})   (live Strategy switch)
  SetState(mjData)               ->  set_state(time, qpos, qvel)  (plain arrays; layout-independent)
  PlanIteration(pool)            ->  planner_step()
  ActionFromPolicy(.., time)     ->  get_action(time)

``model=None`` means NO model is serialized over gRPC (avoids the mjb struct
mismatch between pip mujoco 3.2.3 and the agent_server built at commit 088079e);
``set_state`` sends plain qpos/qvel numbers, which are version-independent.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from mujoco_mpc import agent as agent_lib


class MjpcAgentClient:
    """Owns the gRPC Agent + (optionally) a background planner thread.

    Parameters
    ----------
    task_id : the MJPC task to load (the C++ default is ``"Lean H12"``).
    strategy : initial Lean Strategy index (C++ default 6 = stand).
    strategy_param : the task-parameter key for the strategy. The gRPC
        ``GetTaskParameters`` typically exposes the residual numeric WITHOUT the
        ``residual_`` prefix, so the default is ``"Strategy"``; verify on first run
        with :meth:`task_parameters`.
    extra_flags : extra CLI flags forwarded to the ``agent_server`` child.
    """

    def __init__(
        self,
        task_id: str = "Lean H12",
        strategy: int = 6,
        strategy_param: str = "Strategy",
        extra_flags=(),
    ):
        self._agent = agent_lib.Agent(task_id=task_id, extra_flags=list(extra_flags))
        self._strategy_param = strategy_param

        # Learn the server model size once so set_robot_state can keep any
        # task-object slots at their home values (≙ the C++ "task slots keep home
        # defaults"; the server's task XML may carry extra qpos beyond the robot).
        st = self._agent.get_state()
        self._home_qpos = np.asarray(st.qpos, dtype=float)
        self._home_qvel = np.asarray(st.qvel, dtype=float)
        self._nq = self._home_qpos.shape[0]
        self._nv = self._home_qvel.shape[0]

        self._plan_thread = None
        self._plan_stop = threading.Event()
        self._plan_count = 0

        # gRPC round-trip latency telemetry (EWMA mean, ms) per RPC — the key
        # diagnostic for "how slow is the gRPC planner?". Updated from both the
        # control thread (set_state/get_action) and the planner thread
        # (planner_step); a float-EWMA race is harmless for telemetry.
        self._lat = {"planner_step": 0.0, "set_state": 0.0, "get_action": 0.0, "total_cost": 0.0}
        self._lat_alpha = 0.1

        self.set_strategy(strategy)

    def _record(self, key: str, dt_s: float) -> None:
        ms = dt_s * 1e3
        self._lat[key] += self._lat_alpha * (ms - self._lat[key])

    # -- properties ----------------------------------------------------------
    @property
    def nq(self) -> int:
        return self._nq

    @property
    def nv(self) -> int:
        return self._nv

    @property
    def plan_count(self) -> int:
        return self._plan_count

    def task_parameters(self):
        """Current task parameters (use to discover the exact Strategy key)."""
        return self._agent.get_task_parameters()

    # -- strategy ------------------------------------------------------------
    def set_strategy(self, strategy: int) -> None:
        """Live Strategy switch (≙ C++ ``SetParamByName("residual_Strategy", N)``)."""
        try:
            self._agent.set_task_parameters({self._strategy_param: float(strategy)})
        except Exception:
            # Fall back to the residual-prefixed key if that is what the server wants.
            self._agent.set_task_parameters({"residual_Strategy": float(strategy)})

    # -- state / action ------------------------------------------------------
    def set_robot_state(self, t: float, qpos_robot: np.ndarray, qvel_robot: np.ndarray) -> None:
        """Set the agent state from reconstructed robot dofs, keeping task slots home."""
        qpos = self._home_qpos.copy()
        qvel = self._home_qvel.copy()
        nq = min(len(qpos_robot), self._nq)
        nv = min(len(qvel_robot), self._nv)
        qpos[:nq] = qpos_robot[:nq]
        qvel[:nv] = qvel_robot[:nv]
        t0 = time.perf_counter()
        self._agent.set_state(time=t, qpos=qpos, qvel=qvel)
        self._record("set_state", time.perf_counter() - t0)

    def get_action(self, t=None) -> np.ndarray:
        t0 = time.perf_counter()
        a = self._agent.get_action(time=t)
        self._record("get_action", time.perf_counter() - t0)
        return np.asarray(a)

    def get_total_cost(self) -> float:
        """Sum of weighted cost terms (planner convergence signal)."""
        t0 = time.perf_counter()
        c = self._agent.get_total_cost()
        self._record("total_cost", time.perf_counter() - t0)
        return c

    def latencies(self) -> dict:
        """EWMA gRPC round-trip latency (ms) per RPC, for planner logging."""
        return dict(self._lat)

    # -- planner cadence -----------------------------------------------------
    def plan_n(self, n: int) -> None:
        """Run ``n`` synchronous planning iterations on the current state (≙ the
        C++ ``--sync_plan`` / ``--plan_rate_hz`` modes)."""
        for _ in range(n):
            t0 = time.perf_counter()
            self._agent.planner_step()
            self._record("planner_step", time.perf_counter() - t0)
            self._plan_count += 1

    def start_planner_thread(self) -> None:
        """Continuously replan in the background (≙ the C++ async ``Plan()`` thread)."""
        if self._plan_thread is not None:
            return
        self._plan_stop.clear()

        def _loop():
            while not self._plan_stop.is_set():
                try:
                    t0 = time.perf_counter()
                    self._agent.planner_step()
                    self._record("planner_step", time.perf_counter() - t0)
                    self._plan_count += 1
                except Exception:  # pragma: no cover - keep the loop alive on a transient RPC error
                    pass

        self._plan_thread = threading.Thread(target=_loop, name="mjpc-planner", daemon=True)
        self._plan_thread.start()

    def stop_planner_thread(self) -> None:
        self._plan_stop.set()
        if self._plan_thread is not None:
            self._plan_thread.join(timeout=2.0)
            self._plan_thread = None

    def close(self) -> None:
        self.stop_planner_thread()
        self._agent.close()
