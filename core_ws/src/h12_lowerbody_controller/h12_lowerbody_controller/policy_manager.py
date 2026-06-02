"""Policy registry + safe-handover switching.

The manager owns the set of lower-body policies and exactly one *active* policy.
A switch request does not take effect immediately: the manager enters a SETTLING
state and only commits the swap once a **handover gate** confirms the robot is
standing still with its arms at the nominal pose — so both the outgoing and
incoming policies see in-distribution observations at the switch instant. The
incoming policy is then ``reset`` from the live state for a bumpless start.

The gate is deliberately conservative. ``/lowstate`` exposes no base *linear*
velocity, so "not drifting" is approximated by small joint velocities, small base
angular velocity, and a (near-)zero velocity command — adequate because we only
ever switch *to* standing from a commanded stop. Tune via ``GateConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import NUM_LEG_JOINTS, NUM_POLICY_JOINTS, LegCommand, Policy, RobotState


@dataclass
class GateConfig:
    cmd_eps: float = 0.05        # ||velocity command|| below this = "stop requested"
    gyro_eps: float = 0.30       # rad/s, base angular velocity
    dq_eps: float = 0.50         # rad/s, max |joint velocity|
    arm_eps: float = 0.20        # rad, max |arm joint - 0| (arms at home)
    leg_eps: float = 0.30        # rad, max |leg joint - incoming nominal|
    hold_ticks: int = 10         # consecutive passing ticks required (debounce)
    warn_period_ticks: int = 100  # log a "still waiting" note this often


class PolicyManager:
    def __init__(self, policies: dict[str, Policy], active_name: str,
                 gate: GateConfig | None = None, log=print):
        if active_name not in policies:
            raise KeyError(f"active policy {active_name!r} not in {list(policies)}")
        self._policies = policies
        self._active = active_name
        self._desired = active_name
        self._gate = gate or GateConfig()
        self._log = log
        self._pass_count = 0
        self._settle_ticks = 0

    # -- introspection -------------------------------------------------------
    @property
    def active_name(self) -> str:
        return self._active

    @property
    def desired_name(self) -> str:
        return self._desired

    @property
    def is_settling(self) -> bool:
        return self._desired != self._active

    def names(self) -> list[str]:
        return list(self._policies)

    # -- control -------------------------------------------------------------
    def request_switch(self, name: str) -> bool:
        if name not in self._policies:
            self._log(f"[switch] unknown policy {name!r}; have {list(self._policies)}")
            return False
        if name == self._active and not self.is_settling:
            return True  # already there
        if name != self._desired:
            self._log(f"[switch] requested {self._active!r} -> {name!r}; waiting for handover gate")
        self._desired = name
        self._pass_count = 0
        self._settle_ticks = 0
        return True

    def _gate_ok(self, state: RobotState, incoming: Policy) -> bool:
        g = self._gate
        if np.linalg.norm(state.cmd) > g.cmd_eps:
            return False
        if np.linalg.norm(state.gyro) > g.gyro_eps:
            return False
        if np.max(np.abs(state.dq)) > g.dq_eps:
            return False
        arms = state.q[NUM_LEG_JOINTS:NUM_POLICY_JOINTS]
        if np.max(np.abs(arms)) > g.arm_eps:
            return False
        legs = state.q[:NUM_LEG_JOINTS]
        if np.max(np.abs(legs - incoming.nominal_lower)) > g.leg_eps:
            return False
        return True

    def update(self, state: RobotState) -> str | None:
        """Advance the switch state machine. Returns the new active name if a
        switch committed this tick, else None. Call once per tick before run()."""
        if not self.is_settling:
            return None

        incoming = self._policies[self._desired]
        self._settle_ticks += 1
        if self._gate_ok(state, incoming):
            self._pass_count += 1
        else:
            self._pass_count = 0
            if self._settle_ticks % self._gate.warn_period_ticks == 0:
                self._log(f"[switch] still waiting to enter {self._desired!r} "
                          f"(robot not yet standing-still with arms home)")

        if self._pass_count >= self._gate.hold_ticks:
            incoming.reset(state)
            old = self._active
            self._active = self._desired
            self._pass_count = 0
            self._settle_ticks = 0
            self._log(f"[switch] committed {old!r} -> {self._active!r}")
            return self._active
        return None

    def reset_active(self, state: RobotState) -> None:
        """Restart the active policy's warm-up from the current state. Call this
        at band release so a history-based policy (FAME) discards the band-held
        (out-of-distribution) history and warms up free-standing, matching the
        standalone deploy that the policy was validated against."""
        self._policies[self._active].reset(state)

    def run(self, state: RobotState) -> LegCommand:
        return self._policies[self._active].compute(state)
