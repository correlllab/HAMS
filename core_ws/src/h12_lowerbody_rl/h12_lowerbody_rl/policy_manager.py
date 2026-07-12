"""Policy registry + idle/active state with safe-handover switching.

The manager holds the lower-body policies and at most one *active* policy.
``active_name`` is ``None`` when **idle** — the robot is held by the elastic band
and no policy drives the legs. A requested policy only becomes active once
*committed*:

* **first activation** (idle -> policy): the node releases the band, then calls
  ``commit`` (no gate — the band-held pose is already a stable handover state).
* **switch** (policy -> policy): committed only when the **handover gate** passes
  (robot standing still, arms home) so both policies see in-distribution obs.

The incoming policy is ``reset()`` at commit for a clean free-standing warm-up.
The band itself is owned by the node, not here — this class is pure policy logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import LegCommand, Policy, RobotState, gravity_from_quat_fame


@dataclass
class GateConfig:
    # A switch is a "handover": commit only when the robot is in a stable
    # standing moment so the incoming policy (which resets + warms up) takes over
    # cleanly. We do NOT require a specific arm/leg pose — both policies balance
    # from a standing pose, and the arms are held by the IK at its own home
    # (not at 0), so a pose match is neither achievable nor needed.
    cmd_eps: float = 0.10        # ||velocity command|| below this = "stop requested"
    gyro_eps: float = 1.00       # rad/s, base angular velocity (base roughly still)
    # The walk policy balances dynamically and keeps the legs stepping (~4 rad/s)
    # even at cmd=0, so it never produces joint stillness — gating walk->FAME on
    # low dq would deadlock. Gate the handover on upright + slow base + stop
    # requested instead, and let the incoming FAME policy settle the legs.
    dq_eps: float = 6.0          # rad/s, max |joint velocity| (allow walk stepping)
    upright_eps: float = 0.25    # max horizontal gravity component (~14 deg tilt)
    hold_ticks: int = 5          # consecutive passing ticks required (debounce)
    warn_period_ticks: int = 100  # log a "still waiting" note this often


class PolicyManager:
    def __init__(self, policies: dict[str, Policy], gate: GateConfig | None = None, log=print):
        self._policies = policies
        self._active: str | None = None     # None = idle (band-held, no policy)
        self._desired: str | None = None
        self._gate = gate or GateConfig()
        self._log = log
        self._pass_count = 0
        self._settle_ticks = 0
        # Whether the pending switch's gate requires a stop (||cmd|| <= cmd_eps).
        # True for stop-and-settle handovers (e.g. walk->FAME); False when handing
        # INTO a locomotion policy that must engage while a velocity is commanded
        # (fame->walk) — there the stop check would deadlock (the trigger IS a high
        # cmd), so we gate on upright+still only and let the incoming policy step.
        self._require_stop = True

    # -- introspection -------------------------------------------------------
    @property
    def active_name(self) -> str | None:
        return self._active

    @property
    def desired_name(self) -> str | None:
        return self._desired

    def is_idle(self) -> bool:
        return self._active is None

    def is_pending(self) -> bool:
        return self._desired is not None and self._desired != self._active

    def names(self) -> list[str]:
        return list(self._policies)

    def has(self, name: str) -> bool:
        return name in self._policies

    def desired_policy(self) -> Policy | None:
        """The pending (requested but not yet committed) policy, or None if idle."""
        return self._policies.get(self._desired) if self._desired is not None else None

    # -- control -------------------------------------------------------------
    def request(self, name: str, require_stop: bool = True) -> tuple[bool, str]:
        """Request a policy. Returns (accepted, message). Commit happens later
        (after band release for the first activation, or after the gate for a
        switch). ``require_stop`` controls the switch gate: True = full
        stop-and-settle handover; False = gate on upright+still only (for
        engaging a locomotion policy while a velocity is commanded)."""
        if name not in self._policies:
            return False, f"unknown policy {name!r}; have {self.names()}"
        if name == self._active and not self.is_pending():
            return True, f"{name!r} already active"
        # Idempotent while the same switch is already pending: don't reset the
        # gate's pass_count (an auto-switch caller may re-request every tick).
        if self.is_pending() and name == self._desired:
            self._require_stop = require_stop
            return True, f"{name!r} already requested"
        self._desired = name
        self._require_stop = require_stop
        self._pass_count = 0
        self._settle_ticks = 0
        kind = "activate (idle->%s)" % name if self.is_idle() else f"switch {self._active!r}->{name!r}"
        self._log(f"[policy] requested {kind}")
        return True, f"requested {name!r}"

    def _gate_ok(self, state: RobotState, require_stop: bool = True) -> bool:
        """Stable standing moment: base + joints still and upright, and (when
        ``require_stop``) not commanded to move. Pose-agnostic on purpose (see
        GateConfig). ``require_stop=False`` drops only the velocity-command veto
        so a locomotion policy can engage while a cmd is held (fame->walk)."""
        g = self._gate
        if require_stop and np.linalg.norm(state.cmd) > g.cmd_eps:
            return False
        if np.linalg.norm(state.gyro) > g.gyro_eps:
            return False
        if np.max(np.abs(state.dq)) > g.dq_eps:
            return False
        grav = gravity_from_quat_fame(state.quat)   # [0,0,-1] when upright
        if float(np.hypot(grav[0], grav[1])) > g.upright_eps:
            return False
        return True

    def commit(self, state: RobotState) -> str:
        """Activate the desired policy now (reset for a clean warm-up). Used for
        the first activation right after band release."""
        incoming = self._policies[self._desired]
        incoming.reset(state)
        old = self._active
        self._active = self._desired
        self._pass_count = 0
        self._settle_ticks = 0
        self._log(f"[policy] committed {old!r} -> {self._active!r}")
        return self._active

    def update_switch(self, state: RobotState) -> str | None:
        """Advance a *switch* between two active policies (gated). Returns the new
        active name on commit, else None. Only call when already active and a
        different policy is desired."""
        if self.is_idle() or not self.is_pending():
            return None
        self._settle_ticks += 1
        if self._gate_ok(state, self._require_stop):
            self._pass_count += 1
        else:
            self._pass_count = 0
            if self._settle_ticks % self._gate.warn_period_ticks == 0:
                self._log(f"[policy] waiting to enter {self._desired!r} "
                          f"(need standing-still with arms home)")
        if self._pass_count >= self._gate.hold_ticks:
            return self.commit(state)
        return None

    def reset_active(self, state: RobotState) -> None:
        if self._active is not None:
            self._policies[self._active].reset(state)

    def run(self, state: RobotState) -> LegCommand:
        return self._policies[self._active].compute(state)
