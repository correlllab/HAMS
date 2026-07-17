#!/usr/bin/env python3
"""Two-controller HANDOFF ORCHESTRATOR (piece 3 of the two-controller handoff).

Operator-driven, ONE-WAY sequence that swaps the live robot from the whole-body
controller to lower-body-MPC (legs) + frame-task (arms) WITHOUT a restart or a
gantry, by commanding the selector mux (selector_node.py) over its localhost UDP
control socket.

Sequence (each hand-over waits for an explicit operator ENTER, and every flip is
gated by the selector on the target source being warm + non-stale):

  0. PRECHECK   both channels on A (fullbody), robot standing on the fullbody.
  1. ARMS       wait until the frame-task upper source is warm+fresh (it inits at
                the measured arm pose) -> operator ENTER -> flip UPPER A->B.
                Overlap, not a gap: the fullbody still feeds legs; the 2s upper
                watchdog in the safety layer never trips.
  2. LEGS       wait until the lower-body-MPC + estimator source is warm+fresh
                -> operator ENTER -> flip LOWER A->B.
  3. RELEASE    fullbody is now feeding nothing -> operator ENTER -> stop it
                (frees a whole-body planner; two MJPC planners no longer overlap).
  4. DONE       lower-body stand-6 + frame-task grasp own the robot. No swap-back.

This tool NEVER touches the robot directly and NEVER touches the safety layer.
It only talks UDP to the selector and (optionally, operator-confirmed) sends
SIGINT to the fullbody process. It is safe to run offline against a selector with
no robot attached.
"""

import argparse
import json
import os
import signal
import socket
import sys
import time


class SelectorClient:
    """Tiny request/reply UDP client for the selector control socket."""

    def __init__(self, host, port, timeout=1.0):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(timeout)

    def _rpc(self, line):
        self._sock.sendto(line.encode("utf-8"), self._addr)
        try:
            data, _ = self._sock.recvfrom(4096)
        except socket.timeout:
            return None
        try:
            return json.loads(data.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return None

    def status(self):
        return self._rpc("status")

    def flip(self, channel, target):
        return self._rpc(f"flip {channel} {target}")


def _fmt_status(st):
    if not st:
        return "<no reply from selector>"
    s = st.get("status", st)
    a = s.get("age_ms", {})
    return (f"lower={s.get('lower_src')} upper={s.get('upper_src')} | "
            f"ages(ms) full={a.get('full')} lowerB={a.get('lower_b')} "
            f"upperB={a.get('upper_b')} | stale>{s.get('stale_ms')}ms")


def _prompt(msg):
    try:
        return input(msg)
    except (EOFError, KeyboardInterrupt):
        print("\n[orch] aborted by operator.", flush=True)
        sys.exit(1)


def _wait_source_warm(cli, which, stale_ms, poll_hz=5.0):
    """Block until the named B source (which in {'lower_b','upper_b'}) is fresh."""
    label = {"lower_b": "lower-body-MPC (legs)", "upper_b": "frame-task (arms)"}[which]
    print(f"[orch] waiting for {label} to be warm+fresh (age < {stale_ms:.0f}ms)...",
          flush=True)
    dt = 1.0 / poll_hz
    while True:
        st = cli.status()
        if st and st.get("ok"):
            age = st["status"]["age_ms"].get(which)
            if age is not None and age <= stale_ms:
                print(f"[orch]   {label} warm (age {age:.0f}ms). {_fmt_status(st)}",
                      flush=True)
                return
            shown = "never" if age is None else f"{age:.0f}ms"
            print(f"[orch]   ...{label} not ready (age {shown})", flush=True)
        else:
            print("[orch]   ...no reply from selector (is it up?)", flush=True)
        time.sleep(dt)


def _do_flip(cli, channel, target):
    st = cli.flip(channel, target)
    if not st:
        print(f"[orch] FLIP {channel}->{target}: NO REPLY from selector. Aborting.",
              flush=True)
        sys.exit(2)
    ok = st.get("ok")
    print(f"[orch] FLIP {channel}->{target}: {'OK' if ok else 'REFUSED'} :: "
          f"{st.get('msg')} | {_fmt_status(st)}", flush=True)
    if not ok:
        print("[orch] Gate refused the flip (source not warm/fresh). "
              "Not proceeding. Fix the source and retry this step.", flush=True)
        sys.exit(3)


def _read_pidfile(path):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _stop_fullbody(args):
    pid = args.fullbody_pid or (_read_pidfile(args.fullbody_pidfile)
                                if args.fullbody_pidfile else None)
    if pid is None:
        print("[orch] RELEASE: no --fullbody-pid/--fullbody-pidfile given. "
              "Stop the fullbody controller manually now (Ctrl-C its terminal).",
              flush=True)
        _prompt("[orch] press ENTER once the fullbody controller is stopped... ")
        return
    ans = _prompt(f"[orch] send SIGINT to fullbody pid {pid}? [y/N] ").strip().lower()
    if ans != "y":
        print("[orch] skipped killing fullbody (operator declined).", flush=True)
        return
    try:
        os.kill(pid, signal.SIGINT)
        print(f"[orch] SIGINT -> pid {pid}. Fullbody planner released.", flush=True)
    except ProcessLookupError:
        print(f"[orch] pid {pid} not found (already stopped?).", flush=True)
    except PermissionError:
        print(f"[orch] permission denied signalling pid {pid}.", flush=True)


def run(args):
    cli = SelectorClient(args.control_host, args.control_port)
    stale_ms = args.stale_ms

    print("=" * 72)
    print(" TWO-CONTROLLER HANDOFF  (fullbody -> lowerbody-MPC + frametask)")
    print(" one-way, operator-gated. Ctrl-C aborts before any flip is committed.")
    print("=" * 72)

    # 0. precheck
    st = cli.status()
    if not st or not st.get("ok"):
        print("[orch] PRECHECK FAILED: selector not answering on "
              f"{args.control_host}:{args.control_port}. Is selector_node up?", flush=True)
        sys.exit(4)
    s = st["status"]
    print(f"[orch] PRECHECK: {_fmt_status(st)}", flush=True)
    if s.get("lower_src") != "A" or s.get("upper_src") != "A":
        ans = _prompt("[orch] channels are NOT both on A(fullbody). "
                      "Continue anyway? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit(5)
    if s["age_ms"].get("full") is None or s["age_ms"]["full"] > stale_ms:
        print("[orch] WARNING: fullbody source is not fresh — is the robot actually "
              "standing on the whole-body controller?", flush=True)
    _prompt("[orch] robot standing on fullbody? press ENTER to begin ARM handoff... ")

    # 1. arms: frametask -> upper
    _wait_source_warm(cli, "upper_b", stale_ms)
    _prompt("[orch] ARM handoff — ENTER to flip UPPER A->B (frametask takes arms)... ")
    _do_flip(cli, "upper", "B")

    # 2. legs: lowerbody-MPC -> lower
    print("[orch] arms on frametask. Now warm up lower-body-MPC + estimator.", flush=True)
    _wait_source_warm(cli, "lower_b", stale_ms)
    _prompt("[orch] LEG handoff — ENTER to flip LOWER A->B (lowerbody-MPC takes legs)... ")
    _do_flip(cli, "lower", "B")

    # 3. release fullbody
    print("[orch] both channels on B. Fullbody now feeds nothing.", flush=True)
    _prompt("[orch] ENTER to RELEASE (stop) the fullbody controller... ")
    _stop_fullbody(args)

    # 4. done
    print("[orch] DONE. lower-body stand-6 + frametask own the robot. "
          f"{_fmt_status(cli.status())}", flush=True)
    print("[orch] (one-way: no swap-back path)", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description="Two-controller handoff orchestrator")
    p.add_argument("--control-host", dest="control_host", default="127.0.0.1")
    p.add_argument("--control-port", dest="control_port", type=int, default=47700)
    p.add_argument("--stale-ms", dest="stale_ms", type=float, default=100.0,
                   help="freshness threshold for warm-up polling (match selector --stale-sec)")
    p.add_argument("--fullbody-pid", dest="fullbody_pid", type=int, default=0,
                   help="if set, offer to SIGINT this pid at RELEASE")
    p.add_argument("--fullbody-pidfile", dest="fullbody_pidfile", default="",
                   help="if set, read the fullbody pid from this file at RELEASE")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fullbody_pid == 0:
        args.fullbody_pid = None
    run(args)


if __name__ == "__main__":
    main()
