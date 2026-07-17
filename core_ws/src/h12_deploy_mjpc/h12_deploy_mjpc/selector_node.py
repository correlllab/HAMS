#!/usr/bin/env python3
"""Two-controller SELECTOR / warm mux (piece 2 of the two-controller handoff).

Sits between the three command producers and the safety layer (split_mode) and
decides, per split channel, WHO feeds it -- so the live swap from the whole-body
controller to lower-body-MPC + frame-task arms is just a pointer flip, and the
safety layer never restarts.

  producers (unchanged, hardcoded topics)              selector          safety (split)
  ------------------------------------------           --------          --------------
  fullbody  h12_control_node -> rt/safety/lowcmd_in         \\
     (FULL 27-row: legs 0..11 + arms 12..26)   ============> [A]  ==\\
  lowerbody -> rt/safety/lowcmd_lower_in  =================> [B lower] ==> lowcmd_lower_sel --> low_cmd_lower_in
  frametask -> rt/safety/lowcmd_upper_in  =================> [B upper] ==> lowcmd_upper_sel --> low_cmd_upper_in

The fullbody publishes ONE full command; the selector splits it internally
([0:12] legs, [12:27] arms) so no `--publish_split` flag / deploy rebuild is
needed. Source A therefore feeds BOTH channels; the B sources are the dedicated
split producers. Arm handoff = flip upper A->B; leg handoff = flip lower A->B.

A flip to B is GATED: refused unless that B source is WARM (has published at
least once) and NON-STALE (last message younger than --stale-sec). The mux
holds-last on the un-selected source, so a flip is an overlap, never a gap.

Design notes:
  * Safety layer is UNTOUCHED. It stays split_mode for the whole run; only the
    two INPUT topic names in the split yaml point at *_sel (a config change, not
    code). The per-channel clip still runs on every message the selector emits,
    so the common split yaml must carry locomotion-permissive knee-velocity
    limits or the walk phase estops.
  * NO rclpy: unitree_sdk2py DDS stays the only middleware in-process (matches
    the safety layer / estimator stance). The control plane is a tiny localhost
    UDP socket -- no custom IDL, and it runs/tests with zero DDS.
  * Each *_sel message reproduces the shape the corresponding split producer
    already emits (own rows filled, the rest at LowCmd default), which the
    safety layer's clip is known to accept.
"""

import argparse
import json
import os
import socket
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

# Structural constants -- MUST match the safety layer
# (h12_safety_layer.core.joint_limits.MOTOR_COUNT and the split point). The H1-2
# has 27 actuated joints; legs are motor rows 0..11, torso+arms 12..26.
MOTOR_COUNT = 27
SPLIT_UPPER_START = 12

# Default DDS topics.
FULL_TOPIC = "rt/safety/lowcmd_in"              # fullbody h12_control_node (A, both channels)
LOWER_B_TOPIC = "rt/safety/lowcmd_lower_in"     # lowerbody h12_lower_body_controller (B lower)
UPPER_B_TOPIC = "rt/safety/lowcmd_upper_in"     # frametask upper (B upper)
LOWER_OUT_TOPIC = "rt/safety/lowcmd_lower_sel"  # -> safety split low_cmd_lower_in
UPPER_OUT_TOPIC = "rt/safety/lowcmd_upper_sel"  # -> safety split low_cmd_upper_in


class _Src:
    """Last-seen command from one producer + its arrival time (monotonic)."""

    __slots__ = ("msg", "t")

    def __init__(self):
        self.msg = None
        self.t = 0.0


class Selector:
    def __init__(self, args):
        self._args = args
        self._lock = threading.Lock()
        self._running = False
        self._crc = CRC()

        # per-channel selected source: 'A' = fullbody, 'B' = dedicated producer
        self._lower_src = args.init_lower
        self._upper_src = args.init_upper

        # latest producer commands
        self._full = _Src()        # A: fullbody full 27-row
        self._lower_b = _Src()     # B: lowerbody legs
        self._upper_b = _Src()     # B: frametask arms

        # a safe do-nothing template (all zeros, mode 0) for before-first-command
        self._estop_lower = self._extract_lower(LowCmdDefault())
        self._estop_upper = self._extract_upper(LowCmdDefault())

        # --- DDS ---
        env_domain = os.environ.get("ROS_DOMAIN_ID")
        domain = int(env_domain) if env_domain is not None else args.domain
        if args.iface:
            ChannelFactoryInitialize(domain, args.iface)
        else:
            ChannelFactoryInitialize(domain)

        self._sub_full = ChannelSubscriber(args.full_topic, LowCmd_)
        self._sub_lower_b = ChannelSubscriber(args.lower_in_topic, LowCmd_)
        self._sub_upper_b = ChannelSubscriber(args.upper_in_topic, LowCmd_)
        self._pub_lower = ChannelPublisher(args.lower_out_topic, LowCmd_)
        self._pub_upper = ChannelPublisher(args.upper_out_topic, LowCmd_)

        # --- control plane (localhost UDP) ---
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((args.control_host, args.control_port))
        self._sock.settimeout(0.5)

    # ---- message construction ------------------------------------------------
    def _copy_row(self, dst, src, i):
        d = dst.motor_cmd[i]
        s = src.motor_cmd[i]
        d.mode = s.mode
        d.q = s.q
        d.dq = s.dq
        d.tau = s.tau
        d.kp = s.kp
        d.kd = s.kd

    def _extract_lower(self, src):
        """New LowCmd with only leg rows [0:12] filled from src; rest default."""
        out = LowCmdDefault()
        out.mode_pr = 0
        out.mode_machine = src.mode_machine
        for i in range(SPLIT_UPPER_START):
            self._copy_row(out, src, i)
        out.crc = self._crc.Crc(out)
        return out

    def _extract_upper(self, src):
        """New LowCmd with only torso+arm rows [12:27] filled from src; rest default."""
        out = LowCmdDefault()
        out.mode_pr = 0
        out.mode_machine = src.mode_machine
        for i in range(SPLIT_UPPER_START, MOTOR_COUNT):
            self._copy_row(out, src, i)
        out.crc = self._crc.Crc(out)
        return out

    # ---- subscriber callbacks ------------------------------------------------
    def _on_full(self, msg):
        with self._lock:
            self._full.msg = msg
            self._full.t = time.monotonic()

    def _on_lower_b(self, msg):
        with self._lock:
            self._lower_b.msg = msg
            self._lower_b.t = time.monotonic()

    def _on_upper_b(self, msg):
        with self._lock:
            self._upper_b.msg = msg
            self._upper_b.t = time.monotonic()

    # ---- flip control --------------------------------------------------------
    def _source_for(self, channel, want):
        """Return (Src, name) the given channel would use for source 'A'/'B'."""
        if channel == "lower":
            return (self._full, "fullbody") if want == "A" else (self._lower_b, "lowerbody")
        return (self._full, "fullbody") if want == "A" else (self._upper_b, "frametask")

    def _apply_flip(self, channel, target):
        """Gated flip. Returns (ok, message)."""
        if channel not in ("lower", "upper") or target not in ("A", "B"):
            return False, f"bad flip '{channel} {target}'"
        now = time.monotonic()
        with self._lock:
            src, name = self._source_for(channel, target)
            age = now - src.t if src.msg is not None else None
            if src.msg is None:
                return False, f"refused: {channel}->{target} ({name}) never published"
            if age > self._args.stale_sec:
                return False, (f"refused: {channel}->{target} ({name}) stale "
                               f"{age*1e3:.0f}ms > {self._args.stale_sec*1e3:.0f}ms")
            if channel == "lower":
                self._lower_src = target
            else:
                self._upper_src = target
        return True, f"ok: {channel} -> {target} ({name}, age {age*1e3:.0f}ms)"

    def _status(self):
        now = time.monotonic()
        with self._lock:
            def age(s):
                return round((now - s.t) * 1e3, 1) if s.msg is not None else None
            return {
                "lower_src": self._lower_src,
                "upper_src": self._upper_src,
                "age_ms": {
                    "full": age(self._full),
                    "lower_b": age(self._lower_b),
                    "upper_b": age(self._upper_b),
                },
                "stale_ms": round(self._args.stale_sec * 1e3, 1),
            }

    def _control_loop(self):
        """Parse UDP commands: 'flip <lower|upper> <A|B>' | 'status'."""
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            line = data.decode("utf-8", "replace").strip()
            toks = line.split()
            reply = {"cmd": line, "ok": False, "msg": "unknown command"}
            if toks and toks[0] == "status":
                reply = {"cmd": line, "ok": True, "status": self._status()}
            elif len(toks) == 3 and toks[0] == "flip":
                ok, msg = self._apply_flip(toks[1], toks[2])
                reply = {"cmd": line, "ok": ok, "msg": msg, "status": self._status()}
            try:
                self._sock.sendto(json.dumps(reply).encode("utf-8"), addr)
            except OSError:
                pass
            print(f"[selector] ctrl '{line}' -> {reply.get('msg', reply.get('status'))}",
                  flush=True)

    # ---- publish loop --------------------------------------------------------
    def _publish_loop(self):
        dt = 1.0 / self._args.publish_hz
        while self._running:
            start = time.monotonic()
            with self._lock:
                ls, us = self._lower_src, self._upper_src
                lower_src = self._full if ls == "A" else self._lower_b
                upper_src = self._full if us == "A" else self._upper_b
                lo_msg = lower_src.msg
                up_msg = upper_src.msg
            lower_out = self._extract_lower(lo_msg) if lo_msg is not None else self._estop_lower
            upper_out = self._extract_upper(up_msg) if up_msg is not None else self._estop_upper
            self._pub_lower.Write(lower_out)
            self._pub_upper.Write(upper_out)
            time.sleep(max(0.0, dt - (time.monotonic() - start)))

    # ---- lifecycle -----------------------------------------------------------
    def start(self):
        self._running = True
        self._sub_full.Init(self._on_full, 10)
        self._sub_lower_b.Init(self._on_lower_b, 10)
        self._sub_upper_b.Init(self._on_upper_b, 10)
        self._pub_lower.Init()
        self._pub_upper.Init()
        self._pub_thread = threading.Thread(
            target=self._publish_loop, name="selector_pub", daemon=True)
        self._ctrl_thread = threading.Thread(
            target=self._control_loop, name="selector_ctrl", daemon=True)
        self._pub_thread.start()
        self._ctrl_thread.start()
        a = self._args
        print(f"[selector] up | lower={self._lower_src} upper={self._upper_src} | "
              f"in: full={a.full_topic} lowerB={a.lower_in_topic} upperB={a.upper_in_topic} | "
              f"out: {a.lower_out_topic} {a.upper_out_topic} | "
              f"ctrl udp {a.control_host}:{a.control_port} | {a.publish_hz:.0f}Hz "
              f"stale {a.stale_sec*1e3:.0f}ms", flush=True)

    def spin(self):
        try:
            while self._running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False


def build_parser():
    p = argparse.ArgumentParser(description="Two-controller selector / warm mux")
    p.add_argument("--domain", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", "0")))
    p.add_argument("--iface", default="", help="DDS interface ('' = unitree_sdk2py default)")
    p.add_argument("--full-topic", dest="full_topic", default=FULL_TOPIC)
    p.add_argument("--lower-in-topic", dest="lower_in_topic", default=LOWER_B_TOPIC)
    p.add_argument("--upper-in-topic", dest="upper_in_topic", default=UPPER_B_TOPIC)
    p.add_argument("--lower-out-topic", dest="lower_out_topic", default=LOWER_OUT_TOPIC)
    p.add_argument("--upper-out-topic", dest="upper_out_topic", default=UPPER_OUT_TOPIC)
    p.add_argument("--publish-hz", dest="publish_hz", type=float, default=500.0)
    p.add_argument("--stale-sec", dest="stale_sec", type=float, default=0.1,
                   help="max age for a B source to be flip-eligible")
    p.add_argument("--control-host", dest="control_host", default="127.0.0.1")
    p.add_argument("--control-port", dest="control_port", type=int, default=47700)
    p.add_argument("--init-lower", dest="init_lower", choices=["A", "B"], default="A")
    p.add_argument("--init-upper", dest="init_upper", choices=["A", "B"], default="A")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    sel = Selector(args)
    sel.start()
    sel.spin()


if __name__ == "__main__":
    main()
