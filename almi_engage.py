"""ALMI freeze-hold release, timed in SIM seconds (not wall clock).

The headless sim runs at ~5% real-time, so a wall-clock sleep gives the policy
almost no sim-time to stabilize before the pin releases -> it topples. This
waits a real STAB_SIM sim-seconds pinned, verifies the torso is vertical, then
releases and confirms it stays up. Exit 0 = standing, 1 = fell/failed.
"""
import os, sys, json, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from rosgraph_msgs.msg import Clock

STAB_SIM = float(os.environ.get('STAB_SIM', '40'))    # sim-sec pinned before release
CONFIRM_SIM = float(os.environ.get('CONFIRM_SIM', '15'))  # sim-sec watched after release
UP_MIN = 0.85

class E(Node):
    def __init__(self):
        super().__init__('almi_engage')
        self.pel = None; self.simt = None
        self.create_subscription(String, '/robocasa/object_poses', self.op, 10)
        self.create_subscription(Clock, '/clock', self.ck, 10)
        self.rel = self.create_publisher(Bool, '/hams/freeze_body', 10)
    def op(self, m):
        try: self.pel = json.loads(m.data).get('__pelvis__')
        except Exception: pass
    def ck(self, m): self.simt = m.clock.sec + m.clock.nanosec * 1e-9

def up(p):
    w, x, y, z = p[3:7]
    return 1 - 2 * (x * x + y * y)

rclpy.init(); n = E(); t0 = time.time()
while (n.pel is None or n.simt is None) and time.time() - t0 < 20:
    rclpy.spin_once(n, timeout_sec=0.2)
if n.pel is None or n.simt is None:
    print('ENGAGE_FAIL no pose/clock'); sys.exit(1)

t_eng = n.simt
while n.simt - t_eng < STAB_SIM and time.time() - t0 < 400:
    rclpy.spin_once(n, timeout_sec=0.2)
u_pin = up(n.pel)
print(f'ENGAGE pinned upright={u_pin:.3f} z={n.pel[2]:.3f} simt={n.simt:.1f}', flush=True)
if u_pin < 0.9:
    print('ENGAGE_FAIL not vertical while pinned'); sys.exit(1)

b = Bool(); b.data = False
for _ in range(8):
    n.rel.publish(b); rclpy.spin_once(n, timeout_sec=0.1)
t_rel = n.simt; minu = u_pin
while n.simt - t_rel < CONFIRM_SIM and time.time() - t0 < 500:
    rclpy.spin_once(n, timeout_sec=0.2)
    minu = min(minu, up(n.pel))
    if minu < 0.55:
        break
ok = minu > UP_MIN
print(f'ENGAGE {"OK" if ok else "FAIL"} released min_upright={minu:.3f}', flush=True)
rclpy.shutdown()
sys.exit(0 if ok else 1)
