import rclpy, json, time
from rclpy.node import Node
from std_msgs.msg import String, Bool
from rosgraph_msgs.msg import Clock

STAB_SIM = float(__import__('os').environ.get('STAB_SIM', '20'))   # sim-sec pinned before release
WATCH_SIM = float(__import__('os').environ.get('WATCH_SIM', '40')) # sim-sec watched after release

class D(Node):
    def __init__(self):
        super().__init__('freestand_diag')
        self.pel = None; self.simt = None
        self.create_subscription(String, '/robocasa/object_poses', self.op, 10)
        self.create_subscription(Clock, '/clock', self.ck, 10)
        self.rel = self.create_publisher(Bool, '/hams/freeze_body', 10)
    def op(self, m):
        try: self.pel = json.loads(m.data).get('__pelvis__')
        except Exception: pass
    def ck(self, m): self.simt = m.clock.sec + m.clock.nanosec * 1e-9

def up(p):   # (2,2) of R from quat (w,x,y,z) = body-z . world-up ; 1=vertical, 0=horizontal
    w, x, y, z = p[3:7]
    return 1 - 2 * (x * x + y * y)

rclpy.init(); n = D()
t0 = time.time()
while (n.pel is None or n.simt is None) and time.time() - t0 < 20:
    rclpy.spin_once(n, timeout_sec=0.2)
if n.pel is None or n.simt is None:
    print('DIAG_ERROR no pose/clock'); raise SystemExit

t_eng = n.simt
while n.simt - t_eng < STAB_SIM and time.time() - t0 < 300:
    rclpy.spin_once(n, timeout_sec=0.2)
u_pin = up(n.pel)
print(f'PINNED upright={u_pin:.3f} z={n.pel[2]:.3f} simt={n.simt:.1f}', flush=True)

b = Bool(); b.data = False
for _ in range(8):
    n.rel.publish(b); rclpy.spin_once(n, timeout_sec=0.1)
t_rel = n.simt; minu = u_pin; last = t_rel
while n.simt - t_rel < WATCH_SIM and time.time() - t0 < 600:
    rclpy.spin_once(n, timeout_sec=0.2)
    u = up(n.pel); minu = min(minu, u)
    if n.simt - last >= 4:
        print(f'  +{n.simt - t_rel:4.1f}s upright={u:.3f} z={n.pel[2]:.3f}', flush=True)
        last = n.simt
    if u < 0.55:   # already toppled, stop early
        break
print(f'VERDICT: {"STANDS" if minu > 0.85 else "FALLS"} min_upright={minu:.3f} STAB_SIM={STAB_SIM:.0f}', flush=True)
rclpy.shutdown()
