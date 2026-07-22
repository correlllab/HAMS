"""Bag-based sim-time regrade for the ALMI tier.

The in-run stability check sampled the gripper at +1s/+3s WALL-clock after
close; at the ALMI tier's ~5% realtime that is ~0.05/0.15 SIM-seconds — inside
the settling transient, so slow-settling (diagonal) holds were graded unstable.
Here the recorded /right/gripper/state series is used instead: find the final
contiguous hold segment (aperture in the contact band) and require it to be
STABLE over its last stretch. Sim time is approximated by mapping each bag
message to the nearest /joint_states header stamp (sim clock).

Output: JSON {trial: {settled_mm, stable, regrade_hold}} per method dir.
The final regraded success = regrade_hold AND (in-run tip_to_handle <= 60).
"""
import json, os, sys, glob
import numpy as np
import sqlite3
from rclpy.serialization import deserialize_message
from magpie_msgs.msg import GripperState
from sensor_msgs.msg import JointState

BAND_LO, BAND_HI = 20.0, 85.0
STABLE_MM = 8.0          # max aperture spread over the settled window
SETTLE_WIN = 2.0         # sim-seconds the settled window spans

def read_bag(bag_dir):
    # Bags lack metadata.yaml (copy raced recorder shutdown) -> read the .db3
    # directly: topics(id,name), messages(topic_id,timestamp,data).
    db = glob.glob(os.path.join(bag_dir, '*.db3'))
    if not db:
        raise RuntimeError('no .db3')
    con = sqlite3.connect(f'file:{db[0]}?mode=ro', uri=True)
    tid = dict(con.execute('SELECT name, id FROM topics').fetchall())
    grip, joints = [], []   # (bag_t_ns, aperture_mm), (bag_t_ns, sim_t)
    g_id, j_id = tid.get('/right/gripper/state'), tid.get('/joint_states')
    if g_id is not None:
        for t, data in con.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (g_id,)):
            m = deserialize_message(bytes(data), GripperState)
            grip.append((t, float(m.position)))
    if j_id is not None:
        for t, data in con.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (j_id,)):
            m = deserialize_message(bytes(data), JointState)
            st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            joints.append((t, st))
    con.close()
    return grip, joints

def sim_time_of(bag_t, joints_t, joints_s):
    i = np.searchsorted(joints_t, bag_t)
    i = min(max(i, 0), len(joints_t) - 1)
    return joints_s[i]

def regrade(bag_dir):
    grip, joints = read_bag(bag_dir)
    if len(grip) < 10 or len(joints) < 10:
        return None
    joints_t = np.array([t for t, _ in joints]); joints_s = np.array([s for _, s in joints])
    g_t = np.array([sim_time_of(t, joints_t, joints_s) for t, _ in grip])
    g_a = np.array([a for _, a in grip])
    inb = (g_a > BAND_LO) & (g_a < BAND_HI)
    if not inb.any():
        return {'settled_mm': None, 'stable': False, 'regrade_hold': False}
    # final contiguous in-band segment
    idx = np.where(inb)[0]
    brk = np.where(np.diff(idx) > 1)[0]
    seg = idx[brk[-1] + 1:] if len(brk) else idx
    seg_t, seg_a = g_t[seg], g_a[seg]
    dur = seg_t[-1] - seg_t[0]
    if dur < SETTLE_WIN * 0.5:            # too brief to call a hold
        return {'settled_mm': round(float(np.median(seg_a)), 1),
                'stable': False, 'regrade_hold': False, 'hold_sim_s': round(float(dur), 2)}
    tail = seg_a[seg_t >= seg_t[-1] - SETTLE_WIN]
    spread = float(tail.max() - tail.min()) if len(tail) > 1 else 99.0
    settled = float(np.median(tail))
    stable = spread <= STABLE_MM and BAND_LO < settled < BAND_HI
    return {'settled_mm': round(settled, 1), 'stable': bool(stable),
            'regrade_hold': bool(stable), 'hold_sim_s': round(float(dur), 2),
            'tail_spread_mm': round(spread, 1)}

def main(root):
    out = {}
    for m in ('skill', 'graspgenx', 'centroid', 'topdown_antipodal'):
        for bag in sorted(glob.glob(os.path.join(root, m, 'trial_*_bag'))):
            trial = os.path.basename(bag).replace('_bag', '')
            try:
                r = regrade(bag)
            except Exception as e:
                r = {'error': str(e)[:80]}
            out[f'{m}/{trial}'] = r
    print(json.dumps(out, indent=1))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/home/code/core_ws/benchmark_results/sweep_almi')
