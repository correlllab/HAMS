#!/usr/bin/env python3
"""Aggregate the fridge-handle grasp sweep + render magpie-style grasp-line
overlays. Runs INSIDE hams_ros (needs cv2/numpy/tf2). Reads sweep/<method>/
trial_*.json, writes summary.json + per-method overlay PNGs into <out>/viz/."""
import glob
import json
import os
import sys

import numpy as np
import cv2

SWEEP = sys.argv[1] if len(sys.argv) > 1 else '/home/code/core_ws/benchmark_results/sweep'
TCP_DEPTH_M = 0.1146
METHOD_ORDER = ['centroid', 'topdown_antipodal', 'graspgenx', 'skill']
PRETTY = {'centroid': 'centroid', 'topdown_antipodal': 'pca',
          'graspgenx': 'graspgenx-normal', 'skill': 'new(skill)'}


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def get_pelvis_to_optical():
    """Static pelvis->camera_color_optical_frame (4x4) + K. The body is frozen so
    this is constant; hardcoded from the live TF/caminfo captured during the run
    (pelvis->optical = inverse of the measured optical->pelvis) so aggregation needs
    no ROS and can run anytime, even during bringup restarts."""
    T = np.array([
        [0.0,   -1.0,   0.0,    0.018],
        [-0.775, 0.0,  -0.633,  0.5215],
        [0.633,  0.0,  -0.775,  0.4629],
        [0.0,    0.0,   0.0,    1.0]])
    K = np.array([[230.918, 0, 128.0], [0, 230.918, 128.0], [0, 0, 1.0]])
    return T, K


def project(p_pel, T, K):
    pc = T[:3, :3] @ np.asarray(p_pel, float) + T[:3, 3]
    if pc[2] <= 0.01:
        return None
    uv = K @ (pc / pc[2])
    return int(round(uv[0])), int(round(uv[1]))


def draw_grasp(img, pose, T, K, color=(0, 255, 0), half=0.05):
    """magpie-style: green closing-axis line, orange fingertip ticks, red center."""
    R = quat_to_R(pose['quat']); p = np.array(pose['pos'])
    close_axis = R[:, 0]                 # +X = finger-closing axis
    contact = p + R[:, 2] * TCP_DEPTH_M  # fingertip contact point (+Z approach)
    a = project(contact - close_axis * half, T, K)
    b = project(contact + close_axis * half, T, K)
    c = project(contact, T, K)
    if a and b:
        cv2.line(img, a, b, color, 2)
        for e in (a, b):
            cv2.circle(img, e, 4, (0, 165, 255), -1)   # orange fingertips
    if c:
        cv2.circle(img, c, 3, (0, 0, 255), -1)         # red grip center


def main():
    T, K = get_pelvis_to_optical()
    viz_dir = os.path.join(SWEEP, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    summary = {}
    for m in METHOD_ORDER:
        files = sorted(glob.glob(os.path.join(SWEEP, m, 'trial_*.json')))
        trials = []
        for f in files:
            try:
                trials.append(json.load(open(f)))
            except Exception:
                pass
        n = len(trials)
        if not n:
            continue
        succ = sum(1 for t in trials if t.get('success'))
        execd = sum(1 for t in trials if t.get('executed'))
        gmins = [t['grip_min_mm'] for t in trials if t.get('grip_min_mm') is not None]
        forces = [t['grip_max_force_n'] for t in trials if t.get('grip_max_force_n') is not None]
        etimes = [t.get('exec_time_s', 0) for t in trials if t.get('exec_time_s')]
        summary[m] = dict(
            pretty=PRETTY[m], n=n, success=succ, success_rate=round(succ / n, 3),
            executed=execd, exec_rate=round(execd / n, 3),
            grip_min_mm_mean=round(float(np.mean(gmins)), 1) if gmins else None,
            grip_force_n_mean=round(float(np.mean(forces)), 1) if forces else None,
            exec_time_s_mean=round(float(np.mean(etimes)), 1) if etimes else None)
        # overlay: first successful (or first executed, or first) trial with poses
        pick = next((t for t in trials if t.get('success') and (t.get('chosen_pose') or t.get('candidates'))),
                    next((t for t in trials if t.get('chosen_pose') or t.get('candidates')), None))
        if pick is None:
            continue
        idx = trials.index(pick)
        head = os.path.join(SWEEP, m, f"trial_{idx + 1:02d}_head.jpg")
        img = cv2.imread(head) if os.path.exists(head) else np.full((256, 256, 3), 40, np.uint8)
        if img is None:
            img = np.full((256, 256, 3), 40, np.uint8)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        Kv = K.copy(); Kv[:2] *= 2.0   # scale intrinsics to the 2x image
        for cand in pick.get('candidates', [])[:8]:
            draw_grasp(img, cand, T, Kv, color=(90, 200, 90), half=0.045)   # faint = candidates
        if pick.get('chosen_pose'):
            draw_grasp(img, pick['chosen_pose'], T, Kv, color=(0, 255, 0), half=0.05)  # bright = chosen
        label = f"{PRETTY[m]}  succ={succ}/{n}"
        cv2.rectangle(img, (0, 0), (512, 26), (0, 0, 0), -1)
        cv2.putText(img, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(viz_dir, f"{m}_grasp.png"), img)

    # Combined 2x2 comparison grid of the four method overlays.
    tiles = []
    for m in METHOD_ORDER:
        p = os.path.join(viz_dir, f"{m}_grasp.png")
        t = cv2.imread(p) if os.path.exists(p) else None
        if t is None:
            t = np.full((512, 512, 3), 30, np.uint8)
            cv2.putText(t, PRETTY.get(m, m), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(t, (512, 512)))
    if len(tiles) == 4:
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        cv2.imwrite(os.path.join(viz_dir, 'comparison_grid.png'), grid)

    # Success-rate bar chart.
    bar = np.full((360, 720, 3), 255, np.uint8)
    cv2.putText(bar, 'fridge-handle grasp success rate (contact)', (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    x0, y0, w, gap = 60, 320, 130, 30
    for i, m in enumerate(METHOD_ORDER):
        if m not in summary:
            continue
        r = summary[m]['success_rate']
        h = int(r * 250)
        x = x0 + i * (w + gap)
        cv2.rectangle(bar, (x, y0 - h), (x + w, y0), (150, 120, 40), -1)
        cv2.rectangle(bar, (x, y0 - 250), (x + w, y0), (180, 180, 180), 1)
        cv2.putText(bar, f"{r*100:.0f}%", (x + 35, y0 - h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(bar, PRETTY[m], (x + 2, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(bar, f"{summary[m]['success']}/{summary[m]['n']}",
                    (x + 40, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(viz_dir, 'success_rates.png'), bar)

    json.dump(summary, open(os.path.join(SWEEP, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nviz -> {viz_dir}  (per-method overlays, comparison_grid.png, success_rates.png)")


if __name__ == '__main__':
    main()
