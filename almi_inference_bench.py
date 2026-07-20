#!/usr/bin/env python3
"""One-shot ALMI policy inference-time benchmark: load the TorchScript LSTM the
way AlmiPolicy does and time forward passes on this machine (CPU, as the node
runs it). Writes JSON to stdout. Run INSIDE hams_ros."""
import json
import time

import numpy as np
import torch

P = '/home/code/core_ws/src/h12_lowerbody_rl/policies/almi/policy_lstm_12800.pt'
m = torch.jit.load(P, map_location='cpu')
m.eval()
if hasattr(m, 'reset_memory'):
    m.reset_memory()
obs = torch.zeros(1, 65)
# warmup
for _ in range(50):
    with torch.no_grad():
        m(obs)
ts = []
for _ in range(1000):
    t0 = time.perf_counter()
    with torch.no_grad():
        m(obs)
    ts.append((time.perf_counter() - t0) * 1000.0)
a = np.array(ts)
print(json.dumps({
    'policy': 'almi policy_lstm_12800.pt', 'device': 'cpu', 'n': len(a),
    'mean_ms': round(float(a.mean()), 3), 'p50_ms': round(float(np.percentile(a, 50)), 3),
    'p95_ms': round(float(np.percentile(a, 95)), 3), 'max_ms': round(float(a.max()), 3),
    'budget_ms_at_50hz': 20.0}, indent=2))
