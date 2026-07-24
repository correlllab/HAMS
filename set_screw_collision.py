#!/usr/bin/env python3
"""Rewrite every screw_*_head_collision geom in the battery scene to a target shape,
so the grasp build-up (box -> round -> thinner -> shorter, toward a real nail) is one
clean edit per rung. Keeps mass/friction/solref/solimp/material. Usage:
  set_screw_collision.py box  <hx> <hy> <hz>     # half-extents (m)
  set_screw_collision.py cyl  <r>  <hz>          # radius, half-height (m)
The geom pos z is set to hz so the screw base stays on the table."""
import re, sys

F = 'CL_Assets/battery/scene_h1_2_battery.xml'
kind = sys.argv[1]
if kind == 'box':
    hx, hy, hz = (float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    typ, size = 'box', f'{hx:.6f} {hy:.6f} {hz:.6f}'
elif kind == 'cyl':
    r, hz = (float(sys.argv[2]), float(sys.argv[3]))
    typ, size = 'cylinder', f'{r:.6f} {hz:.6f}'
else:
    sys.exit('kind must be box|cyl')

txt = open(F).read()
# match: type="..." size="..." pos="0 0 ..."  inside a *_head_collision geom line
pat = re.compile(r'(name="screw_\d+_head_collision"\s+)type="[^"]*"\s+size="[^"]*"\s+pos="0 0 [0-9.]+"')
new = pat.sub(lambda m: f'{m.group(1)}type="{typ}" size="{size}" pos="0 0 {hz:.6f}"', txt)
n = len(pat.findall(txt))
open(F, 'w').write(new)
print(f'set {n} screw collisions -> {typ} size="{size}" pos z={hz:.6f}')
