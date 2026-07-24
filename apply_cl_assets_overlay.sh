#!/bin/bash
# Overlay the battery-scene CL_Assets changes onto the submodule WITHOUT needing to
# push the CL_Assets repo. Run after the CL_Assets submodule is checked out at the
# remote base (origin/test/grasping = c727cc6). Copies the 2 changed XMLs in place.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cp -v "$ROOT/cl_assets_overlay/battery/scene_h1_2_battery.xml" "$ROOT/CL_Assets/battery/scene_h1_2_battery.xml"
cp -v "$ROOT/cl_assets_overlay/mujoco_assets/h1_2_magpie.xml"  "$ROOT/CL_Assets/mujoco_assets/h1_2_magpie.xml"
echo "CL_Assets battery-scene overlay applied (raised nail model + fingertip marker)."
