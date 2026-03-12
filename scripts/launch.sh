#/bin/bash
if [ "$1" == "--reset-cache" ]; then 
  rm -rf ~/.cache/ov/texturecache
fi
python /home/code/h12_sim_scripts/dds_bridge.py & disown
python3  /home/code/CL_isaaclab_sim/sim_main.py \
  --device cuda \
  --task Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint \
  --enable_inspire_dds \
  --robot_type h1_2 \
  --enable_cameras
  # --verbose \
