#!/bin/bash
# Recreate the ALMI STANDING sim: frozen BatteryWorkcell (band off, spawn-lower) +
# re-latch ros + restart the ALMI lowerbody controller. Used by the standing watchdog.
set -u
ROOT=/home/guest/HAMS-grasp-2method; PW=Unitreeh12
SUDO(){ echo "$PW" | sudo -S "$@" 2>/dev/null; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [restart-standing] $*" >> "$ROOT/sim_watchdog_standing.log"; }
say "recreating frozen BatteryWorkcell sim"
SUDO docker rm -f hams_sim_robocasa >/dev/null 2>&1
SUDO docker compose -f "$ROOT/docker/docker-compose.yml" --profile robocasa run -d --remove-orphans \
     --name hams_sim_robocasa robocasa /home/code/h12_sim_scripts/launch_robocasa.sh --task BatteryWorkcell >/dev/null 2>&1
for i in $(seq 1 60); do SUDO docker logs hams_sim_robocasa 2>&1 | grep -q "ROS bridges up" && break; sleep 3; done
say "re-latching ros"; SUDO docker restart hams_ros >/dev/null 2>&1
for i in $(seq 1 70); do
  [ "$(SUDO docker exec hams_ros bash -lc 'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash 2>/dev/null; ros2 action list 2>/dev/null | grep -c frame_task')" -ge 1 ] && break; sleep 3; done
say "starting ALMI lowerbody controller"
SUDO docker exec -d hams_ros bash -lc 'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; ros2 run h12_lowerbody_rl lowerbody_controller_node --ros-args -p use_sim_time:=true -p active_policy:=almi -p engage_wait_for_confirm:=false -p disable_elastic_band:=false > /tmp/lowerbody.log 2>&1'
sleep 12; say "restart complete"
