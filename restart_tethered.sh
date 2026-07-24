#!/bin/bash
# Recreate the TETHERED BatteryWorkcell sim (band on, FREEZE_BODY=0 from docker/.env)
# and re-latch the ros bringup onto the fresh clock. Idempotent; used at startup and
# by the watchdog for recovery.
set -u
ROOT=/home/guest/HAMS-grasp-2method
PW=Unitreeh12
LOG=$ROOT/sim_watchdog.log
SUDO(){ echo "$PW" | sudo -S "$@" 2>/dev/null; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [restart] $*" >> "$LOG"; }

say "recreating tethered BatteryWorkcell sim"
SUDO docker rm -f hams_sim_robocasa >/dev/null 2>&1
SUDO docker compose -f "$ROOT/docker/docker-compose.yml" --profile robocasa run -d --remove-orphans \
     --name hams_sim_robocasa robocasa /home/code/h12_sim_scripts/launch_robocasa.sh --task BatteryWorkcell >/dev/null 2>&1
for i in $(seq 1 60); do
  SUDO docker logs hams_sim_robocasa 2>&1 | grep -q "ROS bridges up" && { say "sim bridges up (~$((i*3))s)"; break; }
  sleep 3
done
# confirm tethered (band on, not frozen)
if SUDO docker logs hams_sim_robocasa 2>&1 | grep -q "FREEZE_BODY=1"; then
  say "WARNING: sim came up FROZEN not tethered — check docker/.env HAMS_FREEZE_BODY"
fi
say "re-latching ros bringup"
SUDO docker restart hams_ros >/dev/null 2>&1
for i in $(seq 1 70); do
  n=$(SUDO docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash 2>/dev/null; ros2 action list 2>/dev/null | grep -c frame_task")
  [ "${n:-0}" -ge 1 ] && { say "frame_task up (~$((i*3))s)"; break; }
  sleep 3
done
say "restart complete"
