#!/bin/bash
# Standing-tier watchdog: every 5 min check sim + /clock + frame_task + ALMI node.
# Restart the whole standing stack on 2 consecutive failures.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; PW="${SUDO_PW:-Unitreeh12}"
SUDO(){ echo "$PW" | sudo -S "$@" 2>/dev/null; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$ROOT/sim_watchdog_standing.log"; }
fails=0; say "=== standing watchdog started ==="
while true; do
  up=$(SUDO docker ps --format '{{.Names}}' | grep -c hams_sim_robocasa)
  ft=$(SUDO docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash 2>/dev/null; ros2 action list 2>/dev/null | grep -c frame_task")
  al=$(SUDO docker exec hams_ros bash -lc "pgrep -c -f lib/h12_lowerbody_rl/lowerbody_controller_node")
  if [ "${up:-0}" -ge 1 ] && [ "${ft:-0}" -ge 1 ] && [ "${al:-0}" -ge 1 ]; then
    say "healthy: sim=up frame_task=$ft almi_node=ok"; fails=0
  else
    fails=$((fails+1)); say "UNHEALTHY sim=$up frame_task=$ft almi=$al (fail#$fails)"
    if [ "${up:-0}" -ge 1 ] && [ "${ft:-0}" -ge 1 ] && [ "${al:-0}" -lt 1 ]; then
      say "-> ALMI node died, restarting just it"
      SUDO docker exec -d hams_ros bash -lc 'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; ros2 run h12_lowerbody_rl lowerbody_controller_node --ros-args -p use_sim_time:=true -p active_policy:=almi -p engage_wait_for_confirm:=false -p disable_elastic_band:=false > /tmp/lowerbody.log 2>&1'
      fails=0
    elif [ $fails -ge 2 ]; then say "-> full restart"; bash "$ROOT/restart_standing.sh"; fails=0; fi
  fi
  sleep 300
done
