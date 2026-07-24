#!/bin/bash
# Sim health watchdog. Every 5 min: sim container up + /clock ticking + frame_task
# action present. Restart the tethered sim on 2 consecutive failures. Logs to
# sim_watchdog.log so the health history is visible when Adam is back.
set -u
ROOT=/home/guest/HAMS-grasp-2method
PW=Unitreeh12
LOG=$ROOT/sim_watchdog.log
SUDO(){ echo "$PW" | sudo -S "$@" 2>/dev/null; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
fails=0
say "=== watchdog started (check every 5 min) ==="
while true; do
  up=$(SUDO docker ps --format '{{.Names}}' | grep -c hams_sim_robocasa)
  clk=$(SUDO docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; timeout 8 ros2 topic hz /clock --window 5 2>/dev/null | grep -m1 average" | grep -c average)
  ft=$(SUDO docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash 2>/dev/null; ros2 action list 2>/dev/null | grep -c frame_task")
  if [ "${up:-0}" -ge 1 ] && [ "${clk:-0}" -ge 1 ] && [ "${ft:-0}" -ge 1 ]; then
    say "healthy: sim=up clock=ticking frame_task=$ft"
    fails=0
  else
    fails=$((fails+1))
    say "UNHEALTHY sim=$up clock=$clk frame_task=$ft (fail#$fails)"
    [ $fails -ge 2 ] && { say "-> triggering restart"; bash "$ROOT/restart_tethered.sh"; fails=0; }
  fi
  sleep 300
done
