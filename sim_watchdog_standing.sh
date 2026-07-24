#!/bin/bash
# Standing-tier watchdog: every 5 min check sim + /clock + frame_task + ALMI node.
# Restart the whole standing stack on 2 consecutive failures.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; PW="${SUDO_PW:-Unitreeh12}"
SUDO(){ echo "$PW" | sudo -S "$@" 2>/dev/null; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$ROOT/sim_watchdog_standing.log"; }
fails=0; stall=0; STALL_MIN=${STALL_MIN:-15}
BDIR=/home/code/core_ws/benchmark_results/sweep_battery_standing
say "=== standing watchdog started (stall_min=$STALL_MIN) ==="
while true; do
  up=$(SUDO docker ps --format '{{.Names}}' | grep -c hams_sim_robocasa)
  ft=$(SUDO docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash 2>/dev/null; ros2 action list 2>/dev/null | grep -c frame_task")
  al=$(SUDO docker exec hams_ros bash -lc "pgrep -c -f lib/h12_lowerbody_rl/lowerbody_controller_node")
  if [ "${up:-0}" -ge 1 ] && [ "${ft:-0}" -ge 1 ] && [ "${al:-0}" -ge 1 ]; then
    # Liveness is not enough: the robot can topple (TF corruption -> arm flails ->
    # ALMI loses balance) while every process stays alive, producing trials that
    # write NO json. Detect that: sweep running but no fresh trial_*.json in STALL_MIN.
    sweep=$(pgrep -fc 'sweep_standing\.sh|sweep_random\.sh' || echo 0)
    fresh=$(SUDO docker exec hams_ros bash -lc "find $BDIR -name 'trial_*.json' -mmin -$STALL_MIN 2>/dev/null | wc -l")
    if [ "${sweep:-0}" -ge 1 ] && [ "${fresh:-0}" -eq 0 ]; then
      stall=$((stall+1)); say "STALL: sweep alive but 0 fresh trial json in ${STALL_MIN}min (stall#$stall)"
      if [ $stall -ge 2 ]; then say "-> full restart (stall recovery: robot likely toppled/TF corrupt)"; bash "$ROOT/restart_standing.sh"; stall=0; fi
    else
      say "healthy: sim=up frame_task=$ft almi_node=ok fresh_json=$fresh sweep=$sweep"; stall=0
    fi
    fails=0
  else
    fails=$((fails+1)); say "UNHEALTHY sim=$up frame_task=$ft almi=$al (fail#$fails)"
    if [ "${up:-0}" -ge 1 ] && [ "${ft:-0}" -ge 1 ] && [ "${al:-0}" -lt 1 ]; then
      say "-> ALMI node died, restarting just it"
      SUDO docker exec -d hams_ros bash -lc 'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; ros2 run h12_lowerbody_rl lowerbody_controller_node --ros-args -p use_sim_time:=true -p active_policy:=almi -p engage_wait_for_confirm:=false -p disable_elastic_band:=false -r /safety/heartbeat:=/safety/heartbeat_ignored > /tmp/lowerbody.log 2>&1'
      fails=0
    elif [ $fails -ge 2 ]; then say "-> full restart"; bash "$ROOT/restart_standing.sh"; fails=0; fi
  fi
  sleep 300
done
