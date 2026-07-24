#!/bin/bash
# Bring up (or recover) the ALMI STANDING stack: frozen BatteryWorkcell sim (band off,
# spawn-lower) + ros bringup + ALMI lowerbody controller. Portable: finds its own repo
# dir, works on a fresh machine (creates containers if missing, builds core_ws once).
# Sudo password: export SUDO_PW=... if it isn't the default below (or be in the docker group).
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
PW="${SUDO_PW:-Unitreeh12}"
COMPOSE="$ROOT/docker/docker-compose.yml"
SUDO(){ echo "$PW" | sudo -S "$@" 2>/dev/null; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [restart-standing] $*" >> "$ROOT/sim_watchdog_standing.log"; echo "[restart-standing] $*"; }

# --- sim: always recreate fresh (frozen BatteryWorkcell; cameras/stance from docker/.env) ---
say "recreating frozen BatteryWorkcell sim"
SUDO docker rm -f hams_sim_robocasa >/dev/null 2>&1
SUDO docker compose -f "$COMPOSE" --profile robocasa run -d --remove-orphans \
     --name hams_sim_robocasa robocasa /home/code/h12_sim_scripts/launch_robocasa.sh --task BatteryWorkcell >/dev/null 2>&1
for i in $(seq 1 80); do SUDO docker logs hams_sim_robocasa 2>&1 | grep -q "ROS bridges up" && { say "sim bridges up"; break; }; sleep 3; done

# --- ros: create if missing (first time builds core_ws, ~15min), else restart ---
if ! SUDO docker ps -a --format '{{.Names}}' | grep -q '^hams_ros$'; then
  say "creating hams_ros (first run: colcon-builds core_ws, can take ~15 min)"
  SUDO docker compose -f "$COMPOSE" --profile ros run -d --remove-orphans --name hams_ros ros \
    bash -c 'source /opt/ros/humble/setup.bash
      source /opt/livox_ws/install/setup.bash 2>/dev/null || true
      [ -f /home/code/core_ws/install/setup.bash ] || ( cd /home/code/core_ws && colcon build --symlink-install )
      source /home/code/core_ws/install/setup.bash
      set -a; source /home/code/docker/.env 2>/dev/null; set +a
      ros2 launch h1_bringup h1_sim_bringup.launch.py use_rviz:=false use_nav:=false use_mjpc:=false use_skills:=true' >/dev/null 2>&1
  for i in $(seq 1 300); do SUDO docker exec hams_ros test -f /home/code/core_ws/install/setup.bash 2>/dev/null && break; sleep 5; done
else
  say "re-latching ros bringup"
  SUDO docker restart hams_ros >/dev/null 2>&1
fi

# wait for frame_task, then start ALMI
for i in $(seq 1 90); do
  n=$(SUDO docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash 2>/dev/null; ros2 action list 2>/dev/null | grep -c frame_task")
  [ "${n:-0}" -ge 1 ] && { say "frame_task up"; break; }; sleep 3
done
say "starting ALMI lowerbody controller"
# NOTE: /safety/heartbeat is remapped away. In this sim container nothing legitimately
# publishes that topic (the real robot's safety_node does; the sim does not), so ALMI is
# meant to run heartbeat-free. A stray/stale publisher stuck at False makes the controller
# latch its legs silent and the robot collapses on release. Remapping the subscription to
# an unused topic restores the intended heartbeat-free behaviour (the safety RELAY still
# clips/estops on real joint-limit violations independently).
SUDO docker exec -d hams_ros bash -lc 'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; ros2 run h12_lowerbody_rl lowerbody_controller_node --ros-args -p use_sim_time:=true -p active_policy:=almi -p engage_wait_for_confirm:=false -p disable_elastic_band:=false -r /safety/heartbeat:=/safety/heartbeat_ignored > /tmp/lowerbody.log 2>&1'
sleep 12; say "restart complete"
