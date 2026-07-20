#!/bin/bash
# Host-side STANDING baseline sweep: centroid / topdown_antipodal / graspgenx,
# 20 trials each. Fresh containers per method + gripper-damage check between
# trials (same protections as the skill sweep). Fast: baselines mostly fail at
# pre-grasp (~40-160s/trial).
set -o pipefail
cd /home/guest/Downloads/HAMS-test-grasping
OUTROOT=/home/code/core_ws/benchmark_results/sweep
PW=Unitreeh12
N=20

sudo_do() { echo "$PW" | sudo -S "$@" 2>/dev/null; }

fresh_env() {
  echo "[env] recreating containers $(date +%H:%M:%S)"
  sudo_do docker rm -f hams_ros hams_sim_robocasa >/dev/null; sleep 4
  echo "$PW" | sudo -S -E bash docker/scripts/docker_run.sh robocasa --task OpenFridge --seed 42 > /tmp/sim_batch.log 2>&1 &
  for k in $(seq 1 60); do sudo_do docker logs hams_sim_robocasa 2>&1 | grep -q "ROS bridges up" && break; sleep 3; done
  echo "$PW" | sudo -S -E bash docker/scripts/docker_run.sh ros sleep infinity > /tmp/ros_batch.log 2>&1 &
  for k in $(seq 1 40); do sudo_do docker ps --format '{{.Names}}' | grep -q hams_ros && break; sleep 2; done
  sudo_do docker cp /tmp/grab_head.py hams_ros:/tmp/grab_head.py
  sudo_do docker exec -d hams_ros bash -lc "export HAMS_MAX_GRASP_ATTEMPTS=40; source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash && ros2 launch h1_bringup h1_sim_bringup.launch.py use_rviz:=false use_nav:=false use_mjpc:=false > /tmp/bringup.log 2>&1"
  for k in $(seq 1 80); do sudo_do docker exec hams_ros bash -c "grep -q 'h12_skills ready' /tmp/bringup.log 2>/dev/null" && break; sleep 3; done
  sleep 5
  echo "[env] ready $(date +%H:%M:%S)"
}

gripper_ok() {
  sudo_do docker exec hams_ros bash -lc "
    source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; export ROS_DOMAIN_ID=1
    ros2 service call /right/gripper/open std_srvs/srv/Trigger >/dev/null 2>&1; sleep 3
    timeout 6 ros2 topic echo --once /right/gripper/state 2>/dev/null | python3 -c '
import sys, yaml
d = yaml.safe_load(sys.stdin.read() or \"{}\") or {}
p = float(d.get(\"position\", 0)); f = d.get(\"finger_positions\") or [0, 0]
asym = abs(float(f[0]) - float(f[1]))
sys.exit(0 if (p >= 95.0 and asym <= 6.0) else 1)'"
}

METHODS="${BASELINE_METHODS:-centroid topdown_antipodal graspgenx}"
echo "[baselines] START methods=[$METHODS] $(date +%H:%M:%S)"
for M in $METHODS; do
  fresh_env
  i=1
  while [ $i -le $N ]; do
    T=$(printf "%02d" $i)
    if sudo_do docker exec hams_ros bash -c "test -f $OUTROOT/$M/trial_$T.json" 2>/dev/null; then
      echo "[baselines] skip $M/$T"; i=$((i+1)); continue
    fi
    if ! gripper_ok; then echo "[baselines] gripper DAMAGED -> fresh env"; fresh_env; fi
    echo "[baselines] === $M trial $T $(date +%H:%M:%S) ==="
    sudo_do docker exec hams_ros bash -lc "
      source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; export ROS_DOMAIN_ID=1
      mkdir -p $OUTROOT/$M
      for attempt in 1 2 3; do
        ros2 service call /right/gripper/open std_srvs/srv/Trigger >/dev/null 2>&1; sleep 2
        ros2 topic pub --once /hams/reset_arm std_msgs/msg/Empty '{}' >/dev/null 2>&1; sleep 2
        timeout 330 ros2 run h12_skills grasp_benchmark --method $M \
          --object 'vertical fridge handle' --gt-name door_obj --arm right \
          --success-mode contact --max-attempts 20 --no-plan --out $OUTROOT/$M/trial_$T.json > $OUTROOT/$M/trial_$T.log 2>&1
        [ \$? -eq 124 ] && [ ! -s $OUTROOT/$M/trial_$T.json ] && python3 -c \"import json;json.dump({'method':'$M','success':False,'executed':False,'error':'harness timeout'},open('$OUTROOT/$M/trial_$T.json','w'),indent=2)\"
        P=\$(python3 -c \"import json;d=json.load(open('$OUTROOT/$M/trial_$T.json'));l=(str(d.get('chosen_label',''))+' '+str(d.get('error',''))).lower();print(1 if any(k in l for k in ('no mask','no box','no grasp planned','no candidate','synthesis produced no')) else 0)\" 2>/dev/null)
        [ \"\$P\" != \"1\" ] && break
        echo \"[trial $M/$T] perception no-op; retrying\"
      done
      python3 /tmp/grab_head.py >/dev/null 2>&1 && cp -f /tmp/head_view.jpg $OUTROOT/$M/trial_${T}_head.jpg 2>/dev/null
      python3 -c \"import json;d=json.load(open('$OUTROOT/$M/trial_$T.json'));print('  -> $M succ=%s gfinal=%s | %s'%(d.get('success'),d.get('grip_final_mm'),str(d.get('chosen_label'))[:36]))\"
    "
    i=$((i+1))
  done
done
echo "[baselines] COMPLETE $(date +%H:%M:%S)"
