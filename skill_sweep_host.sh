#!/bin/bash
# Host-side skill sweep: 20 trials in batches of 3, with a FULL container
# recreate (sim + ros + bringup) between batches. Yesterday's evidence: grasps
# succeed on a fresh environment and degrade as in-container restarts accumulate
# zombie nodes — so the clean unit is "fresh containers + fresh bringup + 3 trials".
# Results land on the host bind mount (core_ws/benchmark_results/sweep/skill).
set -o pipefail
cd /home/guest/Downloads/HAMS-test-grasping
SP=/tmp/claude-1001/-home-guest-Downloads-HAMS-test-grasping/9645ef6d-75d8-4b70-b59a-82ef7165e409/scratchpad
OUT=/home/code/core_ws/benchmark_results/sweep/skill
N=20; BATCH=3
PW=Unitreeh12

sudo_do() { echo "$PW" | sudo -S "$@" 2>/dev/null; }

fresh_env() {
  echo "[env] recreating containers $(date +%H:%M:%S)"
  sudo_do docker rm -f hams_ros hams_sim_robocasa >/dev/null; sleep 4
  echo "$PW" | sudo -S -E bash docker/scripts/docker_run.sh robocasa --task OpenFridge --seed 42 > /tmp/sim_batch.log 2>&1 &
  for k in $(seq 1 60); do
    sudo_do docker logs hams_sim_robocasa 2>&1 | grep -q "ROS bridges up" && break; sleep 3
  done
  echo "$PW" | sudo -S -E bash docker/scripts/docker_run.sh ros sleep infinity > /tmp/ros_batch.log 2>&1 &
  for k in $(seq 1 40); do
    sudo_do docker ps --format '{{.Names}}' | grep -q hams_ros && break; sleep 2
  done
  sudo_do docker cp /tmp/grab_head.py hams_ros:/tmp/grab_head.py
  sudo_do docker exec -d hams_ros bash -lc "export HAMS_MAX_GRASP_ATTEMPTS=40; source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash && ros2 launch h1_bringup h1_sim_bringup.launch.py use_rviz:=false use_nav:=false use_mjpc:=false > /tmp/bringup.log 2>&1"
  for k in $(seq 1 80); do
    sudo_do docker exec hams_ros bash -c "grep -q 'h12_skills ready' /tmp/bringup.log 2>/dev/null" && break; sleep 3
  done
  sleep 5
  echo "[env] ready $(date +%H:%M:%S)"
}

# Fingers healthy? Open the gripper (also RELEASES the handle after a successful
# grasp — critical: resetting the arm while still clamped wrenches the fingers),
# then verify it opens wide and symmetric. Bent fingers (from a trial dragging the
# open gripper into the door, or a clamped-teleport) wreck every later trial, so a
# damaged gripper forces a fresh environment.
gripper_ok() {
  sudo_do docker exec hams_ros bash -lc "
    source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; export ROS_DOMAIN_ID=1
    ros2 service call /right/gripper/open std_srvs/srv/Trigger >/dev/null 2>&1; sleep 3
    timeout 6 ros2 topic echo --once /right/gripper/state 2>/dev/null | python3 -c '
import sys, yaml
d = yaml.safe_load(sys.stdin.read() or \"{}\") or {}
p = float(d.get(\"position\", 0)); f = d.get(\"finger_positions\") or [0, 0]
asym = abs(float(f[0]) - float(f[1]))
ok = p >= 95.0 and asym <= 6.0
print(f\"gripper p={p:.1f} asym={asym:.1f} -> {\\\"OK\\\" if ok else \\\"DAMAGED\\\"}\")
sys.exit(0 if ok else 1)'"
}

echo "[host-sweep] START $(date +%H:%M:%S)"
i=1
since_fresh=999
while [ $i -le $N ]; do
  T=$(printf "%02d" $i)
  if sudo_do docker exec hams_ros bash -c "test -f $OUT/trial_$T.json" 2>/dev/null; then
    echo "[host-sweep] skip $T (exists)"; i=$((i+1)); continue
  fi
  need_fresh=0
  [ $since_fresh -ge $BATCH ] && need_fresh=1
  if [ $need_fresh -eq 0 ]; then
    if ! gripper_ok; then echo "[host-sweep] gripper DAMAGED -> fresh env"; need_fresh=1; fi
  fi
  if [ $need_fresh -eq 1 ]; then fresh_env; since_fresh=0; fi
  echo "[host-sweep] === trial $T $(date +%H:%M:%S) ==="
  sudo_do docker exec hams_ros bash -lc "
    source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; export ROS_DOMAIN_ID=1
    mkdir -p $OUT
    for attempt in 1 2 3; do
      # release the handle BEFORE homing the arm (never teleport while clamped)
      ros2 service call /right/gripper/open std_srvs/srv/Trigger >/dev/null 2>&1; sleep 2
      ros2 topic pub --once /hams/reset_arm std_msgs/msg/Empty '{}' >/dev/null 2>&1; sleep 2
      timeout 330 ros2 run h12_skills grasp_benchmark --method skill \
        --object 'vertical fridge handle' --gt-name door_obj --arm right \
        --success-mode contact --out $OUT/trial_$T.json > $OUT/trial_$T.log 2>&1
      [ \$? -eq 124 ] && [ ! -s $OUT/trial_$T.json ] && python3 -c \"import json;json.dump({'method':'skill','success':False,'executed':False,'error':'harness timeout'},open('$OUT/trial_$T.json','w'),indent=2)\"
      P=\$(python3 -c \"import json;d=json.load(open('$OUT/trial_$T.json'));l=(str(d.get('chosen_label',''))+' '+str(d.get('error',''))).lower();print(1 if any(k in l for k in ('no mask','no box','no grasp planned','no candidate','synthesis produced no')) else 0)\" 2>/dev/null)
      [ \"\$P\" != \"1\" ] && break
      echo \"[trial $T] perception no-op; retrying\"
    done
    python3 /tmp/grab_head.py >/dev/null 2>&1 && cp -f /tmp/head_view.jpg $OUT/trial_${T}_head.jpg 2>/dev/null
    python3 -c \"import json;d=json.load(open('$OUT/trial_$T.json'));print('  -> succ=%s gfinal=%s | %s'%(d.get('success'),d.get('grip_final_mm'),str(d.get('chosen_label'))[:40]))\"
  "
  i=$((i+1))
  since_fresh=$((since_fresh+1))
done
echo "[host-sweep] COMPLETE $(date +%H:%M:%S)"
