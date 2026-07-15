# Known issue: headless trial-video recorder renders blank frames

**Where:** `h1_robocasa/h12_mujoco.py` `VideoRecorder` (used by the grasp
benchmark's `--record-video`).
**Status:** open — diagnosed here, not fixed. Two independent bugs.

## Bug 1: frames are blank

The recorder is meant to write a third-person mp4 of each headless
(`MUJOCO_GL=egl`) trial. Every finalized frame is instead blank:

- As written (its OWN `mujoco.Renderer`): a **uniform mid-gray** (mean ~115,
  std ~2). Two `mujoco.Renderer` instances on one thread share the process EGL
  state, and the second one (created after the sensor bridge's) renders into a
  blank framebuffer.
- Reworked to BORROW the sensor bridge's single renderer (one renderer, no
  second context): the gray goes away but the frame is **only the skybox** —
  camera position is correct (verified: `body_id` resolves to the torso, `lookat`
  = torso world pos, dist 2.2 m) but **no scene geometry is drawn**. Two frames
  from separate runs came out byte-identical (the robot sways on its tether, so
  real frames would differ), and changing the recorder's `MjvOption.geomgroup`
  had zero effect on the pixels — so `update_scene(data, camera=<free cam>,
  scene_option=<opt>)` on the borrowed renderer is not populating geoms for the
  recorder's free camera, even though the SAME renderer draws the robot cameras
  fine for perception.

Likely real fix: give the recorder a renderer on its **own thread with its own
EGL context** (MuJoCo's EGL context is thread-affine), rather than a second
renderer on the main thread or a borrowed one. Verify with a standalone script
that loads the CheesyBread env and renders one free-camera frame — if that draws
the robot, the bug is in the recorder/bridge renderer interaction; if it is also
blank, it is in the free-camera `update_scene` path itself.

## Bug 2: mp4 not finalized on hard kill

The benchmark tears the sim down with `docker rm -f` (SIGKILL). The imageio/ffmpeg
writer only flushes the mp4 on a clean shutdown, so a hard kill leaves a ~28-byte
header-only stub even with fragmented-mp4 output params. A **clean SIGINT** to the
sim process (`pkill -INT -f h12_mujoco.py`) DOES finalize it. Fix: shut the sim
down cleanly (SIGTERM/SIGINT + a handler that runs the recorder's `close()`)
before removing the container, and/or flush the ffmpeg pipe periodically.

## How to reproduce / inspect

```bash
# launch a headless episode with recording, let it step, SIGINT for clean finalize
docker compose -f docker/docker-compose.yml --profile robocasa run -d --rm \
  --name hams_sim_robocasa robocasa /home/code/h12_sim_scripts/launch_robocasa.sh \
  --headless --task CheesyBread --seed 45 \
  --record-video /home/code/h1_robocasa/benchmark_videos/vidtest.mp4
sleep 55; docker exec hams_sim_robocasa pkill -INT -f h12_mujoco.py; sleep 8
# video is bind-mounted to the host and survives container removal:
python3 -c "import imageio.v3 as iio; v=iio.imread('h1_robocasa/benchmark_videos/vidtest.mp4',index=None); print(v[len(v)//2].std())"
# std < 5 => blank
```
