# HAMS build system

HAMS (Humanoid Agent Modular Stack) builds into **four Docker images** driven by
two independent build systems that meet at runtime:

1. **Docker** builds the images (system deps, Python envs, C++ libraries baked at
   image-build time).
2. **colcon** builds the ROS 2 workspace(s) at *container start*, from
   bind-mounted source, into a host-persisted cache.

> **MJPC reintegration (branch `mjpc_reintegration`, 2026-07-16):** the fork-era
> MJPC pipeline (baked `agent_server`, `/opt/mjpc-build-seed`, runtime hydration)
> stays removed. In its place: the `mujoco_mpc` submodule (repinned to pristine
> google-deepmind upstream + our `h12_lowerbody` task) is built **at runtime,
> outside colcon**, into a host-persisted `container_cache/mjpc_build` tree via
> the thin `docker/scripts/rebuild_mjpc.sh`; the image bakes only the toolchain.
> `h12_deploy_mjpc` links that build tree directly. See §7.

The guiding principle throughout: **bake stable/heavy things into the image; keep
fast-moving source on the host and bind-mount it in.** Almost every dependency is
a git submodule, and almost every source tree is mounted, not copied — so the
Docker build context stays tiny and iterating on code never requires a rebuild.

---

## 1. Source topology

Nearly everything is a git submodule. `git submodule update --init --recursive`
is required before anything builds.

**Top-level submodules**

| Path | Purpose |
|---|---|
| `CL_Assets` | URDF meshes, MuJoCo XML, Isaac USD (Git-LFS) |
| `CL_isaaclab_sim` | Isaac Sim task/runtime code |
| `unitree_sdk2_python` | Unitree DDS SDK (Python) |
| `mujoco_mpc` | MuJoCo MPC submodule — google-deepmind upstream + our tasks (branch `max_playground`); bind-mounted rw, built at runtime into `container_cache/mjpc_build` by `rebuild_mjpc.sh` — see §7 |

**`core_ws/src` submodules:** `cl_realsense`, `custom_ros_messages`, `estop`,
`h12_ros2_controller`, `h12_ros2_model`, `h12_safety_layer`, `livox_ros_driver2`,
`magpie_control`, `magpie_msgs`, `unitree_ros2`.

**`core_ws/src` in-tree packages** (versioned directly in this repo, *not*
submodules): `FAST_LIO`, `h12_deploy_mjpc`, `h12_lowerbody_rl`,
`h12_skills`, `h1_bringup`, `model_server`.

Large binary assets (meshes, XML, USD) are tracked with **Git-LFS**; run
`git lfs install` first. Some weights are *not* in git and are fetched manually
(e.g. SAM3 `sam3.pt`, GraspGenX checkpoints) — see the root `README.md`.

---

## 2. Image graph

```
nvidia/cuda:12.2.0-devel-ubuntu22.04
        │
        ▼
   hams_base  ────────────┬───────────────┐
   (ROS 2 Humble,         │               │
    Python 3.10, torch    ▼               ▼
    cu130, CycloneDDS   hams_ros      hams_sim_robocasa
    0.10, unitree SDK)  (workspace)   (MuJoCo + RoboCasa)

nvidia/cuda:12.2.0-runtime-ubuntu22.04
        │  (two-stage builder → runtime; NO hams_base)
        ▼
   hams_sim_isaac  (Isaac Sim 5.1 / IsaacLab 2.3.2, conda Python 3.11)
```

- `hams_ros` and `hams_sim_robocasa` inherit from `hams_base`.
- `hams_sim_isaac` is **self-contained** — Isaac Sim 5.x needs Python 3.11, but
  `hams_base` pins 3.10 for the ROS 2 Humble apt packages, so Isaac cannot share
  the base. It gets its Python from a Miniconda env instead.

All containers interoperate over **CycloneDDS on one ROS domain** (default
`ROS_DOMAIN_ID=1`; domain `0` is reserved for the real robot).

---

## 3. `hams_base` (`docker/BaseDockerfile`)

The shared foundation for the ROS and RoboCasa images.

- **Base:** `nvidia/cuda:12.2.0-devel-ubuntu22.04`; env pins `ROS_DISTRO=humble`,
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, US apt mirror.
- **apt:** `gcc-12`/`g++-12` (set as default via update-alternatives), cmake,
  git/git-lfs, Python 3.10 (`python-is-python3`), and the X11/GL/Vulkan/EGL/OSMesa
  runtime libs both simulators dlopen.
- **ROS 2 Humble:** `ros-humble-ros-base` + `rmw-cyclonedds-cpp` + a few msg/TF
  packages (`sensor-msgs-py`, `tf2-ros`, `geometry-msgs`, `cv-bridge`, …).
- **uv:** installed to `/usr/local/bin` for fast, deterministic system pip installs.
- **CycloneDDS 0.10.x from source** → `CYCLONEDDS_HOME=/cyclonedds/install`. Pinned
  because the PyPI `cyclonedds` wheel (a transitive dep of `unitree_sdk2_python`)
  references `dds/ddsi/q_radmin.h`, which was removed after 0.10.
- **`unitree_sdk2_python`:** `COPY`'d from the local submodule checkout, installed
  `uv pip --system`, then its sub-packages copied into site-packages.
- **PyTorch cu130** (`torch`, `torchvision`) — the CUDA-13 wheels have native
  `sm_120`/Blackwell (RTX 5070 Ti) support and bundle their own CUDA runtime.
- **Kinematics libs:** `pin` (pinocchio), `pink`, `mink` — shared IK stack.
- Build-time `http(s)_proxy` ARGs are cleared at the end so they don't leak into
  child images.

---

## 4. `hams_ros` (`docker/RosDockerfile`)

The workspace image. Layers are ordered **most-stable → least-stable** so that
iterating on volatile pins doesn't bust the heavy layers above. `FROM hams_base`.

1. **apt build/workspace deps:** `colcon`, `rosdep`, `vcstool`; apt
   numpy/scipy/yaml/transforms3d (must match system C-extension ABIs); C++ libs
   (`libpcl-dev`, `libeigen3-dev`, `libyaml-cpp-dev`, `libopencv-dev`); the
   ament/rosidl build system; many `ros-humble-*` message/description/PCL/TF/RViz
   packages; and **apt `pinocchio`** (C++ symbols) alongside base's pip `pin`.
2. **Livox-SDK2 from source** — required by `livox_ros_driver2`. Placed before the
   Python layers so pip iteration doesn't rebuild it.
3. **Controller / IK pip stack:** `numpy<2` forced first (binds C-extensions to the
   1.x ABI), `pin-pink`, `qpsolvers` + `quadprog`/`proxsuite`, `meshcat`, `open3d`, …
4. **Vision ML stack** (exact-pinned): `transformers==4.47.1`, `google-genai`,
   `timm`, `ultralytics`, `opencv-python-headless`, CLIP (from a git SHA), etc.
5. **SAM3** — cloned to `/opt/sam3` and exposed via `PYTHONPATH` (its `setup.py`
   produces a bogus version, so it is *not* pip-installed).
6. **setuptools/wheel/numpy clamp:** restores `numpy<2`, `setuptools==59.6.0`,
   `wheel<0.44` (later pip steps bump them). This clamp recurs several times in the
   file — colcon's `python_setup_py` and torch require `setuptools<80`, and the
   80+-era `distutils-precedence.pth` otherwise spams `_distutils_hack` errors.
7. **Nav2 / SLAM / rqt / teleop / rosbag2 / foxglove** — appended late.
8. **GraspGenX** (NVlabs) — installed in-place with an elaborate constraint dance:
   build a modern-setuptools wheel in a staging dir (the system setuptools 59.6 is
   too old for its PEP 621 metadata), install `--no-deps`, freeze the whole env as
   an *additive constraint*, then add deps + downgrade `huggingface-hub` for
   `diffusers`, and add `viser` — all pinned so nothing already-installed moves.
9. **MuJoCo MPC toolchain** — clang-13/llvm-13/lld-13 + ninja + `patch` + GL/X11
   dev headers (apt), then the `mujoco==3.2.3` pip ABI pin under the graspgenx
   constraint freeze (+ explicit setuptools/wheel pins). Toolchain ONLY — mjpc
   itself is never baked; it builds at runtime into `container_cache/mjpc_build`.
   The mujoco pin moves in lockstep with mjpc's `MUJOCO_MPC_MUJOCO_GIT_TAG` and
   intentionally downgrades base's mink-pulled mujoco (see the layer comment).
   See §7.

**Runtime build:** `launch_ros.sh` runs `colcon build --symlink-install` on
`core_ws` **at container start** (see §8). The image bakes only the *toolchain* and
Python deps; the ROS packages themselves are compiled from the bind-mounted source.

---

## 5. `hams_sim_robocasa` (`docker/RobocasaDockerfile`)

Single-stage, `FROM hams_base`. Provides the MuJoCo + RoboCasa kitchen simulator.

- `MUJOCO_GL=egl` baked as default; `launch_robocasa.sh` forces `glfw` for a
  windowed viewer unless `--headless` (then `egl` for offscreen GPU rendering).
- **`mujoco==3.3.1`** (pinned to match RoboCasa's hard pin), `numpy>=2.2.6`, Pillow.
- **RoboCasa + robosuite** installed from git (neither fully on PyPI; robosuite
  from `master`). Also `robosuite_models`, `mimicgen`, `mink==0.0.5` — installed
  only to silence import-time "not installed" warnings.
  - *Side effect:* `lerobot` (a RoboCasa transitive) pins `torch==2.7.1`, which
    downgrades base's cu130 torch to a CPU/cu121 build **in this image only**
    (accepted; Isaac keeps cu130 because it builds from base independently).
- **Kitchen assets (~10 GB)** downloaded at build time so the container is
  ready-to-run (comment out that `RUN` to fetch them manually instead).
- **`msgs_ws` toolchain** (colcon + ament + rosidl) + an empty `/home/code/msgs_ws/src`
  mountpoint — this container builds *only* the IDL packages (`magpie_msgs`,
  `custom_ros_messages`) at start, not the full `core_ws`.
- **Livox baked at `/opt/livox_ws`:** `livox_ros_driver2` is `COPY`'d (not
  bind-mounted — a mount would collide with the `ros` container's `core_ws/src`
  mount, since `build.sh` rewrites its own source tree) and built via upstream
  `build.sh humble`. `build/` is deliberately *not* wiped afterward because
  `--symlink-install` leaves the generated message modules symlinked into it.

---

## 6. `hams_sim_isaac` (`docker/IsaacDockerfile`)

**Two-stage** (builder → runtime), self-contained, from
`nvidia/cuda:12.2.0-runtime-ubuntu22.04`.

**Builder stage:**
- Miniconda + a **Python 3.11** conda env `unitree_sim_env` (all subsequent `RUN`s
  execute inside it via `SHELL [conda run …]`). conda-forge `libgcc`/`libstdcxx`
  keeps the C++ ABI consistent with Isaac's wheels.
- **PyTorch 2.7.0** (+ `torchvision` 0.22.0, `torchaudio` 2.7.0), **cu128** — cu128
  has native `sm_120`; upstream's cu126 does not.
- CycloneDDS 0.10.x from source (same rationale as base — the controller talks to
  the sim over Unitree DDS on domain 1).
- `unitree_sdk2_python` installed **editable** (`pip install -e .`) so its
  sub-packages stay importable.
- **`isaacsim[all,extscache]==5.1.0.0`** (multi-GB, early in layer order).
- **IsaacLab `v2.3.2`** via `./isaaclab.sh --install`, with `PIP_CONSTRAINT`
  pinning `setuptools<80` (a transitive dep, `flatdict==4.0.1`, has a `setup.py`
  that imports `pkg_resources`, dropped in 80+) and `TERM=xterm-256color` (its
  `tput` calls need a real terminfo entry).
- Only `CL_isaaclab_sim/requirements.txt` is `COPY`'d in; the source tree is
  bind-mounted at runtime.

**Runtime stage:** copies the populated `/opt/conda`, `/cyclonedds`, `IsaacLab`,
and `unitree_sdk2_python` from the builder, installs runtime-only X11/EGL libs, and
auto-activates the conda env in `.bashrc`. **ROS publishing uses Isaac Sim's bundled
`isaacsim.ros2.bridge` (OmniGraph)** — there is no `/opt/ros/humble` here and
`rclpy` is not installed. (The bundled `librmw_cyclonedds` is deliberately *not*
put on the global `LD_LIBRARY_PATH`; only Kit's own loader can load it correctly.)

---

## 7. MuJoCo MPC (MJPC) dev loop

> **Status (branch `mjpc_reintegration`):** the fork-era pipeline (baked
> `agent_server` + python `mujoco_mpc` install, `/opt/mjpc-build-seed`, runtime
> hydrate + mtime back-dating, `unitree_sdk2` hydrate) stays removed. This
> section describes the reintroduced, much thinner loop against **pristine
> google-deepmind upstream** (+ our `h12_lowerbody` task on `max_playground`).

MJPC is a CMake/FetchContent project (it fetches and builds its own pinned
MuJoCo 3.2.3, abseil, glfw and gRPC). It is **built standalone, outside colcon**,
entirely at container runtime — the `ros` image bakes only the toolchain
(clang-13/llvm-13/lld-13, ninja, X11/GL dev headers, `patch`; mirrors the only
Linux config upstream CI tests) plus the `mujoco==3.2.3` pip ABI pin.

The pieces:

- **Source** — the `mujoco_mpc` submodule, bind-mounted rw at
  `/home/code/mujoco_mpc` (host-editable; never baked).
- **Build tree** — `container_cache/mjpc_build`, bind-mounted at
  `/home/code/mujoco_mpc/build` (the standard in-tree cmake build dir, covered
  by the submodule's own `build/` gitignore rule). Same pattern as
  `container_cache/msgs_ws`: host-persisted, so rebuilds are incremental across
  `docker compose run --rm` cycles. Wipe the dir for a cold build (~15–25 min,
  dominated by gRPC).
- **Rebuild** — `docker/scripts/rebuild_mjpc.sh` (reachable in-container at
  `/home/code/h12_sim_scripts/rebuild_mjpc.sh`): configures only when the tree
  is cold (clang-13 + Ninja + Release + `MJPC_BUILD_GRPC_SERVICE=ON`, tests
  off), then a single `cmake --build`. Warm loop: edit a task `.cc`, rerun —
  seconds. Default targets: `mjpc agent_server testspeed`; pass others as args.
- **Consumer** — `core_ws/src/h12_deploy_mjpc` links the build tree *directly*
  (`libmjpc.a` + threadpool + abseil archives + the build's own `libmujoco.so`;
  compiled with clang-13 and linked with `ld.lld-13` because upstream leaves
  IPO/LTO on for Release, making the archives clang bitcode). Never compile
  against pip mujoco headers — mjModel layouts skew. The package auto-skips its
  C++ targets (warning only) while the mjpc artifacts are absent, so the
  launch-time colcon build stays green on a fresh host — and it detects them
  via `CONFIGURE_DEPENDS` globs, so the first colcon build after
  `rebuild_mjpc.sh` reconfigures and picks the targets up automatically. (A
  `core_ws/build/h12_deploy_mjpc` configured before the package's clang-13 pin
  existed is stuck on gcc — the configure warning tells you to wipe that one
  dir once.) `agent_server` keeps building so a later switch of the deploy
  layer to the gRPC/python client is cheap.
- **h12_lowerbody task assets** — `task.xml` lives in the submodule; the model
  (`h1_2_magpie.xml`, its two magpie-gripper includes, and `meshes/`) is
  bind-mounted into the task dir from `CL_Assets` (single source of truth; all
  mountpoints gitignored by the task dir's `.gitignore`, pre-created by
  `docker_run.sh`). Binaries resolve task XMLs from the *staged copy* in
  `build/mjpc/tasks` (refreshed on every `rebuild_mjpc.sh` run); set
  `MJPC_TASKS_DIR=/home/code/mujoco_mpc/mjpc/tasks` for zero-rebuild XML
  iteration on h12_lowerbody (source-tree lookup only works for tasks whose
  assets are fully in-tree — menagerie-based upstream tasks need the staged
  copy).

The dev loop, end to end (inside/against a running `hams_ros`):

```bash
# 1. (re)build mjpc — cold: ~15-25 min once per host; warm: seconds
docker exec -it hams_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh

# 2. build + smoke-test the deploy link
docker exec -it hams_ros bash -c 'source /opt/ros/humble/setup.bash \
  && cd /home/code/core_ws \
  && colcon build --symlink-install --packages-select h12_deploy_mjpc \
  && ./install/h12_deploy_mjpc/lib/h12_deploy_mjpc/mjpc_smoketest'
```

> Submodule flow: mjpc changes are committed on `max_playground`, pushed to
> origin (badinkajink), then the HAMS gitlink pin is bumped. Dep/tag bumps in
> mjpc's `CMakeLists.txt` require wiping `container_cache/mjpc_build` first
> (the cached tree pins FetchContent offline).

---

## 8. Build & run orchestration

**Build** — `docker/scripts/docker_build.sh [isaac|robocasa|ros]…` (all three if no
args). Builds `hams_base` first whenever `ros` or `robocasa` is selected, then
`docker compose … build` for the requested profiles.

**Run** — `docker/scripts/docker_run.sh <profile> [cmd…]`:
- Sources `docker/.env` (`GEMINI_API_KEY`, `ROS_DOMAIN_ID`, …).
- Pre-creates the host bind mountpoints (`container_cache/msgs_ws`,
  `container_cache/mjpc_build`, `mujoco_mpc/build`, and the h12_lowerbody
  task-asset files/dir inside the submodule) so dockerd doesn't create them
  root-owned. The `ros` profile hard-fails if the mujoco_mpc task dir or the
  CL_Assets model XMLs are missing (uninitialized submodules) — otherwise
  dockerd would root-create paths inside the submodule working trees.
- Normalizes `ROS_DOMAIN_ID` (empty→1; `0` rejected for sims, confirmed for `ros`).
- `xhost +local:docker`, stable container names (`hams_ros`, `hams_sim_*`), `--rm`.

The Apple-Silicon port mirrors these as `docker/mac/scripts/docker_build_mac.sh` and
`docker/mac/scripts/docker_run_mac.sh` (against `docker/mac/docker-compose.yml`):
services `robocasa`/`ros` only (no `isaac`), base built from
`mac/BaseDockerfile.arm64`, no `xhost`. See the README macOS section.

**`docker-compose.yml`** defines three profiles (`isaac`, `robocasa`, `ros`), each
with `runtime: nvidia`, `network_mode: host`, X11 passthrough, and the bind mounts.
Each profile's `command:` is its `launch_*.sh`.

**`.dockerignore` is deny-all (`*`)** with a tiny whitelist — everything is
bind-mounted at runtime, so only the few files a Dockerfile actually `COPY`s
(`CL_isaaclab_sim/requirements.txt`, `core_ws/src/livox_ros_driver2`,
`unitree_sdk2_python`) are sent to the daemon. This keeps the build context small
(~GB otherwise) and is why `mujoco_mpc` needs no `.dockerignore` entry.

**Launchers:**
- `launch_ros.sh` — sources ROS, then `colcon build
  --symlink-install`s `core_ws` **only if needed** (no `install/`, or any
  `package.xml` newer than `install/setup.bash`), sources the overlay, drops to a
  shell. `livox_ros_driver2` is built via its own `build.sh` (patched idempotently
  to add `--symlink-install`).
- `launch_robocasa.sh` — sources ROS + `/opt/livox_ws`, `colcon build`s the two IDL
  packages in `msgs_ws`, picks `MUJOCO_GL`, runs `h12_mujoco.py`.
- `launch_isaac.sh` — runs the **Unitree DDS relay** (`dds_bridge.py`, a
  CycloneDDS↔CycloneDDS relay of `rt/lowstate`, `rt/lowcmd`, `rt/inspire/*`, …
  between the sim domain and the command domain) in the background, then
  `CL_isaaclab_sim/sim_main.py` from the conda interpreter. (The OmniGraph ROS 2
  bridge is a *separate* mechanism, loaded inside Kit by `sim_main` — see §6.)
  Isaac task selection is WIP.

---

## 9. The colcon workspace(s)

colcon is invoked **at container start**, never at image-build time, against
bind-mounted source, with output persisted on the host:

| Container | Workspace | Built | Cache |
|---|---|---|---|
| `ros` | `core_ws` (full) | every start, gated on staleness | host `core_ws/{build,install,log}` |
| `robocasa` | `msgs_ws` (IDL only) | every start (fast no-op) | host `container_cache/msgs_ws` |

`--symlink-install` is used throughout so Python nodes and model weights resolve
via the install symlinks without a manual copy. Because build/install/log live on
host-side bind mounts, incremental rebuilds across `docker compose run --rm` cycles
are near-instant; wipe the host dir for a clean rebuild.

---

## 10. Cross-cutting invariants & gotchas

- **`setuptools==59.6.0`, `wheel<0.44`, `numpy<2`** are re-clamped after any pip
  step that bumps them — required for colcon's `python_setup_py`, torch, and the
  pinocchio/scipy ABI. Watch for `_distutils_hack` errors (a stale
  `distutils-precedence.pth`).
- **CycloneDDS is pinned to 0.10.x from source** in every base/Isaac image (PyPI
  wheel needs the removed `q_radmin.h`).
- **MuJoCo versions are pinned per image and must not drift:** `3.3.1` in
  `robocasa` (matches RoboCasa's pin); `3.2.3` pip in `ros` (matches the MuJoCo
  commit mjpc's CMake fetches, `mjVERSION 323` — the MJPC ABI pin). Bumping
  mjpc's `MUJOCO_MPC_MUJOCO_GIT_TAG` means moving the `ros` pip pin with it.
- **Layer-order discipline:** heavy/stable layers first; volatile pins appended
  last. New deps go at the *end* of a Dockerfile so they don't bust cached layers.
- **CUDA wheel split:** `ros`/base use torch **cu130**; Isaac uses **cu128**;
  robocasa ends up on a cu121/CPU torch (via `lerobot`). Deliberate, per image.

---

## 11. Common commands

```bash
# one-time
git submodule update --init --recursive
git lfs install
cp docker/.env.example docker/.env      # set GEMINI_API_KEY, ROS_DOMAIN_ID

# build
docker/scripts/docker_build.sh              # base + all profiles
docker/scripts/docker_build.sh robocasa ros # subset

# run
docker/scripts/docker_run.sh robocasa            # windowed sim
docker/scripts/docker_run.sh ros                 # workspace shell (auto colcon build)
docker/scripts/docker_run.sh isaac

# inside the ros container
ros2 launch h1_bringup h1_sim_bringup.launch.py
colcon build --symlink-install                   # force a rebuild

# MJPC (see §7): standalone cmake build into container_cache/mjpc_build
docker exec -it hams_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh
docker exec -it hams_ros bash -c 'source /opt/ros/humble/setup.bash \
  && cd /home/code/core_ws \
  && colcon build --symlink-install --packages-select h12_deploy_mjpc \
  && ./install/h12_deploy_mjpc/lib/h12_deploy_mjpc/mjpc_smoketest'
```
