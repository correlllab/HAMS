# H12 skills

## Spatial memory

`spatial_memory` is an optional, movement-independent ROS 2 skill. It records the
H12 head camera every two seconds with the corresponding TF pose and UTC capture
time, then delegates embedding, live FAISS indexing, persistence, and optional
Gemini reranking to the mentor-maintained `EmbodiedAgent` checkout.

The memory node is deliberately separate from the main `skills` executable. A
missing model, API key, or EmbodiedAgent checkout therefore cannot stop grasp or
the other manipulation actions from starting.

### Runtime contract

- Input image: `/realsense/head/color/image_raw/compressed`
- Pose: TF `odom -> pelvis` in simulation (both frames are configurable)
- Query action: `/skill/retrieve_memory`
- Persistent data: `/data/spatial_memory`
- Gemini is opt-in per goal through `rerank: true`; FAISS still works without a
  key.

The action returns observation poses—the places from which H12 saw the target.
It does not claim to return oracle object coordinates and it never commands the
base. Navigation can consume a returned `memory_id` / pose later.

### Start

The ROS compose service mounts a sibling EmbodiedAgent checkout read-only. Set
`EMBODIED_AGENT_ROOT` only if the checkout is somewhere other than
`../EmbodiedAgent` relative to HAMS.

```bash
docker compose -f docker/docker-compose.yml build ros
docker compose -f docker/docker-compose.yml run --rm ros

# inside the ROS container after the workspace build
ros2 launch h12_skills spatial_memory.launch.py
```

RoboCasa must publish its ground-truth odometry (`HAMS_SIM_ODOM=1`). On the real
H12 with FAST-LIO, launch with `world_frame:=camera_init` instead.

Query the local FAISS index:

```bash
ros2 action send_goal /skill/retrieve_memory \
  custom_ros_messages/action/SkillRetrieveMemory \
  "{query: 'find the current mug', top_k: 3, rerank: false}" --feedback
```

Request Gemini reranking:

```bash
ros2 action send_goal /skill/retrieve_memory \
  custom_ros_messages/action/SkillRetrieveMemory \
  "{query: 'find the current mug', top_k: 3, rerank: true}" --feedback
```

### Scope and validation

The skill owns sensor ingestion, live updates, retrieval, persistence, and
reranking. Frontier exploration, Nav2, and H12 walking are intentionally outside
its boundary. Recorded RoboCasa frames can therefore verify this skill without
depending on locomotion.

Run one frozen episode through the same capture/backend path:

```bash
spatial_memory_replay /data/episodes/episode_000_mug \
  --data-dir /tmp/h12-memory-replay --verify-restart
```
