#!/usr/bin/env python3
"""Generate an algorithm-neutral RoboCasa spatial-memory benchmark.

Each episode contains two passes over the same deterministic virtual-camera
route. A single target object starts on surface A, is moved (without capturing
the move) to surface B, and is then observed during the second pass. Sensor
records and simulator oracle labels are deliberately stored in separate files.

The sensor side remains compatible with EmbodiedAgent's scan importer::

    color/000000.png
    depth/000000.npy             # float32 meters, zero means no return
    robot_xy/000000.txt
    frame_meta/000000.json

The benchmark evaluator uses ``observations.jsonl`` and ``queries.jsonl`` as its
algorithm-neutral contract. Simulator-only visibility and target poses live in
``oracle/episode.json`` and must never be passed to a memory implementation.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mujoco
import numpy as np

import h1_2_robosuite
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.fixtures import FixtureType, fixture_is_type
from robocasa.utils import env_utils as EnvUtils

from spatial_memory_scan import (
    _floor_bounds,
    _hide_robot_geometries,
    _make_waypoints,
    _park_robot_below_scene,
    _scene_option,
    _set_free_camera,
    _write_contact_sheet,
    write_frame_atomic,
)
from spatial_memory_geometry import (
    CAMERA_FRAME,
    camera_intrinsics_from_fovy,
    camera_to_world_from_mujoco,
)


SCHEMA_VERSION = 2
BENCHMARK_ID = "robocasa_object_relocation_v2"
TARGET_NAME = "target"


def _surface_distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a.pos[:2]) - np.asarray(b.pos[:2])))


class SpatialMemoryObjectRelocation(Kitchen):
    """Minimal RoboCasa task used only to construct randomized benchmark scenes."""

    def __init__(
        self,
        *args,
        benchmark_object_group: str = "mug",
        benchmark_episode_seed: int = 0,
        **kwargs,
    ):
        self.benchmark_object_group = benchmark_object_group
        self.benchmark_episode_seed = int(benchmark_episode_seed)
        self.benchmark_surfaces = []
        self.benchmark_surface_a = None
        self.benchmark_surface_b = None
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        eligible = [
            fixture
            for _, fixture in sorted(self.fixtures.items())
            if fixture_is_type(fixture, FixtureType.COUNTER)
            and len(getattr(fixture, "size", ())) >= 2
            and min(float(fixture.size[0]), float(fixture.size[1])) >= 0.55
        ]
        if len(eligible) < 2:
            raise RuntimeError("benchmark requires at least two usable counter surfaces")

        # Keep only well-separated pairs. Cycling this deterministic list across
        # episode seeds gives surface diversity while retaining reproducibility.
        pairs = [
            pair for pair in itertools.combinations(eligible, 2)
            if _surface_distance(*pair) >= 1.25
        ]
        if not pairs:
            raise RuntimeError("benchmark found no pair of sufficiently separated surfaces")
        pairs.sort(key=lambda pair: (pair[0].name, pair[1].name))
        pair_index = self.benchmark_episode_seed % len(pairs)
        surface_a, surface_b = pairs[pair_index]
        if (self.benchmark_episode_seed // len(pairs)) % 2:
            surface_a, surface_b = surface_b, surface_a

        self.benchmark_surfaces = eligible
        self.benchmark_surface_a = surface_a
        self.benchmark_surface_b = surface_b
        self.init_robot_base_ref = surface_a

    def _get_obj_cfgs(self):
        return [
            {
                "name": TARGET_NAME,
                "obj_groups": self.benchmark_object_group,
                "placement": {
                    "fixture": self.benchmark_surface_a,
                    "size": (0.38, 0.34),
                    "pos": (0.0, 0.0),
                    "margin": 0.02,
                },
            }
        ]

    def _check_success(self):
        return False


def _json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def _prepare_root(output: Path, overwrite: bool) -> None:
    resolved = output.resolve()
    if resolved == Path("/") or len(resolved.parts) < 3:
        raise RuntimeError(f"refusing unsafe output path: {resolved}")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RuntimeError(f"benchmark output is not empty: {output} (pass --overwrite)")
        manifest_path = output / "benchmark_manifest.json"
        in_progress_path = output / ".benchmark_in_progress.json"
        identity_path = manifest_path if manifest_path.is_file() else in_progress_path
        if not identity_path.is_file():
            raise RuntimeError(
                "refusing to overwrite a directory without a benchmark identity file"
            )
        with open(identity_path, encoding="utf-8") as file:
            existing = json.load(file)
        if existing.get("benchmark_id") != BENCHMARK_ID:
            raise RuntimeError("refusing to overwrite a different benchmark dataset")
        for name in (
            "episodes",
            "reports",
            "benchmark_manifest.json",
            ".benchmark_in_progress.json",
        ):
            path = output / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    _json_dump(
        output / ".benchmark_in_progress.json",
        {"benchmark_id": BENCHMARK_ID, "status": "generating"},
    )


def _restore_output_owner(output: Path, owner: str | None) -> None:
    """Return root-container output to the invoking host user.

    RoboCasa's Objaverse loader writes a derived XML next to its installed asset,
    so object-bearing tasks must currently run as root inside the image. The
    wrapper passes the host uid:gid here so only this benchmark output tree is
    chowned back before the container exits.
    """
    if owner is None:
        return
    try:
        uid_text, gid_text = owner.split(":", maxsplit=1)
        uid, gid = int(uid_text), int(gid_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError("--output-owner must be numeric UID:GID") from exc
    for path in [output, *output.rglob("*")]:
        os.chown(path, uid, gid)


def _surface_record(surface) -> dict:
    return {
        "name": surface.name,
        "fixture_type": type(surface).__name__,
        "position": [float(value) for value in surface.pos],
        "size": [float(value) for value in surface.size],
        "yaw_rad": float(surface.rot),
    }


def _camera_route(
    candidates: list[tuple[float, float]], surfaces: list, route_points: int
) -> list[dict]:
    """Choose a deterministic route with views of every eligible surface.

    Route selection uses scene geometry, not the target's A/B state. Each surface
    gets one view before any surface gets a second view. Selected positions are
    then nearest-neighbour ordered to make the kinematic motion locally smooth.
    """
    if route_points < 2:
        raise ValueError("route_points must be at least 2")
    remaining = sorted(candidates)
    selected: list[dict] = []
    pass_index = 0
    while remaining and len(selected) < route_points:
        made_progress = False
        for surface in surfaces:
            if len(selected) >= route_points or not remaining:
                break
            sx, sy = (float(surface.pos[0]), float(surface.pos[1]))

            def score(point):
                distance = math.hypot(point[0] - sx, point[1] - sy)
                # About 1.35 m gives an object-sized crop without putting the
                # virtual rover on top of the counter.
                range_cost = abs(distance - (1.35 + 0.25 * pass_index))
                diversity_cost = 0.0
                prior = [item for item in selected if item["focus_surface"] == surface.name]
                if prior:
                    previous_angle = math.atan2(
                        prior[-1]["y"] - sy, prior[-1]["x"] - sx
                    )
                    angle = math.atan2(point[1] - sy, point[0] - sx)
                    separation = abs(math.atan2(
                        math.sin(angle - previous_angle), math.cos(angle - previous_angle)
                    ))
                    diversity_cost = abs(math.pi / 2.0 - separation) * 0.20
                return range_cost + diversity_cost, point

            point = min(remaining, key=score)
            remaining.remove(point)
            selected.append({
                "x": float(point[0]),
                "y": float(point[1]),
                "focus_surface": surface.name,
                "surface_xy": [sx, sy],
            })
            made_progress = True
        if not made_progress:
            break
        pass_index += 1

    if len(selected) < route_points:
        raise RuntimeError(
            f"only {len(selected)} distinct safe route points available; requested {route_points}"
        )

    # Start at a stable lexicographic point, then greedily choose the nearest
    # remaining view. This order is identical for both laps.
    ordered = [min(selected, key=lambda item: (item["x"], item["y"]))]
    pool = [item for item in selected if item is not ordered[0]]
    while pool:
        last = ordered[-1]
        next_item = min(
            pool,
            key=lambda item: (
                (item["x"] - last["x"]) ** 2 + (item["y"] - last["y"]) ** 2,
                item["x"],
                item["y"],
            ),
        )
        pool.remove(next_item)
        ordered.append(next_item)
    for index, item in enumerate(ordered):
        sx, sy = item.pop("surface_xy")
        item["route_index"] = index
        item["yaw_rad"] = math.atan2(sy - item["y"], sx - item["x"])
    return ordered


def _target_geom_ids(model, target_body_id: int) -> set[int]:
    geom_ids = set()
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        while body_id > 0:
            if body_id == target_body_id:
                geom_ids.add(geom_id)
                break
            body_id = int(model.body_parentid[body_id])
    return geom_ids


def _render_rgb_and_visibility(
    renderer: mujoco.Renderer,
    model,
    data,
    camera: mujoco.MjvCamera,
    option: mujoco.MjvOption,
    target_geom_ids: set[int],
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Render RGB-D, target visibility, and the exact rendered camera pose.

    MuJoCo 3.3's segmentation remap can index past its temporary table for some
    Objaverse assets whose scene ``segid`` values are sparse. A paired normal /
    target-hidden render avoids that renderer bug and directly labels whether
    the target contributes visible RGB pixels in this exact camera view.
    """
    renderer.disable_segmentation_rendering()
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera, scene_option=option)
    rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()

    geom_ids = np.asarray(sorted(target_geom_ids), dtype=int)
    original_groups = model.geom_group[geom_ids].copy()
    try:
        # _scene_option enables only visual groups 1 and 2; group 5 is hidden.
        model.geom_group[geom_ids] = 5
        renderer.update_scene(data, camera=camera, scene_option=option)
        background = np.asarray(renderer.render(), dtype=np.uint8).copy()
    finally:
        model.geom_group[geom_ids] = original_groups
    difference = np.abs(rgb.astype(np.int16) - background.astype(np.int16))
    visible_pixels = int((difference.max(axis=2) > 8).sum())

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera, scene_option=option)
    gl_camera = renderer.scene.camera[0]
    camera_to_world = camera_to_world_from_mujoco(
        gl_camera.pos,
        gl_camera.forward,
        gl_camera.up,
    )
    depth_m = np.asarray(renderer.render(), dtype=np.float32).copy()
    renderer.disable_depth_rendering()
    zfar = float(model.vis.map.zfar * model.stat.extent)
    invalid = (
        ~np.isfinite(depth_m)
        | (depth_m <= 0.0)
        | (depth_m >= 0.999 * zfar)
    )
    depth_m[invalid] = 0.0
    return rgb, depth_m, visible_pixels, camera_to_world


def _object_pose(env) -> list[float]:
    qpos = np.asarray(env.sim.data.get_joint_qpos(env.objects[TARGET_NAME].joints[0]))
    return [float(value) for value in qpos]


def _relocate_target(env, surface) -> list[float]:
    cfg = {
        "name": TARGET_NAME,
        "type": "object",
        "placement": {
            "fixture": surface,
            "size": (0.38, 0.34),
            "pos": (0.0, 0.0),
            "margin": 0.02,
        },
    }
    sampler = EnvUtils._get_placement_initializer(env, [copy.deepcopy(cfg)])
    placements = sampler.sample(placed_objects=env.fxtr_placements)
    position, quaternion, obj = placements[TARGET_NAME]
    env.sim.data.set_joint_qpos(
        obj.joints[0], np.concatenate([np.asarray(position), np.asarray(quaternion)])
    )
    qvel_addr = env.sim.model.get_joint_qvel_addr(obj.joints[0])
    if isinstance(qvel_addr, tuple):
        env.sim.data.qvel[qvel_addr[0]:qvel_addr[1]] = 0.0
    else:
        env.sim.data.qvel[qvel_addr] = 0.0
    env.sim.forward()
    return _object_pose(env)


def _target_description(env, requested_group: str) -> dict:
    cfg = next(cfg for cfg in env.object_cfgs if cfg["name"] == TARGET_NAME)
    info = cfg.get("info", {})
    try:
        language = env.get_obj_lang(TARGET_NAME)
    except Exception:
        language = requested_group.replace("_", " ")
    return {
        "name": TARGET_NAME,
        "requested_group": requested_group,
        "language": str(language),
        "category": info.get("cat"),
        "mjcf_path": info.get("mjcf_path"),
    }


def _episode_timestamp(frame_idx: int, interval_seconds: float) -> str:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(seconds=frame_idx * interval_seconds)).isoformat(
        timespec="milliseconds"
    )


def _build_queries(
    observations: list[dict],
    oracle_frames: list[dict],
    target: dict,
    target_pose_a: list[float],
    target_pose_b: list[float],
) -> list[dict]:
    lap1_visible = [
        item["observation_id"] for item in oracle_frames
        if item["lap"] == 1 and item["target_visible"]
    ]
    lap2_visible = [
        item["observation_id"] for item in oracle_frames
        if item["lap"] == 2 and item["target_visible"]
    ]
    if not lap1_visible or not lap2_visible:
        raise RuntimeError(
            "camera route did not visibly observe the target in both laps; "
            f"lap1={len(lap1_visible)} lap2={len(lap2_visible)}"
        )
    language = target["language"]
    queries = [
        {
            "query_id": "static_end_lap1",
            "track": "static",
            "checkpoint_frame": max(
                item["frame_idx"] for item in observations if item["lap"] == 1
            ),
            "text": f"Find the {language}",
            "relevant_observation_ids": lap1_visible,
            "stale_observation_ids": [],
            "target_position_world_xyz": target_pose_a[:3],
            "stale_target_position_world_xyz": None,
        }
    ]
    first_visible_lap2 = next(
        item["frame_idx"] for item in oracle_frames
        if item["lap"] == 2 and item["target_visible"]
    )
    for observation in observations:
        if observation["lap"] != 2 or observation["frame_idx"] < first_visible_lap2:
            continue
        visible_so_far = [
            item["observation_id"] for item in oracle_frames
            if item["lap"] == 2
            and item["target_visible"]
            and item["frame_idx"] <= observation["frame_idx"]
        ]
        if not visible_so_far:
            continue
        queries.append({
            "query_id": f"live_{observation['frame_idx']:06d}",
            "track": "live_current",
            "checkpoint_frame": observation["frame_idx"],
            "first_visible_current_frame": first_visible_lap2,
            "text": (
                f"Find the current location of the {language}; "
                "prefer the newest visual evidence."
            ),
            "relevant_observation_ids": visible_so_far,
            "stale_observation_ids": lap1_visible,
            "target_position_world_xyz": target_pose_b[:3],
            "stale_target_position_world_xyz": target_pose_a[:3],
        })
    queries.extend([
        {
            "query_id": "history_end_lap2",
            "track": "history",
            "checkpoint_frame": observations[-1]["frame_idx"],
            "text": f"Where was the {language} before it moved?",
            "relevant_observation_ids": lap1_visible,
            "stale_observation_ids": [],
            "target_position_world_xyz": target_pose_a[:3],
            "stale_target_position_world_xyz": None,
        },
        {
            "query_id": "absent_end_lap2",
            "track": "absent",
            "checkpoint_frame": observations[-1]["frame_idx"],
            "text": "Find a teddy bear",
            "relevant_observation_ids": [],
            "stale_observation_ids": [],
            "target_position_world_xyz": None,
            "stale_target_position_world_xyz": None,
        },
    ])
    return queries


def generate_episode(args, episode_index: int, object_group: str) -> dict:
    episode_seed = args.seed + episode_index
    episode_id = f"episode_{episode_index:03d}_{object_group}"
    episode_dir = Path(args.output) / "episodes" / episode_id
    if episode_dir.exists():
        raise RuntimeError(f"episode output already exists: {episode_dir}")
    for directory in ("color", "depth", "robot_xy", "frame_meta", "oracle"):
        (episode_dir / directory).mkdir(parents=True, exist_ok=True)

    env = None
    renderer = None
    try:
        env = h1_2_robosuite.make_kitchen_env(
            "SpatialMemoryObjectRelocation",
            layout_ids=args.layout,
            style_ids=args.style,
            seed=episode_seed,
            benchmark_object_group=object_group,
            benchmark_episode_seed=episode_seed,
        )
        env.reset()
        _park_robot_below_scene(env)
        model = env.sim.model._model
        data = env.sim.data._data
        hidden_robot_geoms = _hide_robot_geometries(model)
        model.vis.global_.fovy = args.fovy
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.image_size)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.image_size)

        _, floor_z = _floor_bounds(env)
        camera_z = floor_z + args.camera_height
        all_waypoints, room_bounds, _ = _make_waypoints(
            env,
            model,
            data,
            camera_z=camera_z,
            spacing=args.spacing,
            room_margin=args.room_margin,
            footprint_radius=args.footprint_radius,
            max_positions=0,
        )
        route = _camera_route(
            all_waypoints, env.benchmark_surfaces, args.route_points
        )
        camera = mujoco.MjvCamera()
        option = _scene_option()
        renderer = mujoco.Renderer(
            model, height=args.image_size, width=args.image_size
        )

        target_obj = env.objects[TARGET_NAME]
        target_body_id = env.sim.model.body_name2id(target_obj.root_body)
        target_geom_ids = _target_geom_ids(model, target_body_id)
        if not target_geom_ids:
            raise RuntimeError("could not resolve target object geometry ids")

        target = _target_description(env, object_group)
        surface_a = env.benchmark_surface_a
        surface_b = env.benchmark_surface_b
        pose_a = _object_pose(env)
        observations: list[dict] = []
        oracle_frames: list[dict] = []
        rendered_paths: list[Path] = []
        visibility_threshold = max(
            args.visibility_pixels,
            math.ceil(args.image_size * args.image_size * args.visibility_fraction),
        )
        intrinsics = camera_intrinsics_from_fovy(
            args.image_size,
            args.image_size,
            args.fovy,
        )

        for lap in (1, 2):
            if lap == 2:
                pose_b = _relocate_target(env, surface_b)
                print(
                    f"[memory-benchmark] {episode_id} relocated "
                    f"{surface_a.name} -> {surface_b.name}"
                )
            for route_item in route:
                frame_idx = len(observations)
                x = route_item["x"]
                y = route_item["y"]
                yaw = route_item["yaw_rad"]
                _set_free_camera(
                    camera, x, y, camera_z, yaw, args.pitch, args.look_distance
                )

                rgb, depth_m, target_pixels, camera_to_world = _render_rgb_and_visibility(
                    renderer,
                    model,
                    data,
                    camera,
                    option,
                    target_geom_ids,
                )

                timestamp = _episode_timestamp(frame_idx, args.capture_interval)
                image_path, frame_meta = write_frame_atomic(
                    output=episode_dir,
                    frame_idx=frame_idx,
                    rgb=rgb,
                    pose=(x, y, yaw),
                    captured_at=timestamp,
                    source_type="benchmark_observe",
                    episode_id=episode_id,
                    camera_metadata={
                        "z": camera_z,
                        "pitch_deg": args.pitch,
                        "fovy_deg": args.fovy,
                        "width": args.image_size,
                        "height": args.image_size,
                        "intrinsics": intrinsics.tolist(),
                        "camera_to_world": camera_to_world.tolist(),
                        "coordinate_frame": CAMERA_FRAME,
                        "lap": lap,
                        "route_index": route_item["route_index"],
                    },
                    depth_m=depth_m,
                )
                observation_id = f"obs_{frame_idx:06d}"
                observations.append({
                    "observation_id": observation_id,
                    "frame_idx": frame_idx,
                    "lap": lap,
                    "route_index": route_item["route_index"],
                    "timestamp": timestamp,
                    "image_path": frame_meta["image_path"],
                    "depth_path": frame_meta["depth_path"],
                    "depth_unit": frame_meta["depth_unit"],
                    "pose_path": frame_meta["pose_path"],
                    "robot_pose": frame_meta["robot_pose"],
                    "camera_intrinsics": intrinsics.tolist(),
                    "camera_to_world": camera_to_world.tolist(),
                    "camera_coordinate_frame": CAMERA_FRAME,
                })
                oracle_frames.append({
                    "observation_id": observation_id,
                    "frame_idx": frame_idx,
                    "lap": lap,
                    "current_surface": surface_a.name if lap == 1 else surface_b.name,
                    "camera_focus_surface": route_item["focus_surface"],
                    "target_pixel_count": target_pixels,
                    "target_visible": target_pixels >= visibility_threshold,
                })
                rendered_paths.append(image_path)
                print(
                    f"[memory-benchmark] {episode_id} frame={frame_idx:06d} "
                    f"lap={lap} route={route_item['route_index']} target_pixels={target_pixels}"
                )

        queries = _build_queries(
            observations,
            oracle_frames,
            target,
            pose_a,
            pose_b,
        )
        _write_jsonl(episode_dir / "observations.jsonl", observations)
        _write_jsonl(episode_dir / "queries.jsonl", queries)
        _json_dump(
            episode_dir / "oracle" / "episode.json",
            {
                "schema_version": SCHEMA_VERSION,
                "warning": "simulator ground truth; never provide this file to a memory algorithm",
                "episode_id": episode_id,
                "target": target,
                "surface_a": _surface_record(surface_a),
                "surface_b": _surface_record(surface_b),
                "target_pose_a_xyzw": pose_a,
                "target_pose_b_xyzw": pose_b,
                "relocation": {
                    "after_frame": args.route_points - 1,
                    "before_frame": args.route_points,
                },
                "frames": oracle_frames,
            },
        )
        _write_contact_sheet(rendered_paths, episode_dir / "contact_sheet.jpg")
        return {
            "episode_id": episode_id,
            "episode_seed": episode_seed,
            "object_group": object_group,
            "target_language": target["language"],
            "surface_a": surface_a.name,
            "surface_b": surface_b.name,
            "frame_count": len(observations),
            "query_count": len(queries),
            "visible_frames_lap1": sum(
                item["target_visible"] and item["lap"] == 1 for item in oracle_frames
            ),
            "visible_frames_lap2": sum(
                item["target_visible"] and item["lap"] == 2 for item in oracle_frames
            ),
            "hidden_robot_geoms": hidden_robot_geoms,
            "room_bounds_xy": list(room_bounds),
            "route": route,
        }
    except Exception:
        # A partial episode cannot be evaluated and would prevent a clean retry.
        shutil.rmtree(episode_dir, ignore_errors=True)
        raise
    finally:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()


def generate(args) -> Path:
    output = Path(args.output)
    _prepare_root(output, args.overwrite)
    object_groups = [part.strip() for part in args.objects.split(",") if part.strip()]
    if not object_groups:
        raise ValueError("--objects must contain at least one RoboCasa object group")

    try:
        episodes = []
        for episode_index in range(args.episodes):
            object_group = object_groups[episode_index % len(object_groups)]
            episodes.append(generate_episode(args, episode_index, object_group))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": BENCHMARK_ID,
            "description": "two-pass object relocation with incremental memory queries",
            "oracle_policy": "oracle/ is evaluator-only and excluded from adapter input",
            "scene": {
                "simulator": "RoboCasa",
                "layout": args.layout,
                "style": args.style,
                "base_seed": args.seed,
            },
            "capture": {
                "motion": "two identical deterministic kinematic camera-route laps",
                "modalities": ["rgb", "depth"],
                "depth_unit": "meter",
                "invalid_depth_value": 0.0,
                "camera_coordinate_frame": CAMERA_FRAME,
                "route_points_per_lap": args.route_points,
                "capture_interval_seconds": args.capture_interval,
                "camera_height_m": args.camera_height,
                "pitch_deg": args.pitch,
                "image_size": args.image_size,
                "visibility_threshold_pixels": args.visibility_pixels,
                "visibility_threshold_fraction": args.visibility_fraction,
                "effective_visibility_threshold_pixels": max(
                    args.visibility_pixels,
                    math.ceil(
                        args.image_size * args.image_size * args.visibility_fraction
                    ),
                ),
            },
            "episode_count": len(episodes),
            "episodes": episodes,
        }
        _json_dump(output / "benchmark_manifest.json", manifest)
        (output / ".benchmark_in_progress.json").unlink()
        print(f"[memory-benchmark] wrote {len(episodes)} episodes to {output}")
        return output
    finally:
        if output.exists():
            _restore_output_owner(output, args.output_owner)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the RoboCasa spatial-memory object-relocation benchmark"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-owner", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--objects", default="mug,bowl")
    parser.add_argument("--layout", type=int, default=9)
    parser.add_argument("--style", type=int, default=9)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--route-points", type=int, default=8)
    parser.add_argument("--capture-interval", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--camera-height", type=float, default=1.25)
    parser.add_argument("--pitch", type=float, default=-15.0)
    parser.add_argument("--look-distance", type=float, default=2.0)
    parser.add_argument("--fovy", type=float, default=60.0)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--room-margin", type=float, default=0.35)
    parser.add_argument("--footprint-radius", type=float, default=0.30)
    parser.add_argument("--visibility-pixels", type=int, default=24)
    parser.add_argument(
        "--visibility-fraction",
        type=float,
        default=0.001,
        help="minimum fraction of image pixels visibly affected by the target",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes <= 0:
        raise SystemExit("episodes must be positive")
    if args.route_points < 2:
        raise SystemExit("route-points must be at least 2")
    if args.capture_interval <= 0 or args.image_size <= 0:
        raise SystemExit("capture-interval and image-size must be positive")
    if args.visibility_pixels <= 0:
        raise SystemExit("visibility-pixels must be positive")
    if not 0.0 < args.visibility_fraction < 1.0:
        raise SystemExit("visibility-fraction must be between zero and one")
    generate(args)


if __name__ == "__main__":
    main()
