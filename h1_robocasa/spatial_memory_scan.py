#!/usr/bin/env python3
"""Export a deterministic RoboCasa camera scan for EmbodiedAgent memory.

The camera is intentionally kinematic: it moves on a fixed-height ``x, y, yaw``
grid without a robot, locomotion controller, SLAM, or Nav2.  The output follows
EmbodiedAgent's existing scan contract exactly::

    <output>/color/000000.png
    <output>/robot_xy/000000.txt   # x y yaw_rad

RoboCasa still requires a registered robot while constructing a task.  We build
the existing H1_2 environment, park the robot below the scene, and render through
an independent MuJoCo free camera.  No robot state enters the exported memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import h1_2_robosuite


SCHEMA_VERSION = 2
GENERATED_OUTPUT_NAMES = {
    "color",
    "frame_meta",
    "robot_xy",
    "memory",
    "query_results",
    "scan_manifest.json",
    "poses.csv",
    "contact_sheet.jpg",
    "smoke_results.json",
}


def capture_timestamp() -> str:
    """Return a timezone-aware ISO8601 timestamp for one sensor capture."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def next_frame_index(color_dir: Path) -> int:
    """Continue after every committed or reserved frame id.

    Pose / metadata may survive a process crash before the PNG commit marker. Skipping
    those orphan ids makes the dataset recoverable without overwriting forensic data.
    """
    output = color_dir.parent
    indices = []
    paths = list(color_dir.glob("*.png"))
    paths += list((output / "robot_xy").glob("*.txt"))
    paths += list((output / "frame_meta").glob("*.json"))
    paths += list((output / "frame_meta").glob(".*.reserve"))
    for path in paths:
        stem = path.stem
        if path.name.startswith(".") and path.name.endswith(".reserve"):
            stem = path.name[1:-len(".reserve")]
        try:
            indices.append(int(stem))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def write_frame_atomic(
    output: Path,
    frame_idx: int,
    rgb: np.ndarray,
    pose: tuple[float, float, float],
    captured_at: str,
    source_type: str,
    episode_id: str | None,
    camera_metadata: dict,
) -> tuple[Path, dict]:
    """Persist pose + metadata first, then publish the PNG as the commit marker."""
    stem = f"{frame_idx:06d}"
    memory_id = f"mem_{stem}"
    image_path = output / "color" / f"{stem}.png"
    pose_path = output / "robot_xy" / f"{stem}.txt"
    metadata_path = output / "frame_meta" / f"{stem}.json"
    for directory in (image_path.parent, pose_path.parent, metadata_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    # O_EXCL is the cross-process reservation. Two direct Python callers that choose
    # the same next id cannot both pass this point, even without the wrapper's flock.
    reservation_path = metadata_path.parent / f".{stem}.reserve"
    try:
        reservation_fd = os.open(
            reservation_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError as exc:
        raise FileExistsError(f"frame id is already reserved: {stem}") from exc
    os.close(reservation_fd)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "frame_idx": frame_idx,
        "memory_id": memory_id,
        "image_path": f"color/{stem}.png",
        "pose_path": f"robot_xy/{stem}.txt",
        "robot_pose": list(pose),
        "timestamp": captured_at,
        "source_type": source_type,
        "episode_id": episode_id,
        "camera": camera_metadata,
    }

    suffix = f".{os.getpid()}.tmp"
    pose_tmp = pose_path.with_name(f".{pose_path.name}{suffix}")
    metadata_tmp = metadata_path.with_name(f".{metadata_path.name}{suffix}")
    image_tmp = image_path.with_name(f".{image_path.name}{suffix}")
    try:
        existing = [
            path for path in (image_path, pose_path, metadata_path) if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "refusing to replace an existing frame: "
                + ", ".join(str(path) for path in existing)
            )
        np.savetxt(pose_tmp, np.asarray([pose], dtype=float), fmt="%.9f")
        os.replace(pose_tmp, pose_path)
        with open(metadata_tmp, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
            file.write("\n")
        os.replace(metadata_tmp, metadata_path)
        Image.fromarray(rgb, mode="RGB").save(image_tmp, format="PNG")
        # Consumers only glob final *.png files, so publishing this last guarantees
        # that a visible frame already has both pose and capture metadata.
        os.replace(image_tmp, image_path)
    finally:
        for temporary in (pose_tmp, metadata_tmp, image_tmp):
            temporary.unlink(missing_ok=True)
        reservation_path.unlink(missing_ok=True)
    return image_path, metadata


def _parse_headings(value: str) -> list[float]:
    try:
        headings = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("headings must be comma-separated degrees") from exc
    if not headings:
        raise argparse.ArgumentTypeError("at least one heading is required")
    return headings


def _fixture_points(fixture) -> np.ndarray | None:
    try:
        points = fixture.get_ext_sites(all_points=True, relative=False)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return points if len(points) else None
    except Exception:
        return None


def _floor_bounds(env) -> tuple[tuple[float, float, float, float], float]:
    """Return the largest floor's (xmin, xmax, ymin, ymax) and top z."""
    floors = []
    for fixture in env.fixtures.values():
        if type(fixture).__name__ != "Floor":
            continue
        points = _fixture_points(fixture)
        if points is None:
            continue
        xmin, ymin = points[:, :2].min(axis=0)
        xmax, ymax = points[:, :2].max(axis=0)
        floors.append(((xmin, xmax, ymin, ymax), float(points[:, 2].max())))
    if not floors:
        raise RuntimeError("RoboCasa scene has no readable Floor fixture")
    return max(floors, key=lambda item: (item[0][1] - item[0][0]) * (item[0][3] - item[0][2]))


def _coverage_subset(points: list[tuple[float, float]], limit: int,
                     center: tuple[float, float]) -> list[tuple[float, float]]:
    """Deterministic farthest-point sample for even room coverage."""
    points = sorted(points)
    if limit <= 0 or len(points) <= limit:
        return points

    cx, cy = center
    first = min(points, key=lambda p: ((p[0] - cx) ** 2 + (p[1] - cy) ** 2, p))
    selected = [first]
    remaining = set(points)
    remaining.remove(first)
    while remaining and len(selected) < limit:
        def score(point):
            min_dist = min((point[0] - q[0]) ** 2 + (point[1] - q[1]) ** 2
                           for q in selected)
            return min_dist, -point[0], -point[1]

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _hide_robot_geometries(model) -> int:
    """Move all H1/Magpie geoms into hidden group 5.

    Parking the body below the floor keeps it physically out of the kitchen;
    changing the geom group additionally guarantees that it cannot enter RGB
    frames or the ray test used to choose camera positions.
    """
    robot_geoms = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith(("robot0_", "gripper0_")):
            robot_geoms.append(geom_id)
    if robot_geoms:
        model.geom_group[np.asarray(robot_geoms, dtype=int)] = 5
    return len(robot_geoms)


def _ray_hits_floor(model, data, floor_names: tuple[str, ...],
                    x: float, y: float, ray_z: float) -> bool:
    hit_id = np.array([-1], dtype=np.int32)
    collision_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8).reshape(6, 1)
    distance = mujoco.mj_ray(
        model,
        data,
        np.array([x, y, ray_z], dtype=np.float64),
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
        collision_group,
        1,
        -1,
        hit_id,
    )
    if distance < 0 or hit_id[0] < 0:
        return False
    body_id = int(model.geom_bodyid[int(hit_id[0])])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    return body_name.startswith(floor_names)


def _footprint_is_clear(model, data, floor_names: tuple[str, ...],
                        x: float, y: float, ray_z: float, radius: float) -> bool:
    offsets = [(0.0, 0.0)]
    if radius > 0:
        for ring_radius in (radius / 2.0, radius):
            offsets.extend(
                (ring_radius * math.cos(angle), ring_radius * math.sin(angle))
                for angle in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
            )
    return all(
        _ray_hits_floor(model, data, floor_names, x + dx, y + dy, ray_z)
        for dx, dy in offsets
    )


def _make_waypoints(env, model, data, camera_z: float, spacing: float,
                    room_margin: float, footprint_radius: float, max_positions: int):
    bounds, floor_z = _floor_bounds(env)
    xmin, xmax, ymin, ymax = bounds
    if xmax - xmin <= 2 * room_margin or ymax - ymin <= 2 * room_margin:
        raise RuntimeError(f"room margin {room_margin} leaves no scan area in {bounds}")

    floor_names = tuple(
        fixture.name
        for fixture in env.fixtures.values()
        if type(fixture).__name__ == "Floor"
    )
    if not floor_names:
        raise RuntimeError("RoboCasa scene has no named floor body")
    xs = np.arange(xmin + room_margin, xmax - room_margin + 1e-9, spacing)
    ys = np.arange(ymin + room_margin, ymax - room_margin + 1e-9, spacing)
    candidates = [
        (float(x), float(y))
        for y in ys
        for x in xs
        if _footprint_is_clear(
            model, data, floor_names, float(x), float(y), camera_z, footprint_radius
        )
    ]
    if not candidates:
        raise RuntimeError("fixture clearance rejected every camera position")
    center = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
    return _coverage_subset(candidates, max_positions, center), bounds, floor_z


def _park_robot_below_scene(env) -> None:
    env.sim.data.qvel[:] = 0.0
    if env.sim.data.ctrl.size:
        env.sim.data.ctrl[:] = 0.0
    h1_2_robosuite.place_robot_freejoint(
        env,
        np.array([0.0, 0.0, -10.0], dtype=float),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
    )
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)


def _set_free_camera(camera: mujoco.MjvCamera, x: float, y: float, z: float,
                     yaw_rad: float, pitch_deg: float, look_distance: float) -> None:
    """Place MuJoCo's orbit-style free camera at an exact sensor pose."""
    pitch_rad = math.radians(pitch_deg)
    horizontal = look_distance
    target = np.array([
        x + horizontal * math.cos(yaw_rad),
        y + horizontal * math.sin(yaw_rad),
        z + horizontal * math.tan(pitch_rad),
    ])
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = target
    camera.distance = horizontal / max(math.cos(pitch_rad), 1e-6)
    camera.azimuth = math.degrees(yaw_rad)
    camera.elevation = pitch_deg
    camera.orthographic = 0


def _scene_option() -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    # robosuite convention: group 0 is collision geometry, group 1 is visual.
    option.geomgroup[1] = 1
    option.geomgroup[2] = 1
    option.sitegroup[:] = 0
    return option


def _write_contact_sheet(paths: list[Path], destination: Path) -> None:
    if not paths:
        return
    thumb_w, thumb_h = 160, 160
    label_h = 20
    columns = min(8, len(paths))
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        col, row = index % columns, index // columns
        left = col * thumb_w + (thumb_w - image.width) // 2
        top = row * (thumb_h + label_h) + (thumb_h - image.height) // 2
        sheet.paste(image, (left, top))
        draw.text((col * thumb_w + 4, row * (thumb_h + label_h) + thumb_h + 2),
                  path.stem, fill="black")
    sheet.save(destination, quality=88)


def _prepare_output(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    if resolved == Path("/") or len(resolved.parts) < 3:
        raise RuntimeError(f"refusing unsafe output path: {resolved}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"output is not empty: {path} (pass --overwrite to replace it)")
        unknown = [
            child.name for child in path.iterdir()
            if child.name not in GENERATED_OUTPUT_NAMES
            and not child.name.startswith("retrieval_index_")
        ]
        if unknown:
            raise RuntimeError(
                f"refusing to overwrite directory with unknown files: {', '.join(unknown)}"
            )
        for child in list(path.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for directory in ("color", "robot_xy", "frame_meta"):
        (path / directory).mkdir(parents=True, exist_ok=True)


def scan(args) -> Path:
    output = Path(args.output)
    _prepare_output(output, args.overwrite)

    kwargs = {"layout_ids": args.layout, "style_ids": args.style, "seed": args.seed}
    env = h1_2_robosuite.make_kitchen_env(args.task, **kwargs)
    env.reset()
    _park_robot_below_scene(env)

    model = env.sim.model._model
    data = env.sim.data._data
    hidden_robot_geoms = _hide_robot_geometries(model)
    model.vis.global_.fovy = args.fovy
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.image_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.image_height)
    _, floor_z = _floor_bounds(env)
    camera_z = floor_z + args.camera_height
    waypoints, room_bounds, floor_z = _make_waypoints(
        env,
        model,
        data,
        camera_z=camera_z,
        spacing=args.spacing,
        room_margin=args.room_margin,
        footprint_radius=args.footprint_radius,
        max_positions=args.max_positions,
    )

    camera_z = floor_z + args.camera_height
    camera = mujoco.MjvCamera()
    option = _scene_option()
    rendered_paths: list[Path] = []
    frames = []

    print(f"[memory-scan] hidden_robot_geoms={hidden_robot_geoms}")
    print(f"[memory-scan] room={tuple(round(v, 3) for v in room_bounds)} "
          f"positions={len(waypoints)} headings={len(args.headings)}")
    renderer = mujoco.Renderer(model, height=args.image_height, width=args.image_width)
    try:
        frame_idx = 0
        for x, y in waypoints:
            for heading_deg in args.headings:
                raw_yaw = math.radians(heading_deg)
                # Match Habitat's atan2 convention so sidecars stay in [-pi, pi].
                yaw = math.atan2(math.sin(raw_yaw), math.cos(raw_yaw))
                _set_free_camera(camera, x, y, camera_z, yaw,
                                 args.pitch, args.look_distance)
                renderer.disable_depth_rendering()
                renderer.update_scene(data, camera=camera, scene_option=option)
                rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()

                captured_at = capture_timestamp()
                camera_metadata = {
                    "z": camera_z,
                    "pitch_deg": args.pitch,
                    "fovy_deg": args.fovy,
                    "width": args.image_width,
                    "height": args.image_height,
                }
                image_path, frame_metadata = write_frame_atomic(
                    output=output,
                    frame_idx=frame_idx,
                    rgb=rgb,
                    pose=(x, y, yaw),
                    captured_at=captured_at,
                    source_type="scan_wasd",
                    episode_id=None,
                    camera_metadata=camera_metadata,
                )
                rendered_paths.append(image_path)
                frames.append({
                    **frame_metadata,
                    "frame_meta_path": f"frame_meta/{frame_idx:06d}.json",
                    "camera_z": camera_z,
                    "camera_pitch_deg": args.pitch,
                })
                print(f"[memory-scan] {frame_idx:06d} x={x:.2f} y={y:.2f} "
                      f"yaw={heading_deg:.0f}deg")
                frame_idx += 1
    finally:
        renderer.close()
        env.close()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": (
            "EmbodiedAgent color/<frame>.png + robot_xy/<frame>.txt + "
            "frame_meta/<frame>.json"
        ),
        "scene": {
            "simulator": "RoboCasa",
            "task": args.task,
            "layout": args.layout,
            "style": args.style,
            "seed": args.seed,
        },
        "camera": {
            "motion": "fixed-height planar kinematic camera",
            "width": args.image_width,
            "height": args.image_height,
            "fovy_deg": args.fovy,
            "height_m": args.camera_height,
            "pitch_deg": args.pitch,
        },
        "room_bounds_xy": list(room_bounds),
        "frame_count": len(frames),
        "frames": frames,
    }
    with open(output / "scan_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    with open(output / "poses.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("frame", "memory_id", "x", "y", "yaw_rad"))
        for frame in frames:
            writer.writerow((frame["frame_idx"], frame["memory_id"], *frame["robot_pose"]))
    _write_contact_sheet(rendered_paths, output / "contact_sheet.jpg")
    print(f"[memory-scan] wrote {len(frames)} frames to {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a robot-free RoboCasa scan for EmbodiedAgent memory",
    )
    parser.add_argument("--task", default="Kitchen")
    parser.add_argument("--layout", type=int, default=9)
    parser.add_argument("--style", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/data/layout09_style09_seed42")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--room-margin", type=float, default=0.35)
    parser.add_argument("--footprint-radius", type=float, default=0.30,
                        help="floor-clearance radius for the virtual camera rover")
    parser.add_argument("--max-positions", type=int, default=16,
                        help="0 keeps every collision-filtered grid point")
    parser.add_argument("--headings", type=_parse_headings,
                        default=_parse_headings("0,90,180,270"),
                        help="comma-separated yaw angles in degrees")
    parser.add_argument("--camera-height", type=float, default=1.25)
    parser.add_argument("--pitch", type=float, default=-12.0)
    parser.add_argument("--look-distance", type=float, default=2.0)
    parser.add_argument("--fovy", type=float, default=60.0)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.spacing <= 0 or args.camera_height <= 0 or args.look_distance <= 0:
        raise SystemExit("spacing, camera height, and look distance must be positive")
    if args.footprint_radius < 0:
        raise SystemExit("footprint radius must be non-negative")
    if not -89.0 < args.pitch < 89.0:
        raise SystemExit("camera pitch must be between -89 and 89 degrees")
    if args.max_positions < 0:
        raise SystemExit("max positions must be non-negative")
    scan(args)


if __name__ == "__main__":
    main()
