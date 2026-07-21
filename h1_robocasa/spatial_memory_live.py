#!/usr/bin/env python3
"""Append a real-time RoboCasa before/after sequence to a camera scan.

The virtual camera stays at one collision-free pose aimed at the refrigerator.
Frames are captured on monotonic wall-clock deadlines, while each observation is
stamped with a timezone-aware ISO8601 capture time.  Halfway through the session
the refrigerator changes from fully closed to fully open.  The expected state is
written only to the test-session ledger; EmbodiedAgent memory remains sensor-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import h1_2_robosuite
from spatial_memory_scan import (
    SCHEMA_VERSION,
    _floor_bounds,
    _hide_robot_geometries,
    _make_waypoints,
    _park_robot_below_scene,
    _scene_option,
    _set_free_camera,
    _write_contact_sheet,
    capture_timestamp,
    next_frame_index,
    write_frame_atomic,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _session_id(value: str | None) -> str:
    if value is None:
        value = "live_" + time.strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError("session id must use 1-80 letters, digits, dots, dashes, or underscores")
    return value


def _load_baseline(output: Path, args) -> dict:
    manifest_path = output / "scan_manifest.json"
    if not manifest_path.is_file() or not (output / "color").is_dir():
        raise RuntimeError(f"baseline scan not found at {output}; run the all command first")
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    actual = manifest.get("scene", {})
    expected = {
        "simulator": "RoboCasa",
        "task": args.task,
        "layout": args.layout,
        "style": args.style,
        "seed": args.seed,
    }
    if actual != expected:
        raise RuntimeError(
            "live scene does not match the baseline manifest: "
            f"expected {expected}, found {actual}"
        )
    if not any((output / "color").glob("*.png")):
        raise RuntimeError("baseline scan contains no color frames")
    return manifest


def _get_fridge(env):
    try:
        from robocasa.models.fixtures import FixtureType

        fridge = env.get_fixture(FixtureType.FRIDGE)
        joints = getattr(fridge, "_fridge_door_joint_names", None)
        if fridge is None or not joints:
            raise RuntimeError("no refrigerator door joints were found")
        return fridge
    except Exception as exc:
        raise RuntimeError("this RoboCasa layout has no usable refrigerator fixture") from exc


def _camera_pose_for_fridge(env, model, data, args, fridge):
    _, floor_z = _floor_bounds(env)
    camera_z = floor_z + args.camera_height
    waypoints, _, _ = _make_waypoints(
        env,
        model,
        data,
        camera_z=camera_z,
        spacing=args.spacing,
        room_margin=args.room_margin,
        footprint_radius=args.footprint_radius,
        max_positions=0,
    )
    target = np.asarray(fridge.pos[:2], dtype=float)
    eligible = [
        point for point in waypoints
        if np.linalg.norm(np.asarray(point) - target) >= args.min_target_distance
    ]
    if not eligible:
        raise RuntimeError("no collision-free camera position can view the refrigerator")
    x, y = min(eligible, key=lambda point: np.linalg.norm(np.asarray(point) - target))
    yaw = math.atan2(float(target[1] - y), float(target[0] - x))
    return float(x), float(y), float(yaw), float(camera_z)


def _rgb_change(first: np.ndarray, second: np.ndarray) -> dict:
    first = np.asarray(first, dtype=np.int16)
    second = np.asarray(second, dtype=np.int16)
    difference = np.abs(first - second)
    return {
        "mean_absolute_rgb_difference": float(difference.mean()),
        "fraction_pixels_changed_over_20": float(
            np.mean(np.max(difference, axis=2) > 20)
        ),
    }


def _image_change(before: Path, after: Path) -> dict:
    first = np.asarray(Image.open(before).convert("RGB"), dtype=np.int16)
    second = np.asarray(Image.open(after).convert("RGB"), dtype=np.int16)
    return {
        "before_frame": before.stem,
        "after_frame": after.stem,
        **_rgb_change(first, second),
    }


def capture_live(args) -> dict:
    output = Path(args.output).resolve()
    _load_baseline(output, args)
    session_id = _session_id(args.session_id)
    session_dir = output / "live_sessions" / session_id
    if session_dir.exists():
        raise RuntimeError(f"live session already exists: {session_dir}")

    # Include both t=0 and the final deadline at or before the requested duration.
    capture_count = math.floor(args.duration / args.interval) + 1
    if capture_count < 4:
        raise RuntimeError("duration must allow at least four live frames")
    change_index = capture_count // 2
    first_frame_idx = next_frame_index(output / "color")

    env = None
    renderer = None
    try:
        env = h1_2_robosuite.make_kitchen_env(
            args.task,
            layout_ids=args.layout,
            style_ids=args.style,
            seed=args.seed,
        )
        env.reset()
        _park_robot_below_scene(env)
        model = env.sim.model._model
        data = env.sim.data._data
        hidden_robot_geoms = _hide_robot_geometries(model)
        model.vis.global_.fovy = args.fovy
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.image_size)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.image_size)

        fridge = _get_fridge(env)
        fridge.close_door(env, min=0.0, max=0.0, compartment="fridge")
        env.sim.forward()
        x, y, yaw, camera_z = _camera_pose_for_fridge(env, model, data, args, fridge)
        camera = mujoco.MjvCamera()
        _set_free_camera(camera, x, y, camera_z, yaw, args.pitch, args.look_distance)
        option = _scene_option()
        renderer = mujoco.Renderer(model, height=args.image_size, width=args.image_size)

        # Verify the fixture API and chosen view before appending any durable frame.
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=camera, scene_option=option)
        preview_closed = np.asarray(renderer.render(), dtype=np.uint8).copy()
        fridge.open_door(env, min=1.0, max=1.0, compartment="fridge")
        env.sim.forward()
        renderer.update_scene(data, camera=camera, scene_option=option)
        preview_open = np.asarray(renderer.render(), dtype=np.uint8).copy()
        preview_change = _rgb_change(preview_closed, preview_open)
        if (
            preview_change["mean_absolute_rgb_difference"] < 1.0
            or preview_change["fraction_pixels_changed_over_20"] < 0.01
        ):
            raise RuntimeError(
                f"no clear refrigerator view was found from the safe camera pose: "
                f"{preview_change}"
            )
        fridge.close_door(env, min=0.0, max=0.0, compartment="fridge")
        env.sim.forward()
        # Creating this directory is the point at which live history becomes protected
        # from the baseline overwrite command. All simulator preflight has passed first.
        session_dir.mkdir(parents=True)
    except Exception:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()
        raise

    captured_paths: list[Path] = []
    frame_records: list[dict] = []
    change_event = None
    started_at = capture_timestamp()
    start_monotonic = time.monotonic()
    print(
        f"[memory-live] session={session_id} start_frame={first_frame_idx:06d} "
        f"frames={capture_count} interval={args.interval:.2f}s"
    )
    print(
        f"[memory-live] hidden_robot_geoms={hidden_robot_geoms} "
        f"camera=({x:.2f}, {y:.2f}, yaw={yaw:.3f}) fridge={fridge.name}"
    )
    try:
        for live_idx in range(capture_count):
            deadline = start_monotonic + live_idx * args.interval
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

            phase = "before_change"
            if live_idx >= change_index:
                phase = "after_change"
                if change_event is None:
                    fridge.open_door(env, min=1.0, max=1.0, compartment="fridge")
                    env.sim.forward()
                    change_event = {
                        "event": "fridge_door_opened",
                        "timestamp": capture_timestamp(),
                        "first_after_live_index": live_idx,
                    }
                    print("[memory-live] deterministic scene change: refrigerator opened")

            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=camera, scene_option=option)
            rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
            captured_at = capture_timestamp()
            frame_idx = first_frame_idx + live_idx
            image_path, metadata = write_frame_atomic(
                output=output,
                frame_idx=frame_idx,
                rgb=rgb,
                pose=(x, y, yaw),
                captured_at=captured_at,
                source_type="live_task",
                episode_id=session_id,
                camera_metadata={
                    "z": camera_z,
                    "pitch_deg": args.pitch,
                    "fovy_deg": args.fovy,
                    "width": args.image_size,
                    "height": args.image_size,
                },
            )
            captured_paths.append(image_path)
            # Phase is test-oracle information and deliberately stays outside
            # frame_meta / EpisodicMemory.
            frame_records.append({
                "frame_idx": frame_idx,
                "memory_id": metadata["memory_id"],
                "timestamp": captured_at,
                "phase": phase,
            })
            print(
                f"[memory-live] {frame_idx:06d} {captured_at} "
                f"elapsed={time.monotonic() - start_monotonic:.2f}s"
            )
    finally:
        renderer.close()
        env.close()

    before_paths = captured_paths[:change_index]
    after_paths = captured_paths[change_index:]
    visual_change = _image_change(before_paths[-1], after_paths[0])
    if (
        visual_change["mean_absolute_rgb_difference"] < 1.0
        or visual_change["fraction_pixels_changed_over_20"] < 0.01
    ):
        raise RuntimeError(f"refrigerator change was not visually clear: {visual_change}")

    _write_contact_sheet(captured_paths, session_dir / "contact_sheet.jpg")
    result = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "status": "complete",
        "scene": {
            "simulator": "RoboCasa",
            "task": args.task,
            "layout": args.layout,
            "style": args.style,
            "seed": args.seed,
        },
        "started_at": started_at,
        "completed_at": capture_timestamp(),
        "requested_interval_seconds": args.interval,
        "requested_duration_seconds": args.duration,
        "frame_count": len(frame_records),
        "first_frame_idx": frame_records[0]["frame_idx"],
        "last_frame_idx": frame_records[-1]["frame_idx"],
        "camera_pose": [x, y, yaw],
        "camera_z": camera_z,
        "change_event": change_event,
        "visual_change": visual_change,
        "frames": frame_records,
        "contact_sheet": f"live_sessions/{session_id}/contact_sheet.jpg",
    }
    _atomic_json(session_dir / "session.json", result)
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append a timed closed-to-open refrigerator sequence to a RoboCasa scan",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--task", default="Kitchen")
    parser.add_argument("--layout", type=int, default=9)
    parser.add_argument("--style", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--camera-height", type=float, default=1.25)
    parser.add_argument("--pitch", type=float, default=-12.0)
    parser.add_argument("--look-distance", type=float, default=2.0)
    parser.add_argument("--fovy", type=float, default=60.0)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--room-margin", type=float, default=0.35)
    parser.add_argument("--footprint-radius", type=float, default=0.30)
    parser.add_argument("--min-target-distance", type=float, default=0.75)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.interval <= 0 or args.duration <= 0:
        raise SystemExit("interval and duration must be positive")
    if args.image_size <= 0 or args.camera_height <= 0 or args.look_distance <= 0:
        raise SystemExit("image size, camera height, and look distance must be positive")
    if args.spacing <= 0 or args.footprint_radius < 0 or args.min_target_distance < 0:
        raise SystemExit("invalid camera-grid geometry")
    if not -89.0 < args.pitch < 89.0:
        raise SystemExit("camera pitch must be between -89 and 89 degrees")
    capture_live(args)


if __name__ == "__main__":
    main()
