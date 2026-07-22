"""H12 spatial-memory skill and EmbodiedAgent backend.

HAMS owns sensor capture and the ROS contract. EmbodiedAgent remains the source
of the actual SigLIP encoder, live FAISS index, metadata store, and optional
Gemini reranker. Keeping that boundary explicit avoids forking the mentor's
algorithm into this repository. The ROS node runs as a separate executable so
its optional GPU/model dependencies cannot prevent the main skills node from
starting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
from tf2_ros import Buffer, TransformException, TransformListener

from custom_ros_messages.action import SkillRetrieveMemory


@dataclass(frozen=True)
class MemoryHit:
    """One agent-facing memory result."""

    memory_id: str
    image_path: str
    robot_pose: tuple[float, ...]
    timestamp: str
    score: float
    faiss_score: float
    faiss_rank: int
    rerank_reasoning: str = ''


@dataclass(frozen=True)
class QueryOutcome:
    """Ranked results plus explicit reranker diagnostics."""

    hits: tuple[MemoryHit, ...]
    rerank_attempted: bool = False
    rerank_valid: bool = False
    fallback_reason: str = ''


@dataclass(frozen=True)
class _EmbodiedAgentRuntime:
    """Late-loaded EmbodiedAgent symbols, injectable for unit tests."""

    memory_type: type
    worker_type: type
    retrieve: Callable[..., list]
    gemini_type: type


def _load_runtime(root: Path) -> _EmbodiedAgentRuntime:
    if not (root / 'memory' / 'embedding.py').is_file():
        raise RuntimeError(
            f'EmbodiedAgent checkout not found at {root}; set '
            'EMBODIED_AGENT_ROOT or the embodied_agent_root ROS parameter'
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        episodic = importlib.import_module('agent.episodic_memory')
        embedding = importlib.import_module('memory.embedding')
        retrieval = importlib.import_module('memory.retrieval')
        gemini = importlib.import_module('agent.gemini_client')
    except Exception as exc:
        raise RuntimeError(f'cannot import EmbodiedAgent memory backend: {exc}') from exc
    return _EmbodiedAgentRuntime(
        memory_type=episodic.EpisodicMemory,
        worker_type=embedding.EmbeddingWorker,
        retrieve=retrieval.retrieve_memory_candidates,
        gemini_type=gemini.GeminiClient,
    )


def _validate_rerank(response: Any, candidate_ids: list[str]):
    """Accept only a complete, unambiguous ranking of the supplied IDs."""
    if not isinstance(response, dict):
        return None
    ranked = response.get('ranked_ids')
    analysis = response.get('candidates_analysis')
    expected = set(candidate_ids)
    if (
        not isinstance(ranked, list)
        or len(ranked) != len(candidate_ids)
        or len(set(ranked)) != len(ranked)
        or set(ranked) != expected
        or not isinstance(analysis, list)
        or len(analysis) != len(candidate_ids)
    ):
        return None
    by_id = {}
    for item in analysis:
        if not isinstance(item, dict):
            return None
        memory_id = item.get('memory_id')
        confidence = item.get('confidence')
        if (
            memory_id not in expected
            or memory_id in by_id
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or not isinstance(item.get('object_location'), str)
            or not isinstance(item.get('reasoning'), str)
        ):
            return None
        by_id[memory_id] = dict(item, confidence=float(confidence))
    return ranked, by_id


def _seconds_of_day(value: str) -> Optional[float]:
    """Parse HH:MM[:SS] or take the clock portion of an ISO8601 value."""
    text = (value or '').strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
        return (parsed.hour * 3600.0 + parsed.minute * 60.0
                + parsed.second + parsed.microsecond * 1e-6)
    except ValueError:
        pass
    pieces = text.split(':')
    if len(pieces) not in (2, 3):
        return None
    try:
        hours, minutes = int(pieces[0]), int(pieces[1])
        seconds = float(pieces[2]) if len(pieces) == 3 else 0.0
    except ValueError:
        return None
    if not (0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60):
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds


def _in_time_window(timestamp: str, start: Optional[str], end: Optional[str]) -> bool:
    if not start and not end:
        return True
    value = _seconds_of_day(timestamp)
    lower = _seconds_of_day(start or '')
    upper = _seconds_of_day(end or '')
    if value is None or (start and lower is None) or (end and upper is None):
        return False
    if lower is not None and upper is not None and lower > upper:
        return value >= lower or value <= upper  # window crosses midnight
    return ((lower is None or value >= lower)
            and (upper is None or value <= upper))


class EmbodiedAgentMemory:
    """Own persistent capture, live indexing, retrieval, and optional reranking."""

    def __init__(
        self,
        data_dir: str,
        embodied_agent_root: str,
        model: str = 'siglip_base',
        device: str = 'auto',
        recall_k: int = 12,
        vlm_model: str = 'gemini-3.5-flash',
        runtime: Optional[_EmbodiedAgentRuntime] = None,
    ):
        if int(recall_k) <= 0:
            raise ValueError('recall_k must be positive')
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.capture_dir = self.data_dir / 'capture'
        self.memory_dir = self.data_dir / 'memory'
        self.index_dir = self.data_dir / 'index'
        self.vlm_log_dir = self.data_dir / 'vlm_logs'
        for directory in (
            self.capture_dir / 'color',
            self.capture_dir / 'robot_xy',
            self.capture_dir / 'frame_meta',
            self.memory_dir,
            self.index_dir,
            self.vlm_log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.recall_k = int(recall_k)
        self.vlm_model = vlm_model
        self._runtime = runtime or _load_runtime(
            Path(embodied_agent_root).expanduser().resolve())
        self._memory = self._runtime.memory_type(str(self.memory_dir))
        self._worker = self._runtime.worker_type(
            index_dir=str(self.index_dir), model_name=model, device=device)
        self._gemini = None
        self._capture_lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._closed = False
        self._next_frame = self._discover_next_frame()
        self._recover_capture_files()

    @property
    def ready(self) -> bool:
        """Whether the embedding model and FAISS index are queryable."""
        return bool(self._worker.is_ready)

    def wait_until_ready(self, timeout_sec: float = 180.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while not self.ready and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.ready

    def _discover_next_frame(self) -> int:
        indices = []
        for path in (self.capture_dir / 'color').glob('*.png'):
            try:
                indices.append(int(path.stem))
            except ValueError:
                continue
        return max(indices, default=-1) + 1

    def _recover_capture_files(self) -> None:
        """Make sidecar-complete frames survive a process restart."""
        self._memory.build_from_scan_dir(
            str(self.capture_dir), embedding_model=self.model)
        paths_file = self.index_dir / 'frame_paths.json'
        try:
            indexed = set(json.loads(paths_file.read_text(encoding='utf-8')))
        except (OSError, ValueError, TypeError):
            indexed = set()
        for image_path in sorted((self.capture_dir / 'color').glob('*.png')):
            resolved = str(image_path.resolve())
            if resolved in indexed:
                continue
            pose_path = self.capture_dir / 'robot_xy' / f'{image_path.stem}.txt'
            if not pose_path.is_file():
                continue
            try:
                pose = [float(value) for value in pose_path.read_text().split()]
                from PIL import Image
                import numpy as np
                rgb = np.asarray(Image.open(image_path).convert('RGB'))
            except (OSError, ValueError):
                continue
            yaw = pose[2] if len(pose) >= 3 else 0.0
            self._worker.enqueue(
                rgb, resolved, np.asarray(pose[:2], dtype=np.float64), yaw)

    def add_frame(
        self,
        rgb,
        robot_pose: tuple[float, float, float],
        timestamp: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Persist one RGB observation and enqueue it for live indexing."""
        if self._closed:
            raise RuntimeError('memory backend is closed')
        try:
            pose = tuple(float(value) for value in robot_pose)
        except (TypeError, ValueError) as exc:
            raise ValueError('robot_pose must contain numeric x, y, yaw') from exc
        if len(pose) != 3:
            raise ValueError('robot_pose must contain x, y, yaw')
        captured_at = timestamp or datetime.now(timezone.utc).isoformat()
        try:
            parsed_timestamp = datetime.fromisoformat(
                captured_at.replace('Z', '+00:00'))
        except (AttributeError, ValueError) as exc:
            raise ValueError('timestamp must be ISO8601') from exc
        if (
            parsed_timestamp.tzinfo is None
            or parsed_timestamp.utcoffset() is None
        ):
            raise ValueError('timestamp must include a timezone')

        with self._capture_lock:
            frame_idx = self._next_frame
            self._next_frame += 1
            stem = f'{frame_idx:06d}'
            memory_id = f'mem_{stem}'
            image_path = self.capture_dir / 'color' / f'{stem}.png'
            pose_path = self.capture_dir / 'robot_xy' / f'{stem}.txt'
            meta_path = self.capture_dir / 'frame_meta' / f'{stem}.json'

            from PIL import Image
            import numpy as np
            image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
            image = image.convert('RGB')
            temp_image = image_path.with_suffix('.tmp.png')
            frame_meta = {
                **(metadata or {}),
                'memory_id': memory_id,
                'timestamp': captured_at,
                'source_type': 'agent_observe',
            }
            try:
                image.save(temp_image)
                pose_path.write_text(
                    f'{pose[0]:.9f} {pose[1]:.9f} {pose[2]:.9f}\n',
                    encoding='utf-8')
                meta_path.write_text(
                    json.dumps(frame_meta, sort_keys=True) + '\n',
                    encoding='utf-8')
                # EmbodiedAgent treats the final PNG as the commit marker. Publish
                # it only after both sidecars are durable enough for restart import.
                temp_image.replace(image_path)
            except Exception:
                temp_image.unlink(missing_ok=True)
                raise

            entry = self._memory.create_entry(
                memory_id=memory_id,
                image_path=str(image_path.resolve()),
                robot_pose=list(pose),
                timestamp=captured_at,
                embedding_model=self.model,
                source_type='agent_observe',
                episode_id=frame_meta.get('episode_id'),
            )
            self._memory.add_entry(entry)
            self._worker.enqueue(
                np.asarray(image), str(image_path.resolve()),
                np.asarray(pose[:2], dtype=np.float64), pose[2])
        return memory_id

    def flush(self) -> None:
        """Wait until every captured frame has reached the FAISS index."""
        self._worker.flush()

    def _gemini_client(self):
        if self._gemini is not None:
            return self._gemini
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            return None
        self._gemini = self._runtime.gemini_type(
            api_key=api_key,
            model_name=self.vlm_model,
            log_dir=str(self.vlm_log_dir),
            max_retries=3,
        )
        return self._gemini

    def query(
        self,
        text: str,
        top_k: int = 3,
        rerank: bool = False,
        time_from: str = '',
        time_to: str = '',
    ) -> QueryOutcome:
        """Search live memory, optionally reranking a broader FAISS pool."""
        if self._closed:
            raise RuntimeError('memory backend is closed')
        query_text = str(text).strip()
        if not query_text:
            raise ValueError('query must not be empty')
        top_k = max(1, int(top_k))
        if time_from and _seconds_of_day(time_from) is None:
            raise ValueError(f'invalid time_from: {time_from!r}')
        if time_to and _seconds_of_day(time_to) is None:
            raise ValueError(f'invalid time_to: {time_to!r}')
        if not self.ready:
            raise RuntimeError('embedding model is still loading')

        with self._query_lock:
            fetch_k = -1 if (time_from or time_to) else (
                max(self.recall_k, top_k) if rerank else top_k)
            candidates = self._runtime.retrieve(
                query=query_text,
                index_dir=str(self.index_dir),
                capture_out_dir=str(self.capture_dir),
                top_k=fetch_k,
                model=self.model,
                episodic_memory=self._memory,
                embedding_worker=self._worker,
            )
            candidates = [
                candidate for candidate in candidates
                if _in_time_window(
                    candidate.timestamp or '', time_from or None, time_to or None)
            ]
            if not candidates:
                return QueryOutcome(hits=())
            if time_from or time_to:
                candidates = candidates[:max(self.recall_k, top_k) if rerank else top_k]

            faiss_rank = {
                candidate.memory_id: rank
                for rank, candidate in enumerate(candidates, start=1)
            }
            faiss_score = {
                candidate.memory_id: float(candidate.retrieval_score)
                for candidate in candidates
            }
            analysis_by_id = {}
            attempted = False
            valid = False
            fallback = ''
            if rerank:
                try:
                    client = self._gemini_client()
                except Exception as exc:
                    client = None
                    fallback = f'vlm_setup_error:{type(exc).__name__}'
                if client is None:
                    if not fallback:
                        fallback = 'missing_api_key'
                else:
                    attempted = True
                    try:
                        raw = client.rerank_memory_candidates(
                            query=query_text,
                            candidates=[
                                candidate.to_dict() for candidate in candidates
                            ],
                            image_paths=[
                                candidate.image_path for candidate in candidates
                            ],
                        )
                        validated = _validate_rerank(
                            raw,
                            [candidate.memory_id for candidate in candidates],
                        )
                        if validated is None:
                            fallback = 'invalid_vlm_response'
                        else:
                            ranked_ids, analysis_by_id = validated
                            by_id = {
                                candidate.memory_id: candidate
                                for candidate in candidates
                            }
                            candidates = [
                                by_id[memory_id] for memory_id in ranked_ids
                            ]
                            valid = True
                    except Exception as exc:
                        fallback = f'vlm_error:{type(exc).__name__}'

            hits = []
            for candidate in candidates[:top_k]:
                analysis = analysis_by_id.get(candidate.memory_id, {})
                hits.append(MemoryHit(
                    memory_id=candidate.memory_id,
                    image_path=candidate.image_path,
                    robot_pose=tuple(float(v) for v in candidate.robot_pose),
                    timestamp=candidate.timestamp or '',
                    score=float(analysis.get(
                        'confidence', faiss_score[candidate.memory_id])),
                    faiss_score=faiss_score[candidate.memory_id],
                    faiss_rank=faiss_rank[candidate.memory_id],
                    rerank_reasoning=analysis.get('reasoning', ''),
                ))
            return QueryOutcome(
                hits=tuple(hits),
                rerank_attempted=attempted,
                rerank_valid=valid,
                fallback_reason=fallback,
            )

    def close(self) -> None:
        """Flush and stop the EmbodiedAgent worker exactly once."""
        if self._closed:
            return
        self._closed = True
        self._worker.flush()
        self._worker.stop()
        self._gemini = None


DEFAULT_CAMERA_TOPIC = '/realsense/head/color/image_raw/compressed'


def _yaw_from_quaternion(quaternion) -> float:
    """Return planar yaw without adding a transforms dependency."""
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


class SpatialMemoryNode(Node):
    """Capture timestamped observations and expose text retrieval as a skill."""

    def __init__(self):
        super().__init__('h12_spatial_memory')
        self._cb_group = ReentrantCallbackGroup()

        default_root = os.environ.get('EMBODIED_AGENT_ROOT', '/opt/EmbodiedAgent')
        self.declare_parameter('embodied_agent_root', default_root)
        self.declare_parameter('data_dir', '/data/spatial_memory')
        self.declare_parameter('camera_topic', DEFAULT_CAMERA_TOPIC)
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('robot_frame', 'pelvis')
        self.declare_parameter('capture_interval_sec', 2.0)
        self.declare_parameter('capture_enabled', True)
        self.declare_parameter('model', 'siglip_base')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('recall_k', 12)
        self.declare_parameter('default_top_k', 3)
        self.declare_parameter('vlm_model', 'gemini-3.5-flash')

        self._world_frame = self.get_parameter('world_frame').value
        self._robot_frame = self.get_parameter('robot_frame').value
        self._capture_enabled = bool(self.get_parameter('capture_enabled').value)
        self._default_top_k = max(
            1, int(self.get_parameter('default_top_k').value))
        interval = float(self.get_parameter('capture_interval_sec').value)
        if interval <= 0.0:
            raise ValueError('capture_interval_sec must be positive')

        self._backend = EmbodiedAgentMemory(
            data_dir=self.get_parameter('data_dir').value,
            embodied_agent_root=self.get_parameter('embodied_agent_root').value,
            model=self.get_parameter('model').value,
            device=self.get_parameter('device').value,
            recall_k=int(self.get_parameter('recall_k').value),
            vlm_model=self.get_parameter('vlm_model').value,
        )

        self._image_lock = threading.Lock()
        self._latest_image = None
        self._latest_sequence = 0
        self._captured_sequence = 0
        self._capture_in_progress = threading.Lock()
        camera_topic = self.get_parameter('camera_topic').value
        self.create_subscription(
            CompressedImage,
            camera_topic,
            self._on_image,
            qos_profile_sensor_data,
            callback_group=self._cb_group,
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_timer(
            interval, self._capture_latest, callback_group=self._cb_group)
        self._action = ActionServer(
            self,
            SkillRetrieveMemory,
            '/skill/retrieve_memory',
            execute_callback=self._execute_query,
            cancel_callback=lambda _request: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )
        self.get_logger().info(
            f'spatial memory ready: camera={camera_topic}, '
            f'interval={interval:.1f}s, '
            f'frames={self._world_frame}->{self._robot_frame}, '
            f'data={Path(self.get_parameter("data_dir").value).expanduser()}')

    def _on_image(self, message: CompressedImage) -> None:
        with self._image_lock:
            self._latest_image = message
            self._latest_sequence += 1

    def _capture_latest(self) -> None:
        if not self._capture_enabled:
            return
        if not self._capture_in_progress.acquire(blocking=False):
            return
        try:
            with self._image_lock:
                message = self._latest_image
                sequence = self._latest_sequence
            if message is None or sequence == self._captured_sequence:
                return

            stamp = Time.from_msg(message.header.stamp)
            lookup_stamp = stamp if stamp.nanoseconds else Time()
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._world_frame,
                    self._robot_frame,
                    lookup_stamp,
                    timeout=Duration(seconds=0.2),
                )
            except TransformException as exc:
                self.get_logger().warn(
                    f'skipping memory frame: pose unavailable ({exc})',
                    throttle_duration_sec=5.0)
                return

            encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr is None:
                self.get_logger().warn(
                    'skipping memory frame: JPEG decode failed')
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            translation = transform.transform.translation
            pose = (
                float(translation.x),
                float(translation.y),
                _yaw_from_quaternion(transform.transform.rotation),
            )
            memory_id = self._backend.add_frame(
                rgb,
                pose,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={
                    'ros_stamp_ns': int(stamp.nanoseconds),
                    'camera_frame': message.header.frame_id,
                    'world_frame': self._world_frame,
                    'robot_frame': self._robot_frame,
                },
            )
            self._captured_sequence = sequence
            self.get_logger().info(
                f'captured {memory_id} at ({pose[0]:.2f}, {pose[1]:.2f}, '
                f'yaw={math.degrees(pose[2]):.0f} deg)')
        except Exception as exc:
            self.get_logger().error(f'memory capture failed: {exc}')
        finally:
            self._capture_in_progress.release()

    @staticmethod
    def _feedback(goal_handle, phase: str, progress: float) -> None:
        feedback = SkillRetrieveMemory.Feedback()
        feedback.phase = phase
        feedback.progress = float(progress)
        goal_handle.publish_feedback(feedback)

    def _execute_query(self, goal_handle):
        result = SkillRetrieveMemory.Result()
        request = goal_handle.request
        if not request.query.strip():
            result.success = False
            result.message = 'query must not be empty'
            goal_handle.abort()
            return result
        if goal_handle.is_cancel_requested:
            result.success = False
            result.message = 'canceled'
            goal_handle.canceled()
            return result

        self._feedback(goal_handle, 'recall', 0.1)
        try:
            outcome = self._backend.query(
                request.query,
                top_k=(int(request.top_k) or self._default_top_k),
                rerank=bool(request.rerank),
                time_from=request.time_from,
                time_to=request.time_to,
            )
        except Exception as exc:
            result.success = False
            result.message = str(exc)
            goal_handle.abort()
            return result

        if goal_handle.is_cancel_requested:
            result.success = False
            result.message = 'canceled after current retrieval call'
            goal_handle.canceled()
            return result
        self._feedback(
            goal_handle,
            'rerank' if outcome.rerank_attempted else 'recall',
            0.9,
        )

        for hit in outcome.hits:
            result.memory_ids.append(hit.memory_id)
            result.image_paths.append(hit.image_path)
            pose = Pose2D()
            if len(hit.robot_pose) >= 2:
                pose.x = hit.robot_pose[0]
                pose.y = hit.robot_pose[1]
                pose.theta = (
                    hit.robot_pose[2] if len(hit.robot_pose) >= 3 else 0.0)
            result.robot_poses.append(pose)
            result.scores.append(hit.score)
            result.timestamps.append(hit.timestamp)

        result.rerank_attempted = outcome.rerank_attempted
        result.rerank_valid = outcome.rerank_valid
        result.fallback_reason = outcome.fallback_reason
        result.success = bool(outcome.hits)
        result.message = (
            f'{len(outcome.hits)} memory candidate(s)'
            if outcome.hits else 'no matching memory frames')
        self._feedback(goal_handle, 'done', 1.0)
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def destroy_node(self):
        try:
            self._action.destroy()
            self._backend.close()
        finally:
            return super().destroy_node()


def main(args=None):
    """Run the spatial-memory skill with a multithreaded ROS executor."""
    rclpy.init(args=args)
    node = SpatialMemoryNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
