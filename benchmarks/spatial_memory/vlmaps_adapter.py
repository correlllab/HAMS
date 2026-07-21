"""Streaming VLMaps adapter for the RGB-D object-relocation benchmark.

This is a benchmark-oriented incremental form of the official camera-map
builder. It keeps the core VLMaps recipe—LSeg pixel features, metric-depth
backprojection, distance-weighted 3D voxel fusion, and CLIP text indexing—but
updates the sparse map after every observation so live relocation can be
measured without rebuilding the whole scene at each query checkpoint.
"""

from __future__ import annotations

import importlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .adapter import MemoryAdapter, MemoryCandidate
from .vlmaps_backend import OfficialLSegBackend


DEFAULT_KITCHEN_CATEGORIES = (
    "bowl",
    "plate",
    "bottle",
    "can",
    "sink",
    "stove",
    "refrigerator",
    "counter",
    "cabinet",
    "drawer",
    "floor",
    "wall",
    "table",
    "chair",
    "other",
)


@dataclass
class _Voxel:
    feature: np.ndarray
    position: np.ndarray
    weight: float
    support_observation_id: str
    support_weight: float


def _load_factory(spec: str):
    if ":" not in spec:
        raise ValueError("backend factory must look like package.module:callable")
    module_name, attribute = spec.rsplit(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), attribute)


def _semantic_query(text: str) -> str:
    """Remove benchmark navigation wording while preserving the noun phrase."""
    normalized = " ".join(text.strip().split())
    patterns = (
        r"(?i)^find the current location of the (.+?)(?:;|$)",
        r"(?i)^where was the (.+?) before it moved\??$",
        r"(?i)^find (?:a|an|the) (.+?)\??$",
    )
    for pattern in patterns:
        match = re.match(pattern, normalized)
        if match:
            return match.group(1).strip()
    return normalized


def _component_indices(xy_keys: list[tuple[int, int]]) -> list[list[int]]:
    """Return 8-connected components for sparse top-down voxel cells."""
    key_to_index = {key: index for index, key in enumerate(xy_keys)}
    unvisited = set(range(len(xy_keys)))
    components: list[list[int]] = []
    while unvisited:
        seed = min(unvisited)
        unvisited.remove(seed)
        stack = [seed]
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            x, y = xy_keys[index]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbour = key_to_index.get((x + dx, y + dy))
                    if neighbour in unvisited:
                        unvisited.remove(neighbour)
                        stack.append(neighbour)
        components.append(component)
    return components


class VLMapsAdapter(MemoryAdapter):
    """Incrementally fuse RGB-D LSeg features into a world-coordinate map."""

    name = "vlmaps"

    def __init__(
        self,
        cell_size: float = 0.05,
        depth_sample_rate: int = 100,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        device: str = "auto",
        vlmaps_root: str = "/opt/vlmaps",
        checkpoint_path: str | None = None,
        backend_factory: str | None = None,
        backend: Any | None = None,
        fallback_quantile: float = 0.995,
        category_vocabulary: str = ",".join(DEFAULT_KITCHEN_CATEGORIES),
    ):
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        if depth_sample_rate <= 0:
            raise ValueError("depth_sample_rate must be positive")
        if not 0.0 <= min_depth < max_depth:
            raise ValueError("depth range is invalid")
        if not 0.0 < fallback_quantile < 1.0:
            raise ValueError("fallback_quantile must be between zero and one")
        self.cell_size = float(cell_size)
        self.depth_sample_rate = int(depth_sample_rate)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.fallback_quantile = float(fallback_quantile)
        self.category_vocabulary = tuple(
            category.strip()
            for category in category_vocabulary.split(",")
            if category.strip()
        )
        if not self.category_vocabulary:
            raise ValueError("category_vocabulary must not be empty")
        self.device = device
        self.vlmaps_root = vlmaps_root
        self.checkpoint_path = checkpoint_path
        self.backend_factory = backend_factory
        self._backend = backend
        self._episode_dir: Path | None = None
        self._voxels: dict[tuple[int, int, int], _Voxel] = {}
        self._last_diagnostics: dict = {}
        self._text_feature_cache: dict[str, np.ndarray] = {}

    def _ensure_backend(self):
        if self._backend is None:
            factory = (
                _load_factory(self.backend_factory)
                if self.backend_factory else OfficialLSegBackend
            )
            self._backend = factory(
                vlmaps_root=self.vlmaps_root,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
        return self._backend

    def reset(self, episode_dir: Path, state_dir: Path) -> None:
        self._episode_dir = Path(episode_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self._voxels = {}
        self._last_diagnostics = {}

    def _text_feature(self, text: str) -> np.ndarray:
        feature = self._text_feature_cache.get(text)
        if feature is None:
            feature = self._ensure_backend().encode_text(text)
            self._text_feature_cache[text] = feature
        return feature

    def _text_features(self, texts: list[str]) -> np.ndarray:
        missing = [text for text in texts if text not in self._text_feature_cache]
        if missing:
            backend = self._ensure_backend()
            if hasattr(backend, "encode_texts"):
                encoded = backend.encode_texts(missing)
                self._text_feature_cache.update(zip(missing, encoded))
            else:
                for text in missing:
                    self._text_feature_cache[text] = backend.encode_text(text)
        return np.stack([self._text_feature_cache[text] for text in texts])

    def ingest(self, observation: dict) -> None:
        if self._episode_dir is None:
            raise RuntimeError("reset must be called before ingest")
        required = (
            "image_path",
            "depth_path",
            "camera_intrinsics",
            "camera_to_world",
            "camera_coordinate_frame",
        )
        missing = [key for key in required if observation.get(key) is None]
        if missing:
            raise RuntimeError(
                "VLMaps requires benchmark-v2 RGB-D camera metadata; missing "
                + ", ".join(missing)
            )
        if observation["camera_coordinate_frame"] != (
            "opencv_x_right_y_down_z_forward"
        ):
            raise RuntimeError("VLMaps requires OpenCV x-right/y-down/z-forward poses")

        rgb = np.asarray(
            Image.open(self._episode_dir / observation["image_path"]).convert("RGB")
        )
        depth = np.load(
            self._episode_dir / observation["depth_path"],
            allow_pickle=False,
        ).astype(np.float32, copy=False)
        if depth.shape != rgb.shape[:2]:
            raise RuntimeError("VLMaps RGB and depth dimensions do not match")
        intrinsics = np.asarray(
            observation["camera_intrinsics"], dtype=np.float64
        ).reshape(3, 3)
        camera_to_world = np.asarray(
            observation["camera_to_world"], dtype=np.float64
        ).reshape(4, 4)
        dense_features = self._ensure_backend().encode_image(rgb)
        self._fuse_observation(
            observation["observation_id"],
            depth,
            intrinsics,
            camera_to_world,
            dense_features,
        )

    def _fuse_observation(
        self,
        observation_id: str,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
        dense_features: np.ndarray,
    ) -> None:
        height, width = depth.shape
        frame_number_match = re.search(r"(\d+)$", observation_id)
        random_seed = (
            int(frame_number_match.group(1)) if frame_number_match else 0
        )
        random = np.random.default_rng(random_seed)
        sample_count = math.ceil(height * width / self.depth_sample_rate)
        flat_indices = np.sort(random.choice(
            height * width,
            size=sample_count,
            replace=False,
        ))
        rows = flat_indices // width
        columns = flat_indices % width
        depths = depth[rows, columns].astype(np.float64, copy=False)
        valid = (
            np.isfinite(depths)
            & (depths >= self.min_depth)
            & (depths <= self.max_depth)
        )
        if not valid.any():
            return
        rows = rows[valid]
        columns = columns[valid]
        depths = depths[valid]

        pixels = np.stack(
            (
                columns.astype(np.float64) + 0.5,
                rows.astype(np.float64) + 0.5,
                np.ones_like(depths),
            ),
            axis=1,
        )
        camera_points = (pixels @ np.linalg.inv(intrinsics).T) * depths[:, None]
        homogeneous = np.concatenate(
            (camera_points, np.ones((len(camera_points), 1))), axis=1
        )
        world_points = (homogeneous @ camera_to_world.T)[:, :3]

        feature_height, feature_width = dense_features.shape[:2]
        feature_rows = np.minimum(
            (rows * feature_height // height), feature_height - 1
        )
        feature_columns = np.minimum(
            (columns * feature_width // width), feature_width - 1
        )
        features = dense_features[feature_rows, feature_columns].astype(
            np.float64, copy=False
        )
        radial_distance_sq = np.square(camera_points).sum(axis=1)
        weights = np.exp(-radial_distance_sq / (2.0 * 0.6))
        voxel_keys = np.rint(world_points / self.cell_size).astype(np.int64)
        unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)

        for unique_index, key_array in enumerate(unique_keys):
            member = inverse == unique_index
            member_weights = weights[member]
            frame_weight = float(member_weights.sum())
            if frame_weight <= 1e-12:
                continue
            frame_feature = np.average(
                features[member], axis=0, weights=member_weights
            ).astype(np.float32)
            frame_position = np.average(
                world_points[member], axis=0, weights=member_weights
            )
            key = tuple(int(value) for value in key_array)
            voxel = self._voxels.get(key)
            if voxel is None:
                self._voxels[key] = _Voxel(
                    feature=frame_feature,
                    position=frame_position,
                    weight=frame_weight,
                    support_observation_id=observation_id,
                    support_weight=frame_weight,
                )
                continue
            total_weight = voxel.weight + frame_weight
            voxel.feature = (
                (voxel.feature * voxel.weight + frame_feature * frame_weight)
                / total_weight
            ).astype(np.float32)
            voxel.position = (
                voxel.position * voxel.weight + frame_position * frame_weight
            ) / total_weight
            voxel.weight = total_weight
            if frame_weight >= voxel.support_weight:
                voxel.support_observation_id = observation_id
                voxel.support_weight = frame_weight

    def query(self, text: str, top_k: int) -> list[MemoryCandidate]:
        if not self._voxels or top_k <= 0:
            self._last_diagnostics = {
                "map_native": True,
                "voxel_count": len(self._voxels),
                "component_count": 0,
            }
            return []
        semantic_query = _semantic_query(text)
        backend = self._ensure_backend()
        negative_categories = [
            category for category in self.category_vocabulary
            if category != semantic_query
        ]
        category_features = self._text_features([
            semantic_query,
            *negative_categories,
        ])
        query_feature = category_features[0]
        negative_features = category_features[1:]

        keys = list(self._voxels)
        voxels = [self._voxels[key] for key in keys]
        features = np.stack([voxel.feature for voxel in voxels]).astype(np.float32)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        query_scores = features @ query_feature
        negative_scores = features @ negative_features.T
        best_negative_scores = negative_scores.max(axis=1)
        margins = query_scores - best_negative_scores

        # Match official initialized-category indexing: a voxel belongs to the
        # query only when the query beats every fixed kitchen category.
        foreground_indices = np.flatnonzero(margins > 0.0)
        selection_scores = margins
        selection_mode = "official_initialized_categories"
        selection_threshold = 0.0
        if not len(foreground_indices):
            # Fine-grained kitchen objects can lose the fixed-vocabulary
            # classification even when their query score has
            # a useful spatial peak. Keep that peak measurable using only the
            # top similarity tail and expose the fallback in diagnostics.
            selection_threshold = float(np.quantile(
                query_scores,
                self.fallback_quantile,
            ))
            foreground_indices = np.flatnonzero(
                query_scores >= selection_threshold
            )
            selection_scores = query_scores
            selection_mode = "top_similarity_quantile_fallback"
        topdown: dict[tuple[int, int], int] = {}
        for index in foreground_indices:
            xy = (keys[index][0], keys[index][1])
            prior = topdown.get(xy)
            if prior is None or selection_scores[index] > selection_scores[prior]:
                topdown[xy] = int(index)

        xy_keys = sorted(topdown)
        components = _component_indices(xy_keys) if xy_keys else []
        ranked_components = []
        for component in components:
            voxel_indices = [topdown[xy_keys[index]] for index in component]
            component_score = float(max(
                selection_scores[index] for index in voxel_indices
            ))
            ranked_components.append((component_score, voxel_indices))
        ranked_components.sort(key=lambda item: item[0], reverse=True)

        candidates = []
        used_observation_ids: set[str] = set()
        for component_score, voxel_indices in ranked_components:
            ranked_voxels = sorted(
                voxel_indices,
                key=lambda index: float(selection_scores[index]),
                reverse=True,
            )
            support_id = next(
                (
                    voxels[index].support_observation_id
                    for index in ranked_voxels
                    if voxels[index].support_observation_id not in used_observation_ids
                ),
                None,
            )
            if support_id is None:
                continue
            raw_weights = np.asarray([
                float(selection_scores[index]) for index in voxel_indices
            ])
            positive_weights = raw_weights - raw_weights.min() + 1e-6
            positions = np.stack([voxels[index].position for index in voxel_indices])
            predicted_position = np.average(
                positions, axis=0, weights=positive_weights
            )
            used_observation_ids.add(support_id)
            candidates.append(MemoryCandidate(
                observation_id=support_id,
                score=component_score,
                metadata={
                    "score_label": (
                        "VLMaps category margin"
                        if selection_mode == "official_initialized_categories"
                        else "VLMaps top similarity"
                    ),
                    "predicted_world_xyz": predicted_position.tolist(),
                    "semantic_query": semantic_query,
                    "component_voxel_count": len(voxel_indices),
                    "cell_size_m": self.cell_size,
                    "feature_backend": getattr(backend, "name", type(backend).__name__),
                    "selection_mode": selection_mode,
                },
            ))
            if len(candidates) >= top_k:
                break

        self._last_diagnostics = {
            "map_native": True,
            "incremental_3d_fusion": True,
            "dynamic_point_removal": False,
            "semantic_query": semantic_query,
            "voxel_count": len(voxels),
            "foreground_voxel_count": int(len(foreground_indices)),
            "component_count": len(components),
            "selection_mode": selection_mode,
            "selection_threshold": selection_threshold,
            "query_score_max": float(query_scores.max()),
            "negative_category_count": len(negative_categories),
            "best_negative_score_max": float(best_negative_scores.max()),
            "query_minus_best_category_max": float(margins.max()),
            "feature_backend": getattr(backend, "name", type(backend).__name__),
        }
        return candidates

    def query_diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    def close(self) -> None:
        # Keep the heavyweight LSeg model alive across benchmark episodes.
        self._voxels = {}
