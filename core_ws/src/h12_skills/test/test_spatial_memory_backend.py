"""Movement-free tests for the H12 spatial-memory adapter."""

from dataclasses import dataclass
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from h12_skills.skills.spatial_memory import (
    EmbodiedAgentMemory,
    _EmbodiedAgentRuntime,
    _in_time_window,
    _validate_rerank,
)


@dataclass
class _Entry:
    memory_id: str
    image_path: str
    robot_pose: list
    timestamp: str


class _FakeMemory:
    def __init__(self, _memory_dir):
        self.entries = {}

    def build_from_scan_dir(self, _capture_dir, embedding_model='unknown'):
        del embedding_model
        return 0

    def create_entry(self, **kwargs):
        return _Entry(
            kwargs['memory_id'], kwargs['image_path'], kwargs['robot_pose'],
            kwargs['timestamp'])

    def add_entry(self, entry):
        self.entries[entry.memory_id] = entry


class _Candidate:
    def __init__(self, entry, score):
        self.memory_id = entry.memory_id
        self.image_path = entry.image_path
        self.robot_pose = entry.robot_pose
        self.timestamp = entry.timestamp
        self.retrieval_score = score
        self.frame_idx = int(entry.memory_id.removeprefix('mem_'))

    def to_dict(self):
        return {
            'memory_id': self.memory_id,
            'image_path': self.image_path,
            'robot_pose': self.robot_pose,
            'timestamp': self.timestamp,
            'retrieval_score': self.retrieval_score,
        }


class _FakeWorker:
    latest = None

    def __init__(self, index_dir, model_name, device):
        del index_dir, model_name, device
        self.is_ready = True
        self.frames = []
        self.stopped = False
        _FakeWorker.latest = self

    def enqueue(self, rgb, frame_path, robot_xy, robot_yaw):
        self.frames.append((rgb, frame_path, robot_xy, robot_yaw))

    def flush(self):
        return None

    def stop(self):
        self.stopped = True


class _FakeGemini:
    response = {}

    def __init__(self, **_kwargs):
        return None

    def rerank_memory_candidates(self, **_kwargs):
        return self.response


def _retrieve(**kwargs):
    memory = kwargs['episodic_memory']
    candidates = [
        _Candidate(entry, 0.8 - rank * 0.1)
        for rank, entry in enumerate(memory.entries.values())
    ]
    top_k = kwargs['top_k']
    return candidates if top_k == -1 else candidates[:top_k]


_RUNTIME = _EmbodiedAgentRuntime(
    memory_type=_FakeMemory,
    worker_type=_FakeWorker,
    retrieve=_retrieve,
    gemini_type=_FakeGemini,
)


class SpatialMemoryBackendTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.memory = EmbodiedAgentMemory(
            data_dir=self.temp_dir.name,
            embodied_agent_root='/unused',
            runtime=_RUNTIME,
        )

    def tearDown(self):
        self.memory.close()
        self.temp_dir.cleanup()
        os.environ.pop('GEMINI_API_KEY', None)

    def test_capture_writes_contract_and_is_immediately_queryable(self):
        memory_id = self.memory.add_frame(
            Image.new('RGB', (8, 8), color='red'),
            (1.25, -2.5, 0.75),
            timestamp='2026-07-22T12:34:56+00:00',
        )
        self.assertEqual(memory_id, 'mem_000000')
        capture = Path(self.temp_dir.name) / 'capture'
        self.assertTrue((capture / 'color/000000.png').is_file())
        self.assertTrue((capture / 'robot_xy/000000.txt').is_file())
        self.assertTrue((capture / 'frame_meta/000000.json').is_file())

        outcome = self.memory.query('find the red object', top_k=3)
        self.assertEqual([hit.memory_id for hit in outcome.hits], [memory_id])
        self.assertEqual(outcome.hits[0].robot_pose, (1.25, -2.5, 0.75))
        self.assertFalse(outcome.rerank_attempted)

    def test_valid_rerank_changes_order_and_exposes_confidence(self):
        first = self.memory.add_frame(
            Image.new('RGB', (8, 8), color='red'), (1.0, 0.0, 0.0),
            timestamp='2026-07-22T12:00:00+00:00')
        second = self.memory.add_frame(
            Image.new('RGB', (8, 8), color='blue'), (2.0, 0.0, 0.0),
            timestamp='2026-07-22T12:00:02+00:00')
        os.environ['GEMINI_API_KEY'] = 'test-key'
        _FakeGemini.response = {
            'ranked_ids': [second, first],
            'candidates_analysis': [
                {
                    'memory_id': first,
                    'confidence': 0.1,
                    'object_location': 'not visible',
                    'reasoning': 'wrong frame',
                },
                {
                    'memory_id': second,
                    'confidence': 0.9,
                    'object_location': 'center',
                    'reasoning': 'target visible',
                },
            ],
        }
        outcome = self.memory.query('find blue', top_k=2, rerank=True)
        self.assertTrue(outcome.rerank_attempted)
        self.assertTrue(outcome.rerank_valid)
        self.assertEqual([hit.memory_id for hit in outcome.hits], [second, first])
        self.assertAlmostEqual(outcome.hits[0].score, 0.9)
        self.assertEqual(outcome.hits[0].faiss_rank, 2)

    def test_rerank_without_key_falls_back_to_faiss(self):
        memory_id = self.memory.add_frame(
            Image.new('RGB', (8, 8)), (0.0, 0.0, 0.0),
            timestamp='2026-07-22T12:00:00+00:00')
        outcome = self.memory.query('anything', rerank=True)
        self.assertEqual(outcome.hits[0].memory_id, memory_id)
        self.assertFalse(outcome.rerank_attempted)
        self.assertFalse(outcome.rerank_valid)
        self.assertEqual(outcome.fallback_reason, 'missing_api_key')


class SpatialMemoryUtilityTest(unittest.TestCase):
    def test_time_window_can_cross_midnight(self):
        self.assertTrue(_in_time_window('00:01:00', '23:59:00', '00:02:00'))
        self.assertFalse(_in_time_window('12:00:00', '23:59:00', '00:02:00'))

    def test_rerank_rejects_partial_candidate_lists(self):
        response = {
            'ranked_ids': ['mem_000001'],
            'candidates_analysis': [],
        }
        self.assertIsNone(
            _validate_rerank(response, ['mem_000001', 'mem_000002']))


if __name__ == '__main__':
    unittest.main()
