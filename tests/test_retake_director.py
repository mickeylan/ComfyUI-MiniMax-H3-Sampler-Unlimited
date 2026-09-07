import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "retake_director_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules[PACKAGE] = package

nodes_spec = importlib.util.spec_from_file_location(PACKAGE + ".nodes", PLUGIN_ROOT / "nodes.py")
nodes = importlib.util.module_from_spec(nodes_spec)
sys.modules[nodes_spec.name] = nodes
nodes_spec.loader.exec_module(nodes)

spec = importlib.util.spec_from_file_location(PACKAGE + ".retake_director", PLUGIN_ROOT / "retake_director.py")
retake = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = retake
spec.loader.exec_module(retake)


class RetakeDirectorTests(unittest.TestCase):
    def test_missing_cache_is_reported(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(retake, "_replay_cache_root", return_value=Path(directory)):
            self.assertFalse(retake.replay_cache_snapshot()["available"])

    def test_snapshot_exposes_only_cache_contained_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chunks").mkdir()
            (root / "prompts").mkdir()
            (root / "observations").mkdir()
            (root / "chunks/chunk_0001.pt").write_bytes(b"tensor")
            (root / "observations/chunk_001.jpg").write_bytes(b"jpeg")
            (root / "prompts/chunk_0001.json").write_text(json.dumps({
                "chunk": 1, "frame_start": 0, "frame_end": 48,
                "tensor_path": "chunks/chunk_0001.pt",
                "observation_images": ["observations/chunk_001.jpg", "../outside.jpg"],
                "source_prompt": "source", "effective_h3_prompt": "final",
                "active_revision": 0,
            }), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({
                "format": nodes.REPLAY_CACHE_FORMAT, "status": "complete", "completed_chunks": 1,
                "fingerprint": {"fps": 24.0},
                "chunks": [{"chunk": 1, "metadata_path": "prompts/chunk_0001.json"}],
            }), encoding="utf-8")
            with patch.object(retake, "_replay_cache_root", return_value=root):
                snapshot = retake.replay_cache_snapshot()
                self.assertEqual(snapshot["fps"], 24.0)
                self.assertTrue(snapshot["chunks"][0]["complete"])
                self.assertEqual(snapshot["chunks"][0]["observation_images"], ["observations/chunk_001.jpg"])
                with self.assertRaises(retake.web.HTTPNotFound):
                    retake._asset_path("../outside.jpg")

    def test_build_plan_validates_selection_and_prompt_override(self):
        snapshot = {"available": True, "compatible": True, "cache_identity": "abc", "chunks": [
            {"chunk": 1, "complete": True, "effective_h3_prompt": "original one"},
            {"chunk": 2, "complete": True, "effective_h3_prompt": "original two"},
        ]}
        state = json.dumps({"mode": "video_only", "selected": [2, 1], "overrides": {"2": "edited two"}})
        with patch.object(retake, "replay_cache_snapshot", return_value=snapshot):
            plan = retake.build_retake_plan(state)
        self.assertEqual(plan["cache_identity"], "abc")
        self.assertEqual([item["chunk"] for item in plan["chunks"]], [1, 2])
        self.assertEqual(plan["chunks"][1]["prompt_override"], "edited two")
        self.assertEqual(plan["chunks"][0]["original_h3_prompt"], "original one")

    def test_build_plan_rejects_missing_chunks_and_unknown_modes(self):
        snapshot = {"available": True, "compatible": True, "cache_identity": "abc", "chunks": []}
        with patch.object(retake, "replay_cache_snapshot", return_value=snapshot):
            with self.assertRaisesRegex(ValueError, "Unknown retake mode"):
                retake.build_retake_plan('{"mode":"wrong","selected":[1]}')
            with self.assertRaisesRegex(ValueError, "not available"):
                retake.build_retake_plan('{"mode":"video_only","selected":[1]}')

    def test_node_serializes_state_and_returns_typed_plan(self):
        schema = retake.HREndlessSegmentRetakeDirector.define_schema()
        self.assertEqual([item.id for item in schema.inputs], ["retake_state"])
        self.assertEqual(len(schema.outputs), 2)


if __name__ == "__main__":
    unittest.main()
