import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen36_38 as runtime


class _FakeImageAnalysisLlama:
    def __init__(self):
        self.reset_count = 0
        self.calls = []

    def reset(self):
        self.reset_count += 1

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls)
        return {"choices": [{"message": {"content": json.dumps({
            "asset_id": f"asset_{index}",
            "kind": "character" if index == 1 else "scene",
            "observable_features": f"features {index}",
        })}}]}


class QwenEventLedgerTests(unittest.TestCase):
    def request(self):
        return {
            "director_backend": "qwen3.8",
            "chunk_number": 2,
            "chunk_count": 3,
            "original_prompt": "prompt",
            "target_shots": [],
            "mandatory_coverage": [{"id": "S1.V2", "action": "open the door"}],
            "previous_event_ledger": {
                "completed": [{"id": "S1.V1", "summary": "stood up"}],
                "active": [], "pending": [],
                "forbidden": [{"id": "S1.V1", "summary": "stood up"}],
            },
        }

    def test_prompt_skill_images_are_analyzed_one_at_a_time_with_resets(self):
        llm = _FakeImageAnalysisLlama()
        observations = runtime._analyze_prompt_skill_images(llm, ["image-1", "image-2"])
        self.assertEqual(llm.reset_count, 2)
        self.assertEqual([item["asset_id"] for item in observations], ["asset_1", "asset_2"])
        self.assertEqual([item["kind"] for item in observations], ["character", "scene"])
        for call in llm.calls:
            content = call["messages"][0]["content"]
            images = [item for item in content if item.get("type") == "image_url"]
            self.assertEqual(len(images), 1)

    def test_prompt_skill_disables_mtp_only_for_its_disposable_operation(self):
        request = {"director_mtp": True}
        self.assertFalse(runtime._operation_uses_mtp(request, "prompt_skill_compile"))
        self.assertTrue(runtime._operation_uses_mtp(request, "timing_plan"))
        self.assertTrue(runtime._operation_uses_mtp(request, "chunk"))
        self.assertFalse(runtime._operation_uses_mtp({"director_mtp": False}, "chunk"))

    def test_fresh_prompt_skill_text_runtime_drops_multimodal_and_speculative_state(self):
        original = {
            "model_path": "model.gguf", "chat_handler": object(),
            "speculative": object(), "n_ctx": 32768,
        }
        fresh = runtime._fresh_text_runtime_kwargs(original)
        self.assertIsNone(fresh["chat_handler"])
        self.assertNotIn("speculative", fresh)
        self.assertEqual(fresh["model_path"], "model.gguf")
        self.assertEqual(fresh["n_ctx"], 32768)
        self.assertIn("speculative", original)

    def test_timing_parser_removes_zero_length_terminal_beat(self):
        request = {
            "fps": 24.0,
            "source_shots": [{"shot_number": 2, "shot_start": 0, "shot_end": 144}],
        }
        value = {
            "shots": [{
                "source_shot": 2,
                "visual_beats": [
                    {"start_frame": 0, "end_frame": 144, "action": "hold"},
                    {"start_frame": 144, "end_frame": 144, "action": "duplicate boundary"},
                ],
                "overlays": [],
            }],
        }
        plan = runtime._timing_plan(value, request, json.dumps(value), "system", "prompt")
        self.assertEqual(
            [(beat.start_frame, beat.end_frame, beat.action) for beat in plan.shots[0].visual_beats],
            [(0, 144, "hold")],
        )
        self.assertEqual(plan.validation_warnings, (
            "Removed zero-length Source Shot 2 visual beat 2 at frame 144.",
        ))

    def test_chunk_prompt_includes_previous_event_ownership(self):
        _system, prompt = runtime._chunk_messages(self.request())
        self.assertIn("EVENT OWNERSHIP LEDGER", prompt)
        self.assertIn("S1.V1", prompt)
        self.assertIn("MUST NOT stage, replay", prompt)
        self.assertIn("No typed prompt plan is connected.", prompt)

    def test_chunk_prompt_includes_typed_plan_contract(self):
        request = {**self.request(), "immutable_prompt_plan_contract": "LOCK SHOT 3 AND DIALOGUE S2"}
        _system, prompt = runtime._chunk_messages(request)
        self.assertIn("LOCK SHOT 3 AND DIALOGUE S2", prompt)

    def test_parser_normalizes_and_forbids_completed_events(self):
        result = runtime._chunk_prompt({
            "confidence": "high", "analysis": "ok", "detailed_description": "continue",
            "event_ledger": {
                "completed": [{"id": "S1.V1", "summary": "stood up"}],
                "active": [{"id": "S1.V2", "summary": "walking"}],
                "pending": [{"id": "S1.V3", "summary": "open door"}],
                "forbidden": [],
            },
        }, "{}", "system", "prompt", self.request())
        self.assertEqual(result.event_ledger["completed"][0]["id"], "S1.V1")
        self.assertEqual(result.event_ledger["forbidden"][0]["id"], "S1.V1")

    def test_parser_rejects_conflicting_event_ownership(self):
        with self.assertRaisesRegex(runtime.Qwen35ObservationError, "appears in both"):
            runtime._event_ledger({
                "completed": [],
                "active": [{"id": "S1.V1", "summary": "walking"}],
                "pending": [{"id": "S1.V1", "summary": "walking later"}],
                "forbidden": [],
            })

    def test_payload_round_trip_preserves_ledger(self):
        original = runtime._chunk_prompt({
            "detailed_description": "continue",
            "event_ledger": {"completed": [], "active": [], "pending": [], "forbidden": []},
        }, "{}", "system", "prompt")
        restored = runtime._from_payload(runtime._payload(original), False)
        self.assertEqual(restored.event_ledger, original.event_ledger)


if __name__ == "__main__":
    unittest.main()
