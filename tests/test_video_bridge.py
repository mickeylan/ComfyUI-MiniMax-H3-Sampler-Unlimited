import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import comfy.nested_tensor


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "hr_video_bridge_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules[PACKAGE] = package
MODULE = importlib.import_module(PACKAGE + ".video_bridge")


class VideoBridgeExtractTests(unittest.TestCase):
    def frames(self, count):
        return torch.arange(count, dtype=torch.float32).reshape(count, 1, 1, 1).expand(count, 4, 6, 3)

    def test_extracts_exact_a_tail_and_b_head(self):
        output = MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(40), 24.0, 24.0)
        source, a_tail, b_head, *_rest = output.result
        self.assertEqual(tuple(a_tail.shape), (22, 4, 6, 3))
        self.assertEqual(tuple(b_head.shape), (22, 4, 6, 3))
        self.assertEqual(a_tail[0, 0, 0, 0].item(), 8)
        self.assertEqual(b_head[-1, 0, 0, 0].item(), 21)
        self.assertIs(MODULE.normalize_bridge_source(source), source)

    def test_rejects_short_video_and_fps_mismatch(self):
        with self.assertRaisesRegex(ValueError, "at least 22"):
            MODULE.HRVideoBridgeExtract.execute(self.frames(21), self.frames(22), 24.0, 24.0)
        with self.assertRaisesRegex(ValueError, "FPS mismatch"):
            MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(30), 24.0, 30.0)

    def test_resamples_selected_video_to_common_fps(self):
        output = MODULE.HRVideoBridgeExtract.execute(
            self.frames(30), self.frames(30), 24.0, 30.0, "resample_b_to_a"
        )
        source = output.result[0]
        self.assertEqual(source["fps"], 24.0)
        self.assertEqual(source["source_a_frame_count"], 30)
        self.assertEqual(source["source_b_frame_count"], 24)

    def test_extracts_synchronized_audio_windows(self):
        waveform = torch.arange(48000, dtype=torch.float32).reshape(1, 1, 48000)
        audio = {"waveform": waveform, "sample_rate": 48000}
        output = MODULE.HRVideoBridgeExtract.execute(
            self.frames(30), self.frames(30), 24.0, 24.0, audio_a=audio, audio_b=audio
        )
        a_audio, b_audio = output.result[3:5]
        expected = round(22 * 48000 / 24)
        self.assertEqual(a_audio["waveform"].shape[-1], expected)
        self.assertEqual(b_audio["waveform"].shape[-1], expected)
        self.assertEqual(a_audio["waveform"][0, 0, 0].item(), 48000 - expected)
        self.assertEqual(b_audio["waveform"][0, 0, -1].item(), expected - 1)

    def test_director_uses_backend_dispatching_worker(self):
        self.assertTrue(MODULE._run_worker_once.__module__.endswith(".qwen35"))

    def test_director_uses_qwen_bridge_operation(self):
        source = MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(30), 24.0, 24.0).result[0]
        config = {"version": 1, "backend": "qwen3.8", "model": "model.gguf", "mmproj": "mmproj.gguf",
                  "mtp": False, "mtp_draft_tokens": 2, "reasoning_effort": "medium",
                  "cpu_moe": False, "n_cpu_moe": 0, "debug": False}
        refs = {"version": 1, "images": (), "videos": (), "video_audios": (), "audios": (),
                "ref_image_size": "match", "ref_scale": 1.0}
        selection = types.SimpleNamespace(model_path=Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        response = {"ok": True, "video_bridge": {"version": 1, "transition_frames": 39,
                    "strategy": "match_cut", "analysis": {}, "constraints": ["identity"],
                    "h3_prompt": "subject_definitions: x", "risk_report": "low",
                    "reference_labels": {"pictures": [], "video_a": "<Video 2>", "video_b": "<Video 1>"}}}
        captured = {}
        def worker(request, timeout):
            captured.update(request)
            return types.SimpleNamespace(stderr="", stdout=""), response
        with patch.object(MODULE, "resolve_director_selection", return_value=selection), \
             patch.object(MODULE, "_run_worker_once", side_effect=worker), \
             patch.object(MODULE.comfy.model_management, "unload_all_models"), \
             patch.object(MODULE.comfy.model_management, "soft_empty_cache"):
            output = MODULE.HRVideoBridgeDirector.execute(source, refs, config)
        self.assertEqual(captured["operation"], "video_bridge")
        self.assertEqual(captured["director_backend"], "qwen3.8")
        self.assertEqual(len(captured["image_urls"]), 44)
        self.assertEqual(output.result[0]["type"], "HR_VIDEO_BRIDGE_PLAN")
        self.assertEqual(output.result[0]["reference_labels"]["video_a"], "<Video 2>")
        self.assertEqual(output.result[0]["reference_labels"]["video_b"], "<Video 1>")

    def test_qwen35_bridge_error_preserves_raw_response(self):
        error = MODULE.Qwen35ObservationError(
            "external continuation h3_prompt is incomplete",
            raw_json='{"h3_prompt":"实际返回"}',
        )
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(MODULE.folder_paths, "get_output_directory", return_value=directory):
            path = MODULE._save_qwen35_bridge_error(error)
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["response"]["h3_prompt"], "实际返回")

    def test_director_supports_qwen35_worker(self):
        source = MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(30), 24.0, 24.0).result[0]
        refs = {"version": 1, "images": (), "videos": (), "video_audios": (), "audios": (),
                "ref_image_size": "match", "ref_scale": 1.0}
        config = {"version": 1, "backend": "qwen3.5", "model": "model.gguf", "mmproj": "mmproj.gguf",
                  "mtp": False, "mtp_draft_tokens": 2, "reasoning_effort": "medium",
                  "cpu_moe": False, "n_cpu_moe": 0, "debug": False}
        selection = types.SimpleNamespace(model_path=Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        captured = {}
        directed = types.SimpleNamespace(
            confidence="high", observed_end_state={"must_continue": ["identity"]},
            transition_plan={"first_action": "continue"}, h3_prompt="[Shot 1] Continue into B.",
        )
        class ExistingDirector:
            def __init__(self, model_path, mmproj_path, **kwargs):
                captured["init"] = (model_path, mmproj_path, kwargs)
            def external_video_continuation(self, prompt, frames, **kwargs):
                captured.update(prompt=prompt, frames=frames, kwargs=kwargs)
                return directed
        with patch.object(MODULE, "resolve_director_selection", return_value=selection), \
             patch.object(MODULE, "Qwen35ContinuityDirector", ExistingDirector), \
             patch.object(MODULE.comfy.model_management, "unload_all_models"), \
             patch.object(MODULE.comfy.model_management, "soft_empty_cache"):
            output = MODULE.HRVideoBridgeDirector.execute(source, refs, config)
        self.assertEqual(captured["init"][2]["backend"], "qwen3.5")
        self.assertFalse(captured["init"][2]["mtp_enabled"])
        self.assertEqual(captured["init"][2]["context_tokens"], 131072)
        self.assertEqual(captured["frames"].shape, (22, 4, 12, 3))
        self.assertIn("A on the left and B on the right", captured["prompt"])
        self.assertIn("exact English label [Shot 1]", captured["prompt"])
        self.assertIn("never show a reference sheet", captured["prompt"])
        self.assertIn("Do not use a push-in, pull-back", captured["prompt"])
        self.assertIn("first/last-frame generation", captured["prompt"])
        self.assertIn("minimum visible state change", captured["prompt"])
        self.assertIn("do not invent a new shot", captured["prompt"])
        self.assertIn("identity pictures must never become visible content", captured["kwargs"]["reference_summary"])
        self.assertEqual(output.result[1], directed.h3_prompt)

    def test_qwen35_identity_images_match_paired_boundary_canvas(self):
        source = MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(30), 24.0, 24.0).result[0]
        refs = {"version": 1, "images": (torch.ones(1, 2, 3, 3),), "videos": (),
                "video_audios": (), "audios": (), "ref_image_size": "match", "ref_scale": 1.0}
        config = {"version": 1, "backend": "qwen3.5", "model": "model.gguf", "mmproj": "mmproj.gguf",
                  "mtp": False, "mtp_draft_tokens": 2, "reasoning_effort": "medium",
                  "cpu_moe": False, "n_cpu_moe": 0, "debug": False}
        selection = types.SimpleNamespace(model_path=Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        captured = {}
        directed = types.SimpleNamespace(
            confidence="high", observed_end_state={"must_continue": []},
            transition_plan={"first_action": "continue"}, h3_prompt="[Shot 1] Continue.",
        )
        class ExistingDirector:
            def __init__(self, *_args, **_kwargs):
                pass
            def external_video_continuation(self, _prompt, frames, **_kwargs):
                captured["frames"] = frames
                return directed
        with patch.object(MODULE, "resolve_director_selection", return_value=selection), \
             patch.object(MODULE, "Qwen35ContinuityDirector", ExistingDirector), \
             patch.object(MODULE.comfy.model_management, "unload_all_models"), \
             patch.object(MODULE.comfy.model_management, "soft_empty_cache"):
            MODULE.HRVideoBridgeDirector.execute(source, refs, config)
        self.assertEqual(captured["frames"].shape, (23, 4, 12, 3))
        self.assertTrue(torch.equal(captured["frames"][-1, :, :6], captured["frames"][-1, :, 6:]))

    def test_bridge_reference_keeps_destination_out_of_global_references(self):
        audio = {"waveform": torch.ones(1, 1, 48000), "sample_rate": 48000}
        source = MODULE.HRVideoBridgeExtract.execute(
            self.frames(30), self.frames(30), 24.0, 24.0, audio_a=audio, audio_b=audio
        ).result[0]
        refs = {"version": 1, "images": (), "videos": (), "video_audios": (), "audios": (),
                "ref_image_size": "match", "ref_scale": 1.0}
        bridge_refs = MODULE._bridge_reference_set(refs, source)
        self.assertEqual(bridge_refs["videos"], ())
        self.assertEqual(bridge_refs["video_audios"], ())
        self.assertIsNotNone(source["b_head_audio"])

    def test_conditioning_anchors_a_start_and_b_end(self):
        source = MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(30), 24.0, 24.0).result[0]
        plan = {"type": "HR_VIDEO_BRIDGE_PLAN", "version": 1, "transition_frames": 39,
                "h3_prompt": "subject_definitions: x", "source": source, "analysis": {}}
        refs = {"version": 1, "images": (), "videos": (), "video_audios": (), "audios": (),
                "ref_image_size": "match", "ref_scale": 1.0}
        embedding = torch.ones(1, 2, 3)
        identity_refs = [{"kind": "image", "latent": torch.zeros(1, 24, 1, 4, 6)}]
        latent = {"samples": comfy.nested_tensor.NestedTensor((
            torch.zeros(1, 24, 12, 4, 6), torch.zeros(1, 32, 2, 65)
        ))}
        conditioned = types.SimpleNamespace(result=([[embedding, {"minimax_refs": identity_refs}]], latent, refs))
        fake_node = types.SimpleNamespace(execute=lambda *args, **kwargs: conditioned)
        encoded_frame_counts = []
        def encode(frames):
            encoded_frame_counts.append(int(frames.shape[0]))
            return torch.zeros(1, 24, 2, frames.shape[1] // 16, frames.shape[2] // 16)
        vae = types.SimpleNamespace(encode=encode)
        with patch.object(MODULE, "HRMiniMaxH3ReferenceConditioning", fake_node):
            output = MODULE.HRVideoBridgeConditioning.execute(None, vae, None, plan, refs, 96, 64, "mute")
        positive, result_latent, continuation, *_ = output.result
        self.assertIs(result_latent, latent)
        self.assertIs(positive[0][1]["minimax_refs"], identity_refs)
        self.assertEqual(
            [item["resolved_frame_index"] for item in positive[0][1]["minimax_keyframes"]],
            [0, 38],
        )
        self.assertEqual(continuation["type"], "HR_H3_EXTERNAL_CONTINUATION")
        self.assertTrue(continuation["visual_references_only"])
        self.assertEqual(continuation["audio_mode"], "mute")
        self.assertIsNone(continuation["audio_context"])
        self.assertTrue(all(item.get("audio_latent") is None for item in positive[0][1]["minimax_keyframes"]))
        self.assertEqual(len(positive[0][1]["minimax_keyframes"]), 2)
        self.assertIs(continuation["reference_set"], refs)
        self.assertEqual(continuation["target_frames"], 39)
        self.assertEqual(continuation["tail_frames"], 1)
        self.assertEqual(continuation["tail_images"].shape[0], 1)
        self.assertEqual(encoded_frame_counts, [1, 1])

    def test_auto_seam_preserves_at_least_22_bridge_frames(self):
        self.assertEqual(MODULE._preserve_bridge_span(10, 13, 39, 22), (0, 22))
        self.assertEqual(MODULE._preserve_bridge_span(4, 30, 39, 22), (4, 30))
        self.assertEqual(MODULE._preserve_bridge_span(35, 36, 39, 22), (17, 39))

    def test_sequence_seam_skips_replayed_b_opening(self):
        bridge = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32).reshape(5, 1, 1, 1).expand(5, 2, 2, 3)
        video_b = torch.tensor([30, 40, 50, 60, 70], dtype=torch.float32).reshape(5, 1, 1, 1).expand(5, 2, 2, 3)
        seam = MODULE._best_sequence_seam(bridge, video_b, 5)
        self.assertEqual(seam["left_index"], 4)
        self.assertEqual(seam["right_index"], 2)
        self.assertEqual(seam["overlap_frames"], 3)

    def test_auto_seam_preserves_every_source_a_frame_unchanged(self):
        source = MODULE.HRVideoBridgeExtract.execute(self.frames(30), self.frames(30) + 100, 24.0, 24.0).result[0]
        bridge = self.frames(39) + 50
        output = MODULE.HRVideoBridgeAssemble.execute(
            source, bridge, assemble_policy="auto_seam", seam_search_frames=1,
            video_blend_frames=12, min_bridge_frames=22,
        )
        frames, _audio, timeline, report = output.result
        self.assertTrue(torch.equal(frames[:30], source["video_a"]))
        self.assertEqual(timeline["segments"][0]["source_end"], 29)
        report_value = __import__("json").loads(report)
        self.assertEqual(report_value["video_blend_frames"][0], 0)
        self.assertEqual(report_value["ranges"]["a_end"], 30)

    def test_crossfade_parts_smooths_both_video_seams(self):
        a = torch.zeros(6, 2, 2, 3)
        bridge = torch.ones(6, 2, 2, 3)
        b = torch.zeros(6, 2, 2, 3)
        frames, overlaps = MODULE._crossfade_parts([a, bridge, b], 3, 0)
        self.assertEqual(overlaps, [3, 3])
        self.assertEqual(frames.shape[0], 12)
        self.assertTrue(torch.all(frames[3:6] > 0))
        self.assertTrue(torch.all(frames[3:6] < 1))
        self.assertTrue(torch.all(frames[6:9] > 0))
        self.assertTrue(torch.all(frames[6:9] < 1))

    def test_crossfade_parts_supports_a_longer_bridge_to_b_transition(self):
        a = torch.zeros(12, 2, 2, 3)
        bridge = torch.full((12, 2, 2, 3), 0.5)
        b = torch.ones(12, 2, 2, 3)
        frames, overlaps = MODULE._crossfade_parts([a, bridge, b], [3, 8], 0)
        self.assertEqual(overlaps, [3, 8])
        self.assertEqual(frames.shape[0], 25)
        self.assertTrue(torch.all(frames[13:21] > 0.5))
        self.assertTrue(torch.all(frames[13:21] < 1.0))

    def test_hard_cut_assembly_preserves_all_frames_and_audio_duration(self):
        audio = {"waveform": torch.ones(1, 1, 48000), "sample_rate": 48000}
        source = MODULE.HRVideoBridgeExtract.execute(
            self.frames(30), self.frames(30), 24.0, 24.0, audio_a=audio, audio_b=audio
        ).result[0]
        bridge = self.frames(22)
        bridge_audio = {"waveform": torch.full((1, 1, 44000), 100.0), "sample_rate": 48000}
        output = MODULE.HRVideoBridgeAssemble.execute(
            source, bridge, bridge_audio, assemble_policy="hard_cut_debug", fade_seconds=0
        )
        frames, result_audio, timeline, report = output.result
        self.assertEqual(frames.shape[0], 82)
        self.assertEqual(timeline["total_frames"], 82)
        self.assertEqual(result_audio["waveform"].shape[-1], round(82 * 48000 / 24))
        self.assertLessEqual(float(result_audio["waveform"].abs().max()), 1.0)
        self.assertEqual(len(timeline["segments"]), 3)
        self.assertEqual(len(timeline["seams"]), 2)
        self.assertEqual(__import__("json").loads(report)["output_frames"], 82)


if __name__ == "__main__":
    unittest.main()
