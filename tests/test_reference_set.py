import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]

# reference_set has ComfyUI node definitions, but these tests exercise only its
# deterministic media-slot and protocol validation helpers.
comfy = types.ModuleType("comfy")
comfy.__path__ = []
for name in ("model_management", "nested_tensor", "utils"):
    module = types.ModuleType(f"comfy.{name}")
    setattr(comfy, name, module)
    sys.modules[module.__name__] = module
sys.modules["comfy"] = comfy
sys.modules.setdefault("node_helpers", types.ModuleType("node_helpers"))

class _Custom:
    def __init__(self, name):
        self.name = name

    def Input(self, *args, **kwargs):
        return (args, kwargs)

    def Output(self, *args, **kwargs):
        return (args, kwargs)


class _ComfyNode:
    pass


class _Dummy:
    def __getattr__(self, _name):
        return _Dummy()

    def __call__(self, *args, **kwargs):
        return (args, kwargs)


io = _Dummy()
io.Custom = _Custom
io.ComfyNode = _ComfyNode
latest = types.ModuleType("comfy_api.latest")
latest.io = io
comfy_api = types.ModuleType("comfy_api")
comfy_api.__path__ = []
sys.modules["comfy_api"] = comfy_api
sys.modules["comfy_api.latest"] = latest

spec = importlib.util.spec_from_file_location("hr_reference_set_test", PLUGIN_ROOT / "reference_set.py")
reference_set = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference_set
spec.loader.exec_module(reference_set)


class ReferenceSetTests(unittest.TestCase):
    def test_indexed_preserves_empty_video_audio_slots(self):
        first_video = object()
        second_video = object()
        second_audio = object()
        videos = reference_set._indexed({"ref_video_0": first_video, "ref_video_1": second_video}, "ref_video_", 3)
        audios = reference_set._indexed({"ref_video_audio_1": second_audio}, "ref_video_audio_", 3)
        self.assertEqual(videos, (first_video, second_video, None))
        self.assertEqual(audios, (None, second_audio, None))

    def test_execute_accepts_expanded_autogrow_slot_kwargs(self):
        first = object()
        second = object()
        output = reference_set.HRMiniMaxH3ReferenceSet.execute(
            ref_image_0=first,
            ref_image_1=second,
        )
        value = output[0][0]
        self.assertEqual(value["images"], (first, second))

    def test_execute_merges_aggregated_and_expanded_autogrow_inputs(self):
        first = object()
        second = object()
        output = reference_set.HRMiniMaxH3ReferenceSet.execute(
            ref_images={"ref_image_0": first},
            ref_image_1=second,
        )
        self.assertEqual(output[0][0]["images"], (first, second))

    def test_execute_rejects_unknown_dynamic_input(self):
        with self.assertRaisesRegex(ValueError, "Unknown HR Reference Set inputs"):
            reference_set.HRMiniMaxH3ReferenceSet.execute(unexpected=object())

    def test_story_director_image_batch_expands_to_individual_references(self):
        batch = torch.zeros((4, 32, 32, 3))
        images = reference_set.reference_images({"version": 1, "images": (batch,)})
        self.assertEqual(len(images), 4)
        self.assertTrue(all(tuple(image.shape) == (1, 32, 32, 3) for image in images))

    def test_story_director_image_batch_respects_h3_limit(self):
        with self.assertRaisesRegex(ValueError, "at most 9"):
            reference_set.reference_images({"version": 1, "images": (torch.zeros((10, 8, 8, 3)),)})

    def test_conditioning_skips_empty_fixed_video_slots(self):
        image = torch.zeros((1, 32, 32, 3))
        refs = {
            "version": 1,
            "images": (image,),
            "videos": (None, None, None),
            "video_audios": (None, None, None),
            "audios": (),
            "ref_image_size": "match",
            "ref_scale": 1.0,
        }
        vae = types.SimpleNamespace(encode=lambda frames: torch.zeros((1, 24, 2, 2, 2)))
        clip = types.SimpleNamespace(
            tokenize=lambda prompt, minimax_ref_items: (prompt, minimax_ref_items),
            encode_from_tokens_scheduled=lambda tokens: [[torch.zeros(1), {}]],
        )
        latent = {"samples": object()}
        with patch.object(reference_set, "_empty_av_latent", return_value=(latent, 5)), \
             patch.object(reference_set, "_resize", side_effect=lambda frames, *_args: frames), \
             patch.object(reference_set.node_helpers, "conditioning_set_values", side_effect=lambda conditioning, _values: conditioning, create=True):
            output = reference_set.HRMiniMaxH3ReferenceConditioning.execute(
                clip, vae, None, "prompt", 32, 32, 5, refs
            )
        self.assertIs(output[0][1], latent)

    def test_normalize_accepts_same_index_video_soundtrack(self):
        video = object()
        audio = object()
        result = reference_set.normalize_reference_set({
            "version": 1,
            "videos": (None, video, None),
            "video_audios": (None, audio, None),
        })
        self.assertIs(result["videos"][1], video)
        self.assertIs(result["video_audios"][1], audio)

    def test_normalize_rejects_orphan_video_soundtrack(self):
        with self.assertRaisesRegex(ValueError, "same-index"):
            reference_set.normalize_reference_set({
                "version": 1,
                "videos": (object(), None, None),
                "video_audios": (None, object(), None),
            })

    def test_reference_limits_and_scale_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "limits"):
            reference_set.normalize_reference_set({"version": 1, "images": tuple(object() for _ in range(10))})
        with self.assertRaisesRegex(ValueError, "ref_scale"):
            reference_set.normalize_reference_set({"version": 1, "ref_scale": 5.1})


if __name__ == "__main__":
    unittest.main()
