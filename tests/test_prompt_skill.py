import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import prompt_skill
import qwen35
import qwen36_38


class PromptSkillTests(unittest.TestCase):
    def request(self, continuity_mode="balanced"):
        return prompt_skill.build_prompt_skill_request(
            "A hero enters a temple, then opens a door.", duration_seconds=2.0, fps=24.0,
            image_count=1, style="cinematic", shot_density="medium",
            continuity_mode=continuity_mode, prompt_lang="en",
        )

    def result(self):
        return {
            "image_subjects": [{"entity_id": "asset_1", "kind": "character", "name": "Hero", "observable_features": "black hair"}],
            "summary": "The hero enters and opens the door.",
            "retention_analysis": "Lock identity and never repeat completed actions.",
            "shots": [
                {"start_frame": 0, "end_frame": 22, "pictures": ["asset_1"],
                 "camera": "rear wide tracking shot", "start_state": "outside the temple",
                 "events": [{"id": "S1.V1", "actor": "asset_1", "action": "enters the temple", "phase": "start"}],
                 "end_state": "fully inside", "forbidden_replays": ["outside approach"],
                 "audio": "footsteps", "description": "The hero enters once and ends fully inside."},
                {"start_frame": 22, "end_frame": 56, "pictures": ["asset_1"],
                 "camera": "frontal medium static shot", "start_state": "already fully inside",
                 "events": [{"id": "S2.V1", "actor": "asset_1", "action": "opens the door", "phase": "start"}],
                 "end_state": "door open", "forbidden_replays": ["temple entrance"],
                 "audio": "door creak", "description": "Already inside, the hero opens the door once."},
            ],
            "overall_soundscape": "stone interior ambience",
            "non_diegetic_music": "N/A",
            "warnings": [],
        }

    def test_redistributes_dialogue_from_late_qwen_shot_across_complete_timeline(self):
        story = '<Subject 1> (S1) says: <d>[Chinese] 这是第一句很长的对白，需要使用前面镜头的时间。</d>'
        request = prompt_skill.build_prompt_skill_request(
            story, duration_seconds=2.0, fps=24.0, image_count=1, style="cinematic",
            shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        value = self.result()
        midpoint = 22
        value["shots"][0]["end_frame"] = midpoint
        value["shots"][1]["start_frame"] = midpoint
        value["shots"][1]["end_frame"] = request["total_frames"]
        value["shots"][0]["dialogues"] = []
        value["shots"][1]["dialogues"] = [{
            "id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
            "language": "Chinese", "text": "这是第一句很长的对白，需要使用前面镜头的时间。", "delivery": "自然地",
        }]
        normalized, warnings = prompt_skill._redistribute_dialogues(value, request)
        self.assertTrue(normalized["shots"][0]["dialogues"])
        self.assertEqual(
            "".join(item["text"] for shot in normalized["shots"] for item in shot["dialogues"]),
            request["required_spoken_lines"][0],
        )
        self.assertTrue(any("Redistributed mandatory dialogue" in warning for warning in warnings))

    def test_redistributes_two_long_lines_out_of_one_overloaded_shot(self):
        required = [
            "姐姐，自从你跟太运宗使者比试之后，这十年你都没有怎么好好闭关修炼过。还有不到四十年，太运宗就会派更强的弟子，这样真的来得及吗？",
            "我现在功力已经达到顶峰，再闭关苦修已是无用功。与其毫无头绪的闭关，不如将舒寒当年传授给我的武学反复磨练磨练来得有意思。",
        ]
        story = " ".join(f'<Subject 1> (S1) says: <d>[Chinese] {text}</d>' for text in required)
        request = prompt_skill.build_prompt_skill_request(
            story, duration_seconds=40.0, fps=24.0, image_count=1, style="cinematic",
            shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        value = self.result()
        value["shots"][0]["end_frame"] = 402
        value["shots"][1]["start_frame"] = 402
        value["shots"][1]["end_frame"] = request["total_frames"]
        value["shots"][0]["dialogues"] = [
            {"id": f"S1.D{index}", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": text, "delivery": "自然地"}
            for index, text in enumerate(required, 1)
        ]
        compiled = prompt_skill.compile_prompt_skill(value, request)
        shots = compiled["shot_plan"]["shots"]
        spoken = "".join(item["text"] for shot in shots for item in shot["dialogues"])
        self.assertEqual(spoken, "".join(required))
        self.assertTrue(shots[1]["dialogues"])
        fragments = [item for shot in shots for item in shot["dialogues"]]
        for item in fragments:
            self.assertNotRegex(item["text"], r"^[，,。！？!?；;：:]")
        continued = next(item for item in fragments if item.get("continues_from_previous"))
        self.assertIn("<scenetrans>", prompt_skill._dialogue_description(continued))
        self.assertTrue(any("Redistributed mandatory dialogue" in warning for warning in compiled["warnings"]))

    def test_cross_shot_dialogue_uses_h3_scene_transition_contract(self):
        description = prompt_skill._dialogue_description({
            "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
            "language": "Chinese", "text": "继续说话", "delivery": "自然地",
            "continues_from_previous": True, "continues_to_next": True,
        })
        self.assertEqual(description.count("<scenetrans>"), 2)
        self.assertIn("continues seamlessly across the cut", description)
        self.assertNotIn(" says,", description)

    def test_long_chinese_dialogue_extends_short_requested_duration(self):
        story = (
            '<Subject 1> (S1) says: <d>[Chinese] 姐姐，自从你跟太运宗使者比试之后，'
            '这十年你都没有怎么好好闭关修炼过。还有不到四十年，太运宗就会派更强的弟子，'
            '这样真的来得及吗？</d> '
            '<Subject 2> (S2) says: <d>[Chinese] 我现在功力已经达到顶峰，再闭关苦修已是无用功。'
            '与其毫无头绪的闭关，不如将舒寒当年传授给我的武学反复磨练磨练来得有意思。</d>'
        )
        request = prompt_skill.build_prompt_skill_request(
            story, duration_seconds=6.0, fps=24.0, image_count=2, style="cinematic",
            shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        self.assertGreater(request["minimum_spoken_duration_seconds"], 6.0)
        self.assertGreater(request["duration_seconds"], 6.0)
        self.assertEqual((request["total_frames"] - 5) % 17, 0)
        self.assertEqual(request["duration_source"], "dialogue")
        _system, user = prompt_skill.prompt_skill_messages(request)
        self.assertIn("Estimated duration for the exact spoken content", user)
        self.assertIn("Duration authority: dialogue", user)

    def test_dialogue_duration_can_shorten_longer_user_duration(self):
        request = prompt_skill.build_prompt_skill_request(
            'He says: “Stay.”', duration_seconds=20.0, fps=24.0, image_count=1,
            style="cinematic", shot_density="medium", continuity_mode="balanced", prompt_lang="en",
        )
        self.assertLess(request["duration_seconds"], 20.0)
        self.assertEqual(request["requested_duration_seconds"], 20.0)
        self.assertEqual(request["duration_source"], "dialogue")

    def test_story_without_dialogue_preserves_user_duration(self):
        request = prompt_skill.build_prompt_skill_request(
            "A traveler crosses a bridge.", duration_seconds=20.0, fps=24.0, image_count=1,
            style="cinematic", shot_density="medium", continuity_mode="balanced", prompt_lang="en",
        )
        self.assertGreaterEqual(request["duration_seconds"], 20.0)
        self.assertLess(request["duration_seconds"], 21.0)
        self.assertEqual(request["duration_source"], "user")

    def test_normalizes_qwen38_serialized_image_subject_objects(self):
        value = self.result()
        value["image_subjects"] = [
            ['{"entity_id":"asset_1","kind":"character","name":"Hero","observable_features":"black hair"}']
        ]
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        self.assertEqual(compiled["shot_plan"]["image_subjects"][0]["entity_id"], "asset_1")
        self.assertEqual(compiled["shot_plan"]["image_subjects"][0]["kind"], "character")

    def test_normalizes_unambiguous_compact_image_subject_string(self):
        value = self.result()
        value["image_subjects"] = ["asset_1 | character | black hair"]
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        subject = compiled["shot_plan"]["image_subjects"][0]
        self.assertEqual((subject["entity_id"], subject["kind"]), ("asset_1", "character"))

    def test_normalizes_bare_entity_ids_from_actor_and_non_speaking_use(self):
        request = {
            **self.request(), "image_count": 2,
            "source_image_contract": [
                {"entity_id": "asset_1", "picture": 1, "name": "Hero", "kind": None},
                {"entity_id": "asset_2", "picture": 2, "name": "梵心桃花林", "kind": None},
            ],
        }
        value = self.result()
        value["image_subjects"] = ["asset_1", "asset_2"]
        for shot in value["shots"]:
            shot["pictures"] = ["asset_1", "asset_2"]
        compiled = prompt_skill.compile_prompt_skill(value, request)
        subjects = compiled["shot_plan"]["image_subjects"]
        self.assertEqual([(item["entity_id"], item["kind"]) for item in subjects], [
            ("asset_1", "character"), ("asset_2", "scene"),
        ])
        self.assertTrue(any("Resolved asset_2 kind=scene as a non-speaking visual reference" in item for item in compiled["warnings"]))

    def test_rebuilds_duplicate_omitted_and_out_of_order_subjects_from_source_contract(self):
        request = {
            **self.request(), "image_count": 3,
            "source_image_contract": [
                {"entity_id": "asset_1", "picture": 1, "name": "Hero", "kind": "character"},
                {"entity_id": "asset_2", "picture": 2, "name": "Peach Grove", "kind": None},
                {"entity_id": "asset_3", "picture": 3, "name": "Sword", "kind": "prop"},
            ],
        }
        value = self.result()
        value["image_subjects"] = [
            {"entity_id": "asset_2", "kind": "scene", "name": "Peach Grove", "observable_features": "pink trees"},
            "asset_2",
            {"entity_id": "asset_1", "kind": "scene", "name": "Hero", "observable_features": "black hair"},
        ]
        for shot in value["shots"]:
            shot["pictures"] = ["asset_1", "asset_2", "asset_3"]
        compiled = prompt_skill.compile_prompt_skill(value, request)
        subjects = compiled["shot_plan"]["image_subjects"]
        self.assertEqual(
            [(item["entity_id"], item["picture"], item["name"], item["kind"]) for item in subjects],
            [
                ("asset_1", 1, "Hero", "character"),
                ("asset_2", 2, "Peach Grove", "scene"),
                ("asset_3", 3, "Sword", "prop"),
            ],
        )
        self.assertEqual(subjects[1]["observable_features"], "pink trees")
        self.assertTrue(any("Merged 2 Qwen image_subjects entries for asset_2" in item for item in compiled["warnings"]))
        self.assertTrue(any("Restored omitted image_subjects entry asset_3" in item for item in compiled["warnings"]))
        self.assertTrue(any("resolved kind=character" in item for item in compiled["warnings"]))

    def test_rejects_unknown_subject_even_when_other_entries_are_recoverable(self):
        value = self.result()
        value["image_subjects"] = ["asset_1", "asset_9"]
        with self.assertRaisesRegex(ValueError, "unknown entity_id 'asset_9'"):
            prompt_skill.compile_prompt_skill(value, self.request())

    def test_rejects_ambiguous_bare_image_subject_name_with_actual_value(self):
        value = self.result()
        value["image_subjects"] = ["Hero"]
        with self.assertRaisesRegex(ValueError, "must identify a source asset; got 'Hero'"):
            prompt_skill.compile_prompt_skill(value, self.request())

    def test_allows_dialogue_only_shot_without_fabricating_visual_events(self):
        value = self.result()
        value["shots"][0]["events"] = []
        value["shots"][0].pop("forbidden_replays")
        value["shots"][0]["dialogues"] = [{
            "id": "S1.D1", "kind": "dialogue", "speaker": "asset_1", "speaker_id": "S1",
            "language": "English", "text": "Stay here.", "delivery": "quietly",
        }]
        request = {**self.request(), "required_spoken_lines": ["Stay here."]}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        first = compiled["shot_plan"]["shots"][0]
        self.assertEqual(first["events"], [])
        self.assertEqual(first["forbidden_replays"], [])
        self.assertTrue(any(
            "Normalized missing shots[1].forbidden_replays to an empty array" in warning
            for warning in compiled["warnings"]
        ))

    def test_rejects_non_array_event_contract_fields(self):
        value = self.result()
        value["shots"][0]["events"] = {"id": "S1.V1"}
        with self.assertRaisesRegex(ValueError, r"shots\[1\]\.events must be an array"):
            prompt_skill.compile_prompt_skill(value, self.request())

    def test_compiles_six_fields_and_initial_event_ledger(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        for heading in ("subject_definitions:", "summary:", "retention_analysis:",
                        "detailed_description:", "overall_soundscape:", "non_diegetic_music:"):
            self.assertIn(heading, compiled["prompt"])
        self.assertEqual([item["id"] for item in compiled["initial_event_ledger"]["pending"]], ["S1.V1", "S2.V1"])
        self.assertEqual(compiled["planned_frames"], 56)

    def test_final_identity_gate_rejects_scene_subject_bound_to_character_name(self):
        plan = {
            "image_subjects": [
                {"subject": 2, "picture": 2, "kind": "scene", "name": "Peach Grove"},
                {"subject": 4, "picture": 4, "kind": "character", "name": "Ruolin"},
            ]
        }
        prompt = "detailed_description:\n<Subject 2> (Ruolin) stands center beside <Subject 4>."
        with self.assertRaisesRegex(ValueError, "Ruolin.*Subject 4, not Subject 2"):
            prompt_skill.validate_h3_identity_contract(prompt, plan)

    def test_final_identity_gate_ignores_names_spoken_inside_verbatim_dialogue(self):
        plan = {
            "image_subjects": [
                {"subject": 3, "picture": 3, "kind": "character", "name": "上官若彤"},
                {"subject": 4, "picture": 4, "kind": "character", "name": "上官若琳"},
            ]
        }
        prompt = (
            "subject_definitions:\n<Subject 3> 上官若彤\n<Subject 4> 上官若琳\n"
            "detailed_description:\n<Subject 4> (S2) says: "
            "<d>[Chinese] 上官若彤，你终于来了。</d> with synchronized visible lip movement."
        )
        prompt_skill.validate_h3_identity_contract(prompt, plan)

    def test_final_identity_gate_does_not_infer_binding_from_a_bare_name(self):
        plan = {"image_subjects": [{"subject": 3, "picture": 3, "kind": "character", "name": "上官若彤"}]}
        prompt_skill.validate_h3_identity_contract("summary:\n上官若彤 enters.", plan)

    def test_rejects_model_owned_subject_labels_before_h3_compilation(self):
        value = self.result()
        value["shots"][0]["start_state"] = "<Subject 1> is incorrectly assigned by the model"
        with self.assertRaisesRegex(ValueError, "compiler-owned Subject/Picture labels"):
            prompt_skill.compile_prompt_skill(value, self.request())

    def test_rejects_scene_entity_as_visual_event_actor(self):
        story = "<Picture 1> is a stone temple; <Picture 2> is Hero; Hero says: “Ready.”"
        request = prompt_skill.build_prompt_skill_request(
            story, duration_seconds=3.0, fps=24.0, image_count=2, style="cinematic",
            shot_density="medium", continuity_mode="strict", prompt_lang="en",
        )
        value = self.result()
        value["image_subjects"] = [
            {"entity_id": "asset_1", "kind": "scene", "name": "a stone temple", "observable_features": "stone hall"},
            {"entity_id": "asset_2", "kind": "character", "name": "Hero", "observable_features": "black hair"},
        ]
        value["shots"] = [{
            "start_frame": 0, "end_frame": request["total_frames"], "pictures": ["asset_1", "asset_2"],
            "camera": "static wide shot", "start_state": "<Entity asset_2> stands inside <Entity asset_1>",
            "events": [{"id": "S1.V1", "actor": "asset_1", "action": "turns toward camera", "phase": "start"}],
            "dialogues": [], "end_state": "<Entity asset_2> remains ready", "forbidden_replays": [],
            "audio": "room tone", "description": "<Entity asset_2> waits inside <Entity asset_1>.",
        }]
        with self.assertRaisesRegex(ValueError, "actor asset_1 is not a character"):
            prompt_skill.compile_prompt_skill(value, request)

    def test_preserves_dialogue_monologue_and_voiceover_in_h3_description(self):
        value = self.result()
        value["shots"][0]["dialogues"] = [
            {"id": "S1.D1", "kind": "dialogue", "speaker": "asset_1", "speaker_id": "S1",
             "language": "Chinese", "text": "你终于来了。", "delivery": "平静而清晰地"},
            {"id": "S1.D2", "kind": "monologue", "speaker": "asset_1", "speaker_id": "S1",
             "language": "Chinese", "text": "我不能再等了。", "delivery": "低声自语"},
            {"id": "S1.D3", "kind": "voiceover", "speaker": "asset_1", "speaker_id": "S1",
             "language": "Chinese", "text": "那一天改变了一切。", "delivery": "以克制的回忆语气"},
        ]
        request = {**self.request(), "required_spoken_lines": ["你终于来了。", "我不能再等了。", "那一天改变了一切。"]}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        description = compiled["shot_plan"]["shots"][0]["description"]
        self.assertIn("<Subject 1> (S1) says", description)
        self.assertIn("<d>[Chinese] 你终于来了。</d>", description)
        self.assertIn("speaks an audible monologue", description)
        self.assertIn("<d>[Chinese] 我不能再等了。</d>", description)
        self.assertIn("says in an off-screen voiceover", description)
        self.assertIn("lips remain completely closed", description)
        self.assertIn("<d>[Chinese] 那一天改变了一切。</d>", compiled["prompt"])
        self.assertEqual(
            [item["id"] for item in compiled["initial_event_ledger"]["pending"]],
            ["S1.V1", "S1.D1", "S1.D2", "S1.D3", "S2.V1"],
        )

    def test_rejects_non_character_dialogue_entity_instead_of_guessing(self):
        value = self.result()
        value["image_subjects"][0]["kind"] = "scene"
        value["shots"][1]["dialogues"] = [{
            "id": "S2.D1", "kind": "dialogue", "speaker": "asset_1", "speaker_id": "S1",
            "language": "Chinese", "text": "继续说话。", "delivery": "平静地",
        }]
        request = {**self.request(), "required_spoken_lines": ["继续说话。"]}
        with self.assertRaisesRegex(ValueError, "actor asset_1 is not a character"):
            prompt_skill.compile_prompt_skill(value, request)

    def test_empty_audio_is_normalized_without_inventing_sound(self):
        value = self.result()
        value["shots"][1]["audio"] = ""
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        self.assertEqual(compiled["shot_plan"]["shots"][1]["audio"], "N/A")
        self.assertTrue(any("Normalized empty shots[2].audio to N/A" in warning
                            for warning in compiled["warnings"]))

    def test_fills_deterministic_dialogue_metadata(self):
        value = self.result()
        value["shots"][1]["dialogues"] = [{
            "speaker": "asset_1", "speaker_id": "S1", "text": "继续说话。",
        }]
        request = {**self.request(), "required_spoken_lines": ["继续说话。"]}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        dialogue = compiled["shot_plan"]["shots"][1]["dialogues"][0]
        self.assertEqual(
            {key: dialogue[key] for key in ("id", "kind", "speaker", "speaker_id", "language", "text", "delivery")},
            {"id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "继续说话。", "delivery": "自然清晰地"},
        )
        self.assertEqual((dialogue["start_frame"], dialogue["end_frame"]), (22, 56))
        self.assertTrue(any("fields: id, kind, language, delivery" in warning for warning in compiled["warnings"]))

    def test_incomplete_dialogue_reports_unrecoverable_fields(self):
        value = self.result()
        value["shots"][1]["dialogues"] = [{"text": "无人绑定。"}]
        request = {**self.request(), "required_spoken_lines": ["无人绑定。"]}
        with self.assertRaisesRegex(ValueError, "missing=\\['speaker', 'speaker_id'\\]"):
            prompt_skill.compile_prompt_skill(value, request)

    def test_reused_voice_id_is_reassigned_without_aborting(self):
        value = self.result()
        value["image_subjects"] = [
            {"entity_id": "asset_1", "kind": "character", "name": "Older sister", "observable_features": "red robe"},
            {"entity_id": "asset_2", "kind": "character", "name": "Younger sister", "observable_features": "purple robe"},
        ]
        value["shots"][0]["dialogues"] = [
            {"id": "S1.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "姐姐。", "delivery": "问道"},
        ]
        value["shots"][1]["dialogues"] = [
            {"id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 2>", "speaker_id": "S1",
             "language": "Chinese", "text": "妹妹。", "delivery": "回答"},
        ]
        request = {
            **self.request(), "image_count": 2,
            "source_image_contract": [
                {"entity_id": "asset_1", "picture": 1, "name": "", "kind": None},
                {"entity_id": "asset_2", "picture": 2, "name": "", "kind": None},
            ],
            "required_spoken_lines": ["姐姐。", "妹妹。"],
            "required_speaker_subjects": {},
        }
        compiled = prompt_skill.compile_prompt_skill(value, request)
        dialogues = [item for shot in compiled["shot_plan"]["shots"] for item in shot["dialogues"]]
        self.assertEqual(
            [(item["speaker"], item["speaker_id"]) for item in dialogues],
            [("<Subject 1>", "S1"), ("<Subject 2>", "S2")],
        )
        self.assertTrue(any("normalized <Subject 2> to unused voice id S2" in warning for warning in compiled["warnings"]))

    def test_same_character_keeps_first_voice_id(self):
        value = self.result()
        value["shots"][0]["dialogues"] = [
            {"id": "S1.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "前半句", "delivery": "说道"},
        ]
        value["shots"][1]["dialogues"] = [
            {"id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S2",
             "language": "Chinese", "text": "后半句", "delivery": "继续"},
        ]
        request = {**self.request(), "required_spoken_lines": ["前半句后半句"]}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        dialogues = [item for shot in compiled["shot_plan"]["shots"] for item in shot["dialogues"]]
        self.assertEqual([item["speaker_id"] for item in dialogues], ["S1", "S1"])

    def test_authoritative_source_speaker_binding_corrects_qwen_remap(self):
        value = self.result()
        value["image_subjects"] = [
            {"entity_id": "asset_1", "kind": "character", "name": "Older sister", "observable_features": "red robe"},
            {"entity_id": "asset_2", "kind": "character", "name": "Younger sister", "observable_features": "purple robe"},
        ]
        value["shots"][0]["dialogues"] = [
            {"id": "S1.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "前半句，", "delivery": "平静地"},
        ]
        value["shots"][1]["dialogues"] = [
            {"id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 2>", "speaker_id": "S1",
             "language": "Chinese", "text": "后半句。", "delivery": "继续说道"},
        ]
        request = {
            **self.request(), "image_count": 2,
            "source_image_contract": [
                {"entity_id": "asset_1", "picture": 1, "name": "", "kind": None},
                {"entity_id": "asset_2", "picture": 2, "name": "", "kind": None},
            ],
            "required_spoken_lines": ["前半句，后半句。"],
            "required_speaker_subjects": {"S1": "<Subject 1>"},
        }
        compiled = prompt_skill.compile_prompt_skill(value, request)
        dialogues = [item for shot in compiled["shot_plan"]["shots"] for item in shot["dialogues"]]
        self.assertEqual([item["speaker"] for item in dialogues], ["<Subject 1>", "<Subject 1>"])
        self.assertTrue(any("authoritative source binding" in warning for warning in compiled["warnings"]))

    def test_request_extracts_authoritative_speaker_subject_bindings(self):
        request = prompt_skill.build_prompt_skill_request(
            '<Subject 4> (S1) says: <d>[Chinese] 红衣。</d> <Subject 3> (S2) says: <d>[Chinese] 紫衣。</d>',
            duration_seconds=2.0, fps=24.0, image_count=4, style="cinematic",
            shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        self.assertEqual(request["required_speaker_subjects"], {"S1": "<Subject 4>", "S2": "<Subject 3>"})

    def test_named_story_dialogue_overrides_qwen_wrong_speaker_binding(self):
        story = (
            "<Picture 1>是玉霄峰宫；<Picture 2>是梵心桃花林；"
            "<Picture 3>是上官若彤四视图；<Picture 4>是上官若琳四视图；\n"
            "上官若彤说：“姐姐，自从你跟太运宗使者比试之后，这十年你都没有怎么好好闭关修炼过。”\n"
            "上官若琳说：“我现在功力已经达到顶峰，再闭关苦修已是无用功。”"
        )
        request = prompt_skill.build_prompt_skill_request(
            story, duration_seconds=30.0, fps=24.0, image_count=4, style="cinematic",
            shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        self.assertEqual(request["required_spoken_subjects"], ["<Subject 3>", "<Subject 4>"])
        self.assertEqual(request["required_speaker_subjects"], {"S1": "<Subject 3>", "S2": "<Subject 4>"})
        self.assertEqual(request["source_image_contract"], [
            {"entity_id": "asset_1", "picture": 1, "name": "玉霄峰宫", "kind": None},
            {"entity_id": "asset_2", "picture": 2, "name": "梵心桃花林", "kind": None},
            {"entity_id": "asset_3", "picture": 3, "name": "上官若彤", "kind": "character"},
            {"entity_id": "asset_4", "picture": 4, "name": "上官若琳", "kind": "character"},
        ])
        system, user = prompt_skill.prompt_skill_messages(request)
        self.assertIn("Those H3 labels are private compiler output", system)
        self.assertIn("entity_id=asset_1: immutable source name='玉霄峰宫'", user)
        self.assertIn("entity_id=asset_3: immutable source name='上官若彤'; kind=character", user)

        value = self.result()
        value["image_subjects"] = [
            {"entity_id": "asset_1", "kind": "character", "name": "Wrong woman", "observable_features": "wrong purple robe"},
            {"entity_id": "asset_2", "kind": "character", "name": "梵心桃花林", "observable_features": "pink blossoms"},
            {"entity_id": "asset_3", "kind": "scene", "name": "上官若彤", "observable_features": "purple robe"},
            {"entity_id": "asset_4", "kind": "prop", "name": "上官若琳", "observable_features": "red robe"},
        ]
        with self.assertRaisesRegex(ValueError, "renamed immutable asset_1"):
            prompt_skill.compile_prompt_skill(value, request)

    def test_restores_missing_canonical_marker_for_unique_source_name_without_retry(self):
        request = {
            **self.request(), "image_count": 2,
            "source_image_contract": [
                {"entity_id": "asset_1", "picture": 1, "name": "Hero", "kind": "character"},
                {"entity_id": "asset_2", "picture": 2, "name": "梵心桃花林", "kind": None},
            ],
        }
        value = self.result()
        value["image_subjects"] = [
            {"entity_id": "asset_1", "kind": "character", "name": "Hero", "observable_features": "black hair"},
            {"entity_id": "asset_2", "kind": "scene", "name": "梵心桃花林", "observable_features": "pink blossoms"},
        ]
        for shot in value["shots"]:
            shot["pictures"] = ["asset_1", "asset_2"]
        value["shots"][1]["start_state"] = "Hero pauses in 梵心桃花林."
        value["shots"][1]["description"] = "Hero pauses in 梵心桃花林."
        normalized, warnings = prompt_skill._resolve_entity_contract(value, request)
        self.assertIn("<Subject 2> 梵心桃花林", normalized["shots"][1]["description"])
        self.assertTrue(any(
            "Restored canonical <Entity asset_2> marker before '梵心桃花林' in shots[2].description" in warning
            for warning in warnings
        ))

    def test_rejects_source_name_bound_to_the_wrong_entity_marker(self):
        request = {
            **self.request(), "image_count": 2,
            "source_image_contract": [
                {"entity_id": "asset_1", "picture": 1, "name": "Hero", "kind": "character"},
                {"entity_id": "asset_2", "picture": 2, "name": "梵心桃花林", "kind": None},
            ],
        }
        value = self.result()
        value["image_subjects"] = [
            {"entity_id": "asset_1", "kind": "character", "name": "Hero", "observable_features": "black hair"},
            {"entity_id": "asset_2", "kind": "scene", "name": "梵心桃花林", "observable_features": "pink blossoms"},
        ]
        value["shots"][1]["description"] = "<Entity asset_1> 梵心桃花林 remains still."
        with self.assertRaisesRegex(ValueError, "binds '梵心桃花林' to <Entity asset_1>"):
            prompt_skill._resolve_entity_contract(value, request)

    def test_unknown_dialogue_entity_is_rejected_without_subject_guessing(self):
        value = self.result()
        value["shots"][0]["dialogues"] = [
            {"id": "S1.D1", "kind": "dialogue", "speaker": "asset_9", "speaker_id": "S1",
             "language": "Chinese", "text": "我来了。", "delivery": "平静地"},
        ]
        request = {**self.request(), "required_spoken_lines": ["我来了。"]}
        with self.assertRaisesRegex(ValueError, "unknown entity_id 'asset_9'"):
            prompt_skill.compile_prompt_skill(value, request)

    def test_speaking_character_picture_is_automatically_included_in_shot(self):
        value = self.result()
        value["image_subjects"] = [
            {"entity_id": "asset_1", "kind": "scene", "name": "Palace", "observable_features": "red pillars"},
            {"entity_id": "asset_2", "kind": "character", "name": "Hero", "observable_features": "red and gold robe"},
        ]
        value["shots"][0]["pictures"] = ["asset_1"]
        value["shots"][0]["events"][0]["actor"] = "asset_2"
        value["shots"][1]["events"][0]["actor"] = "asset_2"
        value["shots"][0]["dialogues"] = [
            {"id": "S1.D1", "kind": "dialogue", "speaker": "asset_2", "speaker_id": "S1",
             "language": "Chinese", "text": "我来了。", "delivery": "平静地"},
        ]
        request = {
            **self.request(), "image_count": 2,
            "source_image_contract": [
                {"entity_id": "asset_1", "picture": 1, "name": "", "kind": None},
                {"entity_id": "asset_2", "picture": 2, "name": "", "kind": None},
            ],
            "required_spoken_lines": ["我来了。"],
        }
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertEqual(compiled["shot_plan"]["shots"][0]["pictures"], [1, 2])
        self.assertIn("<Subject 2> (S1)", compiled["prompt"])

    def test_extracts_tagged_dialogue_and_explicit_spoken_quotes(self):
        request = prompt_skill.build_prompt_skill_request(
            '女人说：“不要回头！” 旁白：“风暴已经开始。” <d>[English] Stay close.</d>',
            duration_seconds=2.0, fps=24.0, image_count=1, style="cinematic",
            shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        self.assertEqual(request["required_spoken_lines"], ["不要回头！", "风暴已经开始。", "Stay close."])

    def test_normalizes_qwen_dialogue_order_to_story_order(self):
        value = self.result()
        value["shots"][0]["dialogues"] = [
            {"id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "第二句", "delivery": "回答"},
        ]
        value["shots"][1]["dialogues"] = [
            {"id": "S1.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "第一句", "delivery": "先说"},
        ]
        request = {**self.request(), "required_spoken_lines": ["第一句", "第二句"]}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertEqual(
            [item["text"] for shot in compiled["shot_plan"]["shots"] for item in shot["dialogues"]],
            ["第一句", "第二句"],
        )
        self.assertIn("Restored mandatory spoken lines verbatim", compiled["warnings"][0])
        self.assertIn("<d>[Chinese] 第一句</d>", compiled["shot_plan"]["shots"][0]["description"])

    def test_preserves_repeated_spoken_line_occurrences_in_story_order(self):
        request = prompt_skill.build_prompt_skill_request(
            '女人说：“快走！” 男人回答：“等等。” 女人又喊：“快走！”',
            duration_seconds=2.0, fps=24.0, image_count=1, style="cinematic",
            shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        self.assertEqual(request["required_spoken_lines"], ["快走！", "等等。", "快走！"])

    def test_restores_qwen_rewritten_dialogue_from_source_without_retry(self):
        value = self.result()
        value["shots"][0]["dialogues"] = [
            {"id": "S1.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "被模型改写的第一句", "delivery": "自然地"},
        ]
        value["shots"][1]["dialogues"] = [
            {"id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "被模型改写的第二句", "delivery": "自然地"},
        ]
        required = ["姐姐，原文第一句。", "妹妹，原文第二句。"]
        compiled = prompt_skill.compile_prompt_skill(value, {**self.request(), "required_spoken_lines": required})
        returned = [item["text"] for shot in compiled["shot_plan"]["shots"] for item in shot["dialogues"]]
        self.assertEqual(returned, required)
        self.assertTrue(any("Restored mandatory spoken lines verbatim" in warning for warning in compiled["warnings"]))

    def test_accepts_long_spoken_line_split_across_adjacent_shots(self):
        value = self.result()
        value["shots"][0]["dialogues"] = [
            {"id": "S1.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "姐姐，自从比试之后，", "delivery": "担忧地"},
        ]
        value["shots"][1]["dialogues"] = [
            {"id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "这样真的来得及吗？", "delivery": "继续问道"},
        ]
        request = {**self.request(), "required_spoken_lines": ["姐姐，自从比试之后，这样真的来得及吗？"]}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertIn("<d>[Chinese] 姐姐，自从比试之后，</d>", compiled["prompt"])
        self.assertIn("<d>[Chinese] 这样真的来得及吗？</d>", compiled["prompt"])
        self.assertIn("concatenation exactly preserves", compiled["warnings"][0])

    def test_restores_reordered_spoken_fragments(self):
        value = self.result()
        value["shots"][0]["dialogues"] = [
            {"id": "S1.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "后半句", "delivery": "说"},
        ]
        value["shots"][1]["dialogues"] = [
            {"id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
             "language": "Chinese", "text": "前半句", "delivery": "说"},
        ]
        request = {**self.request(), "required_spoken_lines": ["前半句后半句"]}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        returned = [item["text"] for shot in compiled["shot_plan"]["shots"] for item in shot["dialogues"]]
        self.assertEqual(returned, ["前半句后半句"])
        self.assertTrue(any("Restored mandatory spoken lines verbatim" in warning for warning in compiled["warnings"]))

    def test_rebuilds_omitted_mandatory_line_from_source_speaker_contract(self):
        line = "姐姐，自从你跟太运宗使者比试之后，这十年你都没有怎么好好闭关修炼过。还有不到四十年，太运宗就会派更强的弟子，这样真的来得及吗？"
        request = prompt_skill.build_prompt_skill_request(
            f"<Picture 1>是上官若彤四视图；上官若彤说：“{line}”",
            duration_seconds=2.0, fps=24.0, image_count=1, style="cinematic",
            shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        value = self.result()
        value["image_subjects"][0]["name"] = "上官若彤"
        value["shots"][0]["end_frame"] = 22
        value["shots"][1]["start_frame"] = 22
        value["shots"][1]["end_frame"] = request["total_frames"]
        value["shots"][0]["dialogues"] = []
        value["shots"][1]["dialogues"] = []
        compiled = prompt_skill.compile_prompt_skill(value, request)
        dialogues = [item for shot in compiled["shot_plan"]["shots"] for item in shot["dialogues"]]
        self.assertEqual("".join(item["text"] for item in dialogues), line)
        self.assertTrue(all(item["speaker"] == "<Subject 1>" for item in dialogues))
        self.assertTrue(all(item["speaker_id"] == "S1" for item in dialogues))
        self.assertTrue(any("Rebuilt omitted mandatory dialogue occurrences" in item for item in compiled["warnings"]))

    def test_rejects_omitted_mandatory_spoken_line(self):
        request = {**self.request(), "required_spoken_lines": ["Do not leave me."]}
        with self.assertRaisesRegex(ValueError, "exact story order"):
            prompt_skill.compile_prompt_skill(self.result(), request)

    def test_structure_repair_requires_exact_target_interval(self):
        request = {
            **self.request(),
            "total_frames": 634,
            "prompt_skill_structure_repair": True,
            "prompt_skill_validation_error": "Shot 2 has invalid interval [634,1268)",
            "prompt_skill_previous_response": '{"shots":[{"start_frame":634,"end_frame":1268}]}',
        }
        _system, user = prompt_skill.prompt_skill_messages(request)
        self.assertIn("cover exactly [0,634)", user)
        self.assertIn("no start_frame or end_frame may exceed 634", user)
        self.assertIn("Do not double the requested duration", user)
        self.assertIn("Never return an empty object {}", user)
        self.assertIn("non-empty image_subjects and shots arrays", user)
        self.assertIn("[634,1268)", user)

    def test_prompt_contract_requires_first_class_spoken_content(self):
        system, user = prompt_skill.prompt_skill_messages(self.request())
        self.assertIn("dialogue, monologue, or voiceover", system)
        self.assertIn("never invent a cut merely to make adjacent camera strings different", system)
        self.assertIn("never translate, paraphrase, shorten, or invent dialogue", system)
        self.assertIn('"dialogues"', user)

    def test_normalizes_common_qwen_event_shapes(self):
        value = self.result()
        value["shots"][0]["events"] = [
            {"id": "1_v1", "description": "enters the temple", "phase": "begins"},
        ]
        value["shots"][1]["events"] = [
            {"id": "S1-V1", "action": "continues through the doorway", "phase": "ongoing"},
            {"action": "looks toward the altar"},
        ]
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        events = [item for shot in compiled["shot_plan"]["shots"] for item in shot["events"]]
        self.assertEqual(
            [{key: event[key] for key in ("id", "action", "phase")} for event in events],
            [{"id": "S1.V1", "action": "enters the temple", "phase": "start"},
             {"id": "S1.V1", "action": "continues through the doorway", "phase": "continue"},
             {"id": "S2.V2", "action": "looks toward the altar", "phase": "start"}],
        )
        self.assertEqual([(event["start_frame"], event["end_frame"]) for event in events], [(0, 22), (22, 39), (39, 56)])
        self.assertTrue(any("Normalized shot 2 event 2" in warning for warning in compiled["warnings"]))

    def test_continuation_event_inherits_missing_action_from_first_occurrence(self):
        value = self.result()
        value["shots"][0]["events"] = [
            {"id": "S1.V2", "action": "上官若彤走入桃花林并来到姐姐身旁", "phase": "start"},
        ]
        value["shots"][1]["events"] = [
            {"id": "S1.V2", "phase": "continue"},
        ]
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        continued = compiled["shot_plan"]["shots"][1]["events"][0]
        self.assertEqual(
            {key: continued[key] for key in ("id", "action", "phase")},
            {"id": "S1.V2", "action": "上官若彤走入桃花林并来到姐姐身旁", "phase": "continue"},
        )
        self.assertEqual((continued["start_frame"], continued["end_frame"]), (22, 56))
        self.assertTrue(any("Inherited missing action for continue event S1.V2" in warning
                            for warning in compiled["warnings"]))

    def test_invalid_event_error_reports_the_bad_fields(self):
        value = self.result()
        value["shots"][1]["events"] = [{"id": "nonsense", "phase": "later"}]
        with self.assertRaisesRegex(ValueError, "id='nonsense'.*phase='later'.*action_present=False"):
            prompt_skill.compile_prompt_skill(value, self.request())

    def test_normalizes_duplicate_event_start_to_continuation(self):
        value = self.result()
        value["shots"][1]["events"][0]["id"] = "S1.V1"
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        self.assertEqual(compiled["shot_plan"]["shots"][1]["events"][0]["phase"], "continue")
        self.assertEqual([item["id"] for item in compiled["initial_event_ledger"]["pending"]], ["S1.V1"])
        self.assertIn("normalized to continue", compiled["warnings"][0])

    def test_normalizes_missing_intermediate_end_state_from_next_start_state(self):
        value = self.result()
        value["shots"][0]["end_state"] = ""
        value["shots"][1]["start_state"] = "already inside with both feet planted"
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        self.assertEqual(
            compiled["shot_plan"]["shots"][0]["end_state"],
            "already inside with both feet planted",
        )
        self.assertIn("shared frame boundary", compiled["warnings"][0])

    def test_final_shot_still_requires_end_state(self):
        value = self.result()
        value["shots"][-1]["end_state"] = ""
        with self.assertRaisesRegex(ValueError, r"shots\[2\]\.end_state must be non-empty"):
            prompt_skill.compile_prompt_skill(value, self.request())

    def test_normalizes_inclusive_qwen_shot_end_to_next_start(self):
        value = self.result()
        value["shots"][0]["end_frame"] = 21
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        self.assertEqual(
            [(shot["start_frame"], shot["end_frame"]) for shot in compiled["shot_plan"]["shots"]],
            [(0, 22), (22, 56)],
        )
        self.assertIn("shot 1 [0,21) -> [0,22)", compiled["warnings"][0])

    def test_normalizes_reported_634_frame_boundary_without_retry(self):
        value = self.result()
        value["shots"][0].update(start_frame=0, end_frame=633)
        value["shots"][1].update(start_frame=634, end_frame=1268)
        request = {**self.request(), "total_frames": 1268}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertEqual(compiled["shot_plan"]["shots"][0]["end_frame"], 634)
        self.assertEqual(compiled["shot_plan"]["shots"][1]["start_frame"], 634)
        self.assertEqual(compiled["planned_frames"], 1268)

    def test_merges_zero_length_trailing_shot_without_losing_spoken_content(self):
        value = self.result()
        value["shots"][0].update(start_frame=0, end_frame=634)
        value["shots"][1].update(
            start_frame=634, end_frame=634,
            dialogues=[
                {"id": "S2.D1", "kind": "voiceover", "speaker": "<Subject 1>", "speaker_id": "S1",
                 "language": "Chinese", "text": "故事仍将继续。", "delivery": "平静地"},
            ],
        )
        request = {**self.request(), "total_frames": 634, "required_spoken_lines": ["故事仍将继续。"]}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertEqual(len(compiled["shot_plan"]["shots"]), 1)
        shot = compiled["shot_plan"]["shots"][0]
        self.assertEqual((shot["start_frame"], shot["end_frame"]), (0, 634))
        self.assertEqual(shot["dialogues"][0]["text"], "故事仍将继续。")
        self.assertIn("<d>[Chinese] 故事仍将继续。</d>", compiled["prompt"])
        self.assertIn("Merged zero-length trailing Qwen shot", compiled["warnings"][0])

    def test_normalizes_nonzero_first_shot_origin(self):
        value = self.result()
        value["shots"][0].update(start_frame=5, end_frame=633)
        value["shots"][1].update(start_frame=634, end_frame=1268)
        request = {**self.request(), "total_frames": 1268}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertEqual(
            [(shot["start_frame"], shot["end_frame"]) for shot in compiled["shot_plan"]["shots"]],
            [(0, 634), (634, 1268)],
        )
        self.assertIn("first shot start_frame 5", compiled["warnings"][0])

    def test_normalizes_one_based_inclusive_qwen_intervals(self):
        value = self.result()
        value["shots"][0].update(start_frame=1, end_frame=633)
        value["shots"][1].update(start_frame=634, end_frame=1268)
        request = {**self.request(), "total_frames": 1268}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertEqual(
            [(shot["start_frame"], shot["end_frame"]) for shot in compiled["shot_plan"]["shots"]],
            [(0, 633), (633, 1268)],
        )
        self.assertIn("1-based inclusive", compiled["warnings"][0])

    def test_strict_mode_preserves_identical_adjacent_camera_without_retry(self):
        value = self.result()
        value["shots"][1]["camera"] = value["shots"][0]["camera"]
        compiled = prompt_skill.compile_prompt_skill(value, self.request("strict"))
        self.assertEqual(
            [shot["camera"] for shot in compiled["shot_plan"]["shots"]],
            [value["shots"][0]["camera"], value["shots"][0]["camera"]],
        )
        self.assertTrue(any("preserve the same camera" in warning for warning in compiled["warnings"]))

    def test_shared_messages_are_identical_for_all_backends(self):
        request = self.request()
        request["image_count"] = 1
        self.assertEqual(prompt_skill.prompt_skill_messages(request), prompt_skill.prompt_skill_messages(request))
        self.assertTrue(callable(qwen35.prompt_skill_messages))
        self.assertTrue(callable(qwen36_38.prompt_skill_messages))

    def test_director_uses_configured_backend_for_independent_operation(self):
        captured = {}
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        def worker(payload, timeout=300):
            captured.update(payload)
            return types.SimpleNamespace(returncode=0, stderr="", stdout=""), {
                "ok": True, "prompt_skill_compile": compiled,
            }
        director = qwen35.Qwen35ContinuityDirector(
            Path("model.gguf"), Path("mmproj.gguf"), backend="qwen3.5", mtp_enabled=False
        )
        frame = __import__("torch").zeros(1, 8, 8, 3)
        with patch.object(qwen35, "_run_worker_once", side_effect=worker):
            result = director.compile_prompt_skill(self.request(), (frame,))
        self.assertEqual(captured["operation"], "prompt_skill_compile")
        self.assertEqual(captured["director_backend"], "qwen3.5")
        self.assertEqual(result["planned_frames"], 56)

    def test_builds_versioned_typed_plan_without_changing_legacy_outputs(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        typed_plan = prompt_skill.build_typed_prompt_plan(compiled, fps=24.0)
        self.assertEqual(typed_plan["type"], "HR_H3_PROMPT_PLAN")
        self.assertEqual(typed_plan["version"], 1)
        self.assertEqual(typed_plan["total_frames"], 56)
        self.assertEqual(typed_plan["shots"][0]["events"][0]["id"], "S1.V1")
        self.assertEqual(compiled["prompt"].splitlines()[0], "subject_definitions:")
        self.assertEqual(compiled["initial_event_ledger"]["pending"][0]["id"], "S1.V1")

    def test_typed_plan_copies_mutable_collections(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        typed_plan = prompt_skill.build_typed_prompt_plan(compiled, fps=24.0)
        typed_plan["shots"].append({})
        typed_plan["image_subjects"].append({})
        self.assertEqual(len(compiled["shot_plan"]["shots"]), 2)
        self.assertEqual(len(compiled["shot_plan"]["image_subjects"]), 1)

    def test_chunk_local_prompt_removes_future_subject_and_sound(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        typed_plan = prompt_skill.normalize_prompt_plan(
            prompt_skill.build_typed_prompt_plan(compiled, fps=24.0), fps=24.0, total_frames=56
        )
        current_chunk_prompt = compiled["prompt"].replace(
            "\n[Shot 2] At 00:00.917, Already inside, the hero opens the door once.", ""
        )
        localized = prompt_skill.localize_prompt_from_plan(
            current_chunk_prompt, typed_plan, frame_start=0, frame_end=22
        )
        self.assertIn("Hero", localized)
        self.assertIn("footsteps", localized)
        self.assertNotIn("opens the door", localized)
        self.assertNotIn("door creak", localized)

    def test_chunk_local_summary_restores_explicit_character_actor(self):
        plan = {
            "type": "HR_H3_PROMPT_PLAN", "version": 1, "fps": 24.0, "total_frames": 39,
            "image_subjects": [
                {"entity_id": "asset_2", "picture": 2, "subject": 2, "kind": "scene", "name": "梵心桃花林", "observable_features": "桃花林"},
                {"entity_id": "asset_3", "picture": 3, "subject": 3, "kind": "character", "name": "上官若彤", "observable_features": "淡紫色汉服"},
                {"entity_id": "asset_4", "picture": 4, "subject": 4, "kind": "character", "name": "上官若琳", "observable_features": "深红色汉服"},
            ],
            "shots": [{
                "start_frame": 0, "end_frame": 39, "pictures": [2, 3, 4], "camera": "中景缓慢跟拍",
                "start_state": "桃林深处", "end_state": "来到石台前", "forbidden_replays": [], "audio": "脚步声",
                "events": [{"id": "S1.V1", "actor": "asset_3", "action": "从桃林深处缓步走出，步伐轻盈，神情关切", "phase": "start", "start_frame": 0, "end_frame": 39}],
                "dialogues": [], "visual_description": "桃林中的人物走近石台。",
            }],
            "non_diegetic_music": "N/A",
        }
        localized = prompt_skill.localize_prompt_from_plan(
            "detailed_description:\nunused", plan, frame_start=0, frame_end=39,
        )
        expected = "<Subject 3> 上官若彤 从桃林深处缓步走出，步伐轻盈，神情关切"
        self.assertIn("summary:\n" + expected, localized)
        self.assertIn("[Shot 1] " + expected, localized)
        self.assertNotIn("summary:\n从桃林深处", localized)

    def test_prompt_plan_localization_preserves_global_picture_subject_contract(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        typed_plan = prompt_skill.build_typed_prompt_plan(compiled, fps=24.0)
        typed_plan["image_subjects"].append({
            "picture": 3, "subject": 3, "name": "Future Dragon", "observable_features": "red scales",
        })
        typed_plan["shots"][1]["pictures"] = [3]
        normalized = prompt_skill.normalize_prompt_plan(typed_plan, fps=24.0, total_frames=56)
        self.assertEqual(prompt_skill.active_prompt_plan_pictures(normalized, frame_start=22, frame_end=56), (3,))
        localized = prompt_skill.localize_prompt_from_plan(
            "detailed_description:\n[Shot 1] Future Dragon from <Picture 3> appears.\n\n"
            "overall_soundscape:\nroar\n\nnon_diegetic_music:\nN/A",
            normalized, frame_start=22, frame_end=56,
        )
        self.assertIn("Future Dragon", localized)
        self.assertIn("<Subject 3> is Future Dragon from <Picture 3>", localized)
        self.assertIn("<Subject 1> is Hero from <Picture 1>", localized)
        items = [
            {"kind": "image", "id": 1}, {"kind": "image", "id": 2},
            {"kind": "image", "id": 3}, {"kind": "video", "id": 4},
        ]
        filtered = prompt_skill.filter_prompt_plan_picture_items(items, (3,), kind_key="kind")
        self.assertEqual([item["id"] for item in filtered], [3, 4])

    def test_chunk_after_scripted_dialogue_ends_forbids_invented_speech(self):
        plan = {
            "fps": 24.0,
            "image_subjects": [
                {"entity_id": "asset_3", "picture": 3, "subject": 3, "kind": "character", "name": "上官若彤", "observable_features": "淡紫色汉服"},
                {"entity_id": "asset_4", "picture": 4, "subject": 4, "kind": "character", "name": "上官若琳", "observable_features": "深红色汉服"},
            ],
            "shots": [{
                "start_frame": 520, "end_frame": 804, "pictures": [3, 4], "camera": "static medium shot",
                "start_state": "the sisters face each other", "end_state": "they maintain eye contact",
                "forbidden_replays": [], "audio": "soft wind", "visual_description": "The sisters remain together.",
                "events": [{
                    "id": "S5.V1", "actor": "asset_4", "action": "speaks calmly to <Subject 3>, explaining her decision",
                    "phase": "complete", "start_frame": 520, "end_frame": 804,
                }],
                "dialogues": [{
                    "id": "S5.D1", "kind": "dialogue", "speaker": "<Subject 4>", "speaker_id": "S2",
                    "language": "Chinese", "text": "我已经说完了。", "delivery": "平静地",
                    "start_frame": 520, "end_frame": 770,
                }],
            }],
            "non_diegetic_music": "N/A",
        }
        localized = prompt_skill.localize_prompt_from_plan("", plan, frame_start=770, frame_end=804)
        self.assertNotIn("speaks calmly", localized)
        self.assertNotIn("<d>", localized)
        self.assertIn("All scripted dialogue has ended", localized)
        self.assertIn("Every character keeps their lips sealed with no mouth or jaw movement", localized)
        self.assertIn("subject_definitions are silent identity metadata", localized)
        self.assertIn("never pronounce subject names", localized)
        self.assertIn("<Subject 4> is the silent visual identity from <Picture 4>", localized)
        self.assertNotIn("上官若琳", localized)
        self.assertNotIn("lip movement", localized)
        self.assertIn("overall_soundscape:\nsoft wind", localized)

    def test_long_dialogue_is_sliced_once_across_physical_chunks(self):
        text = "姐姐自从比试之后这十年都没有闭关修炼这样真的来得及吗"
        plan = {
            "fps": 24.0,
            "image_subjects": [{"picture": 1, "subject": 1, "name": "Speaker", "observable_features": ""}],
            "shots": [{
                "start_frame": 0, "end_frame": 160, "pictures": [1], "camera": "locked medium shot",
                "start_state": "speaker already present", "end_state": "speaker finishes the line",
                "forbidden_replays": [], "audio": "衣袂随转身摩擦声", "visual_description": "The speaker turns once, then settles facing her sister.",
                "description": "unused full description", "dialogues": [{
                    "id": "S1.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
                    "language": "Chinese", "text": text, "delivery": "自然地",
                }],
            }],
            "non_diegetic_music": "N/A",
        }
        prompts = [
            prompt_skill.localize_prompt_from_plan("", plan, frame_start=start, frame_end=end)
            for start, end in ((0, 40), (40, 80), (80, 120), (120, 160))
        ]
        fragments = []
        for localized in prompts:
            fragments.extend(
                re.sub(r"^\s*\[[^\]]+\]\s*", "", match).replace("<scenetrans>", "").strip()
                for match in re.findall(r"<d>(.*?)</d>", localized, re.DOTALL)
            )
        self.assertEqual("".join(fragments), text)
        self.assertNotIn("without a cut, reframing, zoom", prompts[0])
        self.assertTrue(all("without a cut, reframing, zoom" in prompt for prompt in prompts[1:]))
        self.assertIn("turns once", prompts[0])
        self.assertTrue(all("turns once" not in prompt for prompt in prompts[1:]))
        self.assertIn("衣袂随转身摩擦声", prompts[0])
        self.assertTrue(all("衣袂随转身摩擦声" not in prompt for prompt in prompts[1:]))

    def test_event_timeline_advances_once_across_physical_chunks(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        plan = prompt_skill.build_typed_prompt_plan(compiled, fps=24.0)
        first = prompt_skill.project_prompt_plan_interval(plan, frame_start=0, frame_end=11)
        second = prompt_skill.project_prompt_plan_interval(plan, frame_start=11, frame_end=22)
        third = prompt_skill.project_prompt_plan_interval(plan, frame_start=22, frame_end=39)
        self.assertEqual([item["id"] for item in first["active"]], ["S1.V1"])
        self.assertEqual([item["interval_phase"] for item in first["active"]], ["start"])
        self.assertEqual([item["id"] for item in second["active"]], ["S1.V1"])
        self.assertEqual([item["interval_phase"] for item in second["active"]], ["complete"])
        self.assertEqual([item["id"] for item in third["completed"]], ["S1.V1"])
        self.assertNotIn("enters the temple", " ".join(third["forbidden"]))
        third_prompt = prompt_skill.localize_prompt_from_plan("", plan, frame_start=22, frame_end=39)
        self.assertNotIn("enters the temple", third_prompt)
        self.assertIn("Do not restart completed event S1.V1", third_prompt)
        self.assertIn("opens the door", third_prompt)

    def test_continuous_beat_does_not_claim_a_camera_cut(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        typed_plan = prompt_skill.build_typed_prompt_plan(compiled, fps=24.0)
        typed_plan["shots"][1]["cut"] = False
        normalized = prompt_skill.normalize_prompt_plan(typed_plan, fps=24.0, total_frames=56)
        segments = prompt_skill.prompt_plan_shots(normalized)
        self.assertTrue(segments[0][4])
        self.assertFalse(segments[1][4])

    def test_prompt_plan_rejects_latent_length_mismatch(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        with self.assertRaisesRegex(ValueError, "total_frames"):
            prompt_skill.normalize_prompt_plan(
                prompt_skill.build_typed_prompt_plan(compiled, fps=24.0), fps=24.0, total_frames=73
            )


if __name__ == "__main__":
    unittest.main()
