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

    def test_source_asset_names_use_ascii_alphanumeric_only(self):
        request = prompt_skill.build_prompt_skill_request(
            "<Picture 1> is Luxury Bedroom!; <Picture 2> is 豪华寝宫。",
            duration_seconds=2.0, fps=24.0, image_count=2, style="cinematic",
            shot_density="low", continuity_mode="balanced", prompt_lang="en",
        )
        self.assertEqual(
            [item["name"] for item in request["source_image_contract"]],
            ["LuxuryBedroom", "Asset2"],
        )
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9]+", item["name"]) for item in request["source_image_contract"]))

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

    def test_same_shot_fragments_merge_without_scene_transition(self):
        shots = [
            {"dialogues": [
                {"speaker": "<Subject 1>", "speaker_id": "S1", "kind": "dialogue", "language": "Chinese", "delivery": "calm", "text": "前半句", "continues_from_previous": True, "continues_to_next": True},
                {"speaker": "<Subject 1>", "speaker_id": "S1", "kind": "dialogue", "language": "Chinese", "delivery": "calm", "text": "后半句。", "continues_from_previous": True, "continues_to_next": True},
            ]},
        ]
        warnings = prompt_skill._normalize_cross_shot_dialogue_flags(shots)
        self.assertEqual(len(shots[0]["dialogues"]), 1)
        self.assertEqual(shots[0]["dialogues"][0]["text"], "前半句后半句。")
        self.assertFalse(shots[0]["dialogues"][0]["continues_from_previous"])
        self.assertFalse(shots[0]["dialogues"][0]["continues_to_next"])
        self.assertTrue(any("real cuts" in warning for warning in warnings))

    def test_final_shot_never_carries_dialogue_to_nonexistent_next_shot(self):
        shots = [
            {"dialogues": [{"text": "第一段", "continues_from_previous": False, "continues_to_next": True}]},
            {"dialogues": [{"text": "最后一段", "continues_from_previous": True, "continues_to_next": True}]},
        ]
        prompt_skill._normalize_cross_shot_dialogue_flags(shots)
        self.assertTrue(shots[0]["dialogues"][-1]["continues_to_next"])
        self.assertTrue(shots[1]["dialogues"][0]["continues_from_previous"])
        self.assertFalse(shots[1]["dialogues"][-1]["continues_to_next"])

    def test_moving_camera_drops_contradictory_static_wording(self):
        camera, warning = prompt_skill._normalize_camera_contract(
            "Medium wide shot, eye-level, static. The camera pulls back slightly"
        )
        self.assertNotIn("static", camera.casefold())
        self.assertIn("pulls back", camera.casefold())
        self.assertIn("contradictory", warning)

    def test_dialogue_does_not_move_before_speaker_first_visible_shot(self):
        story = '<Subject 1> (S1) says: <d>[Chinese] 现在开始说话。</d>'
        request = prompt_skill.build_prompt_skill_request(
            story, duration_seconds=8.0, fps=24.0, image_count=1, style="cinematic",
            shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        value = self.result()
        value["shots"][0].update(end_frame=48, events=[], dialogues=[])
        value["shots"][1].update(
            start_frame=48, end_frame=request["total_frames"],
            start_state="asset_1 enters and becomes visible",
            events=[{"id": "S2.V1", "actor": "asset_1", "action": "enters", "phase": "start"}],
            dialogues=[{"id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1", "language": "Chinese", "text": "现在开始说话。", "delivery": "自然地"}],
        )
        normalized, _warnings = prompt_skill._redistribute_dialogues(value, request)
        self.assertEqual(normalized["shots"][0]["dialogues"], [])
        self.assertEqual(normalized["shots"][1]["dialogues"][0]["text"], "现在开始说话。")

    def test_timeline_extends_when_visible_lead_reduces_dialogue_capacity(self):
        story = '<Subject 1> (S1) says: <d>[Chinese] 这是一句需要完整自然说完而且不能提前开始的对白。</d>'
        request = prompt_skill.build_prompt_skill_request(
            story, duration_seconds=2.0, fps=24.0, image_count=1, style="cinematic",
            shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        old_total = request["total_frames"]
        value = self.result()
        lead_end = old_total - 17
        value["shots"][0].update(end_frame=lead_end, events=[], dialogues=[])
        value["shots"][1].update(
            start_frame=lead_end, end_frame=old_total,
            start_state="asset_1 enters and becomes visible",
            events=[{"id": "S2.V1", "actor": "asset_1", "action": "enters", "phase": "start"}],
            dialogues=[{"id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1", "language": "Chinese", "text": request["required_spoken_lines"][0], "delivery": "自然地"}],
        )
        compiled = prompt_skill.compile_prompt_skill(value, request)
        plan = compiled["shot_plan"]
        self.assertGreater(request["total_frames"], old_total)
        self.assertEqual(request["duration_source"], "dialogue_plus_visual_lead")
        self.assertEqual(plan["shots"][-1]["end_frame"], request["total_frames"])
        self.assertEqual(compiled["planned_frames"], request["total_frames"])
        self.assertEqual("".join(item["text"] for shot in plan["shots"] for item in shot["dialogues"]), request["required_spoken_lines"][0])
        self.assertTrue(any("Extended the H3 timeline" in warning for warning in compiled["warnings"]))

    def test_semantic_dialogue_cut_prefers_nearby_punctuation(self):
        text = "再闭关苦修已是无用。与其毫无头绪的闭关，不如继续。"
        raw_cut = text.index("绪")
        cut = prompt_skill._dialogue_semantic_cut(text, raw_cut)
        self.assertEqual(text[:cut], "再闭关苦修已是无用。")
        self.assertTrue(text[cut:].startswith("与其毫无头绪"))

    def test_extended_timeline_spreads_frames_across_remaining_shots(self):
        shots = [
            {"start_frame": 0, "end_frame": 100},
            {"start_frame": 100, "end_frame": 200},
            {"start_frame": 200, "end_frame": 300},
        ]
        normalized = [dict(shot) for shot in shots]
        added = prompt_skill._extend_shot_intervals(shots, normalized, 1, 80)
        self.assertGreater(added, 0)
        self.assertGreater(shots[1]["end_frame"] - shots[1]["start_frame"], 100)
        self.assertGreater(shots[2]["end_frame"] - shots[2]["start_frame"], 100)
        self.assertEqual(shots[1]["end_frame"], shots[2]["start_frame"])
        self.assertEqual((shots[-1]["end_frame"] - 5) % 17, 0)

    def test_semantic_split_discarded_capacity_gets_minimal_final_extension(self):
        text = "姐姐自从比试之后这十年都没有闭关修炼这样真的来得及吗"
        request = prompt_skill.build_prompt_skill_request(
            f'<Subject 1> (S1) says: <d>[Chinese] {text}</d>', duration_seconds=2.0, fps=24.0,
            image_count=1, style="cinematic", shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        total = request["total_frames"]
        boundaries = [round(total * index / 8) for index in range(9)]
        shots = []
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            shots.append({
                "start_frame": start, "end_frame": end, "pictures": ["asset_1"],
                "camera": "static", "start_state": "asset_1 visible", "end_state": "asset_1 visible",
                "events": [], "forbidden_replays": [], "audio": "room tone", "description": "asset_1 remains visible",
                "dialogues": ([{"id": f"S{index}.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1", "language": "Chinese", "text": text, "delivery": "自然地"}] if index == 1 else []),
            })
        value = {"shots": shots}
        with patch.object(prompt_skill, "_dialogue_semantic_cut", side_effect=lambda _text, cut: min(2, cut)):
            normalized, warnings = prompt_skill._redistribute_dialogues(value, request)
        spoken = "".join(item["text"] for shot in normalized["shots"] for item in shot["dialogues"])
        self.assertEqual(spoken, text)
        self.assertTrue(any("Extended the H3 timeline" in warning for warning in warnings))
        self.assertEqual(normalized["shots"][-1]["end_frame"], request["total_frames"])

    def test_dialogue_avoids_two_character_fragment_at_shot_boundary(self):
        story = '<Subject 1> (S1) says: <d>[Chinese] 达到顶峰。</d>'
        request = prompt_skill.build_prompt_skill_request(
            story, duration_seconds=3.0, fps=24.0, image_count=1, style="cinematic",
            shot_density="medium", continuity_mode="balanced", prompt_lang="zh",
        )
        value = self.result()
        value["shots"][0].update(end_frame=12, dialogues=[])
        value["shots"][1].update(
            start_frame=12, end_frame=request["total_frames"],
            dialogues=[{"id": "S2.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1", "language": "Chinese", "text": "达到顶峰。", "delivery": "自然地"}],
        )
        normalized, _warnings = prompt_skill._redistribute_dialogues(value, request)
        self.assertEqual(normalized["shots"][0]["dialogues"], [])
        self.assertEqual(normalized["shots"][1]["dialogues"][0]["text"], "达到顶峰。")

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

    def test_model_invented_music_is_removed_without_story_music_request(self):
        value = self.result()
        value["non_diegetic_music"] = "A soft flute and guzheng score."
        value["shots"][0]["audio"] = "Wind, ambient chimes, footsteps"
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        self.assertIn("non_diegetic_music:\nN/A", compiled["prompt"])
        self.assertNotIn("ambient chimes", compiled["prompt"])
        self.assertTrue(any("Removed model-invented non-diegetic music" in warning for warning in compiled["warnings"]))

    def test_explicit_story_music_request_is_preserved(self):
        request = prompt_skill.build_prompt_skill_request(
            "A hero enters while background music plays.", duration_seconds=2.0, fps=24.0,
            image_count=1, style="cinematic", shot_density="medium", continuity_mode="balanced", prompt_lang="en",
        )
        value = self.result()
        value["non_diegetic_music"] = "A restrained flute score at slow tempo."
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertIn("A restrained flute score at slow tempo.", compiled["prompt"])

    def test_observable_features_resolve_internal_asset_ids(self):
        value = self.result()
        value["image_subjects"][0]["observable_features"] = "similar facial features to asset_1 with black hair"
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        self.assertNotRegex(compiled["prompt"], r"\basset_\d+\b")
        self.assertIn("similar facial features to <Subject 1>", compiled["prompt"])

    def test_cross_shot_dialogue_uses_h3_scene_transition_contract(self):
        description = prompt_skill._dialogue_description({
            "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
            "language": "Chinese", "text": "继续说话", "delivery": "自然地",
            "continues_from_previous": True, "continues_to_next": True,
        })
        self.assertEqual(description.count("<scenetrans>"), 2)
        self.assertIn("<d>[Chinese] <scenetrans> 继续说话</d>", description)
        self.assertRegex(description, r"</d> <scenetrans> with synchronized visible lip movement;.*The same voice")
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

    def test_restores_terminal_punctuation_in_immutable_source_name(self):
        request = {
            **self.request(), "image_count": 2,
            "source_image_contract": [
                {"entity_id": "asset_1", "picture": 1, "name": "Hero", "kind": "character"},
                {"entity_id": "asset_2", "picture": 2, "name": "豪华寝宫，寝宫中有一张豪华大床。", "kind": "scene"},
            ],
        }
        value = self.result()
        value["image_subjects"] = [
            {"entity_id": "asset_1", "kind": "character", "name": "Hero", "observable_features": "black hair"},
            {"entity_id": "asset_2", "kind": "scene", "name": "豪华寝宫，寝宫中有一张豪华大床", "observable_features": "large bed"},
        ]
        for shot in value["shots"]:
            shot["pictures"] = ["asset_1", "asset_2"]
        compiled = prompt_skill.compile_prompt_skill(value, request)
        subject = compiled["shot_plan"]["image_subjects"][1]
        self.assertEqual(subject["name"], "豪华寝宫，寝宫中有一张豪华大床。")
        self.assertTrue(any("Ignored Qwen name output for immutable asset_2" in item for item in compiled["warnings"]))

    def test_ignores_qwen_renaming_and_restores_ascii_source_name(self):
        request = self.request()
        request["source_image_contract"][0]["name"] = "LuxuryBedroom"
        value = self.result()
        value["image_subjects"][0]["name"] = "豪华寝宫，寝宫中有一张豪华大床。"
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertEqual(compiled["shot_plan"]["image_subjects"][0]["name"], "LuxuryBedroom")
        self.assertTrue(any("Ignored Qwen name output for immutable asset_1" in item for item in compiled["warnings"]))

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

    def test_normalizes_known_model_owned_subject_labels_before_h3_compilation(self):
        value = self.result()
        value["shots"][0]["events"][0]["action"] = "<Subject 1> enters once"
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        event = compiled["shot_plan"]["shots"][0]["events"][0]
        self.assertIn("<Subject 1>", event["action"])
        self.assertTrue(any("Normalized compiler-owned <Subject 1>" in item for item in compiled["warnings"]))

    def test_rejects_unknown_model_owned_subject_labels_before_h3_compilation(self):
        value = self.result()
        value["shots"][0]["events"][0]["action"] = "<Subject 9> enters once"
        with self.assertRaisesRegex(ValueError, "unknown compiler-owned Subject/Picture 9"):
            prompt_skill.compile_prompt_skill(value, self.request())

    def test_allows_scene_entity_as_visual_event_actor(self):
        story = "<Picture 1> is a stone temple; <Picture 2> is Hero waiting silently inside it."
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
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertEqual(compiled["shot_plan"]["image_subjects"][0]["kind"], "scene")
        self.assertEqual(compiled["shot_plan"]["shots"][0]["events"][0]["actor"], "asset_1")

    def test_removes_model_invented_dialogue_when_source_story_has_none(self):
        request = self.request()
        self.assertEqual(request["required_spoken_lines"], [])
        value = self.result()
        value["shots"][0]["dialogues"] = [{
            "id": "S1.D1", "kind": "dialogue", "speaker": "asset_1", "speaker_id": "S1",
            "language": "English", "text": "Invented words.", "delivery": "quietly",
        }]
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertEqual(compiled["shot_plan"]["shots"][0]["dialogues"], [])
        self.assertNotIn("Invented words", compiled["prompt"])
        self.assertTrue(any("Removed 1 model-invented dialogue" in item for item in compiled["warnings"]))

    def test_ignores_format_control_characters_in_empty_dialogue_tags(self):
        request = prompt_skill.build_prompt_skill_request(
            "A silent room. <d>[Chinese] \u200b</d>", duration_seconds=2.0, fps=24.0,
            image_count=1, style="cinematic", shot_density="low",
            continuity_mode="balanced", prompt_lang="en",
        )
        self.assertEqual(request["required_spoken_lines"], [])
        self.assertEqual(request["minimum_spoken_duration_seconds"], 0.0)

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

    def test_normalizes_qwen_dialogue_kind_aliases(self):
        aliases = {
            "conversation": "dialogue", "对白": "dialogue",
            "soliloquy": "monologue", "独白": "monologue",
            "voice-over": "voiceover", "旁白": "voiceover", "inner_monologue": "voiceover",
        }
        for supplied, expected in aliases.items():
            with self.subTest(supplied=supplied):
                value = self.result()
                value["shots"][0]["dialogues"] = [{
                    "id": "S1.D1", "kind": supplied, "speaker": "asset_1", "speaker_id": "S1",
                    "language": "Chinese", "text": "测试台词。", "delivery": "平静地",
                }]
                request = {**self.request(), "required_spoken_lines": ["测试台词。"]}
                compiled = prompt_skill.compile_prompt_skill(value, request)
                self.assertEqual(compiled["shot_plan"]["shots"][0]["dialogues"][0]["kind"], expected)

    def test_rejects_unknown_dialogue_kind_with_actual_value(self):
        value = self.result()
        value["shots"][0]["dialogues"] = [{
            "id": "S1.D1", "kind": "song_lyrics", "speaker": "asset_1", "speaker_id": "S1",
            "language": "Chinese", "text": "测试台词。", "delivery": "平静地",
        }]
        request = {**self.request(), "required_spoken_lines": ["测试台词。"]}
        with self.assertRaisesRegex(ValueError, "invalid dialogue kind: 'song_lyrics'"):
            prompt_skill.compile_prompt_skill(value, request)

    def test_rejects_non_character_dialogue_entity_instead_of_guessing(self):
        value = self.result()
        value["image_subjects"][0]["kind"] = "scene"
        value["shots"][1]["dialogues"] = [{
            "id": "S2.D1", "kind": "dialogue", "speaker": "asset_1", "speaker_id": "S1",
            "language": "Chinese", "text": "继续说话。", "delivery": "平静地",
        }]
        request = {**self.request(), "required_spoken_lines": ["继续说话。"]}
        with self.assertRaisesRegex(ValueError, "dialogue speaker asset_1 is not a character"):
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
            {"entity_id": "asset_1", "picture": 1, "name": "Asset1", "kind": None},
            {"entity_id": "asset_2", "picture": 2, "name": "Asset2", "kind": None},
            {"entity_id": "asset_3", "picture": 3, "name": "Asset3", "kind": "character"},
            {"entity_id": "asset_4", "picture": 4, "name": "Asset4", "kind": "character"},
        ])
        system, user = prompt_skill.prompt_skill_messages(request)
        self.assertIn("Those H3 labels are private compiler output", system)
        self.assertIn("entity_id=asset_1: immutable source name='Asset1'", user)
        self.assertIn("entity_id=asset_3: immutable source name='Asset3'; kind=character", user)

        value = self.result()
        value["image_subjects"] = [
            {"entity_id": "asset_1", "kind": "character", "name": "Wrong woman", "observable_features": "wrong purple robe"},
            {"entity_id": "asset_2", "kind": "character", "name": "梵心桃花林", "observable_features": "pink blossoms"},
            {"entity_id": "asset_3", "kind": "scene", "name": "上官若彤", "observable_features": "purple robe"},
            {"entity_id": "asset_4", "kind": "prop", "name": "上官若琳", "observable_features": "red robe"},
        ]
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertEqual(
            [item["name"] for item in compiled["shot_plan"]["image_subjects"]],
            ["Asset1", "Asset2", "Asset3", "Asset4"],
        )
        self.assertTrue(any("Ignored Qwen name output for immutable asset_1" in item for item in compiled["warnings"]))

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
        self.assertIn("Merged trailing Qwen shot", compiled["warnings"][0])

    def test_merges_positive_length_shot_starting_at_target_end(self):
        value = self.result()
        value["shots"][0].update(start_frame=0, end_frame=804)
        value["shots"][1].update(
            start_frame=804, end_frame=1008,
            events=[{"id": "S6.V1", "actor": "asset_1", "action": "holds the final pose", "phase": "complete"}],
            dialogues=[
                {"id": "S6.D1", "kind": "voiceover", "speaker": "<Subject 1>", "speaker_id": "S1",
                 "language": "Chinese", "text": "故事仍将继续。", "delivery": "平静地"},
            ],
        )
        request = {**self.request(), "total_frames": 804, "required_spoken_lines": ["故事仍将继续。"]}
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertEqual(len(compiled["shot_plan"]["shots"]), 1)
        shot = compiled["shot_plan"]["shots"][0]
        self.assertEqual((shot["start_frame"], shot["end_frame"]), (0, 804))
        self.assertEqual(shot["dialogues"][0]["text"], "故事仍将继续。")
        self.assertTrue(any(event["action"] == "holds the final pose" for event in shot["events"]))
        self.assertIn("beginning at or beyond target frame 804", compiled["warnings"][0])

    def test_rejects_out_of_range_nonterminal_shot(self):
        value = self.result()
        value["shots"][0].update(start_frame=804, end_frame=1008)
        value["shots"][1].update(start_frame=0, end_frame=804)
        request = {**self.request(), "total_frames": 804}
        with self.assertRaisesRegex(ValueError, "Shot 1 has invalid or non-contiguous"):
            prompt_skill.compile_prompt_skill(value, request)

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

    def test_prompt_plan_records_normalized_sampler_chunk_geometry(self):
        request = prompt_skill.build_prompt_skill_request(
            "A hero waits.", duration_seconds=2.0, fps=24.0, image_count=1,
            style="cinematic", shot_density="low", continuity_mode="balanced",
            prompt_lang="en", chunk_frames=60,
        )
        self.assertEqual(request["chunk_frames"], 56)
        self.assertEqual(request["retained_chunk_frames"], 51)
        compiled = prompt_skill.compile_prompt_skill(self.result(), request)
        typed = prompt_skill.build_typed_prompt_plan(compiled, fps=24.0, chunk_frames=60)
        self.assertEqual(typed["chunk_frames"], 56)
        self.assertEqual(typed["retained_chunk_frames"], 51)
        prompt_skill.normalize_prompt_plan(
            typed, fps=24.0, total_frames=typed["total_frames"], chunk_frames=56
        )
        with self.assertRaisesRegex(ValueError, "chunk_frames does not match"):
            prompt_skill.normalize_prompt_plan(
                typed, fps=24.0, total_frames=typed["total_frames"], chunk_frames=39
            )

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

    def test_prompt_plan_edit_allows_camera_and_event_changes(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        original = prompt_skill.normalize_prompt_plan(
            prompt_skill.build_typed_prompt_plan(compiled, fps=24.0), fps=24.0, total_frames=56
        )
        edited = {**original, "shots": [dict(shot) for shot in original["shots"]]}
        edited["shots"][0]["camera"] = "locked profile two-shot"
        edited["shots"][0]["events"] = [dict(event, action="walks inside once") for event in edited["shots"][0]["events"]]
        prompt_skill.validate_prompt_plan_edit(original, edited)

    def test_stale_editor_timeline_rebases_to_connected_compiler_plan(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        original = prompt_skill.normalize_prompt_plan(
            prompt_skill.build_typed_prompt_plan(compiled, fps=24.0), fps=24.0, total_frames=56
        )
        current = {**original, "total_frames": 73, "shots": [dict(shot) for shot in original["shots"]]}
        current["shots"][-1]["end_frame"] = 73
        current = prompt_skill.normalize_prompt_plan(current, fps=24.0, total_frames=73)
        stale = {**original, "shots": [dict(shot) for shot in original["shots"]]}
        stale["shots"][0]["camera"] = "manually edited locked camera"
        rebased, warnings = prompt_skill.rebase_prompt_plan_timeline(current, stale)
        normalized = prompt_skill.normalize_prompt_plan(rebased, fps=24.0, total_frames=73)
        self.assertEqual(normalized["total_frames"], 73)
        self.assertEqual(normalized["shots"][-1]["end_frame"], 73)
        self.assertEqual(normalized["shots"][0]["camera"], "manually edited locked camera")
        self.assertTrue(any("Rebased stale editor timeline" in item for item in warnings))

    def test_stale_editor_with_changed_shot_count_uses_current_plan(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        original = prompt_skill.normalize_prompt_plan(
            prompt_skill.build_typed_prompt_plan(compiled, fps=24.0), fps=24.0, total_frames=56
        )
        stale = {**original, "total_frames": 39, "shots": original["shots"][:1]}
        rebased, warnings = prompt_skill.rebase_prompt_plan_timeline(original, stale)
        self.assertEqual(rebased, original)
        self.assertTrue(any("Replaced stale editor JSON" in item for item in warnings))

    def test_prompt_plan_edit_rejects_dialogue_or_identity_changes(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        original = prompt_skill.normalize_prompt_plan(
            prompt_skill.build_typed_prompt_plan(compiled, fps=24.0), fps=24.0, total_frames=56
        )
        renamed = {**original, "image_subjects": [dict(item) for item in original["image_subjects"]]}
        renamed["image_subjects"][0].update(name="旧中文名称", entity_id="old_asset", kind="scene")
        rebased, warnings = prompt_skill.rebase_prompt_plan_edit(original, renamed)
        self.assertEqual(rebased["image_subjects"], original["image_subjects"])
        self.assertTrue(any("Restored immutable Picture/Subject identity" in item for item in warnings))
        prompt_skill.validate_prompt_plan_edit(original, rebased)

        missing = {**original, "image_subjects": []}
        with self.assertRaisesRegex(ValueError, "complete Picture identity set"):
            prompt_skill.rebase_prompt_plan_edit(original, missing)

        reassigned = {**original, "shots": [dict(shot) for shot in original["shots"]]}
        reassigned["shots"][0] = {
            **reassigned["shots"][0],
            "events": [dict(reassigned["shots"][0]["events"][0], id="OLD.V9", actor="old_asset", action="manual action")],
        }
        event_rebased, event_warnings = prompt_skill.rebase_prompt_plan_edit(original, reassigned)
        self.assertEqual(event_rebased["shots"][0]["events"][0]["id"], original["shots"][0]["events"][0]["id"])
        self.assertEqual(event_rebased["shots"][0]["events"][0]["actor"], original["shots"][0]["events"][0]["actor"])
        self.assertEqual(event_rebased["shots"][0]["events"][0]["action"], "manual action")
        self.assertTrue(any("Restored event IDs and actors" in item for item in event_warnings))
        prompt_skill.validate_prompt_plan_edit(original, event_rebased)

        spoken = self.result()
        spoken["shots"][0]["dialogues"] = [{
            "id": "S1.D1", "kind": "dialogue", "speaker": "asset_1", "speaker_id": "S1",
            "language": "English", "text": "Stay.", "delivery": "quietly",
        }]
        spoken_request = {**self.request(), "required_spoken_lines": ["Stay."]}
        spoken_compiled = prompt_skill.compile_prompt_skill(spoken, spoken_request)
        spoken_original = prompt_skill.normalize_prompt_plan(
            prompt_skill.build_typed_prompt_plan(spoken_compiled, fps=24.0), fps=24.0, total_frames=56
        )
        changed = {**spoken_original, "shots": [dict(shot) for shot in spoken_original["shots"]]}
        changed["shots"][0] = {
            **changed["shots"][0],
            "dialogues": [dict(changed["shots"][0]["dialogues"][0], text="Go.")],
        }
        with self.assertRaisesRegex(ValueError, "preserve dialogue IDs"):
            prompt_skill.validate_prompt_plan_edit(spoken_original, changed)

    def test_prompt_plan_rejects_nested_interval_outside_shot(self):
        compiled = prompt_skill.compile_prompt_skill(self.result(), self.request())
        plan = prompt_skill.build_typed_prompt_plan(compiled, fps=24.0)
        plan["shots"][0]["events"][0]["end_frame"] = plan["shots"][0]["end_frame"] + 1
        with self.assertRaisesRegex(ValueError, r"events\[1\] has invalid interval"):
            prompt_skill.normalize_prompt_plan(plan, fps=24.0, total_frames=56)

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
        self.assertNotIn("Hero", localized)
        self.assertIn("Footsteps", localized)
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
        expected = "<Subject 3> 从桃林深处缓步走出，步伐轻盈，神情关切"
        self.assertIn("summary:\n[video continuation + reference generation] The target chunk continues from <Video 1>", localized)
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
        self.assertNotIn("Future Dragon", localized)
        self.assertIn("<Subject 3> is the visible environment or object defined by <Picture 3>", localized)
        self.assertNotIn("<Subject 1>", localized)
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
        self.assertIn("must never be spoken aloud", localized)
        self.assertIn("<Subject 4> is the character defined by <Picture 4>", localized)
        self.assertNotIn("上官若琳", localized)
        self.assertNotIn("lip movement", localized)
        self.assertIn("overall_soundscape:\nSoft wind.", localized)

    def test_localized_soundscape_removes_voice_instructions_and_keeps_ambience(self):
        plan = {
            "fps": 24.0,
            "image_subjects": [{"picture": 1, "subject": 1, "name": "Speaker", "observable_features": ""}],
            "shots": [{
                "start_frame": 0, "end_frame": 39, "pictures": [1], "camera": "static medium shot",
                "start_state": "speaker present", "end_state": "speaker present", "forbidden_replays": [],
                "audio": "Voice of <Subject 1> is calm and clear, soft wind, distant birds",
                "visual_description": "The speaker remains still.", "events": [], "dialogues": [],
            }],
            "overall_soundscape": "Dialogue, forest ambience",
            "non_diegetic_music": "N/A",
        }
        localized = prompt_skill.localize_prompt_from_plan("", plan, frame_start=0, frame_end=39)
        self.assertIn("overall_soundscape:\nSoft wind, distant birds.", localized)
        self.assertNotIn("Voice of", localized)
        self.assertNotIn("Dialogue", localized)

    def test_localized_soundscape_inherits_nonverbal_bed_instead_of_na(self):
        plan = {
            "fps": 24.0,
            "image_subjects": [{"picture": 1, "subject": 1, "name": "Speaker", "observable_features": ""}],
            "shots": [{
                "start_frame": 0, "end_frame": 39, "pictures": [1], "camera": "static medium shot",
                "start_state": "speaker present", "end_state": "speaker present", "forbidden_replays": [],
                "audio": "Dialogue", "visual_description": "The speaker remains still.", "events": [], "dialogues": [],
            }],
            "overall_soundscape": "steady forest ambience",
            "non_diegetic_music": "N/A",
        }
        localized = prompt_skill.localize_prompt_from_plan("", plan, frame_start=0, frame_end=39)
        self.assertIn("overall_soundscape:\nSteady forest ambience.", localized)
        self.assertNotIn("overall_soundscape:\nN/A", localized)

    def test_first_spoken_fragment_establishes_stable_voice_profile(self):
        description = prompt_skill._dialogue_description({
            "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1",
            "language": "Chinese", "text": "你好。", "delivery": "calmly",
        })
        self.assertIn("stable voice identity", description)
        self.assertIn("consistent timbre, pitch, cadence, and speaking rate", description)

    def test_removes_forbidden_appearance_rule_that_conflicts_with_visible_speaker(self):
        request = prompt_skill.build_prompt_skill_request(
            '<Subject 1> (S1) says: <d>[English] Wait.</d>', duration_seconds=2.0, fps=24.0,
            image_count=1, style="cinematic", shot_density="medium", continuity_mode="balanced", prompt_lang="en",
        )
        value = self.result()
        value["shots"][0]["dialogues"] = [{
            "id": "S1.D1", "kind": "dialogue", "speaker": "asset_1", "speaker_id": "S1",
            "language": "English", "text": "Wait.", "delivery": "quietly",
        }]
        value["shots"][0]["forbidden_replays"] = ["asset_1 appearing in this shot", "repeating the entrance"]
        compiled = prompt_skill.compile_prompt_skill(value, request)
        self.assertNotIn("appearing in this shot", compiled["prompt"])
        self.assertIn("repeating the entrance", compiled["prompt"])
        self.assertTrue(any("Removed a contradictory forbidden-replay rule" in warning for warning in compiled["warnings"]))

    def test_compiler_removes_internal_asset_ids_from_h3_text(self):
        value = self.result()
        value["shots"][1]["events"][0]["action"] = "asset_1 opens the door and takes asset_1's key"
        value["shots"][1]["forbidden_replays"] = ["asset_1 entering again"]
        compiled = prompt_skill.compile_prompt_skill(value, self.request())
        self.assertNotRegex(compiled["prompt"], r"\basset_\d+\b")
        self.assertIn("<Subject 1> opens the door", compiled["prompt"])

    def test_chunk_reference_normalization_preserves_dialogue_and_rejects_unknown_ids(self):
        plan = {
            "image_subjects": [
                {"entity_id": "asset_3", "picture": 3, "subject": 3},
                {"entity_id": "asset_4", "picture": 4, "subject": 4},
            ],
        }
        prompt = (
            "Asset3 approaches asset_4 while <Subject 4> watches. "
            "<d>[English] Keep Asset3 exactly as spoken.</d> unknown asset_9."
        )
        normalized = prompt_skill.normalize_h3_chunk_references(prompt, plan)
        self.assertIn("<Subject 3> approaches <Subject 4>", normalized)
        self.assertIn("<d>[English] Keep Asset3 exactly as spoken.</d>", normalized)
        self.assertIn("asset_9", normalized)

    def test_chunk_transition_normalization_removes_false_scene_transition(self):
        plan = {
            "shots": [
                {"start_frame": 0, "end_frame": 100},
                {"start_frame": 100, "end_frame": 200},
            ],
        }
        prompt = (
            "<Subject 1> (S1) carries over from the previous shot: "
            "<d>[Chinese] <scenetrans> 继续说话。</d> The same voice and utterance continues seamlessly across the cut."
        )
        normalized = prompt_skill.normalize_h3_chunk_transitions(
            prompt, plan, frame_start=120, frame_end=160
        )
        self.assertNotIn("<scenetrans>", normalized)
        self.assertNotIn("across the cut", normalized)
        self.assertIn("<d>[Chinese] 继续说话。</d>", normalized)
        crossing = prompt_skill.normalize_h3_chunk_transitions(
            prompt, plan, frame_start=80, frame_end=120
        )
        self.assertIn("<scenetrans>", crossing)

    def test_chunk_retention_normalization_removes_execution_text_and_rebuilds_active_subjects(self):
        plan = {
            "shots": [{"start_frame": 0, "end_frame": 80, "pictures": [2, 3], "dialogues": []}],
            "image_subjects": [
                {"picture": 2, "subject": 2},
                {"picture": 3, "subject": 3},
                {"picture": 4, "subject": 4},
            ],
        }
        prompt = (
            "subject_definitions:\nNone.\n\nsummary:\nsummary.\n\n"
            "retention_analysis:\nCamera contract: portrait close-up.\nCurrent state: standing.\n"
            "Forbidden replay: entering again.\n<Video 1>: fully_preserved - continuation source.\n\n"
            "detailed_description:\n[Shot 1] Continue.\n\noverall_soundscape:\nSoft wind.\n\n"
            "non_diegetic_music:\nN/A"
        )
        normalized = prompt_skill.normalize_h3_chunk_retention(prompt, plan, frame_start=0, frame_end=40)
        retention = normalized.split("retention_analysis:\n", 1)[1].split("\n\ndetailed_description:", 1)[0]
        self.assertNotIn("Camera contract", retention)
        self.assertNotIn("Current state", retention)
        self.assertNotIn("Forbidden replay", retention)
        self.assertIn("<Video 1>: fully_preserved", retention)
        self.assertIn("<Subject 2>: fully_preserved", retention)
        self.assertIn("<Subject 3>: fully_preserved", retention)
        self.assertNotIn("<Subject 4>", retention)
        prompt_skill.validate_h3_chunk_prompt(normalized, plan, frame_start=0, frame_end=40)

    def test_chunk_prompt_validator_rejects_private_ids_and_false_scene_transitions(self):
        plan = {
            "shots": [{"start_frame": 0, "end_frame": 80, "dialogues": []}],
            "image_subjects": [],
        }
        prompt = (
            "subject_definitions:\nNone.\n\nsummary:\n[video continuation] asset_1 continues.\n\n"
            "retention_analysis:\n<Video 1>: fully_preserved - continuation.\n\n"
            "detailed_description:\n[Shot 1] <d>[Chinese] <scenetrans> 继续。</d>\n\n"
            "overall_soundscape:\nSoft wind.\n\nnon_diegetic_music:\nN/A"
        )
        with self.assertRaisesRegex(ValueError, "unmapped private identifier near:.*asset_1"):
            prompt_skill.validate_h3_chunk_prompt(prompt, plan, frame_start=0, frame_end=40)
        prompt = prompt.replace("asset_1", "<Subject 1>")
        with self.assertRaisesRegex(ValueError, "without a real semantic shot cut"):
            prompt_skill.validate_h3_chunk_prompt(prompt, plan, frame_start=0, frame_end=40)

        dialogue_prompt = prompt.replace("<scenetrans> 继续。", "用户逐字说 Asset3。")
        prompt_skill.validate_h3_chunk_prompt(dialogue_prompt, plan, frame_start=0, frame_end=40)

    def test_localized_event_action_removes_internal_asset_ids(self):
        subjects = {
            "asset_3": {"subject": 3, "kind": "character", "name": "A"},
            "asset_4": {"subject": 4, "kind": "character", "name": "B"},
        }
        action = prompt_skill._localized_event_action(
            {"actor": "asset_4", "action": "asset_4 takes asset_3's hand"}, subjects
        )
        self.assertNotRegex(action, r"\basset_\d+\b")
        self.assertIn("<Subject 4> takes <Subject 3>'s hand", action)

    def test_continuing_visual_event_does_not_repeat_full_action(self):
        subjects = {"asset_4": {"subject": 4, "kind": "character", "name": "B"}}
        action = prompt_skill._localized_event_action({
            "actor": "asset_4", "action": "asset_4 steps forward and takes her hand",
            "interval_phase": "continue",
        }, subjects, "both women hold hands and maintain eye contact")
        self.assertNotIn("steps forward", action)
        self.assertNotIn("takes her hand", action)
        self.assertIn("Continue the motion already visible", action)
        self.assertIn("both women hold hands", action)

    def test_completing_visual_event_holds_final_state(self):
        action = prompt_skill._localized_event_action({
            "action": "steps forward and takes her hand", "interval_phase": "complete",
        }, {}, "their joined hands remain steady")
        self.assertNotIn("steps forward", action)
        self.assertIn("Complete the remaining visible motion", action)
        self.assertIn("their joined hands remain steady", action)

    def test_reply_visual_action_waits_for_previous_speaker_to_finish(self):
        story = (
            '<Subject 1> (S1) says: <d>[English] Are you ready?</d> '
            '<Subject 2> (S2) says: <d>[English] Yes.</d>'
        )
        request = prompt_skill.build_prompt_skill_request(
            story, duration_seconds=4.0, fps=24.0, image_count=2, style="cinematic",
            shot_density="low", continuity_mode="balanced", prompt_lang="en",
        )
        value = {
            "image_subjects": [
                {"entity_id": "asset_1", "kind": "character", "name": "A", "observable_features": "dark hair"},
                {"entity_id": "asset_2", "kind": "character", "name": "B", "observable_features": "red robe"},
            ],
            "shots": [{
                "start_frame": 0, "end_frame": request["total_frames"], "pictures": ["asset_1", "asset_2"],
                "camera": "static two-shot", "start_state": "asset_1 and asset_2 face each other",
                "events": [{"id": "S1.V1", "actor": "asset_2", "action": "asset_2 steps forward, takes asset_1's hand, then speaks", "phase": "start"}],
                "dialogues": [
                    {"id": "S1.D1", "kind": "dialogue", "speaker": "asset_1", "speaker_id": "S1", "language": "English", "text": "Are you ready?", "delivery": "quietly"},
                    {"id": "S1.D2", "kind": "dialogue", "speaker": "asset_2", "speaker_id": "S2", "language": "English", "text": "Yes.", "delivery": "calmly"},
                ],
                "end_state": "they hold hands", "forbidden_replays": [], "audio": "room tone", "description": "two women face each other",
            }],
            "non_diegetic_music": "N/A", "warnings": [],
        }
        plan = prompt_skill.compile_prompt_skill(value, request)["shot_plan"]
        event = plan["shots"][0]["events"][0]
        reply = plan["shots"][0]["dialogues"][1]
        self.assertEqual(event["start_frame"], reply["start_frame"])

    def test_only_current_dialogue_speaker_may_vocalize(self):
        shot = {
            "start_frame": 0, "end_frame": 40, "start_state": "A and B face each other",
            "end_state": "A finishes the question", "visual_description": "B answers before A finishes",
            "dialogues": [{
                "speaker": "<Subject 1>", "speaker_id": "S1", "kind": "dialogue",
                "language": "English", "text": "Are you ready?", "delivery": "quietly",
                "start_frame": 0, "end_frame": 40,
            }],
        }
        subjects = {
            "asset_1": {"subject": 1, "kind": "character"},
            "asset_2": {"subject": 2, "kind": "character"},
        }
        prompt = prompt_skill._localized_shot_description(
            shot, 0, 40, 24.0,
            ({"actor": "asset_2", "action": "asset_2 speaks and reaches forward", "interval_phase": "start"},),
            subjects,
        )
        self.assertNotIn("speaks and reaches", prompt)
        self.assertIn("Only <Subject 1> vocalizes", prompt)
        self.assertIn("<Subject 2> keep their lips and jaws completely still", prompt)
        self.assertIn("<Subject 1> (S1) says", prompt)

    def test_new_speaker_does_not_continue_previous_chunk_utterance(self):
        shot = {
            "start_frame": 0, "end_frame": 80, "start_state": "both women face each other",
            "end_state": "the reply continues", "dialogues": [{
                "speaker": "<Subject 2>", "speaker_id": "S2", "kind": "dialogue",
                "language": "English", "text": "This is the second speaker's reply.", "delivery": "calmly",
                "start_frame": 20, "end_frame": 80,
            }],
        }
        prompt = prompt_skill._localized_shot_description(
            shot, 40, 80, 24.0, (), {}, previous_chunk_speakers=("<Subject 1>",)
        )
        self.assertIn("<Subject 2> (S2) says", prompt)
        self.assertNotIn("continues the same uninterrupted utterance from the previous chunk", prompt)

    def test_visual_state_removes_noncanonical_speaking_words(self):
        state = prompt_skill._visual_state("Subject 3 finishes her speech while Subject 4 answers calmly")
        self.assertNotRegex(state, r"\b(?:speech|speaking|answers)\b")
        self.assertIn("maintains eye contact", state)

    def test_non_speaker_physical_action_waits_during_current_dialogue(self):
        shot = {
            "start_frame": 0, "end_frame": 40, "start_state": "A and B face each other",
            "end_state": "B reaches for A", "dialogues": [{
                "speaker": "<Subject 1>", "speaker_id": "S1", "kind": "dialogue",
                "language": "English", "text": "Are you ready?", "delivery": "quietly",
                "start_frame": 0, "end_frame": 40,
            }],
        }
        subjects = {
            "asset_1": {"subject": 1, "kind": "character"},
            "asset_2": {"subject": 2, "kind": "character"},
        }
        prompt = prompt_skill._localized_shot_description(
            shot, 0, 40, 24.0,
            ({"actor": "asset_2", "action": "asset_2 steps forward and reaches for asset_1", "interval_phase": "start"},),
            subjects,
        )
        self.assertNotIn("steps forward", prompt)
        self.assertNotIn("reaches for", prompt)
        self.assertIn("Only <Subject 1> vocalizes", prompt)

    def test_duplicate_event_phase_controls_are_emitted_once(self):
        subjects = {"asset_1": {"subject": 1, "kind": "character"}}
        shot = {"start_frame": 0, "end_frame": 40, "end_state": "A holds position", "dialogues": []}
        event = {"actor": "asset_1", "action": "asset_1 turns", "interval_phase": "continue"}
        prompt = prompt_skill._localized_shot_description(shot, 10, 30, 24.0, (event, dict(event)), subjects)
        self.assertEqual(prompt.count("Continue the motion already visible"), 1)

    def test_no_dialogue_chunk_does_not_prepare_a_speaker(self):
        shot = {
            "start_frame": 0, "end_frame": 40, "start_state": "both women face each other",
            "end_state": "both women hold position", "dialogues": [],
        }
        prompt = prompt_skill._localized_shot_description(shot, 20, 40, 24.0, (), {}, True)
        self.assertNotIn("first audible word", prompt)
        self.assertNotIn("begin the line", prompt)
        self.assertIn("Every mouth and jaw remains still", prompt)

    def test_no_dialogue_interval_seals_mouths_even_before_later_dialogue(self):
        shot = {
            "start_frame": 0, "end_frame": 80, "start_state": "both women face each other",
            "end_state": "both women hold position", "dialogues": [{
                "speaker": "<Subject 1>", "speaker_id": "S1", "kind": "dialogue",
                "language": "English", "text": "Later.", "delivery": "quietly",
                "start_frame": 40, "end_frame": 80,
            }],
        }
        prompt = prompt_skill._localized_shot_description(shot, 0, 40, 24.0, (), {}, False)
        self.assertIn("No character vocalizes in this interval", prompt)
        self.assertIn("mouth and jaw remains completely still", prompt)
        self.assertNotIn("natural breathing, lip movement", prompt)

    def test_director_dialogue_contract_rejects_invented_or_missing_words(self):
        expected = "detailed_description:\n<Subject 1> (S1) says: <d>[English] Stay.</d>"
        prompt_skill.validate_h3_chunk_dialogue_contract(expected, expected)
        with self.assertRaisesRegex(ValueError, "changed the typed-plan dialogue contract"):
            prompt_skill.validate_h3_chunk_dialogue_contract(
                "detailed_description:\n<Subject 1> (S1) says: <d>[English] Go.</d>", expected
            )
        silent = "detailed_description:\nNo character vocalizes in this interval."
        prompt_skill.validate_h3_chunk_dialogue_contract(silent, silent)

    def test_localized_sections_do_not_reintroduce_future_action_or_internal_ids(self):
        plan = {
            "fps": 24.0, "total_frames": 80, "non_diegetic_music": "N/A",
            "image_subjects": [
                {"entity_id": "asset_1", "picture": 1, "subject": 1, "kind": "character", "name": "A", "observable_features": "resembles asset_2"},
                {"entity_id": "asset_2", "picture": 2, "subject": 2, "kind": "character", "name": "B", "observable_features": "red robe"},
            ],
            "shots": [{
                "start_frame": 0, "end_frame": 80, "pictures": [1, 2],
                "camera": "close-up on asset_2 while asset_1 speaks",
                "start_state": "asset_2 reaches toward asset_1", "end_state": "asset_2 holds asset_1's hand",
                "events": [{"id": "S1.V1", "actor": "asset_2", "action": "asset_2 reaches toward asset_1", "start_frame": 40, "end_frame": 80}],
                "dialogues": [{"id": "S1.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1", "language": "English", "text": "Wait.", "delivery": "firmly", "start_frame": 0, "end_frame": 40}],
                "forbidden_replays": ["Repeating the dialogue"], "audio": "wind", "visual_description": "asset_2 reaches toward asset_1",
            }],
        }
        localized = prompt_skill.localize_prompt_from_plan("", plan, frame_start=0, frame_end=40)
        summary = localized.split("summary:\n", 1)[1].split("\n\nretention_analysis:", 1)[0]
        retention = localized.split("retention_analysis:\n", 1)[1].split("\n\ndetailed_description:", 1)[0]
        self.assertNotIn("reaches toward", summary)
        self.assertNotRegex(localized, r"\basset_\d+\b")
        self.assertNotIn("Camera contract", retention)
        self.assertNotIn("Dialogue continuation contract", retention)

    def test_multi_speaker_chunk_uses_sequential_handoff_without_visual_action(self):
        plan = {
            "fps": 10.0, "total_frames": 40, "non_diegetic_music": "N/A",
            "image_subjects": [
                {"entity_id": "asset_1", "picture": 1, "subject": 1, "kind": "character", "name": "A", "observable_features": "blue robe"},
                {"entity_id": "asset_2", "picture": 2, "subject": 2, "kind": "character", "name": "B", "observable_features": "red robe"},
            ],
            "shots": [{
                "start_frame": 0, "end_frame": 40, "pictures": [1, 2], "camera": "two-shot",
                "start_state": "A and B face each other", "end_state": "B holds A's hand",
                "events": [{"id": "S1.V1", "actor": "asset_2", "action": "asset_2 steps forward and takes asset_1's hand", "start_frame": 0, "end_frame": 40}],
                "dialogues": [
                    {"id": "S1.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1", "language": "English", "text": "Question?", "delivery": "quietly", "start_frame": 0, "end_frame": 20},
                    {"id": "S1.D2", "kind": "dialogue", "speaker": "<Subject 2>", "speaker_id": "S2", "language": "English", "text": "Answer.", "delivery": "calmly", "start_frame": 20, "end_frame": 40},
                ],
                "forbidden_replays": [], "audio": "wind", "visual_description": "B reaches for A",
            }],
        }
        localized = prompt_skill.localize_prompt_from_plan("", plan, frame_start=0, frame_end=40)
        self.assertNotIn("steps forward", localized)
        self.assertNotIn("takes <Subject 1>'s hand", localized)
        self.assertIn("At 0.000 seconds, begin a strict speaker handoff: only <Subject 1> vocalizes", localized)
        self.assertIn("At 2.000 seconds, begin a strict speaker handoff: only <Subject 2> vocalizes", localized)
        self.assertNotIn("Only <Subject 1> vocalizes the current dialogue", localized)
        self.assertNotIn("Only <Subject 2> vocalizes the current dialogue", localized)

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
        self.assertTrue(all("<scenetrans>" not in prompt for prompt in prompts))
        self.assertTrue(all("across the cut" not in prompt for prompt in prompts))
        self.assertTrue(all("into the next shot" not in prompt for prompt in prompts))
        self.assertTrue(all("continues into the next chunk without a pause or restart" in prompt for prompt in prompts[:-1]))
        self.assertNotIn("established shot continues from <Video 1>", prompts[0])
        self.assertTrue(all("established shot continues from <Video 1>" in prompt for prompt in prompts[1:]))
        self.assertIn("turns once", prompts[0])
        self.assertTrue(all("turns once" not in prompt for prompt in prompts[1:]))
        self.assertIn("衣袂随转身摩擦声", prompts[0])
        self.assertTrue(all("衣袂随转身摩擦声" not in prompt for prompt in prompts[1:]))

    def test_physical_chunk_sizes_preserve_one_exact_utterance_without_false_cuts(self):
        text = "姐姐，自从你跟太运宗使者比试之后，这十年你都没有怎么好好闭关修炼过。这样真的来得及吗？"
        total_frames = 360
        plan = {
            "fps": 24.0, "total_frames": total_frames, "non_diegetic_music": "N/A",
            "image_subjects": [{"entity_id": "asset_1", "picture": 1, "subject": 1, "kind": "character", "name": "Asset1", "observable_features": "dark hair"}],
            "shots": [{
                "start_frame": 0, "end_frame": total_frames, "pictures": [1], "camera": "static medium shot",
                "start_state": "the speaker is visible", "end_state": "the speaker completes her question",
                "events": [], "forbidden_replays": [], "audio": "soft wind", "visual_description": "The speaker maintains eye contact.",
                "dialogues": [{"id": "S1.D1", "kind": "dialogue", "speaker": "<Subject 1>", "speaker_id": "S1", "language": "Chinese", "text": text, "delivery": "concerned", "start_frame": 0, "end_frame": total_frames}],
            }],
        }
        for chunk_frames in (39, 56, 73):
            prompts = []
            cursor = 0
            while cursor < total_frames:
                end = min(total_frames, cursor + chunk_frames)
                prompts.append(prompt_skill.localize_prompt_from_plan("<Video 1> is the continuation source.", plan, frame_start=cursor, frame_end=end))
                cursor = end
            fragments = [
                re.sub(r"^\s*\[[^]]+\]\s*", "", item).strip()
                for localized in prompts
                for item in re.findall(r"<d>(.*?)</d>", localized, re.DOTALL)
            ]
            self.assertEqual("".join(fragments), text)
            self.assertTrue(all("<scenetrans>" not in localized for localized in prompts))
            self.assertNotIn("continues into the next chunk", prompts[-1])
            for index, localized in enumerate(prompts):
                prompt_skill.validate_h3_chunk_prompt(
                    localized, plan,
                    frame_start=index * chunk_frames,
                    frame_end=min(total_frames, (index + 1) * chunk_frames),
                )

    def test_dialogue_fragments_form_one_continuous_global_interval(self):
        shots = [
            {"dialogues": [{"text": "first", "start_frame": 100, "end_frame": 160, "continues_to_next": True}]},
            {"dialogues": [{"text": "second", "start_frame": 240, "end_frame": 320, "continues_from_previous": True}]},
        ]
        prompt_skill._close_dialogue_timeline_gaps(shots)
        self.assertEqual(shots[0]["dialogues"][0]["end_frame"], 240)

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
        self.assertNotIn("Do not restart completed event S1.V1", third_prompt)
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
