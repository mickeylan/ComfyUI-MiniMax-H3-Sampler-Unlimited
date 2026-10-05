import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dialogue_timing import dialogue_duration_seconds, dialogue_frame_count, slice_dialogue_for_interval


class DialogueTimingTests(unittest.TestCase):
    def test_punctuation_adds_pause_but_not_spoken_units(self):
        plain = "<d>[Chinese] 你好世界</d>"
        punctuated = "<d>[Chinese] 你好，世界！</d>"
        self.assertAlmostEqual(dialogue_duration_seconds(plain), 0.8)
        self.assertAlmostEqual(dialogue_duration_seconds(punctuated), 1.4)
        self.assertEqual(dialogue_frame_count(punctuated, 24.0), 34)

    def test_long_dialogue_is_split_by_retained_frame_interval(self):
        source = "<Subject 3> (S1) says warmly: <d>[Chinese] 姐姐自从比试之后这十年都没有闭关修炼这样真的来得及吗</d> with synchronized visible lip movement."
        parts = [
            slice_dialogue_for_interval(source, 0, 60, 0, 20),
            slice_dialogue_for_interval(source, 0, 60, 20, 40),
            slice_dialogue_for_interval(source, 0, 60, 40, 60),
        ]
        texts = [part.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "") for part in parts]
        self.assertEqual("".join(texts), "姐姐自从比试之后这十年都没有闭关修炼这样真的来得及吗")
        self.assertIn("says warmly", parts[0])
        self.assertIn("continues the same uninterrupted utterance from the previous chunk", parts[1])
        self.assertIn("continues the same uninterrupted utterance from the previous chunk", parts[2])
        self.assertIn("continues into the next chunk", parts[0])
        self.assertNotIn("continues into the next chunk", parts[2])
        self.assertNotIn("<scenetrans>", parts[0])
        self.assertNotIn("<scenetrans>", parts[1])
        self.assertNotIn("<scenetrans>", parts[2])

    def test_short_vocative_is_not_isolated_at_first_chunk_boundary(self):
        source = "<Subject 1> (S1) says: <d>[Chinese] 姐姐，自从你回来以后我一直很担心。</d>"
        first = slice_dialogue_for_interval(source, 0, 30, 0, 3)
        second = slice_dialogue_for_interval(source, 0, 30, 3, 30)
        self.assertEqual(first, "")
        second_text = second.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0]
        self.assertTrue(second_text.startswith("姐姐，自从"))

    def test_unpunctuated_cjk_boundary_uses_two_character_cadence(self):
        source = "<Subject 1> (S1) says: <d>[Chinese] 不如将舒寒当年传授给我的武学反复磨练</d>"
        first = slice_dialogue_for_interval(source, 0, 30, 0, 15)
        second = slice_dialogue_for_interval(source, 0, 30, 15, 30)
        first_text = first.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0]
        second_text = second.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0]
        self.assertFalse(first_text.endswith("反"))
        self.assertFalse(second_text.startswith("复"))
        self.assertEqual(first_text + second_text, "不如将舒寒当年传授给我的武学反复磨练")

    def test_high_confidence_chinese_pairs_are_not_split(self):
        cases = (
            ("太运宗就会派更强的弟子", "更强"),
            ("与其毫无头绪的闭关", "毫无"),
            ("反复磨练磨练来得有意思", "来得"),
            ("传授给我的武学反复磨练", "反复"),
        )
        for text, protected in cases:
            boundary = text.index(protected) + 1
            source = f"<Subject 1> (S1) says: <d>[Chinese] {text}</d>"
            first = slice_dialogue_for_interval(source, 0, len(text), 0, boundary)
            second = slice_dialogue_for_interval(source, 0, len(text), boundary, len(text))
            first_text = first.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0] if first else ""
            second_text = second.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0]
            self.assertFalse(first_text.endswith(protected[0]))
            self.assertFalse(second_text.startswith(protected[1]))
            self.assertEqual(first_text + second_text, text)

    def test_chunk_boundary_keeps_leading_punctuation_with_previous_fragment(self):
        source = "<Subject 1> (S1) says: <d>[Chinese] 无用。与其</d>"
        first = slice_dialogue_for_interval(source, 0, 10, 0, 4)
        second = slice_dialogue_for_interval(source, 0, 10, 4, 10)
        first_text = first.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "")
        second_text = second.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "")
        self.assertEqual(first_text, "无用。")
        self.assertEqual(second_text, "与其")

    def test_chunk_boundary_moves_nearby_sentence_end_instead_of_splitting_word(self):
        source = "<Subject 1> (S1) says: <d>[Chinese] 苦修已是无用。与其</d>"
        first = slice_dialogue_for_interval(source, 0, 18, 0, 10)
        second = slice_dialogue_for_interval(source, 0, 18, 10, 18)
        first_text = first.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "")
        second_text = second.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "")
        self.assertEqual(first_text, "苦修已是无用。")
        self.assertEqual(second_text, "与其")

    def test_one_character_chunk_fragment_is_deferred_without_loss(self):
        source = "<Subject 1> (S1) says: <d>[Chinese] 已经达到顶峰</d>"
        first = slice_dialogue_for_interval(source, 0, 10, 0, 1)
        second = slice_dialogue_for_interval(source, 0, 10, 1, 10)
        self.assertEqual(first, "")
        second_text = second.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "")
        self.assertEqual(second_text, "已经达到顶峰")

    def test_new_utterance_short_chunk_start_is_deferred_without_loss(self):
        source = "<Subject 4> (S2) says: <d>[Chinese] 我现在功力已经达到顶峰</d>"
        first = slice_dialogue_for_interval(
            source, 0, 24, 0, 5, defer_short_initial_fragment=True,
        )
        second = slice_dialogue_for_interval(
            source, 0, 24, 5, 24, defer_short_initial_fragment=True,
        )
        self.assertEqual(first, "")
        second_text = second.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0]
        self.assertEqual(second_text, "我现在功力已经达到顶峰")

    def test_one_character_tail_moves_to_a_two_character_final_fragment_without_loss(self):
        source = "<Subject 1> (S1) says: <d>[Chinese] 已经达到顶峰</d>"
        first = slice_dialogue_for_interval(source, 0, 10, 0, 9)
        second = slice_dialogue_for_interval(source, 0, 10, 9, 10)
        first_text = first.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "")
        second_text = second.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "")
        self.assertEqual(first_text, "已经达到")
        self.assertEqual(second_text, "顶峰")
        self.assertEqual(first_text + second_text, "已经达到顶峰")

    def test_punctuation_snapping_does_not_create_empty_middle_chunks(self):
        text = "这十年你都没有怎么好好闭关修炼过。还有不到四十年，太运宗就会派更强的弟子，这样真的来得及吗？"
        source = f"<Subject 3> (S1) says: <d>[Chinese] {text}</d>"
        boundaries = (240, 260, 311, 362, 413, 464, 515, 540)
        parts = [
            slice_dialogue_for_interval(source, 240, 540, start, end)
            for start, end in zip(boundaries, boundaries[1:])
        ]
        texts = [part.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0] for part in parts]
        self.assertTrue(all(texts))
        self.assertEqual("".join(texts), text)

    def test_physical_chunk_slice_does_not_invent_scene_transition_markers(self):
        source = "<Subject 1> (S1) says: <d>[Chinese] 继续说话直到下一段</d>"
        part = slice_dialogue_for_interval(source, 0, 20, 5, 15)
        self.assertNotIn("<scenetrans>", part)
        self.assertIn("continues into the next chunk without a pause or restart", part)

    def test_continuation_fragment_resumes_at_chunk_start_without_lead_in(self):
        source = "<Subject 1> (S1) says: <d>[Chinese] 这是一句不能中途停顿的连续对白</d>"
        part = slice_dialogue_for_interval(source, 0, 30, 10, 20)
        self.assertIn("At 0.000 seconds", part)
        self.assertIn("no lead-in silence", part)
        self.assertIn("no new breath", part)
        self.assertIn("do not close the sentence cadence", part)

    def test_fragment_does_not_claim_next_chunk_when_next_actual_speaker_differs(self):
        source = "<Subject 1> (S1) says: <d>[Chinese] 这样真的来得及吗？</d>"
        part = slice_dialogue_for_interval(
            source, 0, 20, 0, 19, continues_to_next_chunk=False,
        )
        self.assertIn("<d>[Chinese] 这样真的来得及吗？</d>", part)
        self.assertNotIn("continues into the next chunk", part)

    def test_adjacent_chunks_share_boundary_when_scene_marker_is_stripped_from_continuation(self):
        text = "还有不到四十年，太运宗就会派更强的弟子，这样真的来得及吗？"
        first_source = f"<Subject 3> (S1) carries over: <d>[Chinese] <scenetrans> {text}</d>"
        second_source = f"<Subject 3> (S1) carries over: <d>[Chinese] {text}</d>"
        first = slice_dialogue_for_interval(first_source, 480, 650, 480, 515)
        second = slice_dialogue_for_interval(second_source, 480, 650, 515, 600)
        first_text = first.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "").strip()
        second_text = second.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "").strip()
        boundary = len(first_text)
        self.assertTrue(first_text)
        self.assertTrue(second_text)
        self.assertEqual(first_text + second_text, text[:boundary + len(second_text)])

    def test_real_scene_transition_marker_stays_only_on_boundary_side(self):
        source = "<Subject 1> (S1) carries over: <d>[Chinese] <scenetrans> 继续说话直到这个镜头结束</d>"
        first = slice_dialogue_for_interval(source, 0, 20, 0, 10)
        second = slice_dialogue_for_interval(source, 0, 20, 10, 20)
        self.assertEqual(first.count("<scenetrans>"), 1)
        self.assertEqual(second.count("<scenetrans>"), 0)

    def test_complete_interval_preserves_original_dialogue(self):
        source = "<Subject 1> (S1) says: <d>[English] Stay close.</d>"
        self.assertEqual(slice_dialogue_for_interval(source, 10, 30, 10, 30), source)


if __name__ == "__main__":
    unittest.main()
