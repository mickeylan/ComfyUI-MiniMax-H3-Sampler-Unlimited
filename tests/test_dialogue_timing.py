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
        self.assertAlmostEqual(dialogue_duration_seconds(plain), 1.0)
        self.assertAlmostEqual(dialogue_duration_seconds(punctuated), 1.6)
        self.assertEqual(dialogue_frame_count(punctuated, 24.0), 39)

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
        self.assertIn("continues speaking", parts[1])
        self.assertIn("continues speaking", parts[2])
        self.assertIn("continues into the next chunk", parts[0])
        self.assertNotIn("continues into the next chunk", parts[2])
        self.assertIn("<scenetrans>", parts[0])
        self.assertIn("<scenetrans>", parts[1])
        self.assertIn("<scenetrans>", parts[2])

    def test_chunk_boundary_keeps_leading_punctuation_with_previous_fragment(self):
        source = "<Subject 1> (S1) says: <d>[Chinese] 无用。与其</d>"
        first = slice_dialogue_for_interval(source, 0, 10, 0, 4)
        second = slice_dialogue_for_interval(source, 0, 10, 4, 10)
        first_text = first.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "")
        second_text = second.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "")
        self.assertEqual(first_text, "无用。")
        self.assertEqual(second_text, "与其")

    def test_existing_scene_transition_markers_are_not_sliced_as_spoken_text(self):
        source = "<Subject 1> (S1) carries over: <d>[Chinese] <scenetrans>继续说话<scenetrans></d>"
        part = slice_dialogue_for_interval(source, 0, 20, 5, 15)
        self.assertEqual(part.count("<scenetrans>"), 2)
        spoken = part.split("<d>[Chinese] ", 1)[1].split("</d>", 1)[0].replace("<scenetrans>", "")
        self.assertNotIn("<", spoken)

    def test_complete_interval_preserves_original_dialogue(self):
        source = "<Subject 1> (S1) says: <d>[English] Stay close.</d>"
        self.assertEqual(slice_dialogue_for_interval(source, 10, 30, 10, 30), source)


if __name__ == "__main__":
    unittest.main()
