import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audio_seam_assemble
from audio_seam_assemble import assemble_audio_chunks


class AudioSeamAssembleTests(unittest.TestCase):
    def test_assembly_preserves_exact_frame_duration(self):
        sample_rate = 4000
        fps = 20.0
        time = np.arange(2000) / sample_rate
        first = np.stack((np.sin(2 * np.pi * 73 * time), np.sin(2 * np.pi * 73 * time)))
        overlap = 200
        second = np.concatenate((first[:, -overlap:], first[:, :1000]), axis=-1)
        assembled, seams = assemble_audio_chunks([first, second], [10, 6], [0, 1], fps, sample_rate)
        self.assertEqual(assembled.shape, (2, 3000))
        self.assertTrue(seams[0]["aligned"])
        self.assertGreater(seams[0]["correlation"], 0.99)

    def test_low_correlation_uses_short_bounded_fade(self):
        rng = np.random.default_rng(9)
        first = rng.normal(size=(2, 2000))
        second = rng.normal(size=(2, 1200))
        assembled, seams = assemble_audio_chunks([first, second], [10, 6], [0, 1], 20.0, 4000)
        self.assertEqual(assembled.shape[-1], 3000)
        self.assertFalse(seams[0]["aligned"])
        self.assertEqual(seams[0]["fade_samples"], 40)

    def test_each_alignment_uses_original_previous_chunk(self):
        chunks = [
            np.full((2, 2000), 1.0),
            np.full((2, 1200), 2.0),
            np.full((2, 1200), 3.0),
        ]
        calls = []

        def analyze(previous, current, _sample_rate, _overlap):
            calls.append((previous.copy(), current.copy()))
            return {"mean_correlation": 0.2, "mean_lag_ms": None}

        with patch.object(audio_seam_assemble, "analyze_audio_seam", side_effect=analyze):
            assemble_audio_chunks(chunks, [10, 6, 6], [0, 1, 1], 20.0, 4000)
        self.assertTrue(np.all(calls[0][0] == 1.0))
        self.assertTrue(np.all(calls[1][0] == 2.0))

    def test_large_lag_is_not_applied_even_with_high_correlation(self):
        first = np.ones((2, 2000))
        second = np.ones((2, 1200))
        result = {"mean_correlation": 0.95, "mean_lag_ms": -20.0}
        with patch.object(audio_seam_assemble, "analyze_audio_seam", return_value=result):
            _assembled, seams = assemble_audio_chunks([first, second], [10, 6], [0, 1], 20.0, 4000)
        self.assertFalse(seams[0]["aligned"])
        self.assertEqual(seams[0]["cut_samples"], 200)
        self.assertEqual(seams[0]["fade_samples"], 40)


if __name__ == "__main__":
    unittest.main()
