import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_seam_probe import analyze_audio_seam


class AudioSeamProbeTests(unittest.TestCase):
    def test_reports_clean_continuation_for_exact_reconstructed_overlap(self):
        sample_rate = 4000
        time = np.arange(sample_rate * 2) / sample_rate
        previous = np.sin(2 * np.pi * 173 * time) + 0.2 * np.sin(2 * np.pi * 41 * time)
        overlap = 1000
        current = np.concatenate((previous[-overlap:], previous[:1000]))
        result = analyze_audio_seam(previous, current, sample_rate, overlap)
        self.assertGreater(result["mean_correlation"], 0.99)
        self.assertLess(abs(result["mean_lag_ms"]), 1.0)
        self.assertEqual(result["reading"], "clean_continuation")

    def test_reports_unrelated_reconstruction(self):
        rng = np.random.default_rng(7)
        previous = rng.normal(size=8000)
        current = rng.normal(size=4000)
        result = analyze_audio_seam(previous, current, 4000, 1000)
        self.assertLess(result["mean_correlation"], 0.3)
        self.assertEqual(result["reading"], "not_continued")

    def test_reports_level_and_floor_steps_separately(self):
        sample_rate = 4000
        previous = np.sin(2 * np.pi * 37 * np.arange(8000) / sample_rate) * 0.5
        overlap = 1000
        current = np.concatenate((previous[-overlap:], previous[:1000] * 0.1))
        result = analyze_audio_seam(previous, current, sample_rate, overlap)
        self.assertGreater(result["broadband_step"], 0.7)
        self.assertGreater(result["floor_step"], 0.7)


if __name__ == "__main__":
    unittest.main()
