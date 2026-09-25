import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


if __name__ == "__main__":
    unittest.main()
