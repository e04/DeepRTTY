"""Integration checks for compiled upstream receivers; build before running."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_demod import NativeDecoder
from test_fldigi_agreement import synthesize_high_snr


class NativeDemodTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.decoders = [NativeDecoder(name) for name in ("fldigi", "minimodem", "mmtty")]

    def test_every_code_and_untranslated_control_sequences(self) -> None:
        # Duplicate CR/LF, NULL, and shift codes must survive text presentation.
        codes = list(range(32)) + [0, 8, 8, 2, 2, 27, 27, 31, 31, 4]
        audio = synthesize_high_snr(codes, seed=301)
        for decoder in self.decoders:
            with self.subTest(decoder=decoder.name):
                self.assertEqual(decoder.demodulate(audio, sample_rate=3200, center_hz=800).codes, codes)

    def test_long_stream_without_state_leaking_between_samples(self) -> None:
        codes = list(range(32)) * 5 + [4]
        audio = synthesize_high_snr(codes, seed=302, duration_seconds=32)
        for decoder in self.decoders:
            with self.subTest(decoder=decoder.name):
                self.assertEqual(decoder.demodulate(audio, sample_rate=3200, center_hz=800).codes, codes)
                silence = np.zeros(3200, dtype=np.float32)
                self.assertEqual(decoder.demodulate(silence, sample_rate=3200, center_hz=800).codes, [])

    def test_unbuilt_native_receiver_requires_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "is not built"):
                NativeDecoder("fldigi", Path(directory))


if __name__ == "__main__":
    unittest.main()
