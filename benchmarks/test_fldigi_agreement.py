"""High-SNR agreement test between DeepRTTY and the fldigi receiver core.

Run from the repository root with::

    python benchmarks/src/build.py --bootstrap-fftw
    python -m unittest benchmarks/test_fldigi_agreement.py

The fldigi oracle is the native upstream C++ receiver, pinned to the upstream
revision documented in src/vendor/manifest.json. Raw ITA2 codes are
compared so text presentation options such as USOS cannot hide a disagreement.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_DIR = BENCHMARK_DIR.parent
PYTHON_EXAMPLE_DIR = REPO_DIR / "examples" / "python"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(PYTHON_EXAMPLE_DIR))

from native_demod import NativeDecoder
from rtty_stream import (
    BOTH,
    FIGS,
    FIGURES,
    LETTERS,
    LTRS,
    audio_to_spectrogram,
    greedy_ctc_codes,
    load_metadata,
    decode_long,
)

SAMPLE_RATE = 3200
WINDOW_SECONDS = 12.0
CENTER_HZ = 800.0
SHIFT_HZ = 170.0
BAUD = 45.45
STOP_BITS = 1.5
LEAD_SECONDS = 1.0
SNR_DB = 30.0


def encode_ita2(text: str) -> list[int]:
    """Encode text with the same USOS convention used by the examples."""
    letters = {char: code for code, char in LETTERS.items()}
    figures = {char: code for code, char in FIGURES.items()}
    common = {char: code for code, char in BOTH.items()}
    codes: list[int] = []
    shift: str | None = None
    for char in text.upper():
        if char in common:
            codes.append(common[char])
            if char == " ":
                shift = "letters"
        elif char in letters:
            if shift != "letters":
                codes.append(LTRS)
            codes.append(letters[char])
            shift = "letters"
        elif char in figures:
            if shift != "figures":
                codes.append(FIGS)
            codes.append(figures[char])
            shift = "figures"
        else:
            raise ValueError(f"Character is not representable in ITA2: {char!r}")
    return codes


def synthesize_high_snr(
    codes: list[int],
    seed: int,
    duration_seconds: float = WINDOW_SECONDS,
) -> np.ndarray:
    """Generate phase-continuous, mark-low RTTY audio at 30 dB RMS SNR."""
    sample_count = round(duration_seconds * SAMPLE_RATE)
    bit_samples = SAMPLE_RATE / BAUD
    character_samples = (1.0 + 5.0 + STOP_BITS) * bit_samples
    end_seconds = LEAD_SECONDS + len(codes) * character_samples / SAMPLE_RATE
    if end_seconds + 1.0 > duration_seconds:
        raise ValueError("Fixture needs at least one second of trailing mark.")

    space = np.zeros(sample_count, dtype=np.float32)
    for character_index, code in enumerate(codes):
        start = LEAD_SECONDS * SAMPLE_RATE + character_index * character_samples
        bits = (0,) + tuple((code >> bit_index) & 1 for bit_index in range(5))
        for bit_index, bit in enumerate(bits):
            if bit:
                continue
            first = math.ceil(start + bit_index * bit_samples - 1e-9)
            last = math.ceil(start + (bit_index + 1) * bit_samples - 1e-9)
            space[max(0, first) : min(sample_count, last)] = 1.0

    # Match the benchmark generator's 0.2-bit raised-cosine transitions.
    transition_samples = round(0.2 * bit_samples)
    transition = np.hanning(transition_samples + 2)[1:-1]
    transition /= transition.sum()
    space = np.convolve(space, transition, mode="same")
    frequency = CENTER_HZ - SHIFT_HZ / 2.0 + SHIFT_HZ * space
    phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE + 0.3
    signal = np.cos(phase).astype(np.float32)

    noise = np.random.default_rng(seed).standard_normal(sample_count).astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(signal**2)))
    noise *= signal_rms / (10.0 ** (SNR_DB / 20.0) * float(np.sqrt(np.mean(noise**2))))
    audio = signal + noise
    return (0.8 * audio / np.max(np.abs(audio))).astype(np.float32)


class FldigiAgreementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fldigi = NativeDecoder("fldigi")
        cls.metadata = load_metadata(REPO_DIR / "model.onnx.json")
        cls.session = ort.InferenceSession(str(REPO_DIR / "model.onnx"), providers=["CPUExecutionProvider"])

    def deep_rtty_codes(self, audio: np.ndarray) -> list[int]:
        spectrogram = audio_to_spectrogram(audio, self.metadata)
        output = self.session.run(
            [self.metadata["onnx_output_name"]],
            {self.metadata["onnx_input_name"]: spectrogram},
        )[0]
        return greedy_ctc_codes(output, int(self.metadata["blank_index"]))

    def assert_matches_fldigi(self, transmitted: list[int], audio: np.ndarray) -> None:
        fldigi_codes = self.fldigi.demodulate(
            audio,
            sample_rate=SAMPLE_RATE,
            center_hz=CENTER_HZ,
            shift_hz=SHIFT_HZ,
            baud=BAUD,
        ).codes

        # Establish that this signal is a valid positive fixture before
        # treating the independent fldigi receiver as the oracle.
        self.assertEqual(fldigi_codes, transmitted)
        self.assertEqual(self.deep_rtty_codes(audio), fldigi_codes)

    def test_raw_ita2_codes_match_fldigi_at_high_snr(self) -> None:
        messages = (
            "RYRYRYRYRY",
            "CQ CQ DE W1AW K",
            "THE QUICK BROWN FOX",
            "599 001 JA1ABC",
            "CQ TEST DE JA1ABC 599 001 K\r\n",
        )
        for seed, message in enumerate(messages, start=1):
            with self.subTest(message=message):
                transmitted = encode_ita2(message)
                audio = synthesize_high_snr(transmitted, seed)
                self.assert_matches_fldigi(transmitted, audio)

    def test_every_raw_ita2_code_matches_fldigi(self) -> None:
        transmitted = list(range(32))
        self.assert_matches_fldigi(transmitted, synthesize_high_snr(transmitted, seed=200))

    def test_every_letters_and_figures_character_matches_fldigi(self) -> None:
        printable_codes = set(range(32)) - {0, *BOTH, FIGS, LTRS}
        self.assertEqual(set(LETTERS), printable_codes)
        self.assertEqual(set(FIGURES), printable_codes)
        self.assertEqual(set(LETTERS.values()), set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.assertEqual(len(set(FIGURES.values())), 26)

        transmitted = [LTRS, *sorted(LETTERS), 8, 2, FIGS, *sorted(FIGURES)]
        self.assert_matches_fldigi(transmitted, synthesize_high_snr(transmitted, seed=201))

    def test_long_stream_matches_fldigi_across_window_boundaries(self) -> None:
        # Five complete codebooks exercise every code on both sides of several
        # overlap cuts. SPACE keeps the final LTRS away from end-of-transmission.
        transmitted = list(range(32)) * 5 + [4]
        audio = synthesize_high_snr(transmitted, seed=202, duration_seconds=32.0)

        fldigi_codes = self.fldigi.demodulate(
            audio,
            sample_rate=SAMPLE_RATE,
            center_hz=CENTER_HZ,
            shift_hz=SHIFT_HZ,
            baud=BAUD,
        ).codes
        deep_rtty_codes = [
            token.code
            for token in decode_long(
                audio,
                REPO_DIR / "model.onnx",
                self.metadata,
                session=self.session,
            )
        ]

        self.assertEqual(fldigi_codes, transmitted)
        self.assertEqual(deep_rtty_codes, fldigi_codes)


if __name__ == "__main__":
    unittest.main()
