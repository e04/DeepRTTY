"""Run from this directory: python -m unittest test_rtty_stream.py"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from rtty_stream import (
    BOTH,
    FIGS,
    FIGURES,
    LETTERS,
    LTRS,
    Ita2Decoder,
    OverlapMerger,
    StreamConfig,
    StreamingDecoder,
    Token,
    audio_to_spectrogram,
    create_session,
    ctc_greedy_tokens,
    decode_ita2,
    greedy_ctc_codes,
    load_metadata,
)

REPO_DIR = Path(__file__).resolve().parents[2]
RATE = 3200
CHAR = 7.5 / 45.45 * RATE


def encode(text: str) -> list[int]:
    letters = {v: k for k, v in LETTERS.items()}
    figures = {v: k for k, v in FIGURES.items()}
    both = {v: k for k, v in BOTH.items()}
    codes, shift = [], None
    for char in text:
        if char in both:
            codes.append(both[char])
            shift = "letters" if char == " " else shift
        elif char in letters:
            codes += ([LTRS] if shift != "letters" else []) + [letters[char]]
            shift = "letters"
        else:
            codes += ([FIGS] if shift != "figures" else []) + [figures[char]]
            shift = "figures"
    return codes


def synthesize(codes: list[int], seconds: float, lead: float = 0.3, snr_noise: float = 0.05) -> np.ndarray:
    """Phase-continuous 45.45 Bd / 170 Hz FSK centered at 800 Hz, mark low."""
    count = round(seconds * RATE)
    space = np.zeros(count)
    bit = RATE / 45.45
    for index, code in enumerate(codes):
        start = (lead * RATE) + index * CHAR
        for bit_index, value in enumerate((0,) + tuple((code >> i) & 1 for i in range(5))):
            if not value:
                space[max(0, math.ceil(start + bit_index * bit)) : min(count, math.ceil(start + (bit_index + 1) * bit))] = 1
    frequency = 715.0 + 170.0 * space
    audio = np.cos(2 * np.pi * np.cumsum(frequency) / RATE)
    audio += snr_noise * np.random.default_rng(1).standard_normal(count)
    return (0.5 * audio / np.max(np.abs(audio))).astype(np.float32)


def tokens(*pairs: tuple[float, int], score: float = 0.9) -> list[Token]:
    return [Token(round(seconds * RATE), code, score) for seconds, code in pairs]


class Ita2DecoderTest(unittest.TestCase):
    def test_figures_shift_persists_across_calls(self) -> None:
        decoder = Ita2Decoder()
        self.assertEqual(decoder.decode([FIGS, 23, 19]), "12")
        self.assertEqual(decoder.decode([1, 10]), "34")
        self.assertEqual(decoder.decode([4, 1]), " E")  # USOS
        self.assertEqual(decode_ita2([1, 10]), "ER")


class CtcTokensTest(unittest.TestCase):
    def test_matches_reference_greedy_decoder(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(200):
            log_probs = np.log(rng.dirichlet(np.full(33, 0.3), size=int(rng.integers(1, 50))))
            log_probs[rng.random(len(log_probs)) < 0.5, 32] = 0.0
            frames, codes, scores = ctc_greedy_tokens(log_probs, 32)
            self.assertEqual(codes.tolist(), greedy_ctc_codes(log_probs[None], 32))
            self.assertTrue(np.all(np.diff(frames) > 0))
            self.assertTrue(np.all((scores > 0) & (scores <= 1)))


class SpectrogramTest(unittest.TestCase):
    def test_muted_prefix_does_not_raise_active_audio(self) -> None:
        metadata = load_metadata(REPO_DIR / "model.onnx.json")
        audio = synthesize(encode("RY" * 80), seconds=12.0)
        muted = audio.copy()
        muted[: 9 * RATE] = 0.0
        reference = audio_to_spectrogram(audio, metadata)[0, 0]
        actual = audio_to_spectrogram(muted, metadata)[0, 0]
        self.assertLess(float(np.max(np.abs(actual[-300:] - reference[-300:]))), 0.11)


class OverlapMergerTest(unittest.TestCase):
    # Windows [0, 12) and [8, 20) s with a 0.5 s guard: reliable overlap [8.5, 11.5], middle 10.0.
    def merge(self, first: list[Token], second: list[Token]) -> list[int]:
        merger = OverlapMerger(round(0.5 * RATE), 0.45 * CHAR)
        out = merger.push(0, 12 * RATE, first) + merger.push(8 * RATE, 12 * RATE, second)
        return [token.code for token in out + merger.finish()]

    def test_character_straddling_the_middle_is_emitted_once(self) -> None:
        # The same character lands on opposite sides of the middle in the two windows.
        self.assertEqual(self.merge(tokens((9.8, 1), (9.99, 3), (11.9, 7)), tokens((8.2, 9), (10.01, 3), (10.2, 5))), [1, 3, 5])
        self.assertEqual(self.merge(tokens((9.8, 1), (10.01, 3)), tokens((9.99, 3), (10.2, 5))), [1, 3, 5])

    def test_guarded_edges_are_ignored(self) -> None:
        # 11.9 s is inside the first window's guard, 8.2 s inside the second window's guard.
        self.assertEqual(self.merge(tokens((11.9, 7)), tokens((8.2, 9))), [])

    def test_unpaired_tokens_belong_to_the_owning_half(self) -> None:
        first = tokens((9.0, 1), (10.5, 5))  # 10.5 s: spurious in the second window's half
        second = tokens((9.2, 6), (10.8, 3))  # 9.2 s: spurious in the first window's half
        self.assertEqual(self.merge(first, second), [1, 3])

    def test_confident_substitution_wins(self) -> None:
        first = [Token(round(9.0 * RATE), 1, 0.55), Token(round(10.5 * RATE), 5, 0.99)]
        second = [Token(round(9.01 * RATE), 3, 0.97), Token(round(10.5 * RATE), 6, 0.51)]
        self.assertEqual(self.merge(first, second), [3, 5])

    def test_spurious_neighbour_does_not_steal_a_pair(self) -> None:
        first = tokens((9.85, 1), (10.0, 3))
        second = tokens((9.85, 1), (9.93, 9), (10.01, 3))
        self.assertEqual(self.merge(first, second), [1, 3])


class StreamingDecoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = load_metadata(REPO_DIR / "model.onnx.json")
        cls.session = create_session(REPO_DIR / "model.onnx")

    def decoder(self, **config) -> StreamingDecoder:
        return StreamingDecoder(REPO_DIR / "model.onnx", self.metadata, StreamConfig(**config), self.session)

    def test_figures_run_across_window_boundaries(self) -> None:
        text = "CQ DE JA1ABC " + "1234567890/" * 22 + " K\r\n"
        codes = encode(text)
        audio = synthesize(codes, seconds=(len(codes) + 4) * CHAR / RATE)
        self.assertGreater(audio.size, 3 * 38400)
        decoder = self.decoder()
        merged = [token.code for token in decoder.push(audio) + decoder.flush()]
        self.assertEqual(merged, codes)
        self.assertEqual(Ita2Decoder().decode(merged), text)

    def test_chunked_push_equals_whole_push(self) -> None:
        audio = synthesize(encode("RYRYRY THE QUICK BROWN FOX 599 001 K\r\n" * 4), seconds=31.3)
        whole = self.decoder(batch_windows=3)
        expected = whole.push(audio) + whole.flush()
        chunked = self.decoder()
        actual = []
        for start in range(0, audio.size, 1001):
            actual += chunked.push(audio[start : start + 1001])
        actual += chunked.flush()
        self.assertEqual([t.code for t in actual], [t.code for t in expected])
        self.assertTrue(all(a.sample < b.sample for a, b in zip(actual, actual[1:])))

    def test_short_stream_is_decoded(self) -> None:
        codes = encode("CQ CQ DE W1AW K")
        decoder = self.decoder()
        merged = decoder.push(synthesize(codes, seconds=5.0)) + decoder.flush()
        self.assertEqual([token.code for token in merged], codes)

    def test_rejects_overlap_too_small(self) -> None:
        with self.assertRaises(ValueError):
            self.decoder(hop_seconds=10.8)


if __name__ == "__main__":
    unittest.main()
