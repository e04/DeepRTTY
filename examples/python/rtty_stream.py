"""Continuous DeepRTTY reception with overlapping windows.

Each 12-second window is decoded to CTC tokens stamped with an absolute sample
time and a confidence. Consecutive windows are merged by aligning the tokens in
their reliable overlap, so each transmitted character is emitted exactly once.
The merged ITA2 code stream then passes through one stateful decoder, which
carries the LTRS/FIGS shift across window boundaries.

Stream raw audio that is already centered at 800 Hz:

    ffmpeg -i input.wav -ac 1 -ar 3200 -f s16le - | python rtty_stream.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import onnxruntime as ort

LETTERS = {
    1: "E", 3: "A", 5: "S", 6: "I", 7: "U", 9: "D", 10: "R", 11: "J", 12: "N", 13: "F",
    14: "C", 15: "K", 16: "T", 17: "Z", 18: "L", 19: "W", 20: "H", 21: "Y", 22: "P",
    23: "Q", 24: "O", 25: "B", 26: "G", 28: "M", 29: "X", 30: "V",
}
FIGURES = {
    1: "3", 3: "-", 5: "\a", 6: "8", 7: "7", 9: "$", 10: "4", 11: "'", 12: ",", 13: "!",
    14: ":", 15: "(", 16: "5", 17: '"', 18: ")", 19: "2", 20: "#", 21: "6", 22: "0",
    23: "1", 24: "9", 25: "?", 26: "&", 28: ".", 29: "/", 30: ";",
}
BOTH = {2: "\n", 4: " ", 8: "\r"}
NUL, FIGS, LTRS = 0, 27, 31


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audio_to_spectrogram(audio: np.ndarray, metadata: dict) -> np.ndarray:
    """Match torch.stft(center=True, pad_mode='reflect') and DeepRTTY normalization."""
    fft_length = int(metadata["fft_length"])
    win_length = int(metadata["win_length"])
    hop_length = int(metadata["hop_length"])
    sample_rate = int(metadata["sample_rate"])
    bin_hz = sample_rate / fft_length
    start_bin = math.ceil(float(metadata["spectrogram_min_freq_hz"]) / bin_hz)
    stop_bin = math.floor(float(metadata["spectrogram_max_freq_hz"]) / bin_hz) + 1

    periodic_hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_length) / win_length)).astype(np.float32)
    window = np.zeros(fft_length, dtype=np.float32)
    window_offset = (fft_length - win_length) // 2
    window[window_offset : window_offset + win_length] = periodic_hann
    padded = np.pad(audio, (fft_length // 2, fft_length // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, fft_length)[::hop_length]
    spectrum = np.fft.rfft(frames * window, axis=-1)[:, start_bin:stop_bin]
    magnitude = np.abs(spectrum).astype(np.float32)

    noise_source = magnitude
    if metadata.get("normalization") == "noise_q25_active_log1p":
        frame_rank = math.floor(float(metadata["noise_level_quantile"]) * (magnitude.shape[1] - 1))
        frame_levels = np.partition(magnitude, frame_rank, axis=1)[:, frame_rank]
        reference_rank = math.floor(
            float(metadata["silent_frame_reference_quantile"]) * (frame_levels.size - 1)
        )
        reference = np.partition(frame_levels, reference_rank)[reference_rank]
        active_frames = frame_levels >= reference * float(metadata["silent_frame_relative_level"])
        noise_source = magnitude[active_frames]

    flat = noise_source.reshape(-1)
    rank = math.floor(float(metadata["noise_level_quantile"]) * (flat.size - 1))
    q25 = np.partition(flat, rank)[rank]
    noise_level = max(
        float(q25) / float(metadata["noise_level_rayleigh_factor"]),
        float(magnitude.max()) * float(metadata["noise_level_min_relative_to_peak"]),
        float(metadata["noise_level_min"]),
    )
    normalized = np.log1p(magnitude / noise_level).astype(np.float32)
    return normalized[np.newaxis, np.newaxis, :, :]


def greedy_ctc_codes(log_probs: np.ndarray, blank_index: int) -> list[int]:
    """Reference CTC collapse used to verify the timestamped stream decoder."""
    path = log_probs[0].argmax(axis=-1)
    codes: list[int] = []
    previous: int | None = None
    for value in path:
        code = int(value)
        if code == blank_index:
            previous = None
        else:
            if code != previous:
                codes.append(code)
            previous = code
    return codes


class Ita2Decoder:
    """Stateful ITA2 decoder whose letters/figures shift persists across calls."""

    def __init__(self, usos: bool = True, shift: str = "letters") -> None:
        self.usos = usos
        self.shift = shift

    def decode(self, codes: Iterable[int]) -> str:
        figures = self.shift == "figures"
        output: list[str] = []
        for code in codes:
            code = int(code)
            if code == LTRS:
                figures = False
            elif code == FIGS:
                figures = True
            elif code == NUL:
                continue
            elif code in BOTH:
                output.append(BOTH[code])
                if code == 4 and self.usos:
                    figures = False
            else:
                output.append((FIGURES if figures else LETTERS)[code])
        self.shift = "figures" if figures else "letters"
        return "".join(output)


def decode_ita2(codes: Iterable[int], usos: bool = True) -> str:
    return Ita2Decoder(usos=usos).decode(codes)


CHARACTER_BITS = 1.0 + 5.0 + 1.5


class Token(NamedTuple):
    sample: int  # absolute stream position of the CTC spike, in samples
    code: int  # raw ITA2 code
    score: float  # peak posterior probability of the spike


@dataclass(frozen=True)
class StreamConfig:
    hop_seconds: float = 8.0
    edge_guard_seconds: float = 0.5
    match_tolerance_seconds: float | None = None  # default: 0.45 character
    batch_windows: int = 1  # >1 batches inference calls; no CPU speedup measured, adds latency


def ctc_greedy_tokens(log_probs: np.ndarray, blank_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized greedy CTC over one [time, class] matrix.

    Returns run-center frame positions (possibly .5), codes, and peak posteriors.
    The code sequence equals greedy_ctc_codes.
    """
    path = log_probs.argmax(axis=-1)
    if path.size == 0:
        empty = np.empty(0)
        return empty, empty.astype(np.int64), empty
    best = np.take_along_axis(log_probs, path[:, None], axis=-1)[:, 0]
    starts = np.flatnonzero(np.diff(path, prepend=-1))
    stops = np.append(starts[1:], path.size)
    peaks = np.maximum.reduceat(best, starts)  # one segment per run, blank runs included
    codes = path[starts]
    keep = codes != blank_index
    starts, stops, codes, peaks = starts[keep], stops[keep], codes[keep], peaks[keep]
    return (starts + stops - 1) * 0.5, codes.astype(np.int64), np.exp(peaks.astype(np.float64))


def _align(a: list[Token], b: list[Token], tolerance: float) -> list[tuple[Token | None, Token | None]]:
    """Time-constrained alignment of two short token sequences.

    Pairs are allowed only within `tolerance` samples. Equal codes cost less than
    substitutions, and both cost less than leaving two tokens unpaired, so a
    spurious neighbour cannot steal the partner of a real character.
    """
    n, m = len(a), len(b)
    gap = 1.0
    cost = np.full((n + 1, m + 1), np.inf)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 pair, 1 a-only, 2 b-only
    cost[:, 0] = np.arange(n + 1) * gap
    cost[0, :] = np.arange(m + 1) * gap
    move[1:, 0] = 1
    move[0, 1:] = 2
    for i in range(1, n + 1):
        ta, ca = a[i - 1].sample, a[i - 1].code
        for j in range(1, m + 1):
            best, step = cost[i - 1, j] + gap, 1
            if cost[i, j - 1] + gap < best:
                best, step = cost[i, j - 1] + gap, 2
            dt = abs(ta - b[j - 1].sample)
            if dt <= tolerance:
                pair = cost[i - 1, j - 1] + 0.5 * dt / tolerance + (0.0 if ca == b[j - 1].code else 0.9)
                if pair <= best:
                    best, step = pair, 0
            cost[i, j], move[i, j] = best, step
    pairs: list[tuple[Token | None, Token | None]] = []
    i, j = n, m
    while i or j:
        step = move[i, j]
        if step == 0:
            pairs.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif step == 1:
            pairs.append((a[i - 1], None))
            i -= 1
        else:
            pairs.append((None, b[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


class OverlapMerger:
    """Merge token streams of overlapping windows into one committed stream.

    Windows must be pushed in order of increasing start. Only the reliable part
    of each window, excluding `edge_guard` samples at interior edges, competes.
    Inside the reliable overlap of two consecutive windows the tokens are
    aligned: a paired token is taken from whichever window is more confident,
    and an unpaired token is kept only by the window that owns its half of the
    overlap. Output is final once returned.
    """

    def __init__(self, edge_guard: int, tolerance: float) -> None:
        self.edge_guard = int(edge_guard)
        self.tolerance = float(tolerance)
        self._pending: list[Token] = []
        self._pending_reliable_end: float = -math.inf
        self._cutoff: float = -math.inf
        self._windows = 0

    def push(self, window_start: int, window_length: int, tokens: Iterable[Token]) -> list[Token]:
        reliable_start = window_start + (self.edge_guard if self._windows else 0)
        tokens = [token for token in tokens if token.sample >= reliable_start]
        self._windows += 1
        if self._windows == 1:
            self._pending = tokens
            self._pending_reliable_end = window_start + window_length - self.edge_guard
            return []

        zone_start = max(reliable_start, self._cutoff)
        zone_end = self._pending_reliable_end
        if zone_end < zone_start:
            # No reliable overlap: trust each window up to the middle of the gap.
            zone_start = zone_end = 0.5 * (zone_start + zone_end)
        middle = 0.5 * (zone_start + zone_end)

        committed = [token for token in self._pending if token.sample < zone_start]
        previous = [token for token in self._pending if zone_start <= token.sample <= zone_end]
        current = [token for token in tokens if zone_start <= token.sample <= zone_end]
        merged: list[Token] = []
        for old, new in _align(previous, current, self.tolerance):
            if old is not None and new is not None:
                merged.append(old if old.score >= new.score else new)
            elif old is not None and old.sample < middle:
                merged.append(old)
            elif new is not None and new.sample >= middle:
                merged.append(new)
        # Unpaired tokens between two pairs come back in arbitrary a/b order.
        merged.sort(key=lambda token: token.sample)
        committed += merged

        self._cutoff = zone_end
        self._pending = [token for token in tokens if token.sample > zone_end]
        self._pending_reliable_end = window_start + window_length - self.edge_guard
        return committed

    def finish(self, stream_end: int | None = None) -> list[Token]:
        committed = [token for token in self._pending if stream_end is None or token.sample < stream_end]
        self._pending = []
        return committed


class StreamingDecoder:
    """Push audio chunks of any size; receive merged tokens as they become final.

    Audio must already be at the model sample rate and centered at 800 Hz.
    """

    def __init__(
        self,
        model_path: Path,
        metadata: dict,
        config: StreamConfig = StreamConfig(),
        session: ort.InferenceSession | None = None,
    ) -> None:
        self.metadata = metadata
        self.config = config
        rate = int(metadata["sample_rate"])
        self.output_hop = int(metadata["output_hop_length"])
        self.window = round(float(metadata["window_seconds"]) * rate)
        # Keep window starts on the output-frame grid so spikes of different windows are comparable.
        self.hop = max(self.output_hop, round(config.hop_seconds * rate / self.output_hop) * self.output_hop)
        edge_guard = round(config.edge_guard_seconds * rate)
        character = CHARACTER_BITS / float(metadata["rtty_baud"]) * rate
        tolerance = (
            0.45 * character if config.match_tolerance_seconds is None else config.match_tolerance_seconds * rate
        )
        if self.window - self.hop - 2 * edge_guard < 2 * character:
            raise ValueError("Window overlap minus both edge guards must cover at least two characters.")
        self.merger = OverlapMerger(edge_guard, tolerance)
        self.blank_index = int(metadata["blank_index"])
        self.session = session or create_session(model_path)
        self.input_name = metadata["onnx_input_name"]
        self.output_name = metadata["onnx_output_name"]

        self._buffer = np.zeros(self.window + self.hop, dtype=np.float32)
        self._fill = 0  # valid samples in _buffer
        self._buffer_start = 0  # absolute sample index of _buffer[0]
        self._next_window = 0  # absolute start of the next window
        self._last_window_end = 0
        self._queued: list[tuple[int, np.ndarray]] = []

    @property
    def total_samples(self) -> int:
        return self._buffer_start + self._fill

    def push(self, audio: np.ndarray) -> list[Token]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        committed: list[Token] = []
        position = 0
        while position < audio.size:
            if self._fill == self._buffer.size:
                self._compact()
            take = min(self._buffer.size - self._fill, audio.size - position)
            self._buffer[self._fill : self._fill + take] = audio[position : position + take]
            self._fill += take
            position += take
            committed += self._cut_windows()
        return committed

    def flush(self) -> list[Token]:
        """Decode the remaining tail and return every outstanding token."""
        total = self.total_samples
        committed = self._run(force=True)
        if total and (total > self._last_window_end or self._last_window_end == 0):
            start = max(0, total - self.window)
            audio = self._buffer[start - self._buffer_start : total - self._buffer_start]
            if audio.size < self.window:
                # Shorter than one window: mirror-pad and drop anything decoded past the end.
                audio = np.pad(audio, (0, self.window - audio.size), mode="symmetric")
            self._queued.append((start, audio.copy()))
            committed += self._run(force=True)
        return committed + self.merger.finish(total)

    def _cut_windows(self) -> list[Token]:
        committed: list[Token] = []
        while self._next_window + self.window <= self.total_samples:
            offset = self._next_window - self._buffer_start
            self._queued.append((self._next_window, self._buffer[offset : offset + self.window].copy()))
            self._last_window_end = self._next_window + self.window
            self._next_window += self.hop
            committed += self._run(force=False)
        return committed

    def _compact(self) -> None:
        # Drop samples no future window needs, keeping enough for a tail window at flush.
        # A full buffer always holds a cut window, so this frees at least one sample.
        keep_from = min(self._next_window, self.total_samples - self.window)
        drop = keep_from - self._buffer_start
        self._buffer[: self._fill - drop] = self._buffer[drop : self._fill]
        self._fill -= drop
        self._buffer_start = keep_from

    def _run(self, force: bool) -> list[Token]:
        if not self._queued or (not force and len(self._queued) < self.config.batch_windows):
            return []
        batch, self._queued = self._queued, []
        starts = [start for start, _ in batch]
        inputs = np.concatenate([audio_to_spectrogram(audio, self.metadata) for _, audio in batch])
        outputs = self.session.run([self.output_name], {self.input_name: inputs})[0]
        committed: list[Token] = []
        for start, log_probs in zip(starts, outputs):
            frames, codes, scores = ctc_greedy_tokens(log_probs, self.blank_index)
            samples = start + np.rint(frames * self.output_hop).astype(np.int64)
            tokens = [Token(int(s), int(c), float(p)) for s, c, p in zip(samples, codes, scores)]
            committed += self.merger.push(start, self.window, tokens)
        return committed


def create_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def decode_long(
    audio: np.ndarray,
    model_path: Path,
    metadata: dict,
    config: StreamConfig = StreamConfig(),
    session: ort.InferenceSession | None = None,
) -> list[Token]:
    """Decode an arbitrarily long, already-centered recording into merged tokens."""
    decoder = StreamingDecoder(model_path, metadata, config, session)
    return decoder.push(audio) + decoder.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode continuous centered RTTY from raw s16le mono PCM on stdin.")
    parser.add_argument("--model", type=Path, default=Path("../../model.onnx"))
    parser.add_argument("--metadata", type=Path, default=Path("../../model.onnx.json"))
    parser.add_argument("--hop-seconds", type=float, default=StreamConfig.hop_seconds)
    parser.add_argument("--edge-guard-seconds", type=float, default=StreamConfig.edge_guard_seconds)
    parser.add_argument("--batch-windows", type=int, default=1, help="Windows per inference call (latency vs throughput).")
    parser.add_argument("--raw-codes", action="store_true", help="Print raw decimal ITA2 codes instead of text.")
    parser.add_argument("--no-usos", action="store_true", help="Do not unshift to letters after a space.")
    args = parser.parse_args()
    metadata = load_metadata(args.metadata)
    config = StreamConfig(args.hop_seconds, args.edge_guard_seconds, batch_windows=args.batch_windows)
    decoder = StreamingDecoder(args.model, metadata, config)
    ita2 = Ita2Decoder(usos=not args.no_usos)

    def emit(tokens: list[Token]) -> None:
        if tokens:
            codes = [token.code for token in tokens]
            sys.stdout.write(" ".join(map(str, codes)) + " " if args.raw_codes else ita2.decode(codes))
            sys.stdout.flush()

    chunk_bytes = 2 * int(metadata["sample_rate"]) // 5
    stdin = sys.stdin.buffer
    remainder = b""
    while data := stdin.read(chunk_bytes):
        data = remainder + data
        usable = len(data) - len(data) % 2
        remainder = data[usable:]
        emit(decoder.push(np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0))
    emit(decoder.flush())
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
