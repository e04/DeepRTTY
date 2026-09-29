"""Conventional non-coherent FSK demodulator for RTTY.

Mark and space are each mixed to baseband and integrated over one bit (the
integrate-and-dump matched filter, sliding and centred so bit edges show up
undelayed). ``d = |mark| - |space|`` is positive on mark. Start bits are
mark-to-space crossings; each candidate frame is checked for a space start
bit, a mark stop bit and carrier at its first and last data bits. Like a
UART, after a frame the receiver hunts for the next edge past its stop bit,
and an edge that fails the checks is a framing error. Because the window can
open in the middle of a continuous stream, the first edge is chosen among
those of the first two characters by the frames minus errors of the chain it
leads to (the earliest on ties), which rejects mis-framed starts.

It serves as a generator check at high SNR and as the baseline in
``rtty.benchmark``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Hunt for the next start bit after the stop-bit sample (one stop bit is 7 bits per frame).
NEXT_FRAME_BITS = 6.6
SYNC_SEARCH_BITS = 15.0
SQUELCH_RELATIVE_POWER = 0.05
FRAMING_ERROR_PENALTY = 2.0


@dataclass(frozen=True)
class DemodResult:
    codes: list[int]
    start_sec: list[float]


def _tone_envelope(audio: np.ndarray, frequency_hz: float, bit_samples: float, sample_rate: int) -> np.ndarray:
    t = np.arange(audio.size) / sample_rate
    baseband = audio.astype(np.float64) * np.exp(-2j * np.pi * frequency_hz * t)
    length = max(1, int(round(bit_samples)))
    kernel = np.ones(length) / length
    return np.abs(np.convolve(baseband, kernel, mode="same"))


def _value_at(values: np.ndarray, position: float) -> float:
    index = int(round(position))
    if index < 0 or index >= values.size:
        return float("nan")
    return float(values[index])


def demodulate(
    audio: np.ndarray,
    *,
    sample_rate: int,
    center_hz: float,
    shift_hz: float = 170.0,
    baud: float = 45.45,
) -> DemodResult:
    bit_samples = sample_rate / baud
    mark = _tone_envelope(audio, center_hz - 0.5 * shift_hz, bit_samples, sample_rate)
    space = _tone_envelope(audio, center_hz + 0.5 * shift_hz, bit_samples, sample_rate)
    decision = mark - space
    power = mark**2 + space**2
    squelch = SQUELCH_RELATIVE_POWER * float(np.percentile(power, 99)) if power.size else 0.0

    # Every mark-to-space edge with carrier is a start-bit attempt; invalid attempts are framing errors.
    falling = np.flatnonzero((decision[:-1] > 0.0) & (decision[1:] <= 0.0))
    edges: list[tuple[float, int | None]] = []
    for index in falling:
        before, after = decision[index], decision[index + 1]
        start = index + before / (before - after)
        if start + 5.5 * bit_samples >= decision.size:
            continue
        if min(_value_at(power, start + 0.5 * bit_samples), _value_at(power, start + 5.5 * bit_samples)) < squelch:
            continue
        stop_bit = _value_at(decision, start + 6.5 * bit_samples)
        code: int | None = None
        if _value_at(decision, start + 0.5 * bit_samples) < 0.0 and not (np.isfinite(stop_bit) and stop_bit <= 0.0):
            code = 0
            for bit_index in range(5):
                if _value_at(decision, start + (1.5 + bit_index) * bit_samples) > 0.0:
                    code |= 1 << bit_index
        edges.append((start, code))

    # UART chaining: after a frame the receiver hunts from its stop-bit sample; after a framing
    # error it tries the next edge. Chains are scored frames minus errors.
    starts = np.array([edge[0] for edge in edges])
    after_stop = np.searchsorted(starts, starts + NEXT_FRAME_BITS * bit_samples, side="left")
    score = [0.0] * (len(edges) + 1)
    for i in range(len(edges) - 1, -1, -1):
        if edges[i][1] is None:
            score[i] = score[i + 1] - FRAMING_ERROR_PENALTY
        else:
            score[i] = 1.0 + score[after_stop[i]]
    chain: list[int] = []
    if edges:
        sync_limit = starts[0] + SYNC_SEARCH_BITS * bit_samples
        # max() keeps the earliest of equally good chains.
        index = max((i for i in range(len(edges)) if starts[i] <= sync_limit), key=lambda i: score[i])
        while index < len(edges):
            if edges[index][1] is None:
                index += 1
            else:
                chain.append(index)
                index = int(after_stop[index])
    return DemodResult(
        codes=[edges[i][1] for i in chain],
        start_sec=[edges[i][0] / sample_rate for i in chain],
    )
