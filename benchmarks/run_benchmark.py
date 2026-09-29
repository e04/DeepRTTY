from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import onnxruntime as ort

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR / "examples" / "python"))

import reference_demod
from native_demod import BUILD_DIR, NativeDecoder
from rtty_stream import StreamConfig, decode_long, load_metadata

SAMPLE_RATE = 3200
WINDOW_SECONDS = 30.0
MODEL_HOP_SECONDS = 8.0
CENTER_HZ = 800.0
SHIFT_HZ = 170.0
BAUD = 45.45
STOP_BITS = 1.5
SNR_REFERENCE_BANDWIDTH_HZ = 2500.0
BAND_MIN_HZ = 650.0
BAND_MAX_HZ = 950.0
BENCHMARK_SEED = 20_000_000

LETTERS = {"E":1,"A":3,"S":5,"I":6,"U":7,"D":9,"R":10,"J":11,"N":12,"F":13,"C":14,"K":15,"T":16,"Z":17,"L":18,"W":19,"H":20,"Y":21,"P":22,"Q":23,"O":24,"B":25,"G":26,"M":28,"X":29,"V":30}
FIGURES = {"3":1,"-":3,"\a":5,"8":6,"7":7,"$":9,"4":10,"'":11,",":12,"!":13,":":14,"(":15,"5":16,'"':17,")":18,"2":19,"#":20,"6":21,"0":22,"1":23,"9":24,"?":25,"&":26,".":28,"/":29,";":30}
COMMON = {"\n":2," ":4,"\r":8}
LABEL_CHARS = ("_", "E", ">", "A", " ", "S", "I", "U", "<", "D", "R", "J", "N", "F", "C", "K", "T", "Z", "L", "W", "H", "Y", "P", "Q", "O", "B", "G", "]", "M", "X", "V", "[")
MACROS = (
    "RYRYRYRYRY RYRYRYRYRY\r\n",
    "CQ CQ CQ DE W1AW W1AW K\r\n",
    "CQ TEST DE JA1ABC JA1ABC K\r\n",
    "W1AW DE JA1ABC 599 001 001 K\r\n",
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 1234567890\r\n",
    "TNX FER CALL UR RST 599 599 NAME IS RADIO QTH TOKYO\r\n",
)
COLORS = {"model":"#2a78d6","reference":"#eb6834","fldigi":"#1baa7d","minimodem":"#8b5cc7","mmtty":"#d49a16"}


@dataclass(frozen=True)
class Sample:
    audio: np.ndarray
    label: str


def encode_text(text: str) -> list[int]:
    codes: list[int] = []
    shift: str | None = None
    for char in text.upper():
        if char in COMMON:
            codes.append(COMMON[char])
        elif char in LETTERS:
            if shift != "letters":
                codes.append(31)
                shift = "letters"
            codes.append(LETTERS[char])
        elif char in FIGURES:
            if shift != "figures":
                codes.append(27)
                shift = "figures"
            codes.append(FIGURES[char])
        if char == " ":
            shift = "letters"
    return codes


def band_power(audio: np.ndarray) -> float:
    spectrum = np.fft.rfft(audio.astype(np.float64))
    frequencies = np.fft.rfftfreq(audio.size, 1.0 / SAMPLE_RATE)
    selected = (frequencies >= BAND_MIN_HZ) & (frequencies <= BAND_MAX_HZ)
    return float(2.0 * np.sum(np.abs(spectrum[selected]) ** 2) / audio.size**2)


def benchmark_sample(seed: int, snr_db: float) -> Sample:
    rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed ^ 0xA5A5_5A5A)
    text = "".join(rng.choice(MACROS) for _ in range(10))
    codes = encode_text(text)
    bit_seconds = 1.0 / BAUD
    character_seconds = (1.0 + 5.0 + STOP_BITS) * bit_seconds
    first_start = -rng.uniform(0.0, character_seconds)
    starts = first_start + np.arange(len(codes), dtype=np.float64) * character_seconds
    count = round(WINDOW_SECONDS * SAMPLE_RATE)
    space = np.zeros(count, dtype=np.float32)
    for code, start_seconds in zip(codes, starts):
        bits = (0,) + tuple((code >> index) & 1 for index in range(5))
        for bit_index, bit in enumerate(bits):
            if bit:
                continue
            start = math.ceil((start_seconds + bit_index * bit_seconds) * SAMPLE_RATE - 1e-9)
            stop = math.ceil((start_seconds + (bit_index + 1) * bit_seconds) * SAMPLE_RATE - 1e-9)
            start, stop = max(0, start), min(count, stop)
            if start < stop:
                space[start:stop] = 1.0
    transition = round(0.2 * SAMPLE_RATE / BAUD)
    kernel = np.hanning(transition + 2)[1:-1]
    kernel /= kernel.sum()
    space = np.convolve(space, kernel, mode="same")
    frequency = CENTER_HZ - SHIFT_HZ / 2.0 + SHIFT_HZ * space
    phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE + rng.uniform(0.0, 2.0 * np.pi)
    signal = np.cos(phase).astype(np.float32)

    target_noise_band_power = 0.5 / (10.0 ** (snr_db / 10.0)) * (
        (BAND_MAX_HZ - BAND_MIN_HZ) / SNR_REFERENCE_BANDWIDTH_HZ
    )
    noise = noise_rng.standard_normal(count).astype(np.float32)
    noise *= math.sqrt(target_noise_band_power / band_power(noise))
    audio = signal + noise
    audio *= 0.8 / max(float(np.max(np.abs(audio))), 1e-12)

    label_codes = [
        code for code, start in zip(codes, starts)
        if start >= 0.0 and start + 6.0 * bit_seconds <= WINDOW_SECONDS
    ]
    return Sample(audio.astype(np.float32), "".join(LABEL_CHARS[code] for code in label_codes))


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def cer(references: list[str], predictions: list[str]) -> float:
    return sum(edit_distance(a, b) for a, b in zip(references, predictions)) / sum(map(len, references))


def model_predictions(
    session: ort.InferenceSession,
    model_path: Path,
    metadata: dict,
    samples: list[Sample],
    batch_size: int,
) -> list[str]:
    config = StreamConfig(hop_seconds=MODEL_HOP_SECONDS, batch_windows=batch_size)
    return [
        "".join(
            LABEL_CHARS[token.code]
            for token in decode_long(sample.audio, model_path, metadata, config, session)
        )
        for sample in samples
    ]


def plot_results(rows: list[dict[str, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for label, key in (("DeepRTTY model","model"),("Reference demodulator","reference"),("fldigi native core","fldigi"),("minimodem native core","minimodem"),("MMTTY native core","mmtty")):
        emphasized = key == "model"
        ax.plot(
            [row["snr_db"] for row in rows],
            [100.0*row[f"{key}_cer"] for row in rows],
            marker="o",
            linewidth=4 if emphasized else 2,
            markersize=7 if emphasized else 5,
            color=COLORS[key],
            label=label,
            alpha=1.0 if emphasized else 0.45,
            zorder=3 if emphasized else 2,
        )
    ax.set(title="RTTY 45.45 baud / 170 Hz, white noise", xlabel="SNR (dB, 2500 Hz reference)", ylabel="Raw ITA2 CER (%)", ylim=(0,100))
    ax.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False, loc="center right")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the public DeepRTTY white-noise benchmark.")
    parser.add_argument("--model", type=Path, default=REPO_DIR / "model.onnx")
    parser.add_argument("--metadata", type=Path, default=REPO_DIR / "model.onnx.json")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--native-build-dir", type=Path, default=BUILD_DIR)
    parser.add_argument("--min-snr-db", type=float, default=-18.0)
    parser.add_argument("--max-snr-db", type=float, default=6.0)
    parser.add_argument("--step-db", type=float, default=2.0)
    parser.add_argument("--samples-per-snr", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.samples_per_snr <= 0 or args.batch_size <= 0 or args.step_db <= 0 or args.min_snr_db > args.max_snr_db:
        parser.error("Samples, batch size, and SNR step must be positive; min SNR must be <= max SNR")
    try:
        decoders = {name: NativeDecoder(name, args.native_build_dir)
                    for name in ("fldigi", "minimodem", "mmtty")}
    except RuntimeError as error:
        parser.error(str(error))
    metadata = load_metadata(args.metadata)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    warmup = benchmark_sample(BENCHMARK_SEED - 1, 0.0)
    for decoder in decoders.values():
        decoder.demodulate(warmup.audio, sample_rate=SAMPLE_RATE, center_hz=CENTER_HZ, shift_hz=SHIFT_HZ, baud=BAUD)
    model_predictions(session, args.model, metadata, [warmup], 1)

    names = ("reference", "fldigi", "minimodem", "mmtty", "model")
    elapsed = {name: 0.0 for name in names}
    rows: list[dict[str, float]] = []
    total_samples = 0
    snrs = np.arange(args.min_snr_db, args.max_snr_db + 1e-9, args.step_db)
    for snr_index, snr_db in enumerate(snrs):
        samples = [benchmark_sample(BENCHMARK_SEED + snr_index*10_000 + index, float(snr_db)) for index in range(args.samples_per_snr)]
        references = [sample.label for sample in samples]
        predictions: dict[str, list[str]] = {}
        for name, decoder in {"reference": reference_demod, **decoders}.items():
            started = time.perf_counter()
            predictions[name] = ["".join(LABEL_CHARS[code] for code in decoder.demodulate(sample.audio, sample_rate=SAMPLE_RATE, center_hz=CENTER_HZ, shift_hz=SHIFT_HZ, baud=BAUD).codes) for sample in samples]
            elapsed[name] += time.perf_counter() - started
        started = time.perf_counter()
        predictions["model"] = model_predictions(session, args.model, metadata, samples, args.batch_size)
        elapsed["model"] += time.perf_counter() - started
        row = {"snr_db":float(snr_db), **{f"{name}_cer":cer(references,predictions[name]) for name in names}}
        rows.append(row)
        total_samples += len(samples)
        print("  ".join(f"{key}={value:.4f}" for key,value in row.items()), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cer_by_snr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    runtime=[]
    for name in names:
        seconds=elapsed[name]
        runtime.append({"decoder":name,"wall_seconds":seconds,"samples_per_second":total_samples/seconds,"realtime_factor":seconds/(total_samples*WINDOW_SECONDS)})
    with (args.output_dir / "runtime.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(runtime[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(runtime)
    config = {
        "seed": BENCHMARK_SEED,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "model_window_seconds": float(metadata["window_seconds"]),
        "model_hop_seconds": MODEL_HOP_SECONDS,
        "baud": BAUD,
        "shift_hz": SHIFT_HZ,
        "center_hz": CENTER_HZ,
        "snr_reference_bandwidth_hz": SNR_REFERENCE_BANDWIDTH_HZ,
        "snr_db": [float(value) for value in snrs],
        "samples_per_snr": args.samples_per_snr,
        "total_samples": total_samples,
        "backend": "native",
        "decoders": {name: decoder.metadata() for name, decoder in decoders.items()},
        "runtime_scope": "End-to-end decode, including audio resampling, PCM transport and subprocess startup for native receivers; excludes build and warmup.",
        "mmtty": {
            "upstream_commit": decoders["mmtty"].metadata()["upstream_commit"],
            "sample_rate": 8000,
            "filter_taps": 512,
            "filter_attenuation_db": 60.0,
            "smoothing_hz": 300.0,
            "limiter_agc": True,
            "majority_logic": True,
            "squelch": False,
            "afc": False,
            "atc": False,
        },
    }
    (args.output_dir / "benchmark_config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    plot_results(rows,args.output_dir/"cer_by_snr.png")


if __name__ == "__main__":
    main()
