"""Python transport for upstream native receivers; no receive DSP runs here."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

import numpy as np
from scipy.signal import resample_poly

BUILD_DIR = Path(__file__).resolve().parent / "src/.build"


@dataclass(frozen=True)
class DemodResult:
    codes: list[int]


class NativeDecoder:
    def __init__(self, name: str, build_dir: Path = BUILD_DIR):
        self.name = name
        self.executable = build_dir / f"{name}_rx"
        if not self.executable.is_file():
            build_command = shlex.join(["python", str(BUILD_DIR.parent / "build.py"), "--bootstrap-fftw"])
            raise RuntimeError(f"Native {name} is not built. Run: "
                               f"{build_command}")
        try:
            self.build_info = json.loads((build_dir / "build_info.json").read_text())
        except (OSError, ValueError) as error:
            raise RuntimeError("Missing native build metadata; rebuild the receivers") from error
        actual = hashlib.sha256(self.executable.read_bytes()).hexdigest()
        if actual != self.build_info["binary_sha256"].get(name):
            raise RuntimeError(f"Native {name} binary changed; rebuild the receivers")

    def metadata(self) -> dict:
        return {
            "backend": "native", "language": "C" if self.name == "minimodem" else "C++",
            "upstream_commit": self.build_info["upstream"][self.name]["commit"],
            "binary_sha256": self.build_info["binary_sha256"][self.name],
            "adapter_sha256": self.build_info["adapter_sha256"],
            "compiler": self.build_info.get("cc_version" if self.name == "minimodem" else "cxx_version"),
            "optimization": self.build_info["optimization"],
            "processing_sample_rate": "input" if self.name == "minimodem" else 8000,
            "transport": "float32 PCM, subprocess per sample, raw ITA2 output",
            "afc": False, "squelch": False,
        }

    def demodulate(self, audio: np.ndarray, *, sample_rate: int, center_hz: float,
                   shift_hz: float = 170.0, baud: float = 45.45) -> DemodResult:
        if sample_rate <= 0 or not all(math.isfinite(v) and v > 0 for v in (center_hz, shift_hz, baud)):
            raise ValueError("Sample rate, center, shift, and baud must be positive")
        if baud != 45.45 or shift_hz != 170 or center_hz != 800:
            raise ValueError("Native adapters are validated for 800 Hz / 170 Hz / 45.45 baud")
        values = np.asarray(audio, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Audio must be finite mono PCM")
        if self.name == "minimodem":
            # --binary-output preserves RTTY start/stop framing (unlike --binary-raw).
            command = [str(self.executable), "--rx", "rtty", "--file", "-",
                       "--float-samples", "--binary-output", "--quiet",
                       "--samplerate", str(sample_rate), "--mark", str(center_hz - shift_hz / 2),
                       "--space", str(center_hz + shift_hz / 2)]
        else:
            common = math.gcd(sample_rate, 8000)
            if sample_rate != 8000:
                # Audio input adaptation only; the decoder itself is upstream C++.
                values = resample_poly(values.astype(np.float64), 8000 // common,
                                       sample_rate // common).astype(np.float32)
            command = [str(self.executable), str(center_hz), str(shift_hz), str(baud)]
        result = subprocess.run(command, input=values.tobytes(), capture_output=True,
                                check=True, timeout=120)
        if self.name == "minimodem":
            lines = result.stdout.splitlines()
            if any(len(line) != 5 or set(line) - {48, 49} for line in lines):
                raise RuntimeError(f"Invalid raw minimodem output: {result.stdout[:100]!r}")
            codes = [sum((bit - 48) << index for index, bit in enumerate(line)) for line in lines]
        else:
            codes = list(result.stdout)
            if any(code > 31 for code in codes):
                raise RuntimeError(f"Invalid raw {self.name} output")
        return DemodResult(codes)
