"""Build pinned upstream C/C++ receivers with headless benchmark adapters."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
BUILD = HERE / ".build"


def read(project: str, path: str) -> str:
    return (VENDOR / project / path).read_text(encoding="latin1")


def section(source: str, first: str, last: str) -> str:
    start = source.index(first)
    return source[start:source.index(last, start)]


def function(source: str, signature: str) -> str:
    start = source.index(signature)
    # Ignore braces in comments/string literals while retaining source offsets.
    stripped = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                      lambda m: " " * len(m[0]), source, flags=re.DOTALL)
    opening = stripped.index("{", start)
    depth = 0
    for index in range(opening, len(stripped)):
        depth += (stripped[index] == "{") - (stripped[index] == "}")
        if depth == 0:
            return source[start:index + 1] + "\n"
    raise ValueError(f"Unclosed upstream function: {signature}")


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Expected one upstream anchor: {old!r}")
    return source.replace(old, new)


def write(path: str, content: str) -> None:
    (BUILD / path).write_text(content, encoding="utf-8")


def prepare_fldigi() -> None:
    for name in ("complex.h", "gfft.h", "fftfilt.h"):
        write(name, read("fldigi", "src/include/" + name))
    misc = read("fldigi", "src/include/misc.h")
    write("misc.h", "#include <cmath>\n" + function(misc, "inline double sinc(")
          + function(misc, "inline double decayavg("))
    write("config.h", "")
    write("fftfilt.cxx", read("fldigi", "src/filters/fftfilt.cxx"))
    source = read("fldigi", "src/cw_rtty/rtty.cxx")
    header = read("fldigi", "src/include/rtty.h")
    enums = section(header, "enum RTTY_RX_STATE", "\n\tstatic const double")
    fields = section(header, "\tdouble shift;", "\n\tvoid Clear_syncscope")
    write("fldigi_fields.h", enums + fields)
    receive = function(source, "int rtty::rx_process(")
    # Channel viewer, waterfall polarity, GUI filter reset, and GUI-derived metric.
    # The adapter supplies fixed tuning, polarity, and disables AFC/squelch.
    start = receive.index("\tif ( !progdefaults.report_when_visible")
    end = receive.index("\n#if FILTER_DEBUG == 1", start)
    receive = receive[:start] + receive[end:]
    signatures = ("void rtty::reset_filters(", "cmplx rtty::mixer(",
                  "bool rtty::is_mark_space(", "bool rtty::is_mark(", "bool rtty::rx(")
    write("fldigi_receive.inc", "\n".join(function(source, s) for s in signatures) + receive)


def prepare_mmtty() -> None:
    source = read("mmtty", "Rtty.cpp")
    header = read("mmtty", "Rtty.h")
    fir = read("mmtty", "fir.cpp")
    fir_header = read("mmtty", "fir.h")
    write("mmtty_fields.h", section(header, "class CSmooz{", "\n//---------------------------------------------------------------------------")
          + section(header, "#define\tDEMBUFMAX", "\n#define\tMODBUFMAX"))
    write("mmtty_fir.h", section(fir_header, "#define\tTAPMAX", "void __fastcall MakeHilbert"))
    signatures = ("CFSKDEM::CFSKDEM()", "void CFSKDEM::SetIIR(",
                  "void CFSKDEM::SetBaudRate(", "void CFSKDEM::SetSmoozFreq(",
                  "void CFSKDEM::SetLPFFreq(", "void CFSKDEM::SetMarkFreq(",
                  "void CFSKDEM::SetSpaceFreq(", "double CFSKDEM::GetFilWidth(",
                  "void CFSKDEM::SetFilterTap(", "void CFSKDEM::DoFSK(",
                  "void CFSKDEM::Do(double", "int CFSKDEM::GetData(" )
    functions = "\n".join(function(source, s) for s in signatures)
    fir_functions = "\n".join(function(fir, s) for s in (
        "double __fastcall DoFIR(", "static double I0(",
        "void MakeFilter(double *HP, int", "void MakeFilter(double *HP, FIR"))
    # Upstream's overlapping memcpy is undefined in portable C++. memmove preserves
    # the intended FIR delay-line shift; the arithmetic remains upstream code.
    fir_functions = replace_once(fir_functions,
        "memcpy(zp, &zp[1], sizeof(double)*tap);",
        "memmove(zp, &zp[1], sizeof(double)*tap);")
    write("mmtty_receive.inc", fir_functions + functions)


def run(command: list[str], **kwargs) -> None:
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def fftw_flags(bootstrap: bool) -> list[str]:
    local = BUILD / "fftw"
    if (local / "lib/libfftw3f.a").exists():
        return ["-I" + str(local / "include"), str(local / "lib/libfftw3f.a")]
    if shutil.which("pkg-config"):
        result = subprocess.run(["pkg-config", "--cflags", "--libs", "fftw3f"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return shlex.split(result.stdout)
    if not bootstrap:
        raise SystemExit("FFTW3 single precision is required. Install fftw3f development files "
                         "or rerun with --bootstrap-fftw (builds locally, no system install).")
    archive = BUILD / "fftw-3.3.10.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve("https://www.fftw.org/fftw-3.3.10.tar.gz", archive)
    expected = "56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise SystemExit("FFTW archive checksum mismatch")
    with tarfile.open(archive) as tar:
        tar.extractall(BUILD, filter="data")
    source = BUILD / "fftw-3.3.10"
    log_path = BUILD / "fftw-build.log"
    try:
        with log_path.open("w") as log:
            run(["./configure", "--enable-float", "--disable-fortran", "--disable-shared",
                 "--enable-static", "--prefix=" + str(local)], cwd=source,
                stdout=log, stderr=subprocess.STDOUT)
            run(["make", "-j" + str(min(os.cpu_count() or 2, 8))], cwd=source,
                stdout=log, stderr=subprocess.STDOUT)
            run(["make", "install"], cwd=source, stdout=log, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"FFTW build failed; see {log_path}") from error
    return ["-I" + str(local / "include"), str(local / "lib/libfftw3f.a")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-fftw", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((VENDOR / "manifest.json").read_text())
    for project, info in manifest.items():
        for path, expected in info["files"].items():
            if hashlib.sha256((VENDOR / project / path).read_bytes()).hexdigest() != expected:
                raise SystemExit(f"Vendored source checksum mismatch: {project}/{path}")
    BUILD.mkdir(exist_ok=True)
    prepare_fldigi()
    prepare_mmtty()
    cc = shlex.split(os.environ.get("CC", "cc"))
    cxx = shlex.split(os.environ.get("CXX", "c++"))
    for name, extra in (("fldigi", [str(BUILD / "fftfilt.cxx")]), ("mmtty", [])):
        run(cxx + ["-std=c++17", "-O3", "-I" + str(BUILD),
                   str(HERE / (name + "_adapter.cpp"))] + extra
            + ["-o", str(BUILD / (name + "_rx"))])
    flags = fftw_flags(args.bootstrap_fftw)
    source = VENDOR / "minimodem/src"
    files = ["minimodem.c", "fsk.c", "baudot.c", "uic_codes.c"]
    files += [p.name for p in sorted(source.glob("databits_*.c"))]
    # Keep the complete upstream CLI receive loop. Only audio I/O is replaced.
    run(cc + ["-std=gnu99", "-O3", "-DUSE_SNDFILE=1", "-I" + str(source)]
        + [str(source / f) for f in files] + [str(HERE / "minimodem_audio.c")]
        + flags + ["-lm", "-o", str(BUILD / "minimodem_rx")])
    binaries = {name: hashlib.sha256((BUILD / (name + "_rx")).read_bytes()).hexdigest()
                for name in manifest}
    (BUILD / "build_info.json").write_text(json.dumps({
        "upstream": manifest, "cc": cc, "cxx": cxx, "optimization": "-O3",
        "cc_version": subprocess.check_output(cc + ["--version"], text=True).splitlines()[0],
        "cxx_version": subprocess.check_output(cxx + ["--version"], text=True).splitlines()[0],
        "platform": platform.platform(), "fftw_flags": flags,
        "binary_sha256": binaries,
        "adapter_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), *HERE.glob("*.cpp"), *HERE.glob("*.c"),
                                     HERE / "input.h"]},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
