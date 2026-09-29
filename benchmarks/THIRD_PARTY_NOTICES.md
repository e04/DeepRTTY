# Third-party notices

## fldigi RTTY receiver

- Upstream: <https://github.com/w1hkj/fldigi>
- Reference revision: `61b97f4133c488063f3de1795c894d22d5032e8a`
- Reference files: `src/cw_rtty/rtty.cxx`, `src/filters/fftfilt.cxx`
- License: GNU GPL v3 or later
- Native sources: `benchmarks/src/vendor/fldigi`; GPL license in `COPYING`.
- The `gfft.h` FFT implementation is LGPL v3 or later (see its retained header).

The benchmark executes the upstream C++ mark/space filters, envelope and noise
trackers, optimal ATC decision, and receive framing state machine at 8 kHz.
AFC and squelch are disabled; raw codes are captured before text presentation.

## minimodem RTTY receiver

- Upstream: <https://github.com/kamalmostafa/minimodem>
- Reference revision: `bb2f34cf5148f101563aa926e201d306edbacbd3`
- Reference files: `src/fsk.c`, `src/minimodem.c`
- License: GNU GPL v3 or later
- Native sources: `benchmarks/src/vendor/minimodem`; GPL license in `COPYING`.

The benchmark executes the upstream C receive loop, mark/space FFT-bin analysis,
whole-frame confidence calculation, frame-position search, and timing tracker.
A headless audio adapter supplies PCM; upstream binary output preserves raw codes.

## MMTTY RTTY receiver

- Upstream: <https://github.com/n5ac/mmtty>
- Reference revision: `5a21f1db8fb5b53486cfa66d6bf0a953a2ae762c`
- Reference files: `Rtty.cpp`, `Rtty.h`, `fir.cpp`
- License: GNU LGPL v3 or later
- Native sources: `benchmarks/src/vendor/mmtty`; licenses in `COPYING.txt`
  and `COPYING.LESSER.txt`.

The benchmark executes MMTTY's C++ 8 kHz, 512-tap FIR mark/space filters,
limiter AGC, envelope smoothing, and majority-logic framing state machine.
AFC, receive BPF/LMS processing, ATC, squelch, and Baudot text presentation
are disabled or omitted; the adapter returns raw ITA2 codes for scoring.

## FFTW

The native minimodem receiver links FFTW3 single precision (GNU GPL v2 or later).
The optional local bootstrap downloads FFTW 3.3.10 from
<https://www.fftw.org/fftw-3.3.10.tar.gz> and verifies SHA-256
`56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467`.
Downloaded FFTW sources and their licensing files are retained in the ignored
`benchmarks/src/.build/fftw-3.3.10` directory.
