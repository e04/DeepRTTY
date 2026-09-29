// SPDX-License-Identifier: GPL-3.0-or-later
// Headless adapter for the vendored fldigi receiver. DSP/framing live in .inc.
#include <algorithm>
#include <cctype>
#include <cstring>
#include <memory>
#include "input.h"
#include "fftfilt.h"
#include "misc.h"

#define TWOPI (2.0 * M_PI)
#define MAXPIPE 1024
#define MAXBITS (2 * 8000 / 23 + 1)
#define OUTBUFSIZE 1
#define FILTER_DEBUG 0

class Cmovavg;
struct view_rtty {};
struct Defaults {
    int rtty_cwi = 0, rtty_afcspeed = 0;
    bool SynopAdifDecoding = false, SynopKmlDecoding = false;
    bool rx_lowercase = false, true_scope = false;
} progdefaults;
struct Status { bool sqlonoff = false, afconoff = false; double sldrSquelchValue = 0; } progStatus;
struct synop {
    static synop* instance() { static synop s; return &s; }
    void add(int) {} void flush(bool) {} bool enabled() { return false; }
};
void put_rx_char(int) {}
void set_zdata(cmplx*, int) {}
int dspcnt = 0;
bool bHighSpeed = true;

class rtty {
public:
#include "fldigi_fields.h"
    double samplerate = 8000, frequency = 800, metric = 0, freqerr = 0;
    int sigsearch = 0;
    bool reverse = true; // Upstream mark-high convention -> benchmark mark-low.
    view_rtty* rttyviewer = nullptr;
    rtty(double center, double spacing, double baud) {
        frequency = center; shift = rtty_shift = spacing; rtty_baud = baud;
        symbollen = static_cast<int>(samplerate / baud + 0.5);
        nbits = 5; filter_length = 512; rtty_parity = RTTY_PARITY_NONE;
        mark_filt = space_filt = nullptr; mark_phase = space_phase = 0;
        mark_mag = space_mag = mark_env = space_env = mark_noise = space_noise = 0;
        rxstate = RTTY_RX_STATE_IDLE; lastchar = 0; xy_phase = 0; inp_ptr = 0;
        pipeptr = 0; clear_zdata = false; rttyviewer = nullptr;
        pipe = new double[MAXPIPE](); dsppipe = new double[MAXPIPE]();
        std::fill(std::begin(bit_buf), std::end(bit_buf), false);
        std::fill(std::begin(mark_history), std::end(mark_history), cmplx(0, 0));
        std::fill(std::begin(space_history), std::end(space_history), cmplx(0, 0));
        reset_filters();
    }
    ~rtty() { delete mark_filt; delete space_filt; delete[] pipe; delete[] dsppipe; }
    void reset_filters();
    cmplx mixer(double&, double, cmplx);
    bool is_mark_space(int&); bool is_mark(); bool rx(bool);
    int rx_process(const double*, int);
    // Capture before text conversion (including NULL, shifts, repeated CR/LF).
    int decode_char() { emit_code(rxdata & 31); return 0; }
    void Update_syncscope() {} void Clear_syncscope() {}
    void set_freq(double) { throw std::runtime_error("AFC is disabled"); }
};
#include "fldigi_receive.inc"

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    rtty receiver(std::atof(argv[1]), std::atof(argv[2]), std::atof(argv[3]));
    return receive([&](double sample) { receiver.rx_process(&sample, 1); });
}
