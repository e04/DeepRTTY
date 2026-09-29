// SPDX-License-Identifier: LGPL-3.0-or-later
// Original FIR receive/framing methods; disabled DSP paths are guarded stubs.
#include <algorithm>
#include <cstring>
#include <new>
#include "input.h"
#define __fastcall
#define BITDEBUG 0
#define PI 3.1415926535897932384626433832795
using BYTE = unsigned char;
double DemSamp = 8000;
int DemOver = 0;

// Configuration entry points for unused paths are inert; execution must fail
// if any of those paths is accidentally activated by the adapter.
struct UnusedDSP {
    int m_Max = 0;
    bool m_fEnabled = false;
    double m_dm = 0, m_ds = 0;
    template<class... A> void SetSampleFreq(A...) {}
    template<class... A> void SetFreq(A...) {}
    template<class... A> void SetFreeFreq(A...) {}
    template<class... A> void SetCarrierFreq(A...) {}
    template<class... A> void SetShift(A...) {}
    template<class... A> void SetMarkFreq(A...) {}
    template<class... A> void SetSpaceFreq(A...) {}
    template<class... A> void MakeIIR(A...) {}
    template<class... A> double Do(A...) { throw std::runtime_error("disabled MMTTY DSP path"); }
    void DoFSK(double) { throw std::runtime_error("disabled MMTTY FFT path"); }
    double GetOut() { return 0; }
};
using CDECM2 = UnusedDSP;
using CIIRTANK = UnusedDSP;
using COVERLIMIT = UnusedDSP;
using CPLL = UnusedDSP;
using CIIR = UnusedDSP;
using CATC = UnusedDSP;
using CPHASE = UnusedDSP;
using CAA6YQ = UnusedDSP;
struct CScope { void WriteData(double) {} void UpdateData(double) {} };
struct CTICK { void Write(double) { throw std::runtime_error("disabled tick path"); } };
#include "mmtty_fir.h"
#include "mmtty_fields.h"
#include "mmtty_receive.inc"

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    // Upstream relies on application-owned initialization for several scalars.
    alignas(CFSKDEM) unsigned char storage[sizeof(CFSKDEM)]{};
    CFSKDEM* receiver = new(storage) CFSKDEM;
    receiver->m_type = 1; receiver->m_atc = 0; receiver->m_limitagc = 200;
    receiver->m_StopLen = 1; receiver->m_BitLen = 5; receiver->m_Parity = 0;
    receiver->SetSQ(0); receiver->SetRev(0);
    receiver->SetFilterTap(512); receiver->SetSmoozFreq(300);
    double center = std::atof(argv[1]), shift = std::atof(argv[2]);
    receiver->SetMarkFreq(center - shift / 2);
    receiver->SetSpaceFreq(center + shift / 2);
    receiver->SetBaudRate(std::atof(argv[3]));
    int result = receive([&](double sample) {
        receiver->Do(std::max(-1.0, std::min(1.0, sample)) * 32767.0);
        int wire;
        while ((wire = receiver->GetData()) >= 0) {
            int code = 0;
            for (int i = 0; i < 5; ++i) { code = (code << 1) | (wire & 1); wire >>= 1; }
            emit_code(code);
        }
    });
    receiver->~CFSKDEM();
    return result;
}
