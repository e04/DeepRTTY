// SPDX-License-Identifier: GPL-3.0-or-later
// Audio transport only. The complete upstream CLI and FFTW receiver are linked.
// --file - --float-samples supplies host-endian float32 PCM on stdin.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "simpleaudio.h"

struct simpleaudio { FILE *input; unsigned int rate; };
static void unsupported(void) { fputs("Unsupported benchmark audio operation\n", stderr); exit(2); }
simpleaudio *simpleaudio_open_stream(sa_backend_t backend, const char *device,
    sa_direction_t direction, sa_format_t format, unsigned int rate,
    unsigned int channels, char *app, char *stream) {
    (void)device; (void)app;
    if (backend != SA_BACKEND_FILE || direction != SA_STREAM_RECORD ||
        format != SA_SAMPLE_FORMAT_FLOAT || channels != 1 || strcmp(stream, "-")) unsupported();
    simpleaudio *sa = calloc(1, sizeof(*sa));
    if (!sa) return NULL;
    sa->input = stdin; sa->rate = rate; return sa;
}
unsigned int simpleaudio_get_rate(simpleaudio *sa) { return sa->rate; }
unsigned int simpleaudio_get_channels(simpleaudio *sa) { (void)sa; return 1; }
unsigned int simpleaudio_get_framesize(simpleaudio *sa) { (void)sa; return sizeof(float); }
unsigned int simpleaudio_get_samplesize(simpleaudio *sa) { (void)sa; return sizeof(float); }
sa_format_t simpleaudio_get_format(simpleaudio *sa) { (void)sa; return SA_SAMPLE_FORMAT_FLOAT; }
ssize_t simpleaudio_read(simpleaudio *sa, void *buf, size_t nframes) {
    size_t count = fread(buf, sizeof(float), nframes, sa->input);
    return ferror(sa->input) ? -1 : (ssize_t)count;
}
void simpleaudio_close(simpleaudio *sa) { free(sa); }
void simpleaudio_set_rxnoise(simpleaudio *sa, float noise) { (void)sa; (void)noise; unsupported(); }
ssize_t simpleaudio_write(simpleaudio *sa, void *buf, size_t count) {
    (void)sa; (void)buf; (void)count; unsupported(); return -1;
}
void simpleaudio_tone_reset(void) { unsupported(); }
void simpleaudio_tone(simpleaudio *sa, float freq, size_t count) {
    (void)sa; (void)freq; (void)count; unsupported();
}
void simpleaudio_tone_init(unsigned int count, float mag) { (void)count; (void)mag; unsupported(); }
