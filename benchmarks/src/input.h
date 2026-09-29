// Benchmark transport: host-endian float32 PCM on stdin, raw ITA2 bytes on stdout.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

inline void emit_code(int code) {
    unsigned char value = static_cast<unsigned char>(code);
    if (std::fwrite(&value, 1, 1, stdout) != 1) throw std::runtime_error("output failed");
}

template<class Consumer> int receive(Consumer consume) {
    float buffer[4096];
    size_t count;
    while ((count = std::fread(buffer, sizeof(float), 4096, stdin))) {
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(buffer[i])) throw std::runtime_error("non-finite audio");
            consume(static_cast<double>(buffer[i]));
        }
    }
    if (std::ferror(stdin)) throw std::runtime_error("input failed");
    return 0;
}
