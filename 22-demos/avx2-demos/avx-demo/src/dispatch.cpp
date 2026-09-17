// dispatch.cpp — Runtime CPU feature detection
#include <immintrin.h>
#include <cstdio>

void add_arrays(const float*, const float*, float*, int);
void add_arrays_avx2(const float*, const float*, float*, int);

void add_arrays_dispatch(const float* a, const float* b, float* c, int n) {
    static bool logged = false;
    if (__builtin_cpu_supports("avx2")) {
        if (!logged) { std::printf("[dispatch] Using AVX2 path\n"); logged = true; }
        add_arrays_avx2(a, b, c, n);
    } else {
        if (!logged) { std::printf("[dispatch] Using scalar path\n"); logged = true; }
        add_arrays(a, b, c, n);
    }
}
