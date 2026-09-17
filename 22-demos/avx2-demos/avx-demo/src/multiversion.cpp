// multiversion.cpp — Function multiversioning (compiler auto-selects)
#include <immintrin.h>

__attribute__((target("default")))
void add_arrays_mv(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__attribute__((target("avx2")))
void add_arrays_mv(const float* a, const float* b, float* c, int n) {
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
