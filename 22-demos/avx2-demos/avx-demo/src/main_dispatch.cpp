// main_dispatch.cpp — Demo: runtime CPU dispatch
#include <cstdio>
#include <chrono>

void add_arrays_dispatch(const float* a, const float* b, float* c, int n);

int main() {
    constexpr int N = 1024 * 1024;
    float* a = new float[N];
    float* b = new float[N];
    float* c = new float[N];

    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i) * 0.5f;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 100; rep++) {
        add_arrays_dispatch(a, b, c, N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("[dispatch] %d reps, %.2f ms total, c[42] = %.1f\n", 100, ms, c[42]);

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}
