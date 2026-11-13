// main.cu — CUDA addition & multiplication demo
#include "kernels.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    constexpr int N = 1024 * 1024;
    size_t bytes = N * sizeof(float);

    // Host arrays
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i) * 0.5f;
    }

    // Device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // --- Addition ---
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 100; rep++) {
        launch_add(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    double ms_add = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("[cuda add]      100 reps, %.2f ms total, c[42] = %.1f\n", ms_add, h_c[42]);

    // --- Multiplication ---
    t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 100; rep++) {
        launch_multiply(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    double ms_mul = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("[cuda multiply] 100 reps, %.2f ms total, c[42] = %.1f\n", ms_mul, h_c[42]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}
