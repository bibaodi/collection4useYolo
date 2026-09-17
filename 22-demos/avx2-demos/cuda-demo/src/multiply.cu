// multiply.cu — Element-wise multiplication kernel
#include "kernels.h"
#include <cuda_runtime.h>

__global__ void multiply_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

void launch_multiply(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    multiply_kernel<<<blocks, threads>>>(a, b, c, n);
}
