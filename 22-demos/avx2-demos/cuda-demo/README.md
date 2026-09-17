# CUDA Demo — Addition & Multiplication

Element-wise vector addition and multiplication using CUDA kernels.

## Prerequisites

Install the CUDA toolkit system-wide on Debian 13:

```bash
sudo apt install nvidia-cuda-toolkit
```

This puts `nvcc` at `/usr/bin/nvcc` and headers under `/usr/include` — CMake finds them automatically.

## Build (standalone)

```bash
cmake -B build
cmake --build build
./build/cuda_demo
```

## Build (via super-project)

From the workspace root:

```bash
cmake -B build
cmake --build build
./build/cuda-demo/cuda_demo
```

## Source Files

| File | Description |
|---|---|
| `src/add.cu` | Vector addition kernel |
| `src/multiply.cu` | Element-wise multiplication kernel |
| `src/main.cu` | Driver — benchmarks both kernels |
| `src/kernels.h` | Shared declarations |

## Expected Output

```
[cuda add]      100 reps, X.XX ms total, c[42] = 63.0
[cuda multiply] 100 reps, X.XX ms total, c[42] = 882.0
```
