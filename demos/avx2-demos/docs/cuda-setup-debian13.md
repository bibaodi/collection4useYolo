# CUDA Setup on Debian 13

Guide for installing the CUDA toolkit and building/running the `cuda-demo` sub-project.

## Prerequisites

- NVIDIA GPU with driver installed (`nvidia-smi` should work)
- Debian 13 (trixie)

## Install CUDA Toolkit

### Option A: Debian package (simplest)

```bash
sudo apt install nvidia-cuda-toolkit
```

This installs `nvcc` at `/usr/bin/nvcc` and headers under `/usr/include`. CMake finds everything automatically.

### Option B: NVIDIA official repo (latest CUDA 12.x)

Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads):
- Linux → x86_64 → Debian → 12 (closest to 13) → deb (network)

Follow the instructions on that page, then:

```bash
sudo apt install cuda-toolkit
```

This installs to `/usr/local/cuda/`. You may need to add it to PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

## Build the CUDA demo

From the workspace root (super-build):

```bash
cmake -B build
cmake --build build
```

If CUDA is found, you'll see:

```
-- CUDA found: /usr/bin/nvcc
```

Run it:

```bash
./build/cuda-demo/cuda_demo
```

### Standalone build

```bash
cd cuda-demo
cmake -B build
cmake --build build
./build/cuda_demo
```

## What it does

The demo runs two CUDA kernels on 1M-element float arrays:

| Kernel | Operation | Expected c[42] |
|---|---|---|
| `add_kernel` | `c[i] = a[i] + b[i]` | 63.0 |
| `multiply_kernel` | `c[i] = a[i] * b[i]` | 882.0 |

Each kernel runs 100 iterations and reports total time in milliseconds.

## Troubleshooting

### "CUDA not found — skipping cuda-demo"

CMake can't find `nvcc`. Check:

```bash
which nvcc
nvcc --version
```

If nvcc is installed but not on PATH, pass it explicitly:

```bash
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Conda nvcc conflicts

The conda-installed `nvcc` often ships without headers (`cuda_runtime.h`). The super-build's `check_language(CUDA)` will skip it if the compiler test fails. Use the system-installed toolkit instead.

### Linker errors with glibc

If you see `undefined reference to __rseq_size@GLIBC_2.35`, conda's `ld` is being used instead of the system one. The top-level `CMakeLists.txt` handles this with `-B/usr/bin`, but if building standalone, pass:

```bash
cmake -B build -DCMAKE_LINKER=/usr/bin/ld
```
