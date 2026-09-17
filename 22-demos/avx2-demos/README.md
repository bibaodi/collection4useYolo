# SIMD & CUDA Demos

CMake super-build workspace with AVX2 and CUDA demo sub-projects.

## Project Layout

```
.
├── CMakeLists.txt          # Super-build — builds all sub-projects
├── avx-demo/               # AVX2 vectorization demos (always built)
├── cuda-demo/              # CUDA add/multiply demo (built when toolkit found)
└── docs/                   # Setup and usage guides
    ├── README.md
    ├── adding-a-subproject.md
    └── cuda-setup-debian13.md
```

## Quick Start

```bash
cmake -B build
cmake --build build
```

AVX2 targets are always built. CUDA targets are included automatically when `nvcc` is available on the system.

## Sub-Projects

| Directory | What it does | Requires |
|---|---|---|
| `avx-demo/` | AVX2 vectorization — auto-vec, intrinsics, dispatch, multiversioning | GCC with AVX2 support |
| `cuda-demo/` | CUDA element-wise add and multiply kernels | NVIDIA CUDA toolkit |

See each sub-project's `README.md` for details.

## Documentation

Guides live in [`docs/`](docs/README.md):

- [Adding a sub-project](docs/adding-a-subproject.md) — how to extend this workspace
- [CUDA setup on Debian 13](docs/cuda-setup-debian13.md) — toolkit install and troubleshooting
