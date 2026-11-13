# AVX2 Demo Project

Practical examples of SIMD vectorization approaches in C++ targeting AVX2 (Intel 13th gen and similar).

## Build Targets

| Target | Description |
|---|---|
| `avx_auto` | Auto-vectorization — scalar code, compiler emits AVX2 via `-O3 -march=native` |
| `avx_intrinsics` | Manual AVX2 intrinsics with `_mm256_*` functions |
| `avx_dispatch` | Runtime CPU detection — picks AVX2 or scalar fallback |
| `avx_mv` | Function multiversioning — compiler auto-selects best version at runtime |
| `avx_eigen` | Eigen library (optional, commented out by default) |

## Quick Start

```bash
cmake -B build
cmake --build build
```

Run any target:

```bash
./build/avx_auto
./build/avx_intrinsics
./build/avx_dispatch
./build/avx_mv
```

## Eigen (optional)

```bash
sudo apt install libeigen3-dev
```

Uncomment the Eigen section in `CMakeLists.txt`, then rebuild.

## Verify AVX2 Usage

```bash
objdump -d build/avx_intrinsics | grep ymm
```

Look for `ymm0`, `ymm1`, etc. — those are AVX2 (256-bit) registers.

## Project Structure

```
avx-demo/
├── CMakeLists.txt
├── src/
│   ├── scalar.cpp              # Plain loop (auto-vectorization target)
│   ├── avx2_intrinsics.cpp     # Manual AVX2 intrinsics
│   ├── dispatch.cpp            # Runtime CPU feature dispatch
│   ├── multiversion.cpp        # Function multiversioning
│   ├── main_auto.cpp           # Driver for auto-vec
│   ├── main_intrinsics.cpp     # Driver for intrinsics
│   ├── main_dispatch.cpp       # Driver for dispatch
│   ├── main_mv.cpp             # Driver for multiversioning
│   └── main_eigen.cpp          # Driver for Eigen
```

## Notes

- Targets your CPU: AVX2 ✅, AVX-512 ❌ (disabled on most 13th gen Intel)
- `-march=native` auto-detects your CPU features
- The dispatch target compiles `avx2_intrinsics.cpp` with `-mavx2` per-source so the rest stays portable
