# Adding a New Sub-Project

This workspace uses a CMake super-build pattern. Each demo lives in its own directory with its own `CMakeLists.txt`, and the top-level `CMakeLists.txt` pulls them together.

## Step-by-step

### 1. Create the sub-project directory

```
my-demo/
├── CMakeLists.txt
└── src/
    └── main.cpp
```

### 2. Write its CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_demo)

set(CMAKE_CXX_STANDARD 17)

add_executable(my_demo src/main.cpp)
```

### 3. Register it in the top-level CMakeLists.txt

Open the root `CMakeLists.txt` and add:

```cmake
add_subdirectory(my-demo)
```

If the sub-project depends on an optional toolchain (like CUDA), wrap it in a check:

```cmake
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    add_subdirectory(my-demo)
else()
    message(STATUS "CUDA not found — skipping my-demo")
endif()
```

### 4. Build

```bash
cmake -B build
cmake --build build
./build/my-demo/my_demo
```

## Conventions

- Sub-project directories sit at the workspace root alongside `avx-demo/` and `cuda-demo/`
- Each sub-project is self-contained — it can also be built standalone with `cmake -B build` from its own directory
- Source files go under `src/`
- Each sub-project should have its own `README.md` and `.gitignore`

## Linker note (conda environments)

The top-level CMake includes a workaround that forces the system `/usr/bin/ld` linker. This avoids glibc version mismatches when conda's `ld` is on PATH. Sub-projects inherit this automatically when built through the super-build.
