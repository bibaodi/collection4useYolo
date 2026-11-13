cmake_minimum_required(VERSION 3.18)

# Force system linker — conda's ld has glibc version mismatches on Debian 13
if(NOT DEFINED CMAKE_LINKER)
    find_program(_SYS_LD ld PATHS /usr/bin NO_DEFAULT_PATH)
    if(_SYS_LD)
        set(CMAKE_LINKER ${_SYS_LD} CACHE FILEPATH "System linker")
    endif()
endif()

# GCC uses -B to find ld, so we also need to set this
set(CMAKE_C_FLAGS_INIT "-B/usr/bin" CACHE STRING "")
set(CMAKE_CXX_FLAGS_INIT "-B/usr/bin" CACHE STRING "")

project(simd_cuda_demos)

# AVX2 sub-project (always built)
add_subdirectory(avx-demo)

# CUDA sub-project (only if CUDA toolkit found)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    add_subdirectory(cuda-demo)
else()
    message(STATUS "CUDA not found — skipping cuda-demo")
endif()
