cmake_minimum_required(VERSION 3.16)
project(avx_demo)

set(CMAKE_CXX_STANDARD 17)

# 1. Auto-vectorization (no intrinsics)
add_executable(avx_auto src/main_auto.cpp src/scalar.cpp)
target_compile_options(avx_auto PRIVATE -O3 -march=native)

# 2. AVX2 Intrinsics (manual control)
add_executable(avx_intrinsics src/main_intrinsics.cpp src/avx2_intrinsics.cpp)
target_compile_options(avx_intrinsics PRIVATE -O3 -mavx2)

# 3. Runtime Dispatch
add_executable(avx_dispatch
    src/main_dispatch.cpp
    src/scalar.cpp
    src/avx2_intrinsics.cpp
    src/dispatch.cpp
)
target_compile_options(avx_dispatch PRIVATE -O3)
# Only the intrinsics source needs -mavx2; scalar/dispatch stay portable
set_source_files_properties(src/avx2_intrinsics.cpp
    TARGET_DIRECTORY avx_dispatch
    PROPERTIES COMPILE_OPTIONS "-mavx2"
)

# 4. Function Multiversioning
add_executable(avx_mv src/main_mv.cpp src/multiversion.cpp)
target_compile_options(avx_mv PRIVATE -O3)

# 5. Eigen (optional — uncomment if libeigen3-dev is installed)
# find_package(Eigen3 QUIET)
# if(Eigen3_FOUND)
#     add_executable(avx_eigen src/main_eigen.cpp)
#     target_link_libraries(avx_eigen Eigen3::Eigen)
#     target_compile_options(avx_eigen PRIVATE -O3 -march=native)
# endif()
