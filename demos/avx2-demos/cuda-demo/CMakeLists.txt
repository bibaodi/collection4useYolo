cmake_minimum_required(VERSION 3.18)
project(cuda_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_demo
    src/main.cu
    src/add.cu
    src/multiply.cu
)

target_include_directories(cuda_demo PRIVATE src)
