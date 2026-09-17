// main_eigen.cpp — Demo: Eigen library (auto-uses AVX2 internally)
#include <Eigen/Dense>
#include <cstdio>
#include <chrono>

using namespace Eigen;

int main() {
    constexpr int N = 1024 * 1024;

    VectorXf a = VectorXf::Random(N);
    VectorXf b = VectorXf::Random(N);

    auto t0 = std::chrono::high_resolution_clock::now();
    VectorXf c;
    for (int rep = 0; rep < 100; rep++) {
        c = a + b;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("[eigen] %d reps, %.2f ms total, c[42] = %.1f\n", 100, ms, c[42]);

    return 0;
}
