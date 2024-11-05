#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <filesystem>
#include "main_opencl.cpp"
#include "main_openmp.cpp"
#include "lodepng.h" // 添加lodepng库头文件

double x_start = -2.0f, x_finish = 2.0f;
double y_start = -2.0f, y_finish = 2.0f;
double center_x = 0.0f, center_y = 0.0f;

template<typename T>
void benchmarkOpenCL(int width, int height, int iterations, double& init_duration, double& compute_duration) {
    auto start_init = std::chrono::high_resolution_clock::now();

    MandelbrotOpenCL mandelbrotOpenCL(width, height);

    auto end_init = std::chrono::high_resolution_clock::now();
    init_duration = std::chrono::duration<double>(end_init - start_init).count();

    std::vector<uint8_t> output(width * height * 3);

    auto start_compute = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        mandelbrotOpenCL.compute(output.data(), static_cast<T>(x_start), static_cast<T>(x_finish), static_cast<T>(y_start), static_cast<T>(y_finish), static_cast<T>(center_x), static_cast<T>(center_y));
    }
    auto end_compute = std::chrono::high_resolution_clock::now();
    compute_duration = std::chrono::duration<double>(end_compute - start_compute).count();

    std::cout << "OpenCL initialization time: " << init_duration << " seconds" << std::endl;
    std::cout << "OpenCL computation time for " << iterations << " iterations: " << compute_duration << " seconds" << std::endl;
}

template<typename T>
void benchmarkOpenMP(int width, int height, int iterations, double& init_duration, double& compute_duration) {
    auto start_init = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> output(width * height * 3);
    omp_set_num_threads(omp_get_max_threads());

    auto end_init = std::chrono::high_resolution_clock::now();
    init_duration = std::chrono::duration<double>(end_init - start_init).count();

    auto start_compute = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        mandelbrot_omp(output.data(), width, height, static_cast<T>(x_start), static_cast<T>(x_finish), static_cast<T>(y_start), static_cast<T>(y_finish), static_cast<T>(center_x), static_cast<T>(center_y));
    }
    auto end_compute = std::chrono::high_resolution_clock::now();
    compute_duration = std::chrono::duration<double>(end_compute - start_compute).count();

    std::cout << "OpenMP initialization time: " << init_duration << " seconds" << std::endl;
    std::cout << "OpenMP computation time for " << iterations << " iterations: " << compute_duration << " seconds" << std::endl;
}

template<typename T>
void benchmarkSingleThread(int width, int height, int iterations, double& init_duration, double& compute_duration) {
    auto start_init = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> output(width * height * 3);
    auto end_init = std::chrono::high_resolution_clock::now();
    init_duration = std::chrono::duration<double>(end_init - start_init).count();

    auto start_compute = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        mandelbrot_single_thread(output.data(), width, height, static_cast<T>(x_start), static_cast<T>(x_finish), static_cast<T>(y_start), static_cast<T>(y_finish), static_cast<T>(center_x), static_cast<T>(center_y));
    }
    auto end_compute = std::chrono::high_resolution_clock::now();
    compute_duration = std::chrono::duration<double>(end_compute - start_compute).count();

    std::cout << "Single-threaded initialization time: " << init_duration << " seconds" << std::endl;
    std::cout << "Single-threaded computation time for " << iterations << " iterations: " << compute_duration << " seconds" << std::endl;
}

bool check_consistency(int width, int height, double x_start, double x_finish, double y_start, double y_finish, double center_x, double center_y) {
    std::vector<uint8_t> output_omp(width * height * 3);
    std::vector<uint8_t> output_single(width * height * 3);
    std::vector<uint8_t> output_opencl(width * height * 3);

    mandelbrot_omp(output_omp.data(), width, height, x_start, x_finish, y_start, y_finish, center_x, center_y);
    mandelbrot_single_thread(output_single.data(), width, height, x_start, x_finish, y_start, y_finish, center_x, center_y);

    MandelbrotOpenCL mandelbrotOpenCL(width, height);
    mandelbrotOpenCL.compute(output_opencl.data(), x_start, x_finish, y_start, y_finish, center_x, center_y);

    // 保存结果为PNG图片
    lodepng::encode("output/output_omp.png", output_omp, width, height, LCT_RGB);
    lodepng::encode("output/output_single.png", output_single, width, height, LCT_RGB);
    lodepng::encode("output/output_opencl.png", output_opencl, width, height, LCT_RGB);

    for (int i = 0; i < width * height * 3; ++i) {
        if (output_omp[i] != output_single[i] || output_omp[i] != output_opencl[i]) {
            return false;
        }
    }
    return true;
}

template<typename T>
void calculate_speedup(int width, int height, int iterations) {
    double single_init_duration, single_compute_duration;
    double omp_init_duration, omp_compute_duration;
    double opencl_init_duration, opencl_compute_duration;

    benchmarkSingleThread<T>(width, height, iterations, single_init_duration, single_compute_duration);
    benchmarkOpenMP<T>(width, height, iterations, omp_init_duration, omp_compute_duration);
    benchmarkOpenCL<T>(width, height, iterations, opencl_init_duration, opencl_compute_duration);

    double omp_speedup = (single_compute_duration ) / (omp_compute_duration + omp_init_duration);
    double opencl_speedup = (single_compute_duration ) / (opencl_compute_duration + opencl_init_duration);

    std::ofstream result_file("output/speedup_result.txt",std::ios::app);

    result_file << "Single-threaded initialization duration: " << single_init_duration << " seconds" << std::endl;
    result_file << "OpenMP initialization duration: " << omp_init_duration << " seconds" << std::endl;
    result_file << "OpenCL initialization duration: " << opencl_init_duration << " seconds" << std::endl;

    result_file << "Single-threaded computation duration: " << single_compute_duration << " seconds" << std::endl;
    result_file << "OpenMP computation duration: " << omp_compute_duration << " seconds" << std::endl;
    result_file << "OpenCL computation duration: " << opencl_compute_duration << " seconds" << std::endl;
    result_file << "OpenMP Speedup: " << omp_speedup << "x" << std::endl;
    result_file << "OpenCL Speedup: " << opencl_speedup << "x" << std::endl;
    result_file.close();

    std::cout << "OpenMP Speedup: " << omp_speedup << "x" << std::endl;
    std::cout << "OpenCL Speedup: " << opencl_speedup << "x" << std::endl;

    std::cout << "Speedup calculation completed. Results saved to speedup_result.txt" << std::endl;
}

int main(int argc, char* argv[]) {
    int width = 1024;
    int height = 1024;
    int iterations = 100;

    if (argc > 1) {
        iterations = std::stoi(argv[1]);
    }
    
    std::filesystem::create_directory("output");

    bool consistent = check_consistency(width, height, x_start, x_finish, y_start, y_finish, center_x, center_y);
    if (consistent) {
        std::cout << "Results are consistent between parallel, single-threaded, and OpenCL computations." << std::endl;
    } else {
        std::cout << "Results are NOT consistent between parallel, single-threaded, and OpenCL computations." << std::endl;
    }


    std::ofstream result_file("output/speedup_result.txt",std::ios::app);
    result_file << "Precision: Double" << std::endl;
    result_file.close();
    calculate_speedup<double>(width, height, iterations);

    result_file.open("output/speedup_result.txt", std::ios::app);
    result_file << "Precision: Float" << std::endl;
    result_file.close();
    calculate_speedup<float>(width, height, iterations);

    return 0;
}
