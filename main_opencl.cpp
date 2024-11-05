#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <exception>
#include <thread>


class MandelbrotOpenCL {
public:
    MandelbrotOpenCL(int width, int height) : width(width), height(height) {
        initOpenCL();
    }

    ~MandelbrotOpenCL() {
        cleanupOpenCL();
    }

    void checkError(cl_int err, const char* operation) {
        if (err != CL_SUCCESS) {
            std::cerr << "Error during operation '" << operation << "': " << err << std::endl;
            exit(1);
        }
    }

    std::string loadKernel(const char* name) {
        std::ifstream in(name);
        if (!in.is_open()) {
            std::cerr << "Failed to open kernel file: " << name << std::endl;
            exit(1);
        }
        std::string result((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return result;
    }

    void printBuildLog(const cl::Program& program, const cl::Device& device) {
        size_t log_size;
        clGetProgramBuildInfo(program(), device(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program(), device(), CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
    }

    template<typename T>
    void compute(uint8_t* output, T x_start, T x_finish, T y_start, T y_finish, T center_x, T center_y) {
        cl::Kernel kernel = kernels[std::is_same<T, double>::value ? 0 : 1];
        kernel.setArg(0, buffers[0]);
        kernel.setArg(1, width);
        kernel.setArg(2, height);
        kernel.setArg(3, x_start);
        kernel.setArg(4, x_finish);
        kernel.setArg(5, y_start);
        kernel.setArg(6, y_finish);
        kernel.setArg(7, center_x);
        kernel.setArg(8, center_y);

        queues[0].enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
        queues[0].enqueueReadBuffer(buffers[0], CL_TRUE, 0, width * height * 3 * sizeof(uint8_t), output);
    }

private:
    int width, height;
    std::vector<cl::Context> contexts;
    std::vector<cl::Program> programs;
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues;
    std::vector<cl::Buffer> buffers;

    void initOpenCL() {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "No OpenCL platforms found." << std::endl;
            exit(1);
        }

        cl::Platform platform = platforms[0];
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            std::cerr << "No OpenCL devices found." << std::endl;
            exit(1);
        }

        cl::Device device = devices[0];
        contexts.push_back(cl::Context(device));
        queues.push_back(cl::CommandQueue(contexts[0], device));

        std::ifstream kernelFile("kernal.cl");
        std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources sources;
        sources.push_back({src.c_str(), src.length()});
        programs.push_back(cl::Program(contexts[0], sources));

        if (programs[0].build({device}) != CL_SUCCESS) {
            std::cerr << "Error building: " << programs[0].getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            exit(1);
        }

        kernels.push_back(cl::Kernel(programs[0], "mandelbrot"));
        kernels.push_back(cl::Kernel(programs[0], "mandelbrot_double"));
        kernels.push_back(cl::Kernel(programs[0], "mandelbrot_float"));
        buffers.push_back(cl::Buffer(contexts[0], CL_MEM_WRITE_ONLY, width * height * 3 * sizeof(uint8_t)));
    }

    void cleanupOpenCL() {
        // OpenCL resources will be automatically released when the vectors go out of scope
    }
};

int device_info() {
    // 获取所有平台
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No platforms found." << std::endl;
        return 1;
    }

    // 打印每个平台上的设备信息
    for (auto& platform : platforms) {
        std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& device : devices) {
            std::cout << "  Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "    Type: " << device.getInfo<CL_DEVICE_TYPE>() << std::endl;
            std::cout << "    Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "    Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
            std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
            if (extensions.find("cl_khr_fp64") != std::string::npos) {
                std::cout << "    Supports double precision doubleing point operations." << std::endl;
            } else {
                std::cout << "    Does not support double precision doubleing point operations." << std::endl;
            }
        }
    }

    return 0;
}