#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


__constant__ double c_center_y, c_center_x;
__constant__ int c_width, c_height;

__global__ void mandelbrot_kernel(uint8_t *output, double x_start, double x_finish, double y_start, double y_finish)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= c_height || col >= c_width)
    {
        return;
    }
    int max_iter = 256;
    int iter = 0;

    double dx = (x_finish - x_start) / c_width;
    double dy = (y_finish - y_start) / c_height;
    int idx = row * c_width + col;
    double real = x_start + col * dx;
    double imag = y_start + row * dy;
    double c_real = real;
    double c_imag = imag;

    double real2, imag2;

    for (int i = 0; i < max_iter; ++i)
    {
        real2 = real * real;
        imag2 = imag * imag;
        if (real2 + imag2 > 4.0)
        {
            break;
        }
        imag = 2 * real * imag + c_imag;
        real = real2 - imag2 + c_real;
        iter++;
    }

    double t = (double)iter / max_iter;
    uint8_t r, g, b;

    if (iter == max_iter)
    {
        r = g = b = 0; // 黑色
    }
    else
    {
        double t1 = 1 - t;
        r = static_cast<uint8_t>(9 * t1 * t * t * t * 255);
        g = static_cast<uint8_t>(15 * t1 * t1 * t * t * 255);
        b = static_cast<uint8_t>(8.5 * t1 * t1 * t1 * t * 255);
    }

    output[idx * 3] = r;
    output[idx * 3 + 1] = g;
    output[idx * 3 + 2] = b;
}

class MandelbrotCUDA
{
public:
    MandelbrotCUDA(int width, int height) : width(width), height(height)
    {
        ratio = static_cast<double>(width) / height;
        initCUDA();
    }

#define TILE_WIDTH 32
    void compute(double x_start, double x_finish, double y_start, double y_finish)
    {
        dim3 dimGrid(ceil((double)width / TILE_WIDTH), ceil((double)height / TILE_WIDTH), 1);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        uint8_t *dataptr = thrust::raw_pointer_cast(&data_device[0]);

        mandelbrot_kernel<<<dimGrid, dimBlock>>>(dataptr, x_start, x_finish, y_start, y_finish);

        data_host = data_device;
    }

    uint8_t *get_data()
    {
        return data_host.data();
    }

    void set_zoom(double zoom)
    {
        this->zoom = zoom;
    }
    void set_center(double center_x, double center_y)
    {
        this->center_x = center_x;
        this->center_y = center_y;
    }
    void set_boundaries(double x_start, double x_finish, double y_start, double y_finish)
    {
        this->x_start = x_start;
        this->x_finish = x_finish;
        this->y_start = y_start;
        this->y_finish = y_finish;
    }

    void update()
    {
        this->scale *= zoom;
        this->x_start = center_x - 0.5 * ratio * scale;
        this->x_finish = center_x + 0.5 * ratio * scale;
        this->y_start = center_y - 0.5 * scale;
        this->y_finish = center_y + 0.5 * scale;
    }

    void initCUDA()
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0)
        {
            std::cerr << "No CUDA devices found." << std::endl;
            exit(1);
        }
        cudaSetDevice(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        std::cout << "Using CUDA device: " << deviceProp.name << std::endl;

        data_host.resize(width * height * 3);
        data_device = data_host;

        cudaMemcpyToSymbol(c_center_x, &this->center_x, sizeof(double));
        cudaMemcpyToSymbol(c_center_y, &this->center_y, sizeof(double));
        cudaMemcpyToSymbol(c_width, &this->width, sizeof(int));
        cudaMemcpyToSymbol(c_height, &this->height, sizeof(int));
    }

private:
    int width, height;
    thrust::host_vector<uint8_t> data_host;
    thrust::device_vector<uint8_t> data_device;
    double center_x, center_y;
    double scale = 1.0;
    double ratio;
    double zoom;
    double x_start, x_finish, y_start, y_finish;
};
