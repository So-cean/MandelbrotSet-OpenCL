#include <iostream>
#include <vector>
#include <omp.h>

template<typename T>
void mandelbrot_omp(uint8_t* output, int width, int height, T x_start, T x_finish, T y_start, T y_finish, T center_x, T center_y) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            T dx = (x_finish - x_start) / width;
            T dy = (y_finish - y_start) / height;
            T real = x_start + x * dx;
            T imag = y_start + y * dy;

            T c_real = real;
            T c_imag = imag;

            int max_iter = 256;
            int iter = 0;
            T real2, imag2;

            for (int i = 0; i < max_iter; ++i) {
                real2 = real * real;
                imag2 = imag * imag;
                if (real2 + imag2 > 4.0) {
                    break;
                }
                imag = 2 * real * imag + c_imag;
                real = real2 - imag2 + c_real;
                iter++;
            }

            double t = static_cast<double>(iter) / max_iter;
            uint8_t r, g, b;

            if (iter == max_iter) {
                r = g = b = 0; // 黑色
            } else {
                double t1 = 1 - t;
                r = static_cast<uint8_t>(9 * t1 * t * t * t * 255);
                g = static_cast<uint8_t>(15 * t1 * t1 * t * t * 255);
                b = static_cast<uint8_t>(8.5 * t1 * t1 * t1 * t * 255);
            }

            int idx = y * width * 3 + x * 3;
            output[idx] = r;
            output[idx + 1] = g;
            output[idx + 2] = b;
        }
    }
}

template<typename T>
void mandelbrot_single_thread(uint8_t* output, int width, int height, T x_start, T x_finish, T y_start, T y_finish, T center_x, T center_y) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            T dx = (x_finish - x_start) / width;
            T dy = (y_finish - y_start) / height;
            T real = x_start + x * dx;
            T imag = y_start + y * dy;

            T c_real = real;
            T c_imag = imag;

            int max_iter = 256;
            int iter = 0;
            T real2, imag2;

            for (int i = 0; i < max_iter; ++i) {
                real2 = real * real;
                imag2 = imag * imag;
                if (real2 + imag2 > 4.0) {
                    break;
                }
                imag = 2 * real * imag + c_imag;
                real = real2 - imag2 + c_real;
                iter++;
            }

            double t = static_cast<double>(iter) / max_iter;
            uint8_t r, g, b;

            if (iter == max_iter) {
                r = g = b = 0; // 黑色
            } else {
                double t1 = 1 - t;
                r = static_cast<uint8_t>(9 * t1 * t * t * t * 255);
                g = static_cast<uint8_t>(15 * t1 * t1 * t * t * 255);
                b = static_cast<uint8_t>(8.5 * t1 * t1 * t1 * t * 255);
            }

            int idx = y * width * 3 + x * 3;
            output[idx] = r;
            output[idx + 1] = g;
            output[idx + 2] = b;
        }
    }
}

