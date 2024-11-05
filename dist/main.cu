#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <chrono>

#include "main_openmp.cpp"
#include "main_opencl.cpp"
#include "main_cuda.cu"

#define WIDTH 800
#define HEIGHT 600


void renderImage(uint8_t* output, GLuint texture) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, output);

    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
    glEnd();

    glDisable(GL_TEXTURE_2D);
}

void updateParameters(double &scale, double& x_start, double& x_finish, double& y_start, double& y_finish, double center_x, double center_y, double ratio, double zoom_factor = 0.95) {
    // 更新缩放和中心位置
    scale *= zoom_factor;
    x_start = center_x - 0.5 * ratio * scale;
    x_finish = center_x + 0.5 * ratio * scale;
    y_start = center_y - 0.5 * scale;
    y_finish = center_y + 0.5 * scale;
}

void initOpenGL(GLFWwindow*& window, GLuint& texture) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(1);
    }

    window = glfwCreateWindow(WIDTH, HEIGHT, "Mandelbrot Set", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(1);
    }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        exit(1);
    }

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 显示窗口
    glfwShowWindow(window);
}

int main() {
    device_info();

    // 打印当前工作目录
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    int choice;
    std::cout << "Choose mode: 1. Single Thread 2. OpenMP 3. OpenCL 4. CUDA" << std::endl;
    std::cin >> choice;

    double x_start = -2.0f, x_finish = 2.0f;
    double y_start = -1.5f, y_finish = 1.5f;
    double center_x = -1.7497591451303665;//-0.10109636384562;//-0.77568377; //-0.748766710846959//-1.6735 //-1.7497591451303665
    double center_y =  -0.0000000036851380;//0.95628651080914;//0.13646737; //0.123640847970064//0.0003318 //-0.0000000036851380
    double zoom_factor = 0.98f;
    double scale = 1.0f;
    double ratio = static_cast<double> (WIDTH) / HEIGHT;

    GLFWwindow* window;
    GLuint texture;

    // 初始化 OpenGL
    initOpenGL(window, texture);

    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    std::vector<uint8_t> output(WIDTH * HEIGHT*3);

    MandelbrotOpenCL mandelbrotOpenCL(WIDTH, HEIGHT);
    MandelbrotCUDA mandelbrotCUDA(WIDTH, HEIGHT);
    mandelbrotCUDA.set_boundaries(x_start, x_finish, y_start, y_finish);
    mandelbrotCUDA.set_center(center_x, center_y);
    mandelbrotCUDA.set_zoom(zoom_factor);


    while (!glfwWindowShouldClose(window)) {
        updateParameters(scale, x_start, x_finish, y_start, y_finish, center_x, center_y, ratio, zoom_factor);

        if (choice == 1) {
            mandelbrot_single_thread(output.data(), WIDTH, HEIGHT, x_start, x_finish, y_start, y_finish, center_x, center_y);
        } else if (choice == 2) {
            mandelbrot_omp(output.data(), WIDTH, HEIGHT, x_start, x_finish, y_start, y_finish, center_x, center_y);
        } else if (choice == 3) {
            mandelbrotOpenCL.compute(output.data(), x_start, x_finish, y_start, y_finish, center_x, center_y);
        } else if (choice == 4) {
            mandelbrotCUDA.compute(x_start, x_finish, y_start, y_finish);
            
        }

        // 渲染图片
        if (choice != 4) {
            renderImage(output.data(), texture);
        } else {
            renderImage(mandelbrotCUDA.get_data(), texture);
        }

        // 显示 FPS
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - last_time;
        if (elapsed.count() >= 1.0f) {
            double fps = frame_count / elapsed.count();
            std::cout << "FPS: " << fps << std::endl;
            frame_count = 0;
            last_time = current_time;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
