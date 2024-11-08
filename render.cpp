#include <CL/opencl.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cstdlib> 
#include "main_opencl.cpp"
#include "lodepng.h"

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

    glfwShowWindow(window);
}

template<typename T>
void computeMandelbrot(uint8_t* output, int width, int height, T x_start, T x_finish, T y_start, T y_finish, T center_x, T center_y, MandelbrotOpenCL& mandelbrotOpenCL) {
    mandelbrotOpenCL.compute(output, x_start, x_finish, y_start, y_finish, center_x, center_y);
}

void flipVertically(uint8_t* data, int width, int height) {
    int row_size = width * 3;
    std::vector<uint8_t> temp(row_size);
    for (int i = 0; i < height / 2; ++i) {
        uint8_t* row1 = data + i * row_size;
        uint8_t* row2 = data + (height - i - 1) * row_size;
        std::memcpy(temp.data(), row1, row_size);
        std::memcpy(row1, row2, row_size);
        std::memcpy(row2, temp.data(), row_size);
    }
}

int main(int argc, char* argv[]) {
    int num_frames = 360; // 默认帧数
    int frame_rate = 60; // 默认帧率

    if (argc > 1) {
        num_frames = std::stoi(argv[1]);
    }
    if (argc > 2) {
        frame_rate = std::stoi(argv[2]);
    }

    double x_start = -2.0, x_finish = 2.0;
    double y_start = -1.5, y_finish = 1.5;
    double center_x = -0.77568377;
    double center_y = 0.13646737;
    double zoom_factor = 0.98;
    double scale = 1.0;
    double ratio = static_cast<double>(WIDTH) / HEIGHT;

    GLFWwindow* window;
    GLuint texture;

    initOpenGL(window, texture);

    std::vector<uint8_t> output(WIDTH * HEIGHT * 3);
    MandelbrotOpenCL mandelbrotOpenCL(WIDTH, HEIGHT);

    std::filesystem::create_directory("frames");

    for (int i = 0; i < num_frames; ++i) {
        updateParameters(scale, x_start, x_finish, y_start, y_finish, center_x, center_y, ratio, zoom_factor);
        computeMandelbrot(output.data(), WIDTH, HEIGHT, x_start, x_finish, y_start, y_finish, center_x, center_y, mandelbrotOpenCL);

        renderImage(output.data(), texture);
        glfwSwapBuffers(window);
        glfwPollEvents();

        // 保存当前帧为 PNG 文件
        flipVertically(output.data(), WIDTH, HEIGHT); // 添加这行代码进行垂直翻转
        std::string filename = "frames/frame_" + std::to_string(i) + ".png";
        lodepng::encode(filename, output, WIDTH, HEIGHT, LCT_RGB);
    }

    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();

    // 使用 ffmpeg 合成 GIF 文件
    std::string ffmpeg_command = "ffmpeg -framerate " + std::to_string(frame_rate) + " -i frames/frame_%d.png -vf \"scale=" + std::to_string(WIDTH) + ":-1:flags=lanczos\" -c:v gif -y output.gif";
    std::system(ffmpeg_command.c_str());

    

    return 0;
}
