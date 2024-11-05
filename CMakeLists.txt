cmake_minimum_required(VERSION 3.10)
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project(Mandelbrot-Set LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# 设置 CUDA 架构
# set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(fmt CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCL CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(lodepng CONFIG REQUIRED)

if (OPENMP_FOUND)
  message("OK, you find OpenMP!")
else()
  message(FATAL_ERROR "OpenMP not found!")
endif()

set(CMAKE_BUILD_TYPE "Release")

# 添加 main.cpp 可执行文件
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::HeadersCpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL)
target_link_libraries(${PROJECT_NAME} PRIVATE GLUT::GLUT)
target_link_libraries(${PROJECT_NAME} PRIVATE GLEW::GLEW)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)


# 添加 benchmark.cu 可执行文件
# add_executable(benchmark_cu benchmark.cu)
# target_link_libraries(benchmark_cu PRIVATE fmt::fmt)
# target_link_libraries(benchmark_cu PRIVATE OpenMP::OpenMP_CXX)
# target_link_libraries(benchmark_cu PRIVATE OpenCL::OpenCL)
# target_link_libraries(benchmark_cu PRIVATE OpenCL::HeadersCpp)
# target_link_libraries(benchmark_cu PRIVATE OpenGL::GL)
# target_link_libraries(benchmark_cu PRIVATE GLUT::GLUT)
# target_link_libraries(benchmark_cu PRIVATE GLEW::GLEW)
# target_link_libraries(benchmark_cu PRIVATE glfw)
# target_link_libraries(benchmark_cu PRIVATE cuda)
# target_link_libraries(benchmark_cu PRIVATE lodepng)


# 添加 benchmark.cpp 可执行文件
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE fmt::fmt)
target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(benchmark PRIVATE OpenCL::OpenCL)
target_link_libraries(benchmark PRIVATE OpenCL::HeadersCpp)
target_link_libraries(benchmark PRIVATE OpenGL::GL)
target_link_libraries(benchmark PRIVATE GLUT::GLUT)
target_link_libraries(benchmark PRIVATE GLEW::GLEW)
target_link_libraries(benchmark PRIVATE glfw)
target_link_libraries(benchmark PRIVATE lodepng)


# 添加 main.cu 可执行文件
# add_executable(main_cu main.cu)
# target_link_libraries(main_cu PRIVATE fmt::fmt)
# target_link_libraries(main_cu PRIVATE OpenGL::GL)
# target_link_libraries(main_cu PRIVATE GLUT::GLUT)
# target_link_libraries(main_cu PRIVATE GLEW::GLEW)
# target_link_libraries(main_cu PRIVATE OpenMP::OpenMP_CXX)
# target_link_libraries(main_cu PRIVATE OpenCL::OpenCL)
# target_link_libraries(main_cu PRIVATE OpenCL::HeadersCpp)
# target_link_libraries(main_cu PRIVATE glfw)
# target_link_libraries(main_cu PRIVATE cuda)


# 将 kernal.cl 文件复制到构建目录
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/kernal.cl"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>)

