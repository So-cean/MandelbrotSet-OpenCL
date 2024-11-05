# Mandelbrot Set Visualization and Benchmark

## 项目简介

这个项目实现了Mandelbrot集合的可视化和性能基准测试。项目包含三种计算模式:单线程、OpenMP并行和OpenCL,并支持单精度和双精度计算。用户可以选不同的模式和精度来生成Mandelbrot集合,并进行性能测试。

## 参考
[link] https://github.com/Dylan8527/Mandelbrot-set-GPU

## 文件结构

- `main.cpp`:主程序文件,负责初始化OpenGL窗口,处理用户输入,并调用相应的计算函数生成Mandelbrot集合。
- `benchmark.cpp`:性能基准测试文件,包含不同计算模式的基准测试函数,并输出性能结果。
- `main_openmp.cpp`:OpenMP并行计算实现文件。
- `main_opencl.cpp`:OpenCL计算实现文件。
- `lodepng.h`:PNG图片编码库头文件。

## 依赖项

- OpenGL
- GLFW
- GLEW
- OpenCL
- OpenMP
- lodepng

## 编译和运行

### 编译

确保安装了所需的依赖项后,可以使用以下命令编译项目:

<!-- cmake -DCMAKE_TOOLCHAIN_FILE:STRING=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLE:STRING=x64-windows -DVCPKG_TARGET_TRIPLET:STRING=x64-windows -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -G "Visual Studio 17 2022" -T host=x64 -A x64 .. -->

```sh
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
```

可视化Mandelbrot集合
运行以下命令启动可视化程序:

```sh
./build/Release/Mandelbrot-Set
```

### 运行基准测试

运行以下命令启动性能基准测试:

```sh
./build/Release/benchmark
```

### 参数选项

程序启动后,用户可以选择以下参数:

#### 计算模式:

1. 单线程
2. OpenMP并行
3. OpenCL

#### 精度:

1. 单精度 (float)
2. 双精度 (double, 默认)