__kernel void mandelbrot(__global uchar* output, const int width, const int height,
                         const double x_start, const double x_finish,
                         const double y_start, const double y_finish,
                         const double center_x, const double center_y) {
    int x = get_global_id(0); 
    int y = get_global_id(1); 

    if (x >= width || y >= height) {
        return;
    }

    double dx = (x_finish - x_start) / width;
    double dy = (y_finish - y_start) / height;
    double real = x_start + x * dx;
    double imag = y_start + y * dy;

    double c_real = real;
    double c_imag = imag;

    int max_iter = 256;
    int iter = 0;
    double real2, imag2;

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

    double t = (double)iter / max_iter;
    uchar r, g, b;

    if (iter == max_iter) {
        r = g = b = 0; // 黑色
    } else {
        double t1 = 1 - t;
        r = (uchar)(9 * t1 * t * t * t * 255);
        g = (uchar)(15 * t1 * t1 * t * t * 255);
        b = (uchar)(8.5 * t1 * t1 * t1 * t * 255);
    }

    int idx = (y * width + x) * 3;
    output[idx] = r;
    output[idx + 1] = g;
    output[idx + 2] = b;
}
