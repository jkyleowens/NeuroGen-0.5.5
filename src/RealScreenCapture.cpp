#include "NeuroGen/RealScreenCapture.h"
#include <iostream>
#include <X11/Xutil.h>

bool RealScreenCapture::initialize(int display_width, int display_height) {
    width_ = display_width;
    height_ = display_height;
    x11_display_ = XOpenDisplay(nullptr);
    if (!x11_display_) {
        std::cerr << "RealScreenCapture: Failed to open X display" << std::endl;
        return false;
    }
    root_window_ = DefaultRootWindow(x11_display_);
    initialized_ = true;
    return true;
}

cv::Mat RealScreenCapture::captureScreen() {
    return captureRegion(0, 0, width_, height_);
}

cv::Mat RealScreenCapture::captureRegion(int x, int y, int width, int height) {
    if (!initialized_) return cv::Mat();
    XImage* img = XGetImage(x11_display_, root_window_, x, y, width, height, AllPlanes, ZPixmap);
    if (!img) {
        std::cerr << "RealScreenCapture: XGetImage failed" << std::endl;
        return cv::Mat();
    }
    cv::Mat mat(height, width, CV_8UC4, img->data);
    cv::Mat mat_bgr;
    cv::cvtColor(mat, mat_bgr, cv::COLOR_BGRA2BGR);
    XDestroyImage(img);
    return mat_bgr.clone();
}

void RealScreenCapture::shutdown() {
    if (x11_display_) {
        XCloseDisplay(x11_display_);
        x11_display_ = nullptr;
    }
    initialized_ = false;
}
