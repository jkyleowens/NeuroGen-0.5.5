#ifndef REAL_SCREEN_CAPTURE_H
#define REAL_SCREEN_CAPTURE_H

#include <opencv2/opencv.hpp>
#include <X11/Xlib.h>

class RealScreenCapture {
public:
    bool initialize(int display_width = 1920, int display_height = 1080);
    cv::Mat captureScreen();
    cv::Mat captureRegion(int x, int y, int width, int height);
    bool isInitialized() const { return initialized_; }
    void shutdown();
private:
    Display* x11_display_ = nullptr;
    Window root_window_ = 0;
    int width_ = 0;
    int height_ = 0;
    bool initialized_ = false;
};

#endif // REAL_SCREEN_CAPTURE_H
