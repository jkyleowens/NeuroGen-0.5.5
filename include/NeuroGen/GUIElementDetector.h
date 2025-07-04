#ifndef GUI_ELEMENT_DETECTOR_H
#define GUI_ELEMENT_DETECTOR_H
#include <vector>
#include <opencv2/opencv.hpp>
#include "NeuroGen/ScreenElement.h"

class GUIElementDetector {
public:
    bool initialize();
    std::vector<ScreenElement> detectElements(const cv::Mat& screen_image);
private:
    cv::HOGDescriptor button_hog_;
};

#endif // GUI_ELEMENT_DETECTOR_H
