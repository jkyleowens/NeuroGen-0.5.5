#ifndef VISUAL_FEATURE_EXTRACTOR_H
#define VISUAL_FEATURE_EXTRACTOR_H
#include <vector>
#include <opencv2/opencv.hpp>
#include "NeuroGen/ScreenElement.h"

class VisualFeatureExtractor {
public:
    std::vector<float> extractFeatures(const cv::Mat& screen_image);
    std::vector<ScreenElement> detectElements(const cv::Mat& screen_image);
    cv::Mat preprocessImage(const cv::Mat& input);
private:
    cv::HOGDescriptor hog_;
};

#endif // VISUAL_FEATURE_EXTRACTOR_H
