#include "NeuroGen/VisualFeatureExtractor.h"

std::vector<float> VisualFeatureExtractor::extractFeatures(const cv::Mat& img) {
    cv::Mat gray = preprocessImage(img);
    std::vector<float> descriptors;
    std::vector<cv::Point> loc;
    hog_.compute(gray, descriptors, cv::Size(8,8), cv::Size(0,0), loc);
    return descriptors;
}

std::vector<ScreenElement> VisualFeatureExtractor::detectElements(const cv::Mat& img) {
    std::vector<ScreenElement> elems;
    cv::Mat gray = preprocessImage(img);
    std::vector<std::vector<cv::Point>> contours;
    cv::Canny(gray, gray, 50,150);
    cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int id=1;
    for(const auto& c : contours) {
        cv::Rect r = cv::boundingRect(c);
        if(r.width>20 && r.height>10) {
            elems.emplace_back(id++,"unknown",r.x,r.y,r.width,r.height,"",false,0.5f);
        }
    }
    return elems;
}

cv::Mat VisualFeatureExtractor::preprocessImage(const cv::Mat& input) {
    cv::Mat gray;
    if(input.channels()==3) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else if(input.channels()==4) cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    else gray=input.clone();
    return gray;
}
