#include "NeuroGen/GUIElementDetector.h"

bool GUIElementDetector::initialize() {
    return true;
}

std::vector<ScreenElement> GUIElementDetector::detectElements(const cv::Mat& img) {
    std::vector<ScreenElement> elems;
    cv::Mat gray;
    if(img.channels()>1) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else gray = img;
    std::vector<std::vector<cv::Point>> contours;
    cv::Canny(gray, gray, 50,150);
    cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int id=1;
    for(const auto& c : contours) {
        cv::Rect r=cv::boundingRect(c);
        if(r.width>20 && r.height>10) {
            elems.emplace_back(id++,"button",r.x,r.y,r.width,r.height,"",true,0.6f);
        }
    }
    return elems;
}
