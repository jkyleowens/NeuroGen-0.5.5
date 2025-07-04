#ifndef OCR_PROCESSOR_H
#define OCR_PROCESSOR_H
#include <string>
#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>

class OCRProcessor {
public:
    bool initialize();
    void shutdown();
    std::string extractText(const cv::Mat& image);
    float getConfidence() const { return last_confidence_; }
private:
    tesseract::TessBaseAPI* tess_api_ = nullptr;
    bool initialized_ = false;
    float last_confidence_ = 0.0f;
};

#endif // OCR_PROCESSOR_H
