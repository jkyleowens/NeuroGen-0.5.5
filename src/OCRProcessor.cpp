#include "NeuroGen/OCRProcessor.h"
#include <iostream>

bool OCRProcessor::initialize() {
    tess_api_ = new tesseract::TessBaseAPI();
    if (tess_api_->Init(nullptr, "eng")) {
        std::cerr << "OCRProcessor: failed to init tesseract" << std::endl;
        delete tess_api_;
        tess_api_ = nullptr;
        return false;
    }
    initialized_ = true;
    return true;
}

void OCRProcessor::shutdown() {
    if (tess_api_) {
        tess_api_->End();
        delete tess_api_;
        tess_api_ = nullptr;
    }
    initialized_ = false;
}

std::string OCRProcessor::extractText(const cv::Mat& image) {
    if (!initialized_) return "";
    tess_api_->SetImage(image.data, image.cols, image.rows, image.channels(), image.step);
    char* out = tess_api_->GetUTF8Text();
    std::string text(out ? out : "");
    if(out) {
        last_confidence_ = tess_api_->MeanTextConf()/100.0f;
        delete [] out;
    } else {
        last_confidence_ = 0.0f;
    }
    return text;
}
