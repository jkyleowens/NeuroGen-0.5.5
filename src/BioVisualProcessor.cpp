#include "NeuroGen/BioVisualProcessor.h"
#ifdef USE_OPENCV
#include <opencv2/imgproc.hpp>
#endif
#include <algorithm>

BioVisualProcessor::BioVisualProcessor(const std::string& name,
                                       const NetworkConfig& config,
                                       size_t neurons)
    : SpecializedModule(name, config, "visual_processor"),
      membrane_potential_(neurons, 0.0f),
      spike_output_(neurons, 0.0f),
      leak_rate_(0.1f),
      threshold_(1.0f),
      reset_potential_(0.0f) {}

bool BioVisualProcessor::initialize() {
    SpecializedModule::initialize();
    std::fill(membrane_potential_.begin(), membrane_potential_.end(), 0.0f);
    std::fill(spike_output_.begin(), spike_output_.end(), 0.0f);
    return true;
}

#ifdef USE_OPENCV
std::vector<float> BioVisualProcessor::processPixels(const cv::Mat& image) {
    if (image.empty()) return {};
    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else
        gray = image;

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(static_cast<int>(spike_output_.size()), 1));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    for (int i = 0; i < resized.cols && i < static_cast<int>(membrane_potential_.size()); ++i) {
        float input_current = resized.at<float>(0, i);
        membrane_potential_[i] += input_current - leak_rate_ * membrane_potential_[i];
        if (membrane_potential_[i] >= threshold_) {
            spike_output_[i] = 1.0f;
            membrane_potential_[i] = reset_potential_;
        } else {
            spike_output_[i] = 0.0f;
        }
    }
    return spike_output_;
}
#else
std::vector<float> BioVisualProcessor::processPixels(const std::vector<float>& pixels) {
    size_t count = std::min(pixels.size(), membrane_potential_.size());
    for (size_t i = 0; i < count; ++i) {
        float input_current = pixels[i];
        membrane_potential_[i] += input_current - leak_rate_ * membrane_potential_[i];
        if (membrane_potential_[i] >= threshold_) {
            spike_output_[i] = 1.0f;
            membrane_potential_[i] = reset_potential_;
        } else {
            spike_output_[i] = 0.0f;
        }
    }
    return spike_output_;
}
#endif
