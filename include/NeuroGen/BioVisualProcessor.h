#ifndef BIO_VISUAL_PROCESSOR_H
#define BIO_VISUAL_PROCESSOR_H

#include <NeuroGen/SpecializedModule.h>
#include <vector>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

/**
 * @brief Biologically inspired visual processor using a simple LIF network.
 */
class BioVisualProcessor : public SpecializedModule {
public:
    BioVisualProcessor(const std::string& name, const NetworkConfig& config,
                       size_t neurons = 128);

    bool initialize() override;

#ifdef USE_OPENCV
    std::vector<float> processPixels(const cv::Mat& image);
#else
    std::vector<float> processPixels(const std::vector<float>& pixels);
#endif

private:
    std::vector<float> membrane_potential_;
    std::vector<float> spike_output_;
    float leak_rate_;
    float threshold_;
    float reset_potential_;
};

#endif // BIO_VISUAL_PROCESSOR_H
