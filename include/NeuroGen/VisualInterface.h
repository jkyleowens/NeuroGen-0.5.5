// ============================================================================
// VISUAL INTERFACE HEADER
// File: include/NeuroGen/VisualInterface.h
// ============================================================================

#ifndef VISUAL_INTERFACE_H
#define VISUAL_INTERFACE_H

#include "NeuroGen/ScreenElement.h"
#include "NeuroGen/RealScreenCapture.h"
#include "NeuroGen/GUIElementDetector.h"
#include "NeuroGen/OCRProcessor.h"
#include "NeuroGen/BioVisualProcessor.h"
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <map>

#include "NeuroGen/SpecializedModule.h"

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

// Forward declarations

// ============================================================================
// VISUAL INTERFACE CLASS DECLARATION
// ============================================================================

/**
 * @brief Visual interface for screen capture and processing
 */
class VisualInterface {
public:
    explicit VisualInterface(int width = 1920, int height = 1080);
    virtual ~VisualInterface();
    
    // Screen capture interface
    bool initialize_capture();
    void start_continuous_capture();
    void stop_capture();
    std::vector<float> capture_and_process_screen();
    
    // Element detection
    std::vector<ScreenElement> detect_screen_elements();
    void update_element_detection();
    ScreenElement find_element_by_type(const std::string& type) const;
    bool is_element_visible(const ScreenElement& element) const;
    
    // Visual processing
    std::vector<float> get_visual_features(const ScreenElement& element) const;
    cv::Mat get_last_frame() const;
    cv::Mat get_attention_map() const;
    std::vector<float> extract_visual_features() const;
    void apply_visual_feature_enhancement(std::vector<float>& features) const;
    void send_to_visual_cortex(const cv::Mat& image);

private:
    // Internal processing
    void capture_loop();
    void preprocess_image();
    void extract_text_elements();
    void detect_interactive_elements();

    // Configuration
    int target_width_, target_height_;
    float detection_threshold_;
    bool enable_preprocessing_, capture_active_;
    
    // State
    std::vector<ScreenElement> detected_elements_;
    std::vector<float> visual_features_;
    std::thread capture_thread_;
    mutable std::mutex screen_mutex_;
    
    cv::Mat current_screen_;
    std::chrono::steady_clock::time_point last_capture_time_;
    
    std::unique_ptr<RealScreenCapture> real_screen_capture_;
    std::unique_ptr<GUIElementDetector> gui_detector_;
    std::unique_ptr<OCRProcessor> ocr_processor_;
    std::unique_ptr<BioVisualProcessor> visual_processor_;
};


#endif // VISUAL_INTERFACE_H