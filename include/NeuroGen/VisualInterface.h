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
#include "NeuroGen/VisualFeatureExtractor.h"
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <map>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

// Forward declarations
class SpecializedModule;

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
    std::vector<float> extract_visual_features() const;
    void apply_visual_feature_enhancement(std::vector<float>& features) const;
    std::vector<float> get_attention_map() const;
    
    // Module integration
    void send_to_visual_cortex(SpecializedModule* visual_cortex);

private:
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
    
    std::unique_ptr<RealScreenCapture> real_screen_capture_;
    std::unique_ptr<GUIElementDetector> gui_detector_;
    std::unique_ptr<OCRProcessor> ocr_processor_;
    std::unique_ptr<VisualFeatureExtractor> feature_extractor_;

    // Internal methods
    void capture_loop();
    void preprocess_image();
    void extract_text_elements();
    void detect_interactive_elements();
};

// ============================================================================
// ATTENTION CONTROLLER CLASS DECLARATION
// ============================================================================

/**
 * @brief Attention Controller for Module Coordination
 */
class AttentionController {
public:
    AttentionController();
    virtual ~AttentionController() = default;
    
    void register_module(const std::string& module_name);
    void update_context(const std::vector<float>& new_context);
    void compute_attention_weights();
    float get_attention_weight(const std::string& module_name) const;
    void set_priority(const std::string& context, float priority);
    void apply_global_inhibition(float strength);
    std::vector<float> get_all_attention_weights() const;

private:
    std::vector<std::string> module_names_;
    std::vector<float> module_attention_weights_;
    std::vector<float> current_context_;
    std::vector<float> context_features_;
    std::map<std::string, float> context_priorities_;
    float attention_decay_rate_;
    float attention_boost_threshold_;
    float global_inhibition_strength_;
};

#endif // VISUAL_INTERFACE_H