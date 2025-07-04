// ============================================================================
// VISUAL INTERFACE AND SCREEN INTERACTION IMPLEMENTATION - FIXED
// File: src/VisualInterface.cpp
// ============================================================================

#include "NeuroGen/VisualInterface.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#endif

// Forward declaration for specialized module
class SpecializedModule {
public:
    virtual void process(const std::vector<float>& input, float attention_weight) = 0;
    virtual ~SpecializedModule() = default;
};

// ============================================================================
// VISUAL INTERFACE IMPLEMENTATION
// ============================================================================

VisualInterface::VisualInterface(int width, int height) 
    : target_width_(width), target_height_(height), detection_threshold_(0.5f),
      enable_preprocessing_(true), capture_active_(false) {
    
    std::cout << "Visual Interface: Initializing visual processing system..." << std::endl;
    std::cout << "  - Target resolution: " << width << "x" << height << std::endl;
    std::cout << "  - Detection threshold: " << detection_threshold_ << std::endl;
}

VisualInterface::~VisualInterface() {
    stop_capture();
    std::cout << "Visual Interface: Visual processing system shutdown complete." << std::endl;
}

bool VisualInterface::initialize_capture() {
    // Initialize screen capture system
    std::cout << "Initializing screen capture system..." << std::endl;
    
    // Platform-specific initialization would go here
    // For now, return success for simulation
    return true;
}

void VisualInterface::start_continuous_capture() {
    if (capture_active_) return;
    
    capture_active_ = true;
    std::cout << "Starting continuous visual capture..." << std::endl;
    
    // Start capture thread
    capture_thread_ = std::thread(&VisualInterface::capture_loop, this);
}

void VisualInterface::stop_capture() {
    if (!capture_active_) return;
    
    capture_active_ = false;
    std::cout << "Stopping visual capture..." << std::endl;
    
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
}

std::vector<float> VisualInterface::capture_and_process_screen() {
#ifdef USE_OPENCV
    // Real OpenCV implementation would go here
    current_screen_ = cv::Mat::zeros(target_height_, target_width_, CV_8UC3);
    
    // Generate some test UI elements
    cv::rectangle(current_screen_, cv::Point(100, 100), cv::Point(220, 140), cv::Scalar(100, 200, 100), -1);
    cv::putText(current_screen_, "Button", cv::Point(120, 125), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    
    cv::rectangle(current_screen_, cv::Point(100, 200), cv::Point(300, 230), cv::Scalar(255, 255, 255), 2);
    cv::putText(current_screen_, "Text Input", cv::Point(110, 220), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
    
    cv::rectangle(current_screen_, cv::Point(300, 150), cv::Point(380, 170), cv::Scalar(0, 100, 200), -1);
    cv::putText(current_screen_, "Link", cv::Point(310, 165), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    
    // Preprocess the image
    if (enable_preprocessing_) {
        preprocess_image();
    }
    
    // Extract visual features
    visual_features_ = extract_visual_features();
    
    // Update element detection
    update_element_detection();
    
#else
    // Simulated visual processing without OpenCV
    visual_features_.resize(target_width_ * target_height_ * 3);
    
    // Generate synthetic visual features
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    for (size_t i = 0; i < visual_features_.size(); ++i) {
        visual_features_[i] = 0.5f + noise(gen); // Baseline + noise
    }
    
    // Add feature patterns around detected elements
    for (const auto& element : detected_elements_) {
        size_t feature_idx = (element.y * target_width_ + element.x) % visual_features_.size();
        if (feature_idx < visual_features_.size()) {
            visual_features_[feature_idx] = element.confidence;
        }
    }
#endif
    
    return visual_features_;
}

std::vector<ScreenElement> VisualInterface::detect_screen_elements() {
    std::lock_guard<std::mutex> lock(screen_mutex_);
    
    // Simulate detection of screen elements
    detected_elements_.clear();
    
    // Add some example elements
    ScreenElement button(1, "button", 100, 100, 120, 40, "Click Me", true, 0.9f);
    detected_elements_.push_back(button);
    
    ScreenElement textbox(2, "textbox", 100, 200, 200, 30, "", false, 0.8f);
    detected_elements_.push_back(textbox);
    
    ScreenElement link(3, "link", 300, 150, 80, 20, "Link", true, 0.7f);
    detected_elements_.push_back(link);
    
    return detected_elements_;
}

void VisualInterface::update_element_detection() {
#ifdef USE_OPENCV
    detected_elements_.clear();
    
    if (current_screen_.empty()) return;
    
    // Convert to grayscale for edge detection
    cv::Mat gray;
    cv::cvtColor(current_screen_, gray, cv::COLOR_BGR2GRAY);
    
    // Detect edges using Canny
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    
    // Find contours (potential UI elements)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Process contours into screen elements
    int element_id = 1;
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        
        // Filter by size to avoid noise
        if (rect.width > 20 && rect.height > 10) {
            ScreenElement element(element_id++, "unknown", rect.x, rect.y, 
                                rect.width, rect.height, "", false, 0.5f);
            
            // Simple classification based on dimensions
            float aspect_ratio = static_cast<float>(rect.width) / rect.height;
            if (aspect_ratio > 2.0f && rect.height < 50) {
                element.type = "button";
                element.is_clickable = true;
                element.confidence = 0.7f;
            } else if (aspect_ratio > 3.0f && rect.height < 40) {
                element.type = "textbox";
                element.confidence = 0.6f;
            }
            
            detected_elements_.push_back(element);
        }
    }
#else
    // Simulated element detection
    detect_screen_elements();
#endif
}

std::vector<float> VisualInterface::extract_visual_features() const {
    std::vector<float> features(256, 0.0f);
    
    // Extract features based on detected elements
    for (size_t i = 0; i < std::min(detected_elements_.size(), size_t(16)); ++i) {
        const auto& element = detected_elements_[i];
        size_t base_idx = i * 16;
        
        if (base_idx + 15 < features.size()) {
            // Position features (normalized)
            features[base_idx + 0] = element.x / static_cast<float>(target_width_);
            features[base_idx + 1] = element.y / static_cast<float>(target_height_);
            features[base_idx + 2] = element.width / static_cast<float>(target_width_);
            features[base_idx + 3] = element.height / static_cast<float>(target_height_);
            
            // Type features
            if (element.type == "button") features[base_idx + 4] = 1.0f;
            else if (element.type == "textbox") features[base_idx + 5] = 1.0f;
            else if (element.type == "link") features[base_idx + 6] = 1.0f;
            
            // Properties
            features[base_idx + 7] = element.is_clickable ? 1.0f : 0.0f;
            features[base_idx + 8] = element.confidence;
        }
    }
    
    return features;
}

void VisualInterface::apply_visual_feature_enhancement(std::vector<float>& features) const {
    // Apply enhancement filters to improve feature quality
    std::vector<float> enhanced_features = features;
    
    // Simple edge enhancement
    for (size_t i = 1; i < features.size() - 1; ++i) {
        float edge_magnitude = std::abs(features[i-1] - features[i+1]);
        enhanced_features[i] = features[i] + (edge_magnitude * 0.2f);
    }
    
    features = enhanced_features;
}

std::vector<float> VisualInterface::get_attention_map() const {
    std::vector<float> attention_map(target_width_ * target_height_, 0.0f);
    
    // Generate attention map based on detected elements
    for (const auto& element : detected_elements_) {
        if (!element.is_clickable) continue;
        
        // Create Gaussian attention blob around element
        float center_x = element.x + element.width / 2.0f;
        float center_y = element.y + element.height / 2.0f;
        float sigma = std::max(element.width, element.height) / 4.0f;
        
        for (int y = 0; y < target_height_; ++y) {
            for (int x = 0; x < target_width_; ++x) {
                float dx = (x - center_x);
                float dy = (y - center_y);
                float distance_sq = dx*dx + dy*dy;
                float attention_value = element.confidence * std::exp(-distance_sq / (2 * sigma * sigma));
                
                size_t idx = y * target_width_ + x;
                if (idx < attention_map.size()) {
                    attention_map[idx] = std::max(attention_map[idx], attention_value);
                }
            }
        }
    }
    
    return attention_map;
}

bool VisualInterface::is_element_visible(const ScreenElement& element) const {
    return element.confidence > detection_threshold_ && 
           element.x >= 0 && element.y >= 0 &&
           element.x + element.width <= target_width_ &&
           element.y + element.height <= target_height_;
}

void VisualInterface::send_to_visual_cortex(SpecializedModule* visual_cortex) {
    if (!visual_cortex) return;
    
    std::vector<float> visual_features = extract_visual_features();
    std::vector<float> attention_map = get_attention_map();
    
    // Combine visual features with attention
    std::vector<float> attended_features;
    attended_features.reserve(visual_features.size());
    
    size_t attention_scale = visual_features.size() / std::max(size_t(1), attention_map.size());
    
    for (size_t i = 0; i < visual_features.size(); ++i) {
        size_t attention_idx = i / std::max(size_t(1), attention_scale);
        float attention_weight = (attention_idx < attention_map.size()) ? 
                                attention_map[attention_idx] : 1.0f;
        attended_features.push_back(visual_features[i] * (0.5f + 0.5f * attention_weight));
    }
    
    // Process through visual cortex
    visual_cortex->process(attended_features, 1.0f);
}

ScreenElement VisualInterface::find_element_by_type(const std::string& type) const {
    for (const auto& element : detected_elements_) {
        if (element.type == type && element.is_clickable) {
            return element;
        }
    }
    
    // Return empty element if not found
    return ScreenElement(-1, "", 0, 0, 0, 0, "", false, 0.0f);
}

void VisualInterface::capture_loop() {
    std::cout << "Visual Interface: Capture loop started" << std::endl;
    
    while (capture_active_) {
        try {
            capture_and_process_screen();
            
            // Update at ~10 FPS for visual processing
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
        } catch (const std::exception& e) {
            std::cerr << "Visual Interface: Error in capture loop: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Visual Interface: Capture loop terminated" << std::endl;
}

void VisualInterface::preprocess_image() {
#ifdef USE_OPENCV
    if (current_screen_.empty()) return;
    
    // Noise reduction
    cv::Mat denoised;
    cv::bilateralFilter(current_screen_, denoised, 5, 50, 50);
    
    // Contrast enhancement
    cv::Mat enhanced;
    denoised.convertTo(enhanced, -1, 1.2, 10); // alpha=1.2 (contrast), beta=10 (brightness)
    
    current_screen_ = enhanced;
#endif
}

void VisualInterface::extract_text_elements() {
    // Text extraction would use OCR here (Tesseract, etc.)
    // For now, simulated based on element detection
    for (auto& element : detected_elements_) {
        if (element.type == "button") {
            element.text = "Button_" + std::to_string(element.id);
        } else if (element.type == "link") {
            element.text = "Link_" + std::to_string(element.id);
        }
    }
}

void VisualInterface::detect_interactive_elements() {
    // Enhanced interactive element detection
    // This would include cursor hover detection, focus detection, etc.
    for (auto& element : detected_elements_) {
        // Simulate interaction detection based on element properties
        if (element.width > 50 && element.height > 20 && element.type != "unknown") {
            element.is_clickable = true;
            element.confidence = std::min(1.0f, element.confidence + 0.1f);
        }
    }
}

// ============================================================================
// ATTENTION CONTROLLER IMPLEMENTATION
// ============================================================================

AttentionController::AttentionController() 
    : attention_decay_rate_(0.95f), attention_boost_threshold_(0.7f), global_inhibition_strength_(0.1f) {
    
    std::cout << "Attention Controller: Initializing attention system..." << std::endl;
}

void AttentionController::register_module(const std::string& module_name) {
    module_names_.push_back(module_name);
    module_attention_weights_.push_back(1.0f); // Start with equal attention
    
    std::cout << "Attention Controller: Registered module '" << module_name 
              << "' (total modules: " << module_names_.size() << ")" << std::endl;
}

void AttentionController::update_context(const std::vector<float>& new_context) {
    current_context_ = new_context;
    
    // Update context features for attention computation
    if (context_features_.size() != new_context.size()) {
        context_features_.resize(new_context.size());
    }
    
    // Exponential moving average of context
    for (size_t i = 0; i < std::min(context_features_.size(), new_context.size()); ++i) {
        context_features_[i] = context_features_[i] * 0.9f + new_context[i] * 0.1f;
    }
    
    // Recompute attention weights based on new context
    compute_attention_weights();
}

void AttentionController::compute_attention_weights() {
    if (module_names_.empty()) return;
    
    std::vector<float> new_weights(module_attention_weights_.size());
    
    // Context-dependent attention computation
    for (size_t i = 0; i < module_names_.size(); ++i) {
        const std::string& module_name = module_names_[i];
        float base_weight = 1.0f;
        
        // Module-specific attention rules based on context
        if (module_name == "visual_cortex") {
            // High visual attention when visual context is rich
            float visual_richness = 0.0f;
            for (size_t j = 0; j < std::min(context_features_.size(), size_t(64)); ++j) {
                visual_richness += context_features_[j];
            }
            visual_richness /= 64.0f;
            base_weight = 0.3f + 1.4f * visual_richness;
            
        } else if (module_name == "working_memory") {
            // Memory attention based on context complexity
            float complexity = 0.0f;
            for (const auto& feature : context_features_) {
                complexity += std::abs(feature - 0.5f);
            }
            complexity /= context_features_.size();
            base_weight = 0.4f + 1.2f * complexity;
            
        } else if (module_name == "action_execution") {
            // Action attention based on urgency
            float action_urgency = context_priorities_.count("action") ? 
                                  context_priorities_["action"] : 0.5f;
            base_weight = 0.2f + 1.3f * action_urgency;
            
        } else {
            // Default attention for other modules
            base_weight = 0.6f + 0.4f * (rand() / float(RAND_MAX)); // Some randomness
        }
        
        // Apply decay to previous weight
        float decayed_weight = module_attention_weights_[i] * attention_decay_rate_;
        
        // Compute new weight with boost if above threshold
        new_weights[i] = std::max(decayed_weight, base_weight);
        if (new_weights[i] > attention_boost_threshold_) {
            new_weights[i] = std::min(2.0f, new_weights[i] * 1.2f); // Boost strong attention
        }
    }
    
    // Apply global inhibition (competition between modules)
    float total_attention = 0.0f;
    for (float weight : new_weights) {
        total_attention += weight;
    }
    
    float inhibition_factor = 1.0f - global_inhibition_strength_ * (total_attention - module_names_.size());
    inhibition_factor = std::max(0.1f, std::min(inhibition_factor, 1.0f));
    
    // Normalize and apply inhibition
    for (size_t i = 0; i < new_weights.size(); ++i) {
        new_weights[i] *= inhibition_factor;
        new_weights[i] = std::max(0.1f, std::min(new_weights[i], 2.0f)); // Bound attention weights
    }
    
    module_attention_weights_ = new_weights;
}

float AttentionController::get_attention_weight(const std::string& module_name) const {
    for (size_t i = 0; i < module_names_.size(); ++i) {
        if (module_names_[i] == module_name) {
            return module_attention_weights_[i];
        }
    }
    return 1.0f; // Default weight if module not found
}

void AttentionController::set_priority(const std::string& context, float priority) {
    context_priorities_[context] = std::max(0.0f, std::min(priority, 2.0f));
    
    std::cout << "Attention Controller: Set priority '" << context << "' = " << priority << std::endl;
}

void AttentionController::apply_global_inhibition(float strength) {
    global_inhibition_strength_ = std::max(0.0f, std::min(strength, 1.0f));
    
    // Recompute weights with new inhibition
    compute_attention_weights();
}

std::vector<float> AttentionController::get_all_attention_weights() const {
    return module_attention_weights_;
}