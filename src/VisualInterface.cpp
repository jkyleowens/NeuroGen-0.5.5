// ============================================================================
// VISUAL INTERFACE AND SCREEN INTERACTION IMPLEMENTATION
// File: src/VisualInterface.cpp
// ============================================================================

#include <NeuroGen/VisualInterface.h>
#include <NeuroGen/SpecializedModule.h>
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

// ============================================================================
// VISUAL INTERFACE IMPLEMENTATION
// ============================================================================

VisualInterface::VisualInterface(int width, int height) 
    : target_width_(width), target_height_(height), detection_threshold_(0.5f),
      enable_preprocessing_(true), capture_active_(false) {
    
    std::cout << "Visual Interface: Initializing breakthrough visual processing system..." << std::endl;
    std::cout << "  - Target resolution: " << width << "x" << height << std::endl;
    std::cout << "  - Detection threshold: " << detection_threshold_ << std::endl;
}

VisualInterface::~VisualInterface() {
    stop_capture();
    std::cout << "Visual Interface: Visual processing system shutdown complete." << std::endl;
}

bool VisualInterface::initialize_capture() {
#ifdef USE_OPENCV
    std::cout << "Visual Interface: Initializing OpenCV screen capture..." << std::endl;
    
    // Initialize screen capture dimensions
    cv::Size screen_size(1920, 1080); // Default HD resolution
    current_screen_ = cv::Mat::zeros(screen_size, CV_8UC3);
    processed_image_ = cv::Mat::zeros(target_height_, target_width_, CV_8UC3);
    
    // Initialize feature maps for hierarchical processing
    feature_maps_.resize(4);
    for (auto& map : feature_maps_) {
        map = cv::Mat::zeros(target_height_ / 4, target_width_ / 4, CV_32F);
    }
    
    std::cout << "✓ OpenCV visual processing initialized successfully" << std::endl;
    return true;
#else
    std::cout << "WARNING: OpenCV not available - using simulated visual input" << std::endl;
    
    // Create simulated visual data
    visual_features_.resize(target_width_ * target_height_ * 3);
    detected_elements_.clear();
    
    // Add some simulated screen elements
    detected_elements_.push_back({1, "button", 100, 100, 120, 40, "Click Me", true, 0.9f});
    detected_elements_.push_back({2, "textbox", 100, 200, 200, 30, "", true, 0.8f});
    detected_elements_.push_back({3, "link", 300, 150, 80, 20, "More Info", true, 0.7f});
    
    return true;
#endif
}

void VisualInterface::start_continuous_capture() {
    if (capture_active_) return;
    
    std::cout << "Visual Interface: Starting continuous screen capture..." << std::endl;
    capture_active_ = true;
    capture_thread_ = std::thread(&VisualInterface::capture_loop, this);
    std::cout << "✓ Continuous visual processing active" << std::endl;
}

void VisualInterface::stop_capture() {
    if (!capture_active_) return;
    
    std::cout << "Visual Interface: Stopping screen capture..." << std::endl;
    capture_active_ = false;
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    std::cout << "✓ Screen capture stopped" << std::endl;
}

std::vector<float> VisualInterface::capture_and_process_screen() {
    std::lock_guard<std::mutex> lock(screen_mutex_);
    
#ifdef USE_OPENCV
    // Simulate screen capture (in real implementation, use platform-specific APIs)
    // For demonstration, create a synthetic screen with various elements
    current_screen_ = cv::Mat::zeros(1080, 1920, CV_8UC3);
    
    // Draw simulated UI elements
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

std::vector<VisualInterface::ScreenElement> VisualInterface::detect_screen_elements() {
    std::lock_guard<std::mutex> lock(screen_mutex_);
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
    
    int element_id = 1;
    for (const auto& contour : contours) {
        cv::Rect bbox = cv::boundingRect(contour);
        
        // Filter by size (reasonable UI element dimensions)
        if (bbox.width > 20 && bbox.height > 10 && bbox.width < 500 && bbox.height < 200) {
            
            // Classify element type based on dimensions and characteristics
            std::string element_type;
            bool is_clickable = true;
            float confidence = 0.8f;
            
            float aspect_ratio = float(bbox.width) / bbox.height;
            
            if (aspect_ratio > 3.0f && bbox.height < 40) {
                element_type = "button";
                confidence = 0.9f;
            } else if (aspect_ratio > 2.0f && bbox.height < 60) {
                element_type = "textbox";
                confidence = 0.8f;
            } else if (bbox.width < 150 && bbox.height < 30) {
                element_type = "link";
                confidence = 0.7f;
            } else {
                element_type = "unknown";
                confidence = 0.5f;
                is_clickable = false;
            }
            
            detected_elements_.push_back({
                element_id++,
                element_type,
                bbox.x, bbox.y, bbox.width, bbox.height,
                "", // Text extraction would go here
                is_clickable,
                confidence
            });
        }
    }
    
    std::cout << "Visual Interface: Detected " << detected_elements_.size() << " screen elements" << std::endl;
    
#else
    // Simulated element detection - elements already initialized in initialize_capture()
    std::cout << "Visual Interface: Using simulated element detection (" 
              << detected_elements_.size() << " elements)" << std::endl;
#endif
}

std::vector<float> VisualInterface::extract_visual_features() const {
    std::vector<float> features;
    
#ifdef USE_OPENCV
    if (current_screen_.empty()) {
        return std::vector<float>(target_width_ * target_height_ * 3, 0.0f);
    }
    
    // Resize to target dimensions
    cv::Mat resized;
    cv::resize(current_screen_, resized, cv::Size(target_width_, target_height_));
    
    // Convert to float and normalize
    cv::Mat float_image;
    resized.convertTo(float_image, CV_32F, 1.0/255.0);
    
    // Extract features in CHW format (channels, height, width)
    features.reserve(target_width_ * target_height_ * 3);
    
    std::vector<cv::Mat> channels;
    cv::split(float_image, channels);
    
    for (const auto& channel : channels) {
        for (int y = 0; y < channel.rows; ++y) {
            for (int x = 0; x < channel.cols; ++x) {
                features.push_back(channel.at<float>(y, x));
            }
        }
    }
    
    // Apply additional feature processing
    apply_visual_feature_enhancement(features);
    
#else
    // Return simulated features
    features = visual_features_;
#endif
    
    return features;
}

void VisualInterface::apply_visual_feature_enhancement(std::vector<float>& features) const {
    if (features.empty()) return;
    
    // Apply edge enhancement
    size_t channels = 3;
    size_t pixels_per_channel = features.size() / channels;
    size_t width = target_width_;
    size_t height = target_height_;
    
    std::vector<float> enhanced_features = features;
    
    // Sobel edge detection-like enhancement
    for (size_t c = 0; c < channels; ++c) {
        size_t channel_offset = c * pixels_per_channel;
        
        for (size_t y = 1; y < height - 1; ++y) {
            for (size_t x = 1; x < width - 1; ++x) {
                size_t idx = channel_offset + y * width + x;
                
                // Sobel X
                float gx = -features[channel_offset + (y-1)*width + (x-1)] + features[channel_offset + (y-1)*width + (x+1)]
                          -2*features[channel_offset + y*width + (x-1)] + 2*features[channel_offset + y*width + (x+1)]
                          -features[channel_offset + (y+1)*width + (x-1)] + features[channel_offset + (y+1)*width + (x+1)];
                
                // Sobel Y
                float gy = -features[channel_offset + (y-1)*width + (x-1)] - 2*features[channel_offset + (y-1)*width + x] - features[channel_offset + (y-1)*width + (x+1)]
                          +features[channel_offset + (y+1)*width + (x-1)] + 2*features[channel_offset + (y+1)*width + x] + features[channel_offset + (y+1)*width + (x+1)];
                
                // Edge magnitude
                float edge_magnitude = std::sqrt(gx*gx + gy*gy);
                enhanced_features[idx] = std::min(1.0f, features[idx] + edge_magnitude * 0.2f);
            }
        }
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
                float dx = (x - center_x) / target_width_ * 1920; // Scale to screen coordinates
                float dy = (y - center_y) / target_height_ * 1080;
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
           element.x + element.width <= 1920 && // Assuming screen width
           element.y + element.height <= 1080;  // Assuming screen height
}

void VisualInterface::send_to_visual_cortex(SpecializedModule* visual_cortex) {
    if (!visual_cortex) return;
    
    std::vector<float> visual_features = extract_visual_features();
    std::vector<float> attention_map = get_attention_map();
    
    // Combine visual features with attention
    std::vector<float> attended_features;
    attended_features.reserve(visual_features.size());
    
    size_t attention_scale = visual_features.size() / attention_map.size();
    
    for (size_t i = 0; i < visual_features.size(); ++i) {
        size_t attention_idx = i / std::max(size_t(1), attention_scale);
        float attention_weight = (attention_idx < attention_map.size()) ? attention_map[attention_idx] : 1.0f;
        attended_features.push_back(visual_features[i] * (0.5f + 0.5f * attention_weight));
    }
    
    // Process through visual cortex
    auto visual_output = visual_cortex->process(attended_features);
    
    std::cout << "Visual Interface: Sent " << attended_features.size() 
              << " features to visual cortex, received " << visual_output.size() 
              << " outputs" << std::endl;
}

VisualInterface::ScreenElement VisualInterface::find_element_by_type(const std::string& type) const {
    for (const auto& element : detected_elements_) {
        if (element.type == type && element.is_clickable) {
            return element;
        }
    }
    
    // Return empty element if not found
    return {-1, "", 0, 0, 0, 0, "", false, 0.0f};
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
    
    std::cout << "Attention Controller: Initializing biologically-inspired attention system..." << std::endl;
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
            for (size_t j = 0; j < std::min(context_features_.size(), size_t(100)); ++j) {
                visual_richness += std::abs(context_features_[j]);
            }
            base_weight = 0.5f + 1.5f * std::tanh(visual_richness / 50.0f);
            
        } else if (module_name == "prefrontal_cortex") {
            // High cognitive attention for complex decisions
            float complexity = 0.0f;
            for (size_t j = 0; j < context_features_.size(); ++j) {
                complexity += context_features_[j] * context_features_[j];
            }
            base_weight = 0.3f + 1.7f * std::tanh(complexity / 100.0f);
            
        } else if (module_name == "hippocampus") {
            // High memory attention for novel situations
            float novelty = 0.0f;
            for (size_t j = 0; j < context_features_.size(); ++j) {
                novelty += std::abs(context_features_[j] - 0.5f); // Deviation from baseline
            }
            base_weight = 0.4f + 1.2f * std::tanh(novelty / 30.0f);
            
        } else if (module_name == "motor_cortex") {
            // High motor attention when action is needed
            float action_urgency = context_priorities_.count("action") ? context_priorities_["action"] : 0.5f;
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
    
    // Debug output occasionally
    static int update_count = 0;
    if (++update_count % 100 == 0) {
        std::cout << "Attention Controller: Updated weights - ";
        for (size_t i = 0; i < module_names_.size(); ++i) {
            std::cout << module_names_[i] << ":" << module_attention_weights_[i] << " ";
        }
        std::cout << std::endl;
    }
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

// ============================================================================
// MEMORY SYSTEM IMPLEMENTATION
// ============================================================================

MemorySystem::MemorySystem(size_t episodic_capacity, size_t working_capacity) 
    : max_episodic_capacity_(episodic_capacity), working_memory_size_(working_capacity),
      memory_decay_rate_(0.001f), consolidation_threshold_(0.8f) {
    
    std::cout << "Memory System: Initializing advanced memory architecture..." << std::endl;
    std::cout << "  - Episodic memory capacity: " << episodic_capacity << " episodes" << std::endl;
    std::cout << "  - Working memory size: " << working_capacity << " elements" << std::endl;
    
    episodic_memory_.reserve(max_episodic_capacity_);
    working_memory_.resize(working_memory_size_, 0.0f);
}

void MemorySystem::store_episode(const MemoryTrace& trace) {
    // Store the trace directly
    if (episodic_memory_.size() >= max_episodic_capacity_) {
        // Remove oldest episode
        episodic_memory_.erase(episodic_memory_.begin());
    }
    
    episodic_memory_.push_back(trace);
    
    std::cout << "Memory System: Stored new episode (total: " 
              << episodic_memory_.size() << "/" << max_episodic_capacity_ << ")" << std::endl;
    
    static int episode_count = 0;
    if (++episode_count % 1000 == 0) {
        std::cout << "Memory System: Stored " << episode_count << " episodes (capacity: " 
                  << episodic_memory_.size() << "/" << max_episodic_capacity_ << ")" << std::endl;
    }
}

void MemorySystem::update_working_memory(const std::vector<float>& new_information) {
    if (new_information.empty()) return;
    
    size_t update_size = std::min(new_information.size(), working_memory_.size());
    
    // Exponential moving average update
    for (size_t i = 0; i < update_size; ++i) {
        working_memory_[i] = working_memory_[i] * 0.9f + new_information[i] * 0.1f;
    }
}

std::vector<MemorySystem::MemoryTrace> MemorySystem::retrieve_similar_episodes(
    const std::vector<float>& query_state, size_t max_results) const {
    
    if (episodic_memory_.empty() || query_state.empty()) {
        return {};
    }
    
    // Compute similarities and sort
    std::vector<std::pair<float, size_t>> similarities;
    similarities.reserve(episodic_memory_.size());
    
    for (size_t i = 0; i < episodic_memory_.size(); ++i) {
        const auto& trace = episodic_memory_[i];
        float similarity = compute_cosine_similarity(query_state, trace.state_features);
        
        // Weight by importance and recency 
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - trace.timestamp);
        float recency_weight = std::exp(-memory_decay_rate_ * duration.count());
        
        float weighted_similarity = similarity * trace.confidence_level * recency_weight;
        similarities.emplace_back(weighted_similarity, i);
    }
    
    // Sort by similarity (highest first)
    std::partial_sort(similarities.begin(), 
                     similarities.begin() + std::min(max_results, similarities.size()),
                     similarities.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Return top episodes
    std::vector<MemoryTrace> result;
    result.reserve(max_results);
    
    for (size_t i = 0; i < max_results && i < similarities.size(); ++i) {
        size_t idx = similarities[i].second;
        result.push_back(episodic_memory_[idx]);
    }
    
    return result;
}

float MemorySystem::compute_cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) const {
    if (a.empty() || b.empty()) return 0.0f;
    
    size_t min_size = std::min(a.size(), b.size());
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < min_size; ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    float denominator = std::sqrt(norm_a * norm_b);
    return (denominator > 1e-8f) ? (dot_product / denominator) : 0.0f;
}

std::vector<float> MemorySystem::get_working_memory() const {
    return working_memory_;
}

void MemorySystem::consolidate_memories() {
    if (episodic_memory_.empty()) return;
    
    // Simple consolidation: boost confidence of important memories
    for (auto& trace : episodic_memory_) {
        // Decay confidence based on age 
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - trace.timestamp);
        
        if (duration.count() > 300000) { // 5 minutes old
            trace.confidence_level *= 0.95f;
        }
    }
    
    std::cout << "Memory System: Consolidated " << episodic_memory_.size() << " memory traces" << std::endl;
}

size_t MemorySystem::get_episodic_memory_size() const {
    return episodic_memory_.size();
}

float MemorySystem::get_memory_utilization() const {
    return static_cast<float>(episodic_memory_.size()) / static_cast<float>(max_episodic_capacity_);
}