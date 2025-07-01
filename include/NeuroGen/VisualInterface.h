// ============================================================================
// VISUAL INTERFACE AND SCREEN INTERACTION HEADER
// File: include/NeuroGen/VisualInterface.h
// ============================================================================

#ifndef VISUAL_INTERFACE_H
#define VISUAL_INTERFACE_H

#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <memory>
#include <unordered_map>

// Forward declarations
class SpecializedModule;

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

// ============================================================================
// VISUAL INTERFACE CLASS
// ============================================================================

/**
 * @brief Advanced visual interface for screen processing and interaction
 * 
 * This class provides breakthrough visual processing capabilities including
 * screen capture, element detection, feature extraction, and attention-based
 * visual processing for autonomous agents.
 */
class VisualInterface {
public:
    /**
     * @brief Screen element structure for detected UI elements
     */
    struct ScreenElement {
        int id;                 // Unique identifier for the element
        std::string type;       // Type: "button", "textbox", "link", "image", etc.
        int x, y;              // Position coordinates (top-left corner)
        int width, height;     // Dimensions
        std::string text;      // Text content (if any)
        bool is_clickable;     // Whether the element can be clicked
        float confidence;      // Detection confidence (0.0 - 1.0)
        
        // Default constructor
        ScreenElement() : id(0), x(0), y(0), width(0), height(0), is_clickable(false), confidence(0.0f) {}
        
        // Main constructor
        ScreenElement(int _id, const std::string& _type, int _x, int _y, int _w, int _h, 
                      const std::string& _text, bool _clickable, float _conf = 0.8f)
            : id(_id), type(_type), x(_x), y(_y), width(_w), height(_h), 
              text(_text), is_clickable(_clickable), confidence(_conf) {}
    };

public:
    // Constructor and destructor
    VisualInterface(int width = 224, int height = 224);
    ~VisualInterface();

    // Core visual processing methods
    bool initialize_capture();
    void start_continuous_capture();
    void stop_capture();
    std::vector<float> capture_and_process_screen();

    // Element detection and analysis
    std::vector<ScreenElement> detect_screen_elements();
    void update_element_detection();
    ScreenElement find_element_by_type(const std::string& type) const;
    bool is_element_visible(const ScreenElement& element) const;

    // Feature extraction and processing
    std::vector<float> extract_visual_features() const;
    std::vector<float> get_attention_map() const;
    void apply_visual_feature_enhancement(std::vector<float>& features) const;

    // Integration with neural modules
    void send_to_visual_cortex(SpecializedModule* visual_cortex);

private:
    // Internal processing methods
    void capture_loop();
    void preprocess_image();
    void extract_text_elements();
    void detect_interactive_elements();

    // Core parameters
    int target_width_;
    int target_height_;
    float detection_threshold_;
    bool enable_preprocessing_;
    bool capture_active_;

    // Threading and synchronization
    std::thread capture_thread_;
    mutable std::mutex screen_mutex_;

    // Visual data storage
    std::vector<float> visual_features_;
    std::vector<ScreenElement> detected_elements_;

#ifdef USE_OPENCV
    // OpenCV-specific members
    cv::Mat current_screen_;
    cv::Mat processed_image_;
    std::vector<cv::Mat> feature_maps_;
#endif
};

// ============================================================================
// ATTENTION CONTROLLER CLASS
// ============================================================================

/**
 * @brief Biologically-inspired attention controller for visual processing
 * 
 * Manages attention allocation across multiple neural modules based on
 * context, novelty, and task relevance using neurobiological principles.
 */
class AttentionController {
public:
    AttentionController();

    // Module management
    void register_module(const std::string& module_name);
    void update_context(const std::vector<float>& new_context);

    // Attention computation and retrieval
    void compute_attention_weights();
    float get_attention_weight(const std::string& module_name) const;
    std::vector<float> get_all_attention_weights() const;

    // Attention modulation
    void set_priority(const std::string& context, float priority);
    void apply_global_inhibition(float strength);

private:
    // Module tracking
    std::vector<std::string> module_names_;
    std::vector<float> module_attention_weights_;

    // Context and features
    std::vector<float> current_context_;
    std::vector<float> context_features_;
    std::unordered_map<std::string, float> context_priorities_;

    // Attention parameters
    float attention_decay_rate_;
    float attention_boost_threshold_;
    float global_inhibition_strength_;
};

// ============================================================================
// MEMORY SYSTEM CLASS
// ============================================================================

/**
 * @brief Advanced memory system with episodic and working memory
 * 
 * Implements biologically-inspired memory mechanisms including episodic
 * memory storage, working memory, and memory consolidation processes.
 */
class MemorySystem {
public:
    /**
     * @brief Memory trace structure for episodic memory
     */
    struct MemoryTrace {
        std::vector<float> state_features;    // State representation
        std::vector<float> action_taken;      // Action that was taken
        float reward_received;                // Reward from this episode
        float confidence_level;               // Confidence in this memory
        float temporal_discount;              // Temporal discount factor
        std::chrono::high_resolution_clock::time_point timestamp;  // When this occurred
        
        MemoryTrace() : reward_received(0.0f), confidence_level(0.0f), temporal_discount(0.95f) {}
        
        MemoryTrace(const std::vector<float>& state, const std::vector<float>& action, 
                   float reward, float confidence, float discount = 0.95f)
            : state_features(state), action_taken(action), reward_received(reward), 
              confidence_level(confidence), temporal_discount(discount),
              timestamp(std::chrono::high_resolution_clock::now()) {}
    };

public:
    // Constructor
    MemorySystem(size_t episodic_capacity = 10000, size_t working_capacity = 128);

    // Memory operations
    void store_episode(const MemoryTrace& trace);
    std::vector<MemoryTrace> retrieve_similar_episodes(const std::vector<float>& query_state, 
                                                      size_t max_results = 5) const;
    void update_working_memory(const std::vector<float>& new_information);
    void consolidate_memories();

    // Memory access
    std::vector<float> get_working_memory() const;
    size_t get_episodic_memory_size() const;
    float get_memory_utilization() const;

private:
    // Helper methods
    float compute_cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) const;
    // Memory storage
    std::vector<MemoryTrace> episodic_memory_;
    std::vector<float> working_memory_;

    // Memory parameters
    size_t max_episodic_capacity_;
    size_t working_memory_size_;
    float memory_decay_rate_;
    float consolidation_threshold_;
};

#endif // VISUAL_INTERFACE_H
