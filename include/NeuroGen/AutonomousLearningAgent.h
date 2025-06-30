#ifndef AUTONOMOUS_LEARNING_AGENT_H
#define AUTONOMOUS_LEARNING_AGENT_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

// Core NeuroGen includes
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/NeuralModule.h>
#include <NeuroGen/TaskAutomationModules.h>
#include <NeuroGen/EnhancedLearningSystem.h>
#include <NeuroGen/Phase3IntegrationFramework.h>

// Vision and screen capture
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

// Forward declarations
struct ScreenData;
struct ModuleState;
struct AttentionState;
struct MemoryState;

/**
 * @brief Specialized neural module with independent state management
 * 
 * Each module is self-contained with its own neural network, memory,
 * and specialized processing capabilities that mirror biological brain regions.
 */
class SpecializedModule {
public:
    enum ModuleType {
        VISUAL_CORTEX,      // Visual processing and pattern recognition
        PREFRONTAL_CORTEX,  // Executive control and planning
        HIPPOCAMPUS,        // Memory formation and retrieval
        MOTOR_CORTEX,       // Motor planning and execution
        ATTENTION_SYSTEM,   // Central attention and coordination
        REWARD_SYSTEM,      // Dopaminergic reward processing
        WORKING_MEMORY      // Temporary information storage
    };

private:
    ModuleType type_;
    std::string name_;
    std::unique_ptr<NeuralModule> neural_network_;
    std::unique_ptr<DynamicNeuralNetwork> advanced_network_;
    NetworkConfig config_;
    
    // Module-specific state
    std::vector<float> internal_state_;
    std::vector<float> output_buffer_;
    std::vector<float> memory_traces_;
    
    // Inter-module connections
    std::unordered_map<std::string, float> input_connections_;
    std::unordered_map<std::string, float> output_connections_;
    
    // Learning and adaptation
    float attention_weight_;
    float learning_rate_;
    float activation_threshold_;
    bool is_active_;
    
    mutable std::mutex state_mutex_;

public:
    SpecializedModule(ModuleType type, const std::string& name, const NetworkConfig& config);
    ~SpecializedModule();
    
    bool initialize();
    std::vector<float> process(const std::vector<float>& input, float attention_weight = 1.0f);
    
    // Module-specific processing methods (MISSING DECLARATIONS ADDED)
    std::vector<float> process_visual_cortex(const std::vector<float>& visual_input);
    std::vector<float> process_prefrontal_cortex(const std::vector<float>& cognitive_input);
    std::vector<float> process_hippocampus(const std::vector<float>& memory_input);
    std::vector<float> process_motor_cortex(const std::vector<float>& motor_input);
    std::vector<float> process_attention_system(const std::vector<float>& attention_input);
    std::vector<float> process_reward_system(const std::vector<float>& reward_input);
    std::vector<float> process_working_memory(const std::vector<float>& memory_input);
    
    // Inter-module communication
    void connect_to_module(const std::string& target_module, float connection_strength);
    void receive_signal(const std::string& source_module, const std::vector<float>& signal);
    std::vector<float> get_output_for_module(const std::string& target_module) const;
    
    // State management
    bool save_module_state(const std::string& filename) const;
    bool load_module_state(const std::string& filename);
    void reset_to_baseline();
    
    // Monitoring and introspection
    float get_activation_level() const;
    float get_learning_progress() const;
    std::vector<float> get_internal_state() const;
    ModuleType get_type() const { return type_; }
    const std::string& get_name() const { return name_; }
    
    // Adaptation and plasticity
    void adjust_learning_rate(float new_rate);
    void apply_reward_signal(float reward);
    void enable_neuroplasticity(bool enable);
};

/**
 * @brief Central attention and control mechanism for module orchestration
 * 
 * Implements a biologically-inspired attention system that dynamically
 * allocates processing resources and coordinates inter-module communication.
 */
class AttentionController {
private:
    std::vector<float> module_attention_weights_;
    std::vector<std::string> module_names_;
    std::vector<float> context_features_;
    
    // Attention dynamics
    float attention_decay_rate_;
    float attention_boost_threshold_;
    float global_inhibition_strength_;
    
    // Context-dependent weighting
    std::unordered_map<std::string, float> context_priorities_;
    std::vector<float> current_context_;
    
public:
    AttentionController();
    
    void register_module(const std::string& module_name);
    void update_context(const std::vector<float>& new_context);
    void compute_attention_weights();
    
    float get_attention_weight(const std::string& module_name) const;
    void set_priority(const std::string& context, float priority);
    void apply_global_inhibition(float strength);
    
    std::vector<float> get_all_attention_weights() const;
};

/**
 * @brief Advanced memory system with episodic and working memory
 * 
 * Implements multiple memory systems that enable learning from experience
 * and maintaining context across extended interactions.
 */
class MemorySystem {
private:
    struct MemoryTrace {
        std::vector<float> state_vector;
        std::vector<float> action_vector;
        float reward;
        float timestamp;
        float importance_weight;
        int access_count;
    };
    
    std::vector<MemoryTrace> episodic_memory_;
    std::vector<float> working_memory_;
    std::unordered_map<std::string, std::vector<float>> semantic_memory_;
    
    // Memory parameters
    size_t max_episodic_capacity_;
    size_t working_memory_size_;
    float memory_decay_rate_;
    float consolidation_threshold_;
    
public:
    MemorySystem(size_t episodic_capacity = 10000, size_t working_capacity = 256);
    
    void store_episode(const std::vector<float>& state, const std::vector<float>& action, 
                      float reward, float importance = 1.0f);
    void update_working_memory(const std::vector<float>& new_information);
    void store_semantic_knowledge(const std::string& concept, const std::vector<float>& representation);
    
    std::vector<MemoryTrace> retrieve_similar_episodes(const std::vector<float>& query_state, 
                                                      int num_episodes = 5) const;
    std::vector<float> get_working_memory() const;
    std::vector<float> recall_semantic_knowledge(const std::string& concept) const;
    
    void consolidate_memories();
    void forget_low_importance_memories();
    
    bool save_memory_state(const std::string& filename) const;
    bool load_memory_state(const std::string& filename);
};

/**
 * @brief Screen capture and visual processing interface
 * 
 * Handles real-time screen capture, image preprocessing, and integration
 * with the visual cortex module for autonomous operation.
 */
class VisualInterface {
private:
    struct ScreenElement {
        int id;
        std::string type;
        int x, y, width, height;
        std::string text;
        bool is_clickable;
        float confidence;
    };
    
#ifdef USE_OPENCV
    cv::Mat current_screen_;
    cv::Mat processed_image_;
    std::vector<cv::Mat> feature_maps_;
#endif
    
    std::vector<ScreenElement> detected_elements_;
    std::vector<float> visual_features_;
    
    // Visual processing parameters
    int target_width_, target_height_;
    float detection_threshold_;
    bool enable_preprocessing_;
    
    std::mutex screen_mutex_;
    std::atomic<bool> capture_active_;
    std::thread capture_thread_;

public:
    VisualInterface(int width = 224, int height = 224);
    ~VisualInterface();
    
    bool initialize_capture();
    void start_continuous_capture();
    void stop_capture();
    
    std::vector<float> capture_and_process_screen();
    std::vector<ScreenElement> detect_screen_elements();
    void update_element_detection();
    
    // Visual feature extraction
    std::vector<float> extract_visual_features() const;
    std::vector<float> get_attention_map() const;
    bool is_element_visible(const ScreenElement& element) const;
    
    // Integration with neural modules
    void send_to_visual_cortex(SpecializedModule* visual_cortex);
    ScreenElement find_element_by_type(const std::string& type) const;
    
private:
    void capture_loop();
    void preprocess_image();
    void extract_text_elements();
    void detect_interactive_elements();
};

/**
 * @brief Main autonomous learning agent with modular brain architecture
 * 
 * Integrates all specialized modules into a cohesive system that can
 * autonomously browse the internet, learn from experience, and adapt
 * its behavior based on rewards and environmental feedback.
 */
class AutonomousLearningAgent {
private:
    // Core modular architecture
    std::unordered_map<std::string, std::unique_ptr<SpecializedModule>> modules_;
    std::unique_ptr<AttentionController> attention_controller_;
    std::unique_ptr<MemorySystem> memory_system_;
    std::unique_ptr<VisualInterface> visual_interface_;
    std::unique_ptr<EnhancedLearningSystem> learning_system_;
    
    // Global state and coordination
    std::vector<float> global_state_;
    std::vector<float> current_goals_;
    std::vector<float> environmental_context_;
    
    // Learning and adaptation
    float global_reward_signal_;
    float exploration_rate_;
    float learning_rate_;
    std::atomic<bool> is_learning_enabled_;
    
    // Autonomous operation
    std::atomic<bool> is_running_;
    std::thread main_loop_thread_;
    std::chrono::milliseconds update_interval_;
    
    // Performance monitoring
    struct PerformanceMetrics {
        float average_reward;
        float learning_progress;
        float exploration_efficiency;
        float memory_utilization;
        int successful_actions;
        int total_actions;
        std::chrono::steady_clock::time_point start_time;
    } metrics_;
    
    mutable std::mutex agent_mutex_;

    // NetworkConfig objects
    NetworkConfig create_visual_cortex_config();
    NetworkConfig create_cognitive_config();
    NetworkConfig create_motor_config();
    NetworkConfig create_memory_config();

public:
    AutonomousLearningAgent();
    ~AutonomousLearningAgent();
    
    // ========================================================================
    // INITIALIZATION AND CONFIGURATION
    // ========================================================================
    
    bool initialize();
    void configure_modules(const std::unordered_map<std::string, NetworkConfig>& module_configs);
    void setup_inter_module_connections();
    void initialize_visual_system();
    
    // ========================================================================
    // AUTONOMOUS OPERATION
    // ========================================================================
    
    void start_autonomous_operation();
    void stop_autonomous_operation();
    void set_update_frequency(int frequency_hz);
    
    // Main cognitive cycle
    void cognitive_cycle();
    void process_visual_input();
    void update_working_memory();
    void make_decision();
    void execute_action();
    void learn_from_feedback();
    
    // ========================================================================
    // LEARNING AND ADAPTATION
    // ========================================================================
    
    void set_reward_signal(float reward);
    void enable_learning(bool enable);
    void adjust_exploration_rate(float rate);
    void set_learning_goals(const std::vector<float>& goals);
    
    // Experience replay and consolidation
    void replay_experiences();
    void consolidate_learning();
    void transfer_knowledge_between_modules();
    
    // ========================================================================
    // INTERNET BROWSING AND INTERACTION
    // ========================================================================
    
    struct BrowsingAction {
        enum Type { CLICK, SCROLL, TYPE, NAVIGATE, WAIT };
        Type type;
        int x, y;  // Coordinates for click actions
        std::string text;  // Text for typing actions
        std::string url;   // URL for navigation
        float confidence;  // Action confidence
    };
    
    BrowsingAction plan_next_action();
    void execute_browsing_action(const BrowsingAction& action);
    void analyze_page_content();
    void extract_and_store_knowledge();
    
    // Autonomous browsing strategies
    void explore_website(const std::string& base_url);
    void focused_information_gathering(const std::string& topic);
    void adaptive_browsing_strategy();
    
    // ========================================================================
    // STATE MANAGEMENT AND PERSISTENCE
    // ========================================================================
    
    bool save_complete_state(const std::string& base_filename) const;
    bool load_complete_state(const std::string& base_filename);
    
    // Individual module state management
    bool save_module_state(const std::string& module_name, const std::string& filename) const;
    bool load_module_state(const std::string& module_name, const std::string& filename);
    
    // Memory and experience persistence
    bool save_experiences(const std::string& filename) const;
    bool load_experiences(const std::string& filename);
    
    // ========================================================================
    // MONITORING AND INTROSPECTION
    // ========================================================================
    
    PerformanceMetrics get_performance_metrics() const;
    std::vector<float> get_module_activation_levels() const;
    std::vector<float> get_attention_distribution() const;
    float get_overall_learning_progress() const;
    
    // Detailed system analysis
    void generate_performance_report(const std::string& filename) const;
    void log_cognitive_state() const;
    void visualize_module_interactions() const;
    
    // Real-time monitoring
    void enable_real_time_monitoring(bool enable);
    void set_monitoring_callback(std::function<void(const PerformanceMetrics&)> callback);
    
    // ========================================================================
    // ADVANCED FEATURES
    // ========================================================================
    
    // Meta-learning and self-improvement
    void enable_meta_learning(bool enable);
    void optimize_module_architecture();
    void adaptive_resource_allocation();
    
    // Multi-modal processing
    void enable_audio_processing(bool enable);
    void integrate_text_understanding();
    void cross_modal_learning();
    
    // Social learning and interaction
    void enable_social_learning(bool enable);
    void learn_from_user_demonstrations();
    void adapt_to_user_preferences();

private:
    // Internal processing methods
    void main_loop();
    void update_global_state();
    void coordinate_modules();
    void process_inter_module_signals();
    void update_attention_weights();
    void handle_environmental_changes();
    
    // Learning internals
    void compute_reward_prediction_error();
    void update_value_functions();
    void adjust_exploration_strategy();
    void prune_ineffective_connections();
    
    // Utility methods
    std::vector<float> normalize_vector(const std::vector<float>& input) const;
    float compute_similarity(const std::vector<float>& a, const std::vector<float>& b) const;
    void log_message(const std::string& message) const;
};

#endif // AUTONOMOUS_LEARNING_AGENT_H