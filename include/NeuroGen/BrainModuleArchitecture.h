#ifndef BRAIN_MODULE_ARCHITECTURE_H
#define BRAIN_MODULE_ARCHITECTURE_H

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include "NeuroGen/NeuralModule.h"
#include "NeuroGen/ModularNeuralNetwork.h"
#include "NeuroGen/NetworkConfig.h"

/**
 * @brief Brain-inspired modular architecture for autonomous learning agent
 * 
 * This class implements the complete brain-inspired modular architecture
 * with proper initialization, dynamic sizing, and inter-module connections
 * based on the design document specifications.
 */
class BrainModuleArchitecture {
public:
    // ========================================================================
    // MODULE TYPES (Based on Design Document)
    // ========================================================================
    
    enum class ModuleType {
        VISUAL_CORTEX,          // Perception Module (Visual processing)
        COMPREHENSION_MODULE,   // Language & Symbol Interpretation
        EXECUTIVE_FUNCTION,     // Prefrontal Cortex (Goal management & planning)
        MEMORY_MODULE,          // Hippocampus & Neocortex (Memory systems)
        CENTRAL_CONTROLLER,     // Neuromodulator Regulation
        OUTPUT_MODULE,          // Spike-to-Data Translation
        MOTOR_CORTEX,          // Action execution
        REWARD_SYSTEM,         // Dopaminergic reward processing
        ATTENTION_SYSTEM       // Attention and resource allocation
    };
    
    // ========================================================================
    // MODULE CONFIGURATION STRUCTURES
    // ========================================================================
    
    struct ModuleConfig {
        ModuleType type;
        std::string name;
        size_t input_size;
        size_t output_size;
        size_t internal_neurons;
        size_t cortical_columns;
        float learning_rate;
        float plasticity_strength;
        bool enable_stdp;
        bool enable_homeostasis;
        std::vector<std::string> input_connections;
        std::vector<std::string> output_connections;
    };
    
    struct InterModuleConnection {
        std::string source_module;
        std::string source_port;
        std::string target_module;
        std::string target_port;
        float connection_strength;
        bool plastic;
        size_t connection_size;
    };
    
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Constructor with base configuration
     */
    BrainModuleArchitecture();
    
    /**
     * @brief Destructor
     */
    ~BrainModuleArchitecture();
    
    /**
     * @brief Initialize the complete brain architecture
     * @param screen_width Width of visual input
     * @param screen_height Height of visual input
     * @return Success status
     */
    bool initialize(int screen_width = 1920, int screen_height = 1080);
    
    /**
     * @brief Shutdown the architecture
     */
    void shutdown();
    
    // ========================================================================
    // MODULE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Create and configure a brain module
     * @param config Module configuration
     * @return Success status
     */
    bool createModule(const ModuleConfig& config);
    
    /**
     * @brief Connect two modules with specified parameters
     * @param connection Connection specification
     * @return Success status
     */
    bool connectModules(const InterModuleConnection& connection);
    
    /**
     * @brief Get module by name
     * @param name Module name
     * @return Pointer to module or nullptr
     */
    NeuralModule* getModule(const std::string& name);
    
    /**
     * @brief Get module by name (const version)
     * @param name Module name
     * @return Const pointer to module or nullptr
     */
    const NeuralModule* getModule(const std::string& name) const;
    
    /**
     * @brief Get all module names
     * @return Vector of module names
     */
    std::vector<std::string> getModuleNames() const;
    
    // ========================================================================
    // PROCESSING PIPELINE
    // ========================================================================
    
    /**
     * @brief Process visual input through the architecture
     * @param visual_input Raw visual data
     * @return Processed visual features
     */
    std::vector<float> processVisualInput(const std::vector<float>& visual_input);
    
    /**
     * @brief Process text/symbolic input
     * @param text_input Text or symbolic data
     * @return Comprehension output
     */
    std::vector<float> processTextInput(const std::vector<float>& text_input);
    
    /**
     * @brief Execute decision-making process
     * @param context_input Current context
     * @param goals Current goals
     * @return Decision output
     */
    std::vector<float> executeDecisionMaking(const std::vector<float>& context_input,
                                           const std::vector<float>& goals);
    
    /**
     * @brief Generate motor output
     * @param decision_input Decision from executive function
     * @return Motor commands
     */
    std::vector<float> generateMotorOutput(const std::vector<float>& decision_input);
    
    /**
     * @brief Update all modules with time step
     * @param dt Time step
     */
    void update(float dt);
    
    // ========================================================================
    // MEMORY OPERATIONS
    // ========================================================================
    
    /**
     * @brief Store experience in memory modules
     * @param experience Experience vector
     * @param context Context information
     */
    void storeExperience(const std::vector<float>& experience, const std::string& context);
    
    /**
     * @brief Retrieve similar experiences
     * @param query Query vector
     * @param max_results Maximum results to return
     * @return Retrieved experiences
     */
    std::vector<std::vector<float>> retrieveExperiences(const std::vector<float>& query, 
                                                       size_t max_results = 5);
    
    // ========================================================================
    // ATTENTION AND CONTROL
    // ========================================================================
    
    /**
     * @brief Update attention weights based on context
     * @param context Current context
     */
    void updateAttention(const std::vector<float>& context);
    
    /**
     * @brief Apply neuromodulation to all modules
     * @param reward_signal Global reward signal
     * @param attention_signal Attention modulation
     */
    void applyNeuromodulation(float reward_signal, const std::vector<float>& attention_signal);
    
    /**
     * @brief Get current attention weights
     * @return Map of module names to attention weights
     */
    std::map<std::string, float> getAttentionWeights() const;
    
    // ========================================================================
    // LEARNING AND ADAPTATION
    // ========================================================================
    
    /**
     * @brief Apply learning signal to all modules
     * @param reward Global reward signal
     * @param prediction_error Prediction error signal
     */
    void applyLearning(float reward, float prediction_error);
    
    /**
     * @brief Update synaptic connections between modules
     */
    void updateInterModuleConnections();
    
    /**
     * @brief Get learning statistics
     * @return Learning metrics for all modules
     */
    std::map<std::string, std::map<std::string, float>> getLearningStats() const;
    
    // ========================================================================
    // PERFORMANCE MONITORING
    // ========================================================================
    
    /**
     * @brief Get architecture performance metrics
     * @return Performance metrics
     */
    std::map<std::string, float> getPerformanceMetrics() const;
    
    /**
     * @brief Check if architecture is stable
     * @return Stability status
     */
    bool isStable() const;
    
    /**
     * @brief Get total network activity
     * @return Average activity across all modules
     */
    float getTotalActivity() const;

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    // Core modular network
    std::unique_ptr<ModularNeuralNetwork> modular_network_;
    
    // Module configurations
    std::map<std::string, ModuleConfig> module_configs_;
    std::vector<InterModuleConnection> connections_;
    
    // Dynamic sizing parameters
    int visual_input_width_;
    int visual_input_height_;
    size_t visual_feature_size_;
    size_t context_size_;
    size_t goal_size_;
    size_t action_size_;
    
    // Attention and control
    std::map<std::string, float> attention_weights_;
    std::vector<float> global_context_;
    float global_reward_signal_;
    
    // Performance tracking
    bool is_initialized_;
    float total_activity_;
    size_t update_count_;
    
    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================
    
    /**
     * @brief Create default module configurations
     */
    void createDefaultConfigurations();
    
    /**
     * @brief Calculate dynamic sizes based on input dimensions
     */
    void calculateDynamicSizes();
    
    /**
     * @brief Initialize inter-module connections
     */
    void initializeConnections();
    
    /**
     * @brief Create visual cortex module
     */
    bool createVisualCortex();
    
    /**
     * @brief Create comprehension module
     */
    bool createComprehensionModule();
    
    /**
     * @brief Create executive function module
     */
    bool createExecutiveFunction();
    
    /**
     * @brief Create memory module
     */
    bool createMemoryModule();
    
    /**
     * @brief Create central controller
     */
    bool createCentralController();
    
    /**
     * @brief Create output module
     */
    bool createOutputModule();
    
    /**
     * @brief Create motor cortex
     */
    bool createMotorCortex();
    
    /**
     * @brief Create reward system
     */
    bool createRewardSystem();
    
    /**
     * @brief Create attention system
     */
    bool createAttentionSystem();
    
    /**
     * @brief Validate module configuration
     * @param config Configuration to validate
     * @return Validation result
     */
    bool validateModuleConfig(const ModuleConfig& config) const;
    
    /**
     * @brief Update performance metrics
     */
    void updatePerformanceMetrics();
    
    /**
     * @brief Process inter-module signals
     */
    void processInterModuleSignals();
};

#endif // BRAIN_MODULE_ARCHITECTURE_H
