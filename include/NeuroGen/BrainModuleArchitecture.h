// ============================================================================
// BRAIN MODULE ARCHITECTURE HEADER - NLP-FOCUSED (FIXED CIRCULAR DEPENDENCY)
// File: include/NeuroGen/BrainModuleArchitecture.h
// ============================================================================

#ifndef BRAIN_MODULE_ARCHITECTURE_H
#define BRAIN_MODULE_ARCHITECTURE_H

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <set>
#include <queue>
#include <thread>
#include <condition_variable>

// NeuroGen Framework Includes
#include "NeuroGen/NeuralModule.h"
#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/ModularNeuralNetwork.h"
#include "NeuroGen/NetworkConfig.h"
// REMOVED: #include "NeuroGen/LearningStateManager.h"  // This was causing circular dependency

// Forward declarations to avoid circular dependencies
class LearningStateManager;  // Forward declaration only
class NetworkCUDA;
class ContinuousLearningAgent;

/**
 * @brief Simplified Brain-inspired Architecture for Natural Language Processing
 * 
 * This class implements a streamlined brain-inspired modular architecture
 * specifically designed for natural language processing. It consists of
 * five core modules:
 * 
 * 1. Central Controller - Neuromodulatory control and orchestration
 * 2. Input Module - Text tokenization and input processing  
 * 3. Language Processing Module - Deep language understanding
 * 4. Reasoning Module - Logical reasoning and inference
 * 5. Output Module - Spike-to-action conversion for response generation
 * 
 * Key Features:
 * - Simplified 5-module architecture optimized for NLP
 * - Inter-module connections form processing pipeline
 * - Neuromodulatory control from central controller
 * - Attention mechanisms for language processing
 * - Independent state saving/loading for each module
 * - Continuous learning through language interaction
 */
class BrainModuleArchitecture : public std::enable_shared_from_this<BrainModuleArchitecture> {
public:
    // ========================================================================
    // CORE CONFIGURATION STRUCTURES
    // ========================================================================
    
    struct ArchitectureConfig {
        int max_modules = 5;                    // Fixed to 5 for NLP architecture
        bool enable_inter_module_learning = true;
        bool enable_attention_mechanism = true;
        bool enable_memory_consolidation = true;
        bool enable_structural_plasticity = false; // Simplified for NLP
        float global_inhibition_strength = 0.1f;
        float attention_update_rate = 0.01f;
        float memory_consolidation_rate = 0.005f;
        std::string architecture_type = "nlp_focused";
    };
    
    struct ModuleConfig {
        std::string module_name;
        std::string module_type;
        int num_neurons = 1024;
        int input_size = 512;
        int output_size = 512;
        float learning_rate = 0.01f;
        float attention_weight = 0.5f;
        bool enable_plasticity = true;
        bool is_excitatory = true;
    };
    
    struct InterModuleConnection {
        std::string source_module;
        std::string target_module;
        float connection_strength = 0.5f;
        std::string connection_type = "excitatory"; // excitatory, inhibitory, modulatory
        bool is_active = true;
        bool is_plastic = true;
        float plasticity_rate = 0.001f;
    };
    
    struct InterModuleConnectionState {
        float current_strength = 0.5f;
        float baseline_strength = 0.5f;
        float activity_trace = 0.0f;
        std::chrono::steady_clock::time_point last_update;
        bool is_potentiated = false;
        float efficacy_history = 0.0f;
    };
    
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Construct NLP-focused brain architecture
     */
    BrainModuleArchitecture();
    
    /**
     * @brief Destructor
     */
    ~BrainModuleArchitecture();
    
    /**
     * @brief Initialize architecture for NLP processing
     * @return Success status
     */
    bool initializeForNLP();
    
    /**
     * @brief Initialize with legacy interface (calls initializeForNLP)
     * @param input_width Ignored in NLP mode
     * @param input_height Ignored in NLP mode
     * @return Success status
     */
    bool initialize(int input_width = 0, int input_height = 0);
    
    /**
     * @brief Shutdown and cleanup architecture
     */
    void shutdown();

    // ====================================================================== 
    // NLP-SPECIFIC PROCESSING INTERFACE
    // ======================================================================

    /**
     * @brief Process natural language input through the architecture
     * @param text_input Raw text input to process
     * @return Map of module names to their output vectors
     */
    std::map<std::string, std::vector<float>> processNLPInput(
        const std::string& text_input);

    /**
     * @brief Convert text to neural tokens
     * @param text Input text
     * @return Tokenized representation
     */
    std::vector<float> tokenizeText(const std::string& text);

    /**
     * @brief Apply neuromodulatory control to input
     * @param input Base input vector
     * @param control_signals Control signals from central controller
     * @return Modulated input vector
     */
    std::vector<float> applyNeuromodulation(
        const std::vector<float>& input,
        const std::vector<float>& control_signals);

    /**
     * @brief Update attention weights based on module outputs
     * @param module_outputs Current module outputs
     */
    void updateAttentionWeights(
        const std::map<std::string, std::vector<float>>& module_outputs);

    // ======================================================================
    // MODULE MANAGEMENT
    // ======================================================================

    std::vector<std::string> getModuleNames() const;
    size_t getModuleCount() const;
    bool hasModule(const std::string& module_name) const;
    std::shared_ptr<EnhancedNeuralModule> getModule(
        const std::string& module_name) const;
    ModuleConfig getModuleConfig(const std::string& module_name) const;
    std::vector<float> getModuleOutput(const std::string& module_name) const;

    // ======================================================================
    // CONNECTION MANAGEMENT
    // ======================================================================

    bool createConnection(const std::string& source_module,
                          const std::string& target_module,
                          float strength);
    std::vector<InterModuleConnection> getConnections() const;
    bool hasConnection(const std::string& source_module,
                       const std::string& target_module) const;
    std::vector<InterModuleConnection> getModuleConnections(
        const std::string& module_name, bool incoming = true) const;

    // ======================================================================
    // ATTENTION AND CONTROL
    // ======================================================================

    float getAttentionWeight(const std::string& module_name) const;
    void setAttentionWeight(const std::string& module_name, float weight);
    std::vector<float> getGlobalContext() const;
    void updateGlobalContext(const std::vector<float>& new_context);
    std::map<std::string, float> getNeuromodulatorLevels() const;

    // ======================================================================
    // LEARNING AND ADAPTATION
    // ======================================================================

    void update(float dt, float global_reward = 0.0f);
    void applyLearningUpdates(float reward_signal, float dt);

    struct GlobalLearningState {
        uint64_t total_learning_steps;
        float cumulative_reward;
        float average_module_performance;
        std::chrono::steady_clock::time_point last_update;
    };

    GlobalLearningState getGlobalLearningState() const;

    // ======================================================================
    // STATE PERSISTENCE
    // ======================================================================

    bool saveLearningState(const std::string& save_directory,
                           const std::string& checkpoint_name = "latest");
    bool loadLearningState(const std::string& save_directory,
                           const std::string& checkpoint_name = "latest");
    bool saveModuleLearningState(const std::string& module_name,
                                 const std::string& save_directory);
    bool loadModuleLearningState(const std::string& module_name,
                                 const std::string& save_directory);

    // ======================================================================
    // CONFIGURATION AND CONTROL
    // ======================================================================

    ArchitectureConfig getArchitectureConfig() const;
    bool updateArchitectureConfig(const ArchitectureConfig& config);

private:
    // ========================================================================
    // CORE NEURAL ARCHITECTURE COMPONENTS
    // ========================================================================
    
    // Module management
    std::unordered_map<std::string, std::shared_ptr<EnhancedNeuralModule>> modules_;
    std::unordered_map<std::string, ModuleConfig> module_configs_;
    std::vector<InterModuleConnection> connections_;
    std::unordered_map<std::string, InterModuleConnectionState> connection_states_;
    
    // Architecture configuration and state
    ArchitectureConfig architecture_config_;
    std::unique_ptr<ModularNeuralNetwork> modular_network_;
    
    // Attention and control systems
    std::unordered_map<std::string, float> attention_weights_;
    std::unordered_map<std::string, std::vector<float>> attention_history_;
    std::vector<float> global_context_vector_;
    
    // Learning state manager (forward declared, defined in .cpp)
    std::shared_ptr<LearningStateManager> learning_state_manager_;
    
    // Rest of private members...
    mutable std::mutex learning_state_mutex_;
    // ... other members
};

#endif // BRAIN_MODULE_ARCHITECTURE_H