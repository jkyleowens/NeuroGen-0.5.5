// ============================================================================
// BRAIN MODULE ARCHITECTURE HEADER - NLP-FOCUSED
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
#include "NeuroGen/LearningStateManager.h"

// Forward declarations
class LearningStateManager;
class NetworkCUDA;
class ContinuousLearningAgent;

/**
 * @brief Simplified Brain-inspired Architecture for Natural Language Processing
 * 
 * This class implements a streamlined brain-inspired modular architecture
 * specifically designed for natural language processing. It consists of
 * five core modules:
 * 
 * 1. Language Perception Module - reading and tokenization
 * 2. Comprehension Module - semantic integration
 * 3. Reasoning Module - logical inference
 * 4. Output Generation Module - language production
 * 5. Neuromodulation Module - adaptive control
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
    
    // ========================================================================
    // NLP-SPECIFIC PROCESSING INTERFACE
    // ========================================================================
    
    /**
     * @brief Process natural language input through the architecture
     * @param text_input Raw text input to process
     * @return Map of module names to their output vectors
     */
    std::map<std::string, std::vector<float>> processNLPInput(const std::string& text_input);
    
    /**
     * @brief Process tokenized language input with learning
     * @param tokenized_input Pre-tokenized input vector
     * @param reward Global reward signal for learning
     * @return Map of module outputs
     */
    std::map<std::string, std::vector<float>> processTokenizedInput(
        const std::vector<float>& tokenized_input, 
        float reward = 0.0f);
    
    /**
     * @brief Generate language response from current state
     * @return Generated response text
     */
    std::string generateLanguageResponse();
    
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
    
    // ========================================================================
    // MODULE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Get all module names
     * @return Vector of module names
     */
    std::vector<std::string> getModuleNames() const;
    
    /**
     * @brief Get total number of modules
     * @return Module count
     */
    size_t getModuleCount() const;
    
    /**
     * @brief Check if module exists
     * @param module_name Module name to check
     * @return True if module exists
     */
    bool hasModule(const std::string& module_name) const;
    
    /**
     * @brief Get module by name
     * @param module_name Module name
     * @return Shared pointer to module (nullptr if not found)
     */
    std::shared_ptr<EnhancedNeuralModule> getModule(const std::string& module_name) const;
    
    /**
     * @brief Get module configuration
     * @param module_name Module name
     * @return Module configuration
     */
    ModuleConfig getModuleConfig(const std::string& module_name) const;
    
    /**
     * @brief Get outputs from specific module
     * @param module_name Module name
     * @return Output vector (empty if module not found)
     */
    std::vector<float> getModuleOutput(const std::string& module_name) const;
    
    // ========================================================================
    // CONNECTION MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Create connection between modules
     * @param source_module Source module name
     * @param target_module Target module name
     * @param strength Connection strength
     * @return Success status
     */
    bool createConnection(const std::string& source_module, 
                         const std::string& target_module, 
                         float strength);
    
    /**
     * @brief Get all inter-module connections
     * @return Vector of connection configurations
     */
    std::vector<InterModuleConnection> getConnections() const;
    
    /**
     * @brief Check if connection exists
     * @param source_module Source module name
     * @param target_module Target module name
     * @return True if connection exists
     */
    bool hasConnection(const std::string& source_module, const std::string& target_module) const;
    
    /**
     * @brief Get connections for a specific module
     * @param module_name Module name
     * @param incoming If true, get incoming connections; if false, get outgoing
     * @return Vector of connections
     */
    std::vector<InterModuleConnection> getModuleConnections(
        const std::string& module_name, bool incoming = true) const;
    
    // ========================================================================
    // ATTENTION AND CONTROL
    // ========================================================================
    
    /**
     * @brief Update attention weights based on module outputs
     * @param module_outputs Current module outputs
     */
    void updateAttentionWeights(const std::map<std::string, std::vector<float>>& module_outputs);
    
    /**
     * @brief Get current attention weight for module
     * @param module_name Module name
     * @return Attention weight (0.0 to 1.0)
     */
    float getAttentionWeight(const std::string& module_name) const;
    
    /**
     * @brief Set attention weight for module
     * @param module_name Module name
     * @param weight New attention weight
     */
    void setAttentionWeight(const std::string& module_name, float weight);
    
    /**
     * @brief Get global context vector
     * @return Current global context
     */
    std::vector<float> getGlobalContext() const;
    
    /**
     * @brief Update global context from processing results
     * @param new_context Context update vector
     */
    void updateGlobalContext(const std::vector<float>& new_context);
    
    // ========================================================================
    // LEARNING AND ADAPTATION
    // ========================================================================
    
    /**
     * @brief Update all modules with time step and learning
     * @param dt Time step in seconds
     * @param global_reward Global reward signal
     */
    void update(float dt, float global_reward = 0.0f);
    
    /**
     * @brief Apply learning updates to all modules
     * @param reward_signal Global reward signal
     * @param dt Time step
     */
    void applyLearningUpdates(float reward_signal, float dt);
    
    /**
     * @brief Get global learning statistics
     * @return Learning statistics
     */
    struct GlobalLearningState {
        uint64_t total_learning_steps;
        float cumulative_reward;
        float average_module_performance;
        std::chrono::steady_clock::time_point last_update;
    };
    
    GlobalLearningState getGlobalLearningState() const;
    
    // ========================================================================
    // STATE PERSISTENCE
    // ========================================================================
    
    /**
     * @brief Save architecture state to directory
     * @param save_directory Directory path for saving
     * @param checkpoint_name Optional checkpoint name
     * @return Success status
     */
    bool saveLearningState(const std::string& save_directory, 
                          const std::string& checkpoint_name = "latest");
    
    /**
     * @brief Load architecture state from directory
     * @param save_directory Directory path for loading
     * @param checkpoint_name Optional checkpoint name
     * @return Success status
     */
    bool loadLearningState(const std::string& save_directory, 
                          const std::string& checkpoint_name = "latest");
    
    /**
     * @brief Save individual module state
     * @param module_name Module name
     * @param save_directory Directory path
     * @return Success status
     */
    bool saveModuleLearningState(const std::string& module_name, 
                                const std::string& save_directory);
    
    /**
     * @brief Load individual module state
     * @param module_name Module name
     * @param save_directory Directory path
     * @return Success status
     */
    bool loadModuleLearningState(const std::string& module_name, 
                                const std::string& save_directory);
    
    // ========================================================================
    // CONFIGURATION AND CONTROL
    // ========================================================================
    
    /**
     * @brief Get architecture configuration
     * @return Current architecture configuration
     */
    ArchitectureConfig getArchitectureConfig() const;
    
    /**
     * @brief Update architecture configuration
     * @param config New configuration
     * @return Success status
     */
    bool updateArchitectureConfig(const ArchitectureConfig& config);
    
    /**
     * @brief Get neuromodulator levels
     * @return Map of neuromodulator names to levels
     */
    std::map<std::string, float> getNeuromodulatorLevels() const;

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    // Core configuration
    ArchitectureConfig architecture_config_;
    
    // Core modular network
    std::unique_ptr<ModularNeuralNetwork> modular_network_;
    
    // Module management (5 modules for NLP)
    std::map<std::string, std::shared_ptr<EnhancedNeuralModule>> modules_;
    std::map<std::string, ModuleConfig> module_configs_;
    std::vector<InterModuleConnection> connections_;
    
    // Inter-module connection tracking
    std::map<std::pair<std::string, std::string>, InterModuleConnectionState> inter_module_connections_;
    std::map<std::pair<std::string, std::string>, float> connection_usage_history_;
    
    // Attention and control
    std::map<std::string, float> attention_weights_;
    std::map<std::string, float> attention_history_;
    std::vector<float> global_context_vector_;
    float global_inhibition_level_ = 0.1f;
    
    // Neuromodulation for language processing
    float global_dopamine_level_ = 0.2f;        // Higher for language reward
    float global_acetylcholine_level_ = 0.3f;   // Higher for language attention
    float global_norepinephrine_level_ = 0.15f;
    float global_serotonin_level_ = 0.1f;
    
    // Learning session information
    mutable std::mutex learning_state_mutex_;
    uint64_t global_learning_steps_ = 0;
    float global_reward_accumulator_ = 0.0f;
    std::chrono::steady_clock::time_point last_update_time_;
    std::chrono::steady_clock::time_point creation_time_;
    
    // Performance tracking for NLP
    std::map<std::string, std::vector<float>> module_performance_history_;
    std::map<std::string, float> module_prediction_errors_;
    std::map<std::string, float> module_stability_scores_;
    size_t performance_history_size_ = 1000; // Reduced for NLP focus
    
    // External interfaces
    std::shared_ptr<NetworkCUDA> cuda_network_;
    std::shared_ptr<LearningStateManager> learning_state_manager_;
    std::weak_ptr<ContinuousLearningAgent> continuous_learning_agent_;
    
    // ========================================================================
    // INTERNAL NLP METHODS
    // ========================================================================
    
    // Module creation and setup
    void createNLPModules();
    void setupNLPConnections();
    void initializeNLPAttentionSystem();
    
    // Processing pipeline
    void processModulePipeline(const std::vector<float>& input,
                              std::map<std::string, std::vector<float>>& outputs);
    
    // Learning and adaptation
    void updateNeuromodulatorLevels(float reward, float dt);
    void updateGlobalAttention(float dt);
    void consolidateMemoryTraces(float dt);
    
    // Utility methods
    std::vector<float> convertTextToTokens(const std::string& text);
    std::string convertTokensToText(const std::vector<float>& tokens);
    float computeModuleActivation(const std::vector<float>& output);
    
    // Validation and compatibility
    std::pair<bool, std::string> validateArchitectureCompatibility(const std::string& loaded_hash);
    std::string generateArchitectureHash() const;
};

#endif // BRAIN_MODULE_ARCHITECTURE_H