// ============================================================================
// BRAIN MODULE ARCHITECTURE - NATURAL LANGUAGE PROCESSING FOCUSED
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
#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/NetworkConfig.h"

// Forward declarations
class LearningStateManager;
class NetworkCUDA;
class ContinuousLearningAgent;

/**
 * @brief Brain-inspired modular architecture for natural language processing
 * 
 * This class implements a language-focused brain-inspired modular architecture
 * with proper initialization, dynamic sizing, inter-module connections,
 * persistent learning capabilities, memory consolidation, and attention
 * mechanisms specifically designed for natural language understanding and generation.
 */
class BrainModuleArchitecture : public std::enable_shared_from_this<BrainModuleArchitecture> {
public:
    // ========================================================================
    // LANGUAGE-FOCUSED MODULE TYPES
    // ========================================================================
    
    enum class ModuleType {
        LANGUAGE_COMPREHENSION,   // Wernicke's area - Language understanding
        LANGUAGE_PRODUCTION,      // Broca's area - Language generation
        SEMANTIC_MEMORY,          // Temporal cortex - Conceptual knowledge
        SYNTACTIC_PROCESSOR,      // Left frontal - Grammar and structure
        PHONOLOGICAL_PROCESSOR,   // Superior temporal - Sound patterns
        WORKING_MEMORY,          // Dorsolateral PFC - Temporary storage
        EXECUTIVE_FUNCTION,      // Prefrontal Cortex - Goal management
        EPISODIC_MEMORY,         // Hippocampus - Experience-based memories
        CENTRAL_CONTROLLER,      // Thalamus - Attention and routing
        PRAGMATIC_PROCESSOR,     // Right hemisphere - Context understanding
        LEXICAL_ACCESS,          // Left temporal - Word retrieval
        DISCOURSE_INTEGRATION,   // Bilateral frontal - Multi-sentence processing
        MOTOR_CORTEX,           // M1/Premotor - Speech production (adapted for text)
        REWARD_SYSTEM,          // VTA/Nucleus accumbens - Learning reinforcement
        ATTENTION_SYSTEM,       // Parietal cortex - Attention allocation
        EMOTIONAL_PROCESSING    // Amygdala/Limbic - Emotional language
    };
    
    // ========================================================================
    // CONFIGURATION STRUCTURES
    // ========================================================================
    
    struct ModuleConfig {
        ModuleType type;
        std::string name;
        std::string description;
        
        // Network topology
        size_t input_size;
        size_t output_size;
        size_t internal_neurons;
        size_t linguistic_layers;
        size_t semantic_dimensions;
        
        // Learning parameters
        float learning_rate;
        float linguistic_plasticity;
        float attention_sensitivity;
        float semantic_decay_rate;
        float syntactic_strength;
        
        // Processing characteristics
        bool supports_sequential;
        bool supports_hierarchical;
        bool bidirectional;
        
        std::map<std::string, float> custom_params;
        
        ModuleConfig() : type(ModuleType::LANGUAGE_COMPREHENSION), input_size(512), 
                        output_size(256), internal_neurons(1024), linguistic_layers(4),
                        semantic_dimensions(300), learning_rate(0.001f), 
                        linguistic_plasticity(0.8f), attention_sensitivity(0.7f),
                        semantic_decay_rate(0.99f), syntactic_strength(0.9f),
                        supports_sequential(true), supports_hierarchical(false),
                        bidirectional(false) {}
    };
    
    struct InterModuleConnection {
        std::string source_module;
        std::string target_module;
        float connection_strength;
        std::string connection_type;
        bool is_bidirectional;
        float delay_ms;
        
        InterModuleConnection(const std::string& src, const std::string& tgt, 
                            float strength, const std::string& type = "semantic",
                            bool bidirectional = false, float delay = 1.0f)
            : source_module(src), target_module(tgt), connection_strength(strength),
              connection_type(type), is_bidirectional(bidirectional), delay_ms(delay) {}
    };
    
    struct ArchitectureConfig {
        size_t max_sequence_length;
        size_t vocabulary_size;
        size_t embedding_dimensions;
        
        float global_attention_strength;
        size_t working_memory_capacity;
        float memory_consolidation_rate;
        
        float global_learning_rate;
        float inter_module_transfer_rate;
        bool enable_meta_learning;
        bool enable_continual_learning;
        
        bool use_gpu_acceleration;
        size_t batch_processing_size;
        float processing_timeout_ms;
        
        ArchitectureConfig() : max_sequence_length(512), vocabulary_size(50000),
                              embedding_dimensions(300), global_attention_strength(0.8f),
                              working_memory_capacity(1024), memory_consolidation_rate(0.1f),
                              global_learning_rate(0.001f), inter_module_transfer_rate(0.05f),
                              enable_meta_learning(true), enable_continual_learning(true),
                              use_gpu_acceleration(false), batch_processing_size(32),
                              processing_timeout_ms(1000.0f) {}
    };

    // ========================================================================
    // LANGUAGE PROCESSING STRUCTURES
    // ========================================================================
    
    struct LanguageInput {
        std::string text;
        std::vector<std::string> tokens;
        std::vector<float> embeddings;
        std::string language_code;
        std::string input_type;
        float confidence;
        std::map<std::string, float> linguistic_features;
        
        LanguageInput(const std::string& input_text = "") 
            : text(input_text), language_code("en"), input_type("sentence"), confidence(1.0f) {}
    };
    
    struct LanguageOutput {
        std::string generated_text;
        std::vector<std::string> tokens;
        float confidence;
        std::vector<float> semantic_representation;
        std::map<std::string, float> linguistic_scores;
        std::string generation_strategy;
        
        LanguageOutput() : confidence(0.0f), generation_strategy("greedy") {}
    };

    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    explicit BrainModuleArchitecture(const ArchitectureConfig& config = ArchitectureConfig{});
    virtual ~BrainModuleArchitecture();
    
    bool initialize(size_t vocabulary_size = 50000, size_t max_sequence_length = 512);
    bool initializeCustom(const std::vector<ModuleConfig>& module_configs,
                         const std::vector<InterModuleConnection>& connections);
    
    std::pair<bool, std::string> initializeLanguagePipeline();
    std::pair<bool, std::string> validateConfiguration() const;
    
    void reset(bool preserve_language_knowledge = true);

    // ========================================================================
    // LANGUAGE PROCESSING INTERFACE
    // ========================================================================
    
    LanguageOutput processLanguage(const LanguageInput& input, bool learning_enabled = true);
    std::string processText(const std::string& text, bool learning_enabled = true);
    
    LanguageOutput generateResponse(const LanguageInput& context, 
                                  size_t max_length = 100, 
                                  float temperature = 0.7f);
    
    LanguageOutput processConversation(const std::vector<LanguageInput>& conversation_history,
                                     const LanguageInput& current_input);

    // ========================================================================
    // MODULE MANAGEMENT
    // ========================================================================
    
    std::pair<bool, std::string> addLanguageModule(const ModuleConfig& config);
    bool removeModule(const std::string& module_name, bool cleanup_connections = true);
    bool updateModuleConfig(const std::string& module_name, const ModuleConfig& config);
    
    std::shared_ptr<EnhancedNeuralModule> getModule(const std::string& module_name) const;
    std::vector<std::shared_ptr<EnhancedNeuralModule>> getModulesByType(ModuleType type) const;
    std::vector<std::string> getModuleNames() const;
    
    std::shared_ptr<EnhancedNeuralModule> getTaskOptimizedModule(const std::string& task) const;

    // ========================================================================
    // ATTENTION AND CONTROL MECHANISMS
    // ========================================================================
    
    void updateLanguageAttention(const LanguageInput& input, const std::string& task_context = "");
    std::map<std::string, float> getAttentionWeights() const;
    void focusAttentionOn(const std::string& aspect, float strength);

    // ========================================================================
    // LEARNING AND ADAPTATION
    // ========================================================================
    
    void applyLanguageLearning(float reward, const LanguageInput& language_context);
    void consolidateLanguageLearning(float consolidation_strength = 0.1f);
    
    void transferLanguageKnowledge(const std::string& source_task, 
                                 const std::string& target_task,
                                 float transfer_strength = 0.1f);

    // ========================================================================
    // STATE MANAGEMENT AND PERSISTENCE
    // ========================================================================
    
    bool saveState(const std::string& filepath, bool include_language_knowledge = true);
    bool loadState(const std::string& filepath, bool merge_with_current = false);
    
    std::pair<bool, std::string> exportLanguageModel(const std::string& export_path,
                                                   const std::string& format = "onnx");

    // ========================================================================
    // MONITORING AND DIAGNOSTICS
    // ========================================================================
    
    std::map<std::string, float> getLanguageProcessingStats() const;
    std::string getDetailedStatus() const;
    void setPerformanceMonitoring(bool enable_monitoring);
    ArchitectureConfig getConfiguration() const { return config_; }

    // ========================================================================
    // CONNECTION MANAGEMENT
    // ========================================================================
    
    bool addConnection(const std::string& source_module, const std::string& target_module, 
                      float strength);
    bool removeConnection(const std::string& source_module, const std::string& target_module);
    std::vector<InterModuleConnection> getConnections() const;
    
    bool hasConnection(const std::string& source_module, const std::string& target_module) const;

    // ========================================================================
    // PROCESSING CONTROL
    // ========================================================================
    
    std::map<std::string, std::vector<float>> processInput(const std::vector<float>& inputs);
    
    std::map<std::string, std::vector<float>> processInputWithLearning(
        const std::vector<float>& inputs, 
        float reward = 0.0f, 
        float novelty_signal = 0.0f);
    
    void update(float dt, float global_reward = 0.0f);
    
    std::vector<float> getModuleOutput(const std::string& module_name) const;

private:
    // ========================================================================
    // INTERNAL STATE MANAGEMENT
    // ========================================================================
    
    ArchitectureConfig config_;
    
    std::map<std::string, std::shared_ptr<EnhancedNeuralModule>> modules_;
    std::vector<InterModuleConnection> connections_;
    std::map<std::string, ModuleConfig> module_configs_;
    
    std::atomic<bool> is_processing_;
    std::atomic<bool> is_learning_enabled_;
    std::map<std::string, std::vector<float>> module_outputs_;
    std::vector<float> global_linguistic_state_;
    
    std::chrono::high_resolution_clock::time_point last_update_time_;
    std::map<std::string, float> processing_times_;
    std::map<std::string, float> language_metrics_;
    
    mutable std::recursive_mutex architecture_mutex_;
    mutable std::mutex processing_mutex_;

    // ========================================================================
    // INTERNAL HELPER METHODS
    // ========================================================================
    
    bool initializeLanguageModules();
    bool createDefaultConnections();
    void updateInternalState(float dt);
    
    std::vector<float> extractLanguageFeatures(const LanguageInput& input);
    std::string convertNeuralToText(const std::vector<float>& neural_output);
    float calculateOutputConfidence(const std::vector<float>& output);
    
    void routeLanguageSignals();
    void optimizeModulePerformance();
    
    bool validateConnection(const InterModuleConnection& connection) const;
    void propagateLanguageSignal(const std::string& source_module, 
                               const std::vector<float>& signal);
};

#endif // BRAIN_MODULE_ARCHITECTURE_H