// ============================================================================
// SPECIALIZED MODULE CLASS - LANGUAGE PROCESSING UNITS
// File: include/NeuroGen/SpecializedModule.h
// ============================================================================

#ifndef SPECIALIZED_MODULE_H
#define SPECIALIZED_MODULE_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <functional>
#include <NeuroGen/EnhancedNeuralModule.h>

/**
 * @brief Specialized Module for Language Processing
 * 
 * Implements specialized neural processing units that mimic different language-related
 * cortical areas and cognitive functions:
 * 
 * Language-Focused Specializations:
 * - Language comprehension for semantic understanding
 * - Language production for text generation
 * - Semantic memory for conceptual knowledge storage
 * - Syntactic processor for grammar and structure
 * - Working memory for temporary linguistic information
 * - Executive function for high-level language control
 * - Attention system for linguistic resource allocation
 * - Reward system for language learning reinforcement
 * 
 * Each specialized module has its own internal state, processing characteristics,
 * and biological parameters optimized for specific language processing functions.
 */
class SpecializedModule : public EnhancedNeuralModule {
public:
    // ========================================================================
    // LANGUAGE-FOCUSED SPECIALIZATION TYPES
    // ========================================================================
    
    enum class LanguageSpecialization {
        LANGUAGE_COMPREHENSION,    // Semantic understanding and interpretation
        LANGUAGE_PRODUCTION,       // Text generation and language output
        SEMANTIC_MEMORY,          // Conceptual knowledge and word meanings
        SYNTACTIC_PROCESSOR,      // Grammar, syntax, and sentence structure
        PHONOLOGICAL_PROCESSOR,   // Sound patterns and pronunciation (adapted for text)
        LEXICAL_ACCESS,           // Word retrieval and vocabulary management
        PRAGMATIC_PROCESSOR,      // Context and conversational understanding
        DISCOURSE_INTEGRATION,    // Multi-sentence and document-level processing
        WORKING_MEMORY,           // Temporary linguistic information storage
        EXECUTIVE_FUNCTION,       // High-level language control and planning
        ATTENTION_SYSTEM,         // Linguistic attention and resource allocation
        REWARD_SYSTEM,           // Language learning and motivation
        EMOTIONAL_LANGUAGE,       // Emotional aspects of language processing
        GENERAL_PURPOSE          // Flexible general processing
    };

    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Construct specialized module with configuration
     * @param name Module name for identification
     * @param config Network configuration parameters
     * @param specialization Type of language specialization
     */
    SpecializedModule(const std::string& name, 
                     const NetworkConfig& config, 
                     LanguageSpecialization specialization = LanguageSpecialization::GENERAL_PURPOSE);
    
    /**
     * @brief Construct with string-based specialization (backward compatibility)
     * @param name Module name for identification
     * @param config Network configuration parameters
     * @param module_type String description of specialization
     */
    SpecializedModule(const std::string& name, 
                     const NetworkConfig& config, 
                     const std::string& module_type);
    
    /**
     * @brief Virtual destructor
     */
    virtual ~SpecializedModule() = default;
    
    /**
     * @brief Initialize the specialized module
     * @return Success status of initialization
     */
    bool initialize() override;

    // ========================================================================
    // LANGUAGE PROCESSING METHODS
    // ========================================================================
    
    /**
     * @brief Process language comprehension (understanding and interpretation)
     * @param language_input Vector of language features for comprehension
     * @return Vector of comprehension outputs (semantic representations)
     */
    std::vector<float> process_language_comprehension(const std::vector<float>& language_input);
    
    /**
     * @brief Process language production (generation and output)
     * @param generation_input Vector of semantic inputs for text generation
     * @return Vector of generation outputs (language patterns)
     */
    std::vector<float> process_language_production(const std::vector<float>& generation_input);
    
    /**
     * @brief Process semantic memory (conceptual knowledge)
     * @param semantic_input Vector of semantic query inputs
     * @return Vector of retrieved semantic knowledge
     */
    std::vector<float> process_semantic_memory(const std::vector<float>& semantic_input);
    
    /**
     * @brief Process syntactic structures (grammar and sentence structure)
     * @param syntax_input Vector of syntactic analysis inputs
     * @return Vector of processed syntactic patterns
     */
    std::vector<float> process_syntactic_processor(const std::vector<float>& syntax_input);
    
    /**
     * @brief Process lexical access (word retrieval and vocabulary)
     * @param lexical_input Vector of word access requests
     * @return Vector of lexical activation patterns
     */
    std::vector<float> process_lexical_access(const std::vector<float>& lexical_input);
    
    /**
     * @brief Process pragmatic understanding (context and conversational cues)
     * @param pragmatic_input Vector of contextual and pragmatic inputs
     * @return Vector of pragmatic interpretation outputs
     */
    std::vector<float> process_pragmatic_processor(const std::vector<float>& pragmatic_input);
    
    /**
     * @brief Process discourse integration (multi-sentence understanding)
     * @param discourse_input Vector of discourse-level inputs
     * @return Vector of integrated discourse representations
     */
    std::vector<float> process_discourse_integration(const std::vector<float>& discourse_input);

    // ========================================================================
    // COGNITIVE PROCESSING METHODS
    // ========================================================================
    
    /**
     * @brief Process working memory computations (temporary information storage)
     * @param memory_input Vector of memory content inputs
     * @return Vector of processed memory outputs
     */
    std::vector<float> process_working_memory(const std::vector<float>& memory_input);
    
    /**
     * @brief Process executive function (high-level control and planning)
     * @param executive_input Vector of executive control inputs
     * @return Vector of executive control outputs
     */
    std::vector<float> process_executive_function(const std::vector<float>& executive_input);
    
    /**
     * @brief Process attention system computations (resource allocation)
     * @param attention_input Vector of attention control inputs
     * @return Vector of processed attention outputs
     */
    std::vector<float> process_attention_system(const std::vector<float>& attention_input);
    
    /**
     * @brief Process reward system computations (motivation and learning)
     * @param reward_input Vector of reward signal inputs
     * @return Vector of processed reward prediction outputs
     */
    std::vector<float> process_reward_system(const std::vector<float>& reward_input);
    
    /**
     * @brief Process emotional language aspects (affective language processing)
     * @param emotion_input Vector of emotional context inputs
     * @return Vector of emotional language outputs
     */
    std::vector<float> process_emotional_language(const std::vector<float>& emotion_input);

    // ========================================================================
    // SPECIALIZED PROCESSING DISPATCH
    // ========================================================================
    
    /**
     * @brief Process input based on current specialization type
     * @param input Vector of input values
     * @return Vector of specialized processing outputs
     */
    std::vector<float> processSpecialized(const std::vector<float>& input) override;
    
    /**
     * @brief Process with specific linguistic context
     * @param input Vector of input values
     * @param linguistic_context Context information for processing
     * @return Vector of context-aware outputs
     */
    std::vector<float> processWithLinguisticContext(const std::vector<float>& input,
                                                   const std::map<std::string, float>& linguistic_context);

    // ========================================================================
    // MODULE CONFIGURATION AND CONTROL
    // ========================================================================
    
    /**
     * @brief Set module specialization type
     * @param specialization New specialization type
     */
    void setSpecialization(LanguageSpecialization specialization);
    
    /**
     * @brief Set module specialization type (string-based)
     * @param type Specialization type string
     */
    void setSpecializationType(const std::string& type);
    
    /**
     * @brief Get module specialization type
     * @return Current specialization type
     */
    LanguageSpecialization getSpecialization() const { return specialization_; }
    
    /**
     * @brief Get module specialization type as string
     * @return Current specialization type as string
     */
    std::string getSpecializationString() const;
    
    /**
     * @brief Set attention weight for this module
     * @param weight New attention weight value [0.0, 1.0]
     */
    void setAttentionWeight(float weight);
    
    /**
     * @brief Get current attention weight
     * @return Current attention weight
     */
    float getAttentionWeight() const { return attention_weight_; }
    
    /**
     * @brief Set activation threshold for module outputs
     * @param threshold New activation threshold
     */
    void setActivationThreshold(float threshold);
    
    /**
     * @brief Get current activation threshold
     * @return Current activation threshold
     */
    float getActivationThreshold() const { return activation_threshold_; }
    
    /**
     * @brief Set language-specific processing parameters
     * @param parameters Map of parameter names to values
     */
    void setLanguageParameters(const std::map<std::string, float>& parameters);
    
    /**
     * @brief Get current language processing parameters
     * @return Map of current parameters
     */
    std::map<std::string, float> getLanguageParameters() const { return language_parameters_; }

    // ========================================================================
    // LEARNING AND ADAPTATION
    // ========================================================================
    
    /**
     * @brief Apply language-specific reinforcement learning
     * @param reward Reward signal for language performance
     * @param linguistic_context Context that generated the reward
     */
    void applyLanguageReinforcement(float reward, const std::map<std::string, float>& linguistic_context);
    
    /**
     * @brief Adapt processing based on language task performance
     * @param task_name Name of the language task
     * @param performance_score Performance score [0.0, 1.0]
     */
    void adaptToLanguageTask(const std::string& task_name, float performance_score);
    
    /**
     * @brief Enable or disable online learning during processing
     * @param enabled Whether to enable online learning
     */
    void setOnlineLearning(bool enabled) { online_learning_enabled_ = enabled; }
    
    /**
     * @brief Check if online learning is enabled
     * @return True if online learning is enabled
     */
    bool isOnlineLearningEnabled() const { return online_learning_enabled_; }

    // ========================================================================
    // DIAGNOSTIC AND MONITORING
    // ========================================================================
    
    /**
     * @brief Get processing statistics for language tasks
     * @return Map of statistic names to values
     */
    std::map<std::string, float> getLanguageProcessingStats() const;
    
    /**
     * @brief Get current activation patterns
     * @return Vector of current internal activations
     */
    std::vector<float> getCurrentActivations() const;
    
    /**
     * @brief Reset module statistics
     */
    void resetStatistics();
    
    /**
     * @brief Get detailed module status
     * @return Human-readable status string
     */
    std::string getDetailedStatus() const;

    // ========================================================================
    // OVERRIDDEN VIRTUAL FUNCTIONS
    // ========================================================================
    
    /**
     * @brief Update module with time step, inputs, and reward signal
     * @param dt Time step in seconds
     * @param inputs Input vector to process
     * @param reward Reward signal for learning
     */
    void update(float dt, const std::vector<float>& inputs = {}, float reward = 0.0f) override;
    
    /**
     * @brief Process input through the module
     * @param input Input vector
     * @return Processed output vector
     */
    std::vector<float> process(const std::vector<float>& input) override;

private:
    // ========================================================================
    // INTERNAL STATE AND CONFIGURATION
    // ========================================================================
    
    // Specialization configuration
    LanguageSpecialization specialization_;
    std::string specialization_type_string_;  // For backward compatibility
    std::map<std::string, float> language_parameters_;
    
    // Processing parameters
    float attention_weight_;
    float activation_threshold_;
    float learning_rate_;
    float decay_rate_;
    float noise_level_;
    bool online_learning_enabled_;
    
    // Performance tracking
    std::map<std::string, float> processing_statistics_;
    std::map<std::string, float> task_performance_history_;
    size_t processing_count_;
    float cumulative_reward_;
    
    // Internal processing state
    std::vector<float> internal_state_;
    std::vector<float> output_buffer_;
    std::vector<float> attention_state_;
    std::vector<float> memory_trace_;

    // ========================================================================
    // INTERNAL HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Initialize internal state based on specialization
     * @param state_size Size of internal state
     * @param output_size Size of output buffer
     */
    void initializeInternalState(size_t state_size, size_t output_size);
    
    /**
     * @brief Apply activation function based on specialization
     * @param input Input value
     * @return Activated output value
     */
    float applyActivationFunction(float input) const;
    
    /**
     * @brief Apply noise for biological realism
     * @param base_value Base value to add noise to
     * @return Value with added noise
     */
    float applyBiologicalNoise(float base_value) const;
    
    /**
     * @brief Update internal statistics
     * @param processing_time Time taken for processing
     * @param output_quality Quality score of output
     */
    void updateStatistics(float processing_time, float output_quality = 0.5f);
    
    /**
     * @brief Convert specialization enum to string
     * @param spec Specialization enum value
     * @return String representation
     */
    static std::string specializationToString(LanguageSpecialization spec);
    
    /**
     * @brief Convert string to specialization enum
     * @param type String representation
     * @return Specialization enum value
     */
    static LanguageSpecialization stringToSpecialization(const std::string& type);
    
    /**
     * @brief Sigmoid activation function
     * @param x Input value
     * @return Sigmoid output
     */
    static float sigmoid(float x);
    
    /**
     * @brief Hyperbolic tangent activation function
     * @param x Input value
     * @return Tanh output
     */
    static float tanh_activation(float x);
    
    /**
     * @brief ReLU activation function
     * @param x Input value
     * @return ReLU output
     */
    static float relu(float x);
    
    /**
     * @brief Softmax activation for attention distributions
     * @param inputs Input vector
     * @return Softmax-normalized output vector
     */
    static std::vector<float> softmax(const std::vector<float>& inputs);
};

// ============================================================================
// INLINE IMPLEMENTATIONS
// ============================================================================

inline float SpecializedModule::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-std::max(-50.0f, std::min(50.0f, x))));
}

inline float SpecializedModule::tanh_activation(float x) {
    return std::tanh(std::max(-50.0f, std::min(50.0f, x)));
}

inline float SpecializedModule::relu(float x) {
    return std::max(0.0f, x);
}

#endif // SPECIALIZED_MODULE_H