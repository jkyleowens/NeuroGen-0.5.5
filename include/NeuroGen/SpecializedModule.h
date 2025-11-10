// ============================================================================
// SPECIALIZED MODULE HEADER - NLP-FOCUSED
// File: include/NeuroGen/SpecializedModule.h
// ============================================================================

#ifndef SPECIALIZED_MODULE_H
#define SPECIALIZED_MODULE_H

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <map>
#include <deque>
#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/BiologicalNeuronModule.h"  // NEW: Real spiking neurons
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/NetworkStats.h"

/**
 * @brief Specialized Neural Module for NLP-Focused Modular Architecture
 * 
 * This class extends EnhancedNeuralModule to provide specialized processing
 * capabilities for natural language processing tasks. It supports five
 * specialized module types:
 * 
 * 1. neuromodulatory_control - Central controller with neuromodulation
 * 2. text_input_processing - Text tokenization and input processing
 * 3. language_understanding - Deep language comprehension
 * 4. logical_reasoning - Inference and logical operations
 * 5. spike_to_action - Convert neural spikes to actionable responses
 * 
 * Each specialization provides unique processing algorithms optimized
 * for specific aspects of language processing and understanding.
 */
class SpecializedModule : public EnhancedNeuralModule {
public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Construct specialized module with configuration
     * @param name Module name for identification
     * @param config Network configuration parameters
     * @param module_type Type of specialization
     */
    SpecializedModule(const std::string& name, const NetworkConfig& config, 
                     const std::string& module_type = "general");
    
    /**
     * @brief Virtual destructor for polymorphic inheritance
     */
    virtual ~SpecializedModule() = default;
    
    /**
     * @brief Initialize the specialized module
     * @return Success status of initialization
     */
    bool initialize() override;
    
    // ========================================================================
    // CORE PROCESSING INTERFACE
    // ========================================================================
    
    /**
     * @brief Process input through specialized neural computation
     * @param input Input vector to process
     * @return Processed output vector
     */
    std::vector<float> process(const std::vector<float>& input) override;
    
    /**
     * @brief Update module with time step, inputs, and reward signal
     * @param dt Time step in seconds
     * @param inputs Input vector to process (optional)
     * @param reward Reward signal for learning (optional)
     */
    void update(float dt, const std::vector<float>& inputs = {}, float reward = 0.0f) override;
    
    // ========================================================================
    // SPECIALIZED PROCESSING METHODS
    // ========================================================================
    
    /**
     * @brief Process neuromodulatory control signals
     * @param input Control input signals
     * @return Neuromodulatory output signals
     */
    std::vector<float> processNeuromodulatoryControl(const std::vector<float>& input);
    
    /**
     * @brief Process text input tokenization and encoding
     * @param input Raw text token input
     * @return Encoded neural representation
     */
    std::vector<float> processTextInput(const std::vector<float>& input);
    
    /**
     * @brief Process language understanding and comprehension
     * @param input Language feature input
     * @return Language understanding output
     */
    std::vector<float> processLanguageUnderstanding(const std::vector<float>& input);
    
    /**
     * @brief Process logical reasoning and inference
     * @param input Reasoning input data
     * @return Logical reasoning output
     */
    std::vector<float> processLogicalReasoning(const std::vector<float>& input);
    
    /**
     * @brief Process spike-to-action conversion
     * @param input Neural spike patterns
     * @return Actionable response data
     */
    std::vector<float> processSpikeToAction(const std::vector<float>& input);
    
    // ========================================================================
    // MODULE CONFIGURATION AND CONTROL
    // ========================================================================
    
    /**
     * @brief Set module specialization type
     * @param type Specialization type string
     */
    void set_specialization_type(const std::string& type);
    
    /**
     * @brief Get module specialization type
     * @return Current specialization type
     */
    const std::string& get_specialization_type() const;
    
    /**
     * @brief Set attention weight for this module
     * @param weight New attention weight value (0.0 to 1.0)
     */
    void set_attention_weight(float weight);
    
    /**
     * @brief Get current attention weight
     * @return Current attention weight
     */
    float get_attention_weight() const;
    
    /**
     * @brief Apply reinforcement signal to the module
     * @param reward Reward signal value
     */
    void apply_reinforcement_signal(float reward);

private:
    // ========================================================================
    // SPECIALIZED MODULE STATE
    // ========================================================================
    
    // Core specialization
    std::string specialization_type_;
    float learning_rate_modifier_ = 1.0f;
    float attention_weight_ = 0.5f;
    float excitability_level_ = 0.7f;

    // NEW: Biological neural substrate - REAL spiking neurons!
    std::unique_ptr<BiologicalNeuronModule> biological_module_;
    bool use_biological_neurons_ = true;  // Enable biologically realistic processing

    // Processing buffers (for legacy compatibility)
    std::vector<float> processing_buffer_;
    std::vector<float> integration_buffer_;
    std::vector<float> output_buffer_;
    
    // Input history for novelty detection
    std::deque<std::vector<float>> input_history_;
    
    // ========================================================================
    // NEUROMODULATORY CONTROL STATE
    // ========================================================================
    
    // Neuromodulator signals
    std::vector<float> dopamine_signals_;
    std::vector<float> acetylcholine_signals_;
    std::vector<float> norepinephrine_signals_;
    std::vector<float> serotonin_signals_;
    
    // ========================================================================
    // TEXT INPUT PROCESSING STATE
    // ========================================================================
    
    // Token embeddings and position encodings
    std::vector<std::vector<float>> token_embeddings_;     // 1024 tokens x 256 dims
    std::vector<std::vector<float>> position_encodings_;   // 512 positions x 256 dims
    
    // ========================================================================
    // LANGUAGE UNDERSTANDING STATE
    // ========================================================================
    
    // Language comprehension systems
    std::vector<float> semantic_memory_;
    std::vector<float> syntactic_patterns_;
    std::vector<float> context_integrator_;
    
    // Self-attention mechanism
    std::vector<std::vector<float>> self_attention_weights_;
    
    // ========================================================================
    // LOGICAL REASONING STATE
    // ========================================================================
    
    // Reasoning and inference systems
    std::vector<float> logical_state_;
    std::vector<float> inference_chains_;
    std::vector<float> contradiction_detector_;
    
    // Reasoning rule templates
    std::vector<std::vector<float>> reasoning_rules_;
    
    // ========================================================================
    // SPIKE-TO-ACTION STATE
    // ========================================================================
    
    // Spike decoding and action generation
    std::vector<float> spike_decoder_;
    std::vector<float> action_primitives_;
    std::vector<float> confidence_estimator_;
    
    // Spike-to-action mapping matrix
    std::vector<std::vector<float>> spike_to_action_matrix_;
    
    // ========================================================================
    // INITIALIZATION METHODS
    // ========================================================================
    
    /**
     * @brief Initialize specialization-specific parameters
     */
    void initializeSpecializationParameters();
    
    /**
     * @brief Initialize specialized components based on module type
     */
    void initializeSpecializedComponents();
    
    /**
     * @brief Initialize neuromodulatory control systems
     */
    void initializeNeuromodulatoryControl();
    
    /**
     * @brief Initialize text input processing systems
     */
    void initializeTextInputProcessing();
    
    /**
     * @brief Initialize language understanding systems
     */
    void initializeLanguageUnderstanding();
    
    /**
     * @brief Initialize logical reasoning systems
     */
    void initializeLogicalReasoning();
    
    /**
     * @brief Initialize spike-to-action conversion systems
     */
    void initializeSpikeToAction();
    
    // ========================================================================
    // UTILITY METHODS
    // ========================================================================
    
    /**
     * @brief Compute novelty of input compared to history
     * @param input Current input vector
     * @return Novelty score (0.0 to 1.0)
     */
    float computeInputNovelty(const std::vector<float>& input);
    
    /**
     * @brief Compute complexity of input
     * @param input Current input vector
     * @return Complexity score (0.0 to 1.0)
     */
    float computeInputComplexity(const std::vector<float>& input);
    
    /**
     * @brief Compute attention demand of input
     * @param input Current input vector
     * @return Attention demand (0.0 to 1.0)
     */
    float computeAttentionDemand(const std::vector<float>& input);
    
    /**
     * @brief Compute similarity between two vectors
     * @param a First vector
     * @param b Second vector
     * @return Similarity score (0.0 to 1.0)
     */
    float computeVectorSimilarity(const std::vector<float>& a, const std::vector<float>& b);
    
    /**
     * @brief Compute activation spread in vector
     * @param input Input vector
     * @return Activation spread measure
     */
    float computeActivationSpread(const std::vector<float>& input);
    
    // ========================================================================
    // NEUROMODULATORY METHODS
    // ========================================================================
    
    /**
     * @brief Update neuromodulator levels based on input characteristics
     * @param novelty Input novelty score
     * @param complexity Input complexity score
     * @param attention Attention demand score
     */
    void updateNeuromodulatorLevels(float novelty, float complexity, float attention);
    
    // ========================================================================
    // LANGUAGE PROCESSING METHODS
    // ========================================================================
    
    /**
     * @brief Update semantic memory with new input
     * @param input New input to integrate
     */
    void updateSemanticMemory(const std::vector<float>& input);
    
    /**
     * @brief Apply self-attention mechanism to input
     * @param input Input to apply attention to
     * @return Attention-weighted output
     */
    std::vector<float> applySelfAttention(const std::vector<float>& input);
    
    /**
     * @brief Integrate input with existing context
     * @param input Input to integrate
     * @return Context-integrated output
     */
    std::vector<float> integrateWithContext(const std::vector<float>& input);
    
    /**
     * @brief Extract semantic features from input
     * @param input Input to extract features from
     * @param output Output vector to fill with features
     */
    void extractSemanticFeatures(const std::vector<float>& input, std::vector<float>& output);
    
    // ========================================================================
    // REASONING METHODS
    // ========================================================================
    
    /**
     * @brief Update logical reasoning state
     * @param input New logical input
     */
    void updateLogicalState(const std::vector<float>& input);
    
    /**
     * @brief Apply reasoning rules to input
     * @param input Input to apply rules to
     * @return Rule application results
     */
    std::vector<float> applyReasoningRules(const std::vector<float>& input);
    
    /**
     * @brief Detect contradictions in input
     * @param input Input to check for contradictions
     * @return Contradiction detection results
     */
    std::vector<float> detectContradictions(const std::vector<float>& input);
    
    /**
     * @brief Build inference chains from rule outputs
     * @param rule_outputs Outputs from reasoning rules
     * @return Inference chain results
     */
    std::vector<float> buildInferenceChains(const std::vector<float>& rule_outputs);
    
    // ========================================================================
    // ACTION CONVERSION METHODS
    // ========================================================================
    
    /**
     * @brief Decode spike patterns to intermediate representation
     * @param input Spike pattern input
     * @return Decoded representation
     */
    std::vector<float> decodeSpikes(const std::vector<float>& input);
    
    /**
     * @brief Map decoded spikes to action primitives
     * @param decoded_spikes Decoded spike patterns
     * @return Action primitive activations
     */
    std::vector<float> mapToActionPrimitives(const std::vector<float>& decoded_spikes);
    
    /**
     * @brief Estimate confidence for action candidates
     * @param action_candidates Action candidate activations
     * @return Confidence estimates
     */
    std::vector<float> estimateActionConfidence(const std::vector<float>& action_candidates);
    
    // ========================================================================
    // LEARNING METHODS
    // ========================================================================
    
    /**
     * @brief Apply specialized learning updates
     * @param reward Reward signal
     * @param dt Time step
     */
    void applySpecializedLearning(float reward, float dt);
    
    /**
     * @brief Update attention based on performance
     * @param reward Performance reward signal
     */
    void updateAttentionBasedOnPerformance(float reward);
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Sigmoid activation function
 * @param x Input value
 * @return Sigmoid output
 */
float sigmoid(float x);

#endif // SPECIALIZED_MODULE_H