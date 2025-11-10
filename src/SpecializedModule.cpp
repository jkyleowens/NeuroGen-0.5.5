// ============================================================================
// NLP-FOCUSED SPECIALIZED MODULE IMPLEMENTATION
// File: src/SpecializedModule.cpp
// ============================================================================

#include "NeuroGen/SpecializedModule.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <chrono>

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

SpecializedModule::SpecializedModule(const std::string& name, 
                                   const NetworkConfig& config, 
                                   const std::string& module_type)
    : EnhancedNeuralModule(name, config), 
      specialization_type_(module_type) {
    
    std::cout << "🔧 Creating specialized module: " << name 
              << " (type: " << module_type << ")" << std::endl;
    
    // Initialize specialized processing parameters based on type
    initializeSpecializationParameters();
}

bool SpecializedModule::initialize() {
    // Call parent initialization
    if (!EnhancedNeuralModule::initialize()) {
        return false;
    }
    
    // Initialize specialized components based on module type
    initializeSpecializedComponents();
    
    std::cout << "✅ Specialized module '" << module_name_ 
              << "' initialized for " << specialization_type_ << std::endl;
    
    return true;
}

void SpecializedModule::initializeSpecializationParameters() {
    if (specialization_type_ == "neuromodulatory_control") {
        // Central Controller parameters
        attention_weight_ = 1.0f;
        learning_rate_modifier_ = 1.2f; // Higher learning for control
        excitability_level_ = 0.8f;
        
    } else if (specialization_type_ == "text_input_processing") {
        // Input Module parameters
        attention_weight_ = 0.7f;
        learning_rate_modifier_ = 1.0f;
        excitability_level_ = 0.9f; // High sensitivity to input
        
    } else if (specialization_type_ == "language_understanding") {
        // Language Processing Module parameters
        attention_weight_ = 0.9f;
        learning_rate_modifier_ = 0.8f; // Stable learning for language
        excitability_level_ = 0.7f;
        
    } else if (specialization_type_ == "logical_reasoning") {
        // Reasoning Module parameters
        attention_weight_ = 0.8f;
        learning_rate_modifier_ = 0.9f;
        excitability_level_ = 0.6f; // More controlled activation
        
    } else if (specialization_type_ == "spike_to_action") {
        // Output Module parameters
        attention_weight_ = 0.6f;
        learning_rate_modifier_ = 1.1f; // Quick adaptation for output
        excitability_level_ = 0.8f;
    }
}

// ============================================================================
// REMAINING UTILITY METHODS FOR SPECIALIZED PROCESSING
// ============================================================================

float SpecializedModule::computeVectorSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    
    float dot_product = std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
    float norm_a = std::sqrt(std::inner_product(a.begin(), a.end(), a.begin(), 0.0f));
    float norm_b = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0f));
    
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    
    return dot_product / (norm_a * norm_b);
}

float SpecializedModule::computeActivationSpread(const std::vector<float>& input) {
    if (input.empty()) return 0.0f;
    
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float variance = 0.0f;
    
    for (float val : input) {
        variance += (val - mean) * (val - mean);
    }
    
    return std::sqrt(variance / input.size());
}

void SpecializedModule::updateSemanticMemory(const std::vector<float>& input) {
    // Update semantic memory with new input patterns
    size_t memory_size = std::min(input.size(), semantic_memory_.size());
    
    for (size_t i = 0; i < memory_size; ++i) {
        // Integrate new information with existing memory
        semantic_memory_[i] = 0.95f * semantic_memory_[i] + 0.05f * input[i];
    }
}

std::vector<float> SpecializedModule::integrateWithContext(const std::vector<float>& input) {
    std::vector<float> integrated(input.size(), 0.0f);
    
    // Integrate input with stored context
    for (size_t i = 0; i < integrated.size(); ++i) {
        integrated[i] = input[i];
        
        // Add context contribution
        if (i < context_integrator_.size()) {
            integrated[i] += 0.3f * context_integrator_[i];
        }
    }
    
    // Update context integrator
    size_t context_size = std::min(input.size(), context_integrator_.size());
    for (size_t i = 0; i < context_size; ++i) {
        context_integrator_[i] = 0.9f * context_integrator_[i] + 0.1f * input[i];
    }
    
    return integrated;
}

void SpecializedModule::extractSemanticFeatures(const std::vector<float>& input, std::vector<float>& output) {
    // Extract semantic features from integrated input
    size_t feature_count = std::min(input.size(), output.size());
    
    for (size_t i = 0; i < feature_count; ++i) {
        // Apply semantic transformation
        float semantic_activation = input[i];
        
        // Add semantic memory contribution
        if (i < semantic_memory_.size()) {
            semantic_activation += 0.2f * semantic_memory_[i];
        }
        
        // Add syntactic pattern contribution
        if (i < syntactic_patterns_.size()) {
            semantic_activation += 0.1f * syntactic_patterns_[i];
        }
        
        output[i] = std::tanh(semantic_activation);
    }
}

void SpecializedModule::updateLogicalState(const std::vector<float>& input) {
    // Update logical reasoning state
    size_t state_size = std::min(input.size(), logical_state_.size());
    
    for (size_t i = 0; i < state_size; ++i) {
        // Integrate new logical information
        logical_state_[i] = 0.8f * logical_state_[i] + 0.2f * input[i];
    }
}

std::vector<float> SpecializedModule::applyReasoningRules(const std::vector<float>& input) {
    std::vector<float> rule_outputs(config_.output_size, 0.0f);
    
    // Apply reasoning rules to input
    for (size_t rule_idx = 0; rule_idx < reasoning_rules_.size(); ++rule_idx) {
        float rule_activation = 0.0f;
        
        // Compute rule activation
        for (size_t i = 0; i < input.size() && i < reasoning_rules_[rule_idx].size(); ++i) {
            rule_activation += input[i] * reasoning_rules_[rule_idx][i];
        }
        
        rule_activation = std::tanh(rule_activation);
        
        // Apply rule output to result
        if (rule_idx < rule_outputs.size()) {
            rule_outputs[rule_idx] = rule_activation;
        }
    }
    
    return rule_outputs;
}

std::vector<float> SpecializedModule::detectContradictions(const std::vector<float>& input) {
    std::vector<float> contradictions(contradiction_detector_.size(), 0.0f);
    
    // Simple contradiction detection based on opposing patterns
    for (size_t i = 0; i < contradictions.size(); ++i) {
        if (i * 2 + 1 < input.size()) {
            // Check for opposing values
            float opposition = input[i * 2] * input[i * 2 + 1];
            if (opposition < -0.5f) {
                contradictions[i] = std::abs(opposition);
            }
        }
    }
    
    return contradictions;
}

std::vector<float> SpecializedModule::buildInferenceChains(const std::vector<float>& rule_outputs) {
    std::vector<float> inferences(inference_chains_.size(), 0.0f);
    
    // Build simple inference chains
    for (size_t i = 0; i < inferences.size(); ++i) {
        if (i < rule_outputs.size()) {
            // Current inference is based on rule output
            inferences[i] = rule_outputs[i];
            
            // Add contribution from previous inference
            if (i > 0) {
                inferences[i] += 0.3f * inferences[i - 1];
            }
            
            // Update stored inference chain
            inference_chains_[i] = 0.7f * inference_chains_[i] + 0.3f * inferences[i];
        }
    }
    
    return inferences;
}

std::vector<float> SpecializedModule::decodeSpikes(const std::vector<float>& input) {
    std::vector<float> decoded(spike_decoder_.size(), 0.0f);
    
    // Decode spike patterns to intermediate representation
    for (size_t i = 0; i < decoded.size() && i < input.size(); ++i) {
        // Apply spike decoding transformation
        decoded[i] = input[i];
        
        // Apply temporal integration
        if (i < spike_decoder_.size()) {
            spike_decoder_[i] = 0.8f * spike_decoder_[i] + 0.2f * input[i];
            decoded[i] += 0.3f * spike_decoder_[i];
        }
        
        // Apply activation function
        decoded[i] = std::tanh(decoded[i]);
    }
    
    return decoded;
}

std::vector<float> SpecializedModule::mapToActionPrimitives(const std::vector<float>& decoded_spikes) {
    std::vector<float> actions(action_primitives_.size(), 0.0f);
    
    // Map decoded spikes to action primitives using transformation matrix
    for (size_t i = 0; i < actions.size(); ++i) {
        for (size_t j = 0; j < decoded_spikes.size() && j < spike_to_action_matrix_.size(); ++j) {
            if (i < spike_to_action_matrix_[j].size()) {
                actions[i] += decoded_spikes[j] * spike_to_action_matrix_[j][i];
            }
        }
        
        // Update action primitive state
        if (i < action_primitives_.size()) {
            action_primitives_[i] = 0.9f * action_primitives_[i] + 0.1f * actions[i];
        }
    }
    
    return actions;
}

std::vector<float> SpecializedModule::estimateActionConfidence(const std::vector<float>& action_candidates) {
    std::vector<float> confidence(confidence_estimator_.size(), 0.0f);
    
    // Estimate confidence for each action based on consistency and strength
    for (size_t i = 0; i < confidence.size() && i < action_candidates.size(); ++i) {
        // Base confidence on action strength
        confidence[i] = std::abs(action_candidates[i]);
        
        // Modulate with historical consistency
        if (i < confidence_estimator_.size()) {
            float consistency = 1.0f - std::abs(confidence_estimator_[i] - confidence[i]);
            confidence[i] *= consistency;
            
            // Update confidence estimator
            confidence_estimator_[i] = 0.85f * confidence_estimator_[i] + 0.15f * confidence[i];
        }
        
        // Apply sigmoid to normalize
        confidence[i] = std::sigmoid(confidence[i]);
    }
    
    return confidence;
}

void SpecializedModule::updateAttentionBasedOnPerformance(float reward) {
    // Adjust attention weight based on performance feedback
    if (reward > 0.5f) {
        // Increase attention for good performance
        attention_weight_ = std::min(1.0f, attention_weight_ + 0.01f);
    } else if (reward < -0.5f) {
        // Decrease attention for poor performance
        attention_weight_ = std::max(0.1f, attention_weight_ - 0.01f);
    }
}

// ============================================================================
// HELPER FUNCTION IMPLEMENTATIONS
// ============================================================================

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void SpecializedModule::initializeSpecializedComponents() {
    // Initialize processing buffers based on specialization
    size_t buffer_size = config_.num_neurons;

    processing_buffer_.resize(buffer_size, 0.0f);
    integration_buffer_.resize(buffer_size, 0.0f);
    output_buffer_.resize(config_.output_size, 0.0f);

    // NEW: Initialize biological neural substrate with REAL spiking neurons!
    if (use_biological_neurons_) {
        biological_module_ = std::make_unique<BiologicalNeuronModule>(
            module_name_ + "_biological",
            config_.num_neurons,
            config_
        );

        // Initialize with 80% excitatory neurons and sparse connectivity
        float connection_prob = 0.1f; // 10% connection probability
        biological_module_->initialize(0.8f, connection_prob);

        std::cout << "✅ Biological neurons enabled for module: " << module_name_ << std::endl;
    }

    // Initialize specialized state based on module type
    if (specialization_type_ == "neuromodulatory_control") {
        initializeNeuromodulatoryControl();
    } else if (specialization_type_ == "text_input_processing") {
        initializeTextInputProcessing();
    } else if (specialization_type_ == "language_understanding") {
        initializeLanguageUnderstanding();
    } else if (specialization_type_ == "logical_reasoning") {
        initializeLogicalReasoning();
    } else if (specialization_type_ == "spike_to_action") {
        initializeSpikeToAction();
    }
}

// ============================================================================
// SPECIALIZED INITIALIZATION METHODS
// ============================================================================

void SpecializedModule::initializeNeuromodulatoryControl() {
    // Initialize neuromodulatory state vectors
    dopamine_signals_.resize(config_.output_size, 0.2f);
    acetylcholine_signals_.resize(config_.output_size, 0.3f);
    norepinephrine_signals_.resize(config_.output_size, 0.15f);
    serotonin_signals_.resize(config_.output_size, 0.1f);
    
    std::cout << "🧠 Initialized neuromodulatory control systems" << std::endl;
}

void SpecializedModule::initializeTextInputProcessing() {
    // Initialize tokenization and encoding systems
    token_embeddings_.resize(1024, std::vector<float>(256, 0.0f)); // 1024 tokens, 256-dim
    position_encodings_.resize(512, std::vector<float>(256, 0.0f)); // 512 positions
    
    // Initialize with simple embeddings
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (auto& embedding : token_embeddings_) {
        for (float& val : embedding) {
            val = dist(gen);
        }
    }
    
    std::cout << "📝 Initialized text input processing systems" << std::endl;
}

void SpecializedModule::initializeLanguageUnderstanding() {
    // Initialize language comprehension systems
    semantic_memory_.resize(config_.num_neurons, 0.0f);
    syntactic_patterns_.resize(config_.num_neurons / 2, 0.0f);
    context_integrator_.resize(config_.num_neurons / 4, 0.0f);
    
    // Initialize attention matrices for language understanding
    self_attention_weights_.resize(config_.num_neurons, 
                                  std::vector<float>(config_.num_neurons, 0.0f));
    
    std::cout << "🔤 Initialized language understanding systems" << std::endl;
}

void SpecializedModule::initializeLogicalReasoning() {
    // Initialize reasoning and inference systems
    logical_state_.resize(config_.num_neurons, 0.0f);
    inference_chains_.resize(config_.num_neurons / 2, 0.0f);
    contradiction_detector_.resize(config_.num_neurons / 4, 0.0f);
    
    // Initialize reasoning rule templates
    reasoning_rules_.resize(100, std::vector<float>(64, 0.0f));
    
    std::cout << "🤔 Initialized logical reasoning systems" << std::endl;
}

void SpecializedModule::initializeSpikeToAction() {
    // Initialize spike-to-action conversion systems
    spike_decoder_.resize(config_.input_size, 0.0f);
    action_primitives_.resize(config_.output_size, 0.0f);
    confidence_estimator_.resize(config_.output_size, 0.0f);
    
    // Initialize action mapping matrices
    spike_to_action_matrix_.resize(config_.input_size, 
                                  std::vector<float>(config_.output_size, 0.0f));
    
    std::cout << "⚡ Initialized spike-to-action conversion systems" << std::endl;
}

// ============================================================================
// SPECIALIZED PROCESSING METHODS
// ============================================================================

std::vector<float> SpecializedModule::process(const std::vector<float>& input) {
    if (!is_initialized_ || !active_) {
        return std::vector<float>(config_.output_size, 0.0f);
    }

    std::lock_guard<std::mutex> lock(module_mutex_);

    // NEW: Use biological spiking neurons if enabled!
    if (use_biological_neurons_ && biological_module_) {
        // Process through real spiking neural network
        std::vector<float> bio_output = biological_module_->process(input);

        // Apply specialization-specific post-processing if needed
        if (specialization_type_ == "text_input_processing") {
            // For input module, pass spikes through encoding
            return processTextInput(bio_output);
        } else if (specialization_type_ == "spike_to_action") {
            // For output module, decode spikes to actions
            return processSpikeToAction(bio_output);
        }

        // For other modules, return biological output directly or with scaling
        if (bio_output.size() != config_.output_size) {
            // Resize if needed
            bio_output.resize(config_.output_size, 0.0f);
        }

        return bio_output;
    }

    // LEGACY: Abstract processing (fallback if biological neurons disabled)
    if (specialization_type_ == "neuromodulatory_control") {
        return processNeuromodulatoryControl(input);
    } else if (specialization_type_ == "text_input_processing") {
        return processTextInput(input);
    } else if (specialization_type_ == "language_understanding") {
        return processLanguageUnderstanding(input);
    } else if (specialization_type_ == "logical_reasoning") {
        return processLogicalReasoning(input);
    } else if (specialization_type_ == "spike_to_action") {
        return processSpikeToAction(input);
    }

    // Fallback to base processing
    return EnhancedNeuralModule::process(input);
}

std::vector<float> SpecializedModule::processNeuromodulatoryControl(const std::vector<float>& input) {
    // Central Controller: Generate neuromodulatory signals for other modules

    // Analyze input to determine appropriate neuromodulation
    float input_novelty = computeInputNovelty(input);
    float input_complexity = computeInputComplexity(input);
    float attention_demand = computeAttentionDemand(input);

    // Update neuromodulator levels
    updateNeuromodulatorLevels(input_novelty, input_complexity, attention_demand);

    // NEW: Apply neuromodulation to biological neurons
    if (use_biological_neurons_ && biological_module_) {
        float avg_dopamine = std::accumulate(dopamine_signals_.begin(), dopamine_signals_.end(), 0.0f)
                           / dopamine_signals_.size();
        float avg_acetylcholine = std::accumulate(acetylcholine_signals_.begin(), acetylcholine_signals_.end(), 0.0f)
                                / acetylcholine_signals_.size();
        float avg_norepinephrine = std::accumulate(norepinephrine_signals_.begin(), norepinephrine_signals_.end(), 0.0f)
                                 / norepinephrine_signals_.size();

        biological_module_->setNeuromodulators(avg_dopamine, avg_acetylcholine, avg_norepinephrine);
    }

    // Generate control signals
    std::vector<float> control_output(config_.output_size, 0.0f);

    for (size_t i = 0; i < control_output.size(); ++i) {
        // Combine multiple neuromodulatory signals
        float dopamine_component = dopamine_signals_[i % dopamine_signals_.size()];
        float acetylcholine_component = acetylcholine_signals_[i % acetylcholine_signals_.size()];
        float norepinephrine_component = norepinephrine_signals_[i % norepinephrine_signals_.size()];

        control_output[i] = 0.4f * dopamine_component +
                           0.3f * acetylcholine_component +
                           0.3f * norepinephrine_component;

        // Apply sigmoid activation for smooth control
        control_output[i] = std::tanh(control_output[i]);
    }

    return control_output;
}

std::vector<float> SpecializedModule::processTextInput(const std::vector<float>& input) {
    // Input Module: Convert raw text tokens to neural representations
    
    std::vector<float> processed_output(config_.output_size, 0.0f);
    
    // Extract tokens from input (assuming first half is character codes, second half is positions)
    size_t token_count = std::min(input.size() / 2, static_cast<size_t>(512));
    
    for (size_t i = 0; i < token_count; ++i) {
        if (i >= input.size()) break;
        
        // Get token and position
        int token_id = static_cast<int>(input[i] * 255.0f) % token_embeddings_.size();
        int position = static_cast<int>(input[i + 512] * 512.0f) % position_encodings_.size();
        
        // Combine token embedding and position encoding
        for (size_t j = 0; j < 256 && j < processed_output.size(); ++j) {
            float token_component = token_embeddings_[token_id][j];
            float position_component = position_encodings_[position][j];
            
            processed_output[j] += token_component + 0.1f * position_component;
        }
    }
    
    // Normalize output
    float norm = std::sqrt(std::inner_product(processed_output.begin(), processed_output.end(),
                                             processed_output.begin(), 0.0f));
    if (norm > 0.0f) {
        for (float& val : processed_output) {
            val /= norm;
        }
    }
    
    return processed_output;
}

std::vector<float> SpecializedModule::processLanguageUnderstanding(const std::vector<float>& input) {
    // Language Processing Module: Deep language comprehension and semantic analysis
    
    std::vector<float> language_output(config_.output_size, 0.0f);
    
    // Update semantic memory with input
    updateSemanticMemory(input);
    
    // Apply self-attention mechanism
    std::vector<float> attended_input = applySelfAttention(input);
    
    // Integrate with existing context
    std::vector<float> context_integrated = integrateWithContext(attended_input);
    
    // Extract semantic features
    extractSemanticFeatures(context_integrated, language_output);
    
    // Apply language-specific activations
    for (float& val : language_output) {
        val = std::tanh(val * excitability_level_);
    }
    
    return language_output;
}

std::vector<float> SpecializedModule::processLogicalReasoning(const std::vector<float>& input) {
    // Reasoning Module: Logical inference and reasoning operations
    
    std::vector<float> reasoning_output(config_.output_size, 0.0f);
    
    // Update logical state with new input
    updateLogicalState(input);
    
    // Apply reasoning rules
    std::vector<float> rule_outputs = applyReasoningRules(input);
    
    // Check for contradictions
    std::vector<float> contradiction_signals = detectContradictions(input);
    
    // Build inference chains
    std::vector<float> inference_results = buildInferenceChains(rule_outputs);
    
    // Combine reasoning components
    for (size_t i = 0; i < reasoning_output.size(); ++i) {
        if (i < rule_outputs.size()) reasoning_output[i] += 0.4f * rule_outputs[i];
        if (i < inference_results.size()) reasoning_output[i] += 0.4f * inference_results[i];
        if (i < contradiction_signals.size()) reasoning_output[i] -= 0.2f * contradiction_signals[i];
        
        // Apply reasoning-specific activation
        reasoning_output[i] = std::tanh(reasoning_output[i] * 0.8f);
    }
    
    return reasoning_output;
}

std::vector<float> SpecializedModule::processSpikeToAction(const std::vector<float>& input) {
    // Output Module: Convert neural spike patterns to actionable responses
    
    std::vector<float> action_output(config_.output_size, 0.0f);
    
    // Decode spike patterns
    std::vector<float> decoded_spikes = decodeSpikes(input);
    
    // Map to action primitives
    std::vector<float> action_candidates = mapToActionPrimitives(decoded_spikes);
    
    // Estimate confidence for each action
    std::vector<float> confidence_scores = estimateActionConfidence(action_candidates);
    
    // Generate final output with confidence weighting
    for (size_t i = 0; i < action_output.size(); ++i) {
        if (i < action_candidates.size() && i < confidence_scores.size()) {
            action_output[i] = action_candidates[i] * confidence_scores[i];
        }
        
        // Apply output activation function
        action_output[i] = std::sigmoid(action_output[i]);
    }
    
    return action_output;
}

// ============================================================================
// UTILITY METHODS FOR SPECIALIZED PROCESSING
// ============================================================================

float SpecializedModule::computeInputNovelty(const std::vector<float>& input) {
    // Compute how novel the input is compared to recent history
    if (input_history_.empty()) {
        input_history_.push_back(input);
        return 1.0f; // First input is maximally novel
    }
    
    // Compare with recent inputs
    float min_similarity = 1.0f;
    for (const auto& historical_input : input_history_) {
        float similarity = computeVectorSimilarity(input, historical_input);
        min_similarity = std::min(min_similarity, similarity);
    }
    
    // Update history (keep last 10 inputs)
    input_history_.push_back(input);
    if (input_history_.size() > 10) {
        input_history_.erase(input_history_.begin());
    }
    
    return 1.0f - min_similarity; // Novelty is inverse of similarity
}

float SpecializedModule::computeInputComplexity(const std::vector<float>& input) {
    // Compute input complexity based on variance and entropy
    if (input.empty()) return 0.0f;
    
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float variance = 0.0f;
    
    for (float val : input) {
        variance += (val - mean) * (val - mean);
    }
    variance /= input.size();
    
    return std::min(1.0f, variance * 10.0f); // Scale and clamp
}

float SpecializedModule::computeAttentionDemand(const std::vector<float>& input) {
    // Compute how much attention this input requires
    float max_activation = *std::max_element(input.begin(), input.end());
    float activation_spread = computeActivationSpread(input);
    
    return std::min(1.0f, max_activation + 0.5f * activation_spread);
}

void SpecializedModule::updateNeuromodulatorLevels(float novelty, float complexity, float attention) {
    // Update dopamine (reward/novelty)
    float dopamine_target = 0.1f + 0.4f * novelty;
    for (float& val : dopamine_signals_) {
        val = 0.9f * val + 0.1f * dopamine_target;
    }
    
    // Update acetylcholine (attention)
    float acetylcholine_target = 0.2f + 0.3f * attention;
    for (float& val : acetylcholine_signals_) {
        val = 0.95f * val + 0.05f * acetylcholine_target;
    }
    
    // Update norepinephrine (arousal/complexity)
    float norepinephrine_target = 0.1f + 0.2f * complexity;
    for (float& val : norepinephrine_signals_) {
        val = 0.98f * val + 0.02f * norepinephrine_target;
    }
}

std::vector<float> SpecializedModule::applySelfAttention(const std::vector<float>& input) {
    std::vector<float> attended_output(input.size(), 0.0f);
    
    // Simplified self-attention mechanism
    for (size_t i = 0; i < input.size(); ++i) {
        float attention_sum = 0.0f;
        
        for (size_t j = 0; j < input.size(); ++j) {
            float attention_weight = 0.0f;
            if (i < self_attention_weights_.size() && j < self_attention_weights_[i].size()) {
                attention_weight = self_attention_weights_[i][j];
            }
            
            attended_output[i] += attention_weight * input[j];
            attention_sum += attention_weight;
        }
        
        // Normalize
        if (attention_sum > 0.0f) {
            attended_output[i] /= attention_sum;
        }
    }
    
    return attended_output;
}

// ============================================================================
// MODULE CONFIGURATION AND CONTROL
// ============================================================================

void SpecializedModule::set_specialization_type(const std::string& type) {
    specialization_type_ = type;
    initializeSpecializationParameters();
    std::cout << "🔄 Module '" << module_name_ 
              << "' specialization changed to: " << type << std::endl;
}

const std::string& SpecializedModule::get_specialization_type() const {
    return specialization_type_;
}

void SpecializedModule::set_attention_weight(float weight) {
    attention_weight_ = std::max(0.0f, std::min(1.0f, weight));
}

float SpecializedModule::get_attention_weight() const {
    return attention_weight_;
}

// ============================================================================
// LEARNING AND ADAPTATION
// ============================================================================

void SpecializedModule::update(float dt, const std::vector<float>& inputs, float reward) {
    // NEW: Update biological neurons with reward-modulated learning
    if (use_biological_neurons_ && biological_module_) {
        biological_module_->update(dt, inputs, reward);
    }

    // Call parent update
    EnhancedNeuralModule::update(dt, inputs, reward);

    // Apply specialized learning updates
    applySpecializedLearning(reward, dt);

    // Update attention weights based on performance
    updateAttentionBasedOnPerformance(reward);
}

void SpecializedModule::applySpecializedLearning(float reward, float dt) {
    float effective_learning_rate = learning_rate_modifier_ * dt;
    
    if (specialization_type_ == "language_understanding") {
        // Update semantic memory with reward-modulated learning
        for (float& memory_val : semantic_memory_) {
            memory_val += effective_learning_rate * reward * 0.1f;
            memory_val = std::max(-1.0f, std::min(1.0f, memory_val)); // Clamp
        }
    } else if (specialization_type_ == "logical_reasoning") {
        // Update reasoning rules based on success
        if (reward > 0.5f) {
            // Strengthen successful reasoning patterns
            for (auto& rule : reasoning_rules_) {
                for (float& weight : rule) {
                    weight += effective_learning_rate * reward * 0.05f;
                    weight = std::max(-2.0f, std::min(2.0f, weight)); // Clamp
                }
            }
        }
    }
}