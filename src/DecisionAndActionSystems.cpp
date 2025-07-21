// ============================================================================
// DECISION AND ACTION SYSTEMS - FIXED IMPLEMENTATION
// File: src/DecisionAndActionSystems.cpp
// ============================================================================

#include "NeuroGen/DecisionAndActionSystems.h"
#include "NeuroGen/SpecializedModule.h"
#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/MemorySystem.h"
#include "NeuroGen/AttentionController.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>

// ============================================================================
// SPECIALIZED MODULE - EXECUTIVE FUNCTION PROCESSING
// ============================================================================

std::vector<float> SpecializedModule::process_executive_function(const std::vector<float>& executive_input) {
    if (executive_input.empty()) {
        return std::vector<float>(output_buffer_.size(), 0.0f);
    }
    
    // Executive function integrates information for strategic language use
    for (size_t i = 0; i < internal_state_.size() && i < executive_input.size(); ++i) {
        // Executive function integrates information for strategic language use
        internal_state_[i] = internal_state_[i] * 0.75f + executive_input[i] * attention_weight_ * 0.25f;
    }
    
    // Generate executive control signals for language processing
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float control_signal = 0.0f;
        
        // Integrate executive control patterns
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            // Executive control uses strategic weighting
            float strategic_weight = 1.0f + 0.4f * attention_weight_;
            control_signal += internal_state_[j] * strategic_weight;
        }
        
        control_signal /= (state_end - state_start);
        
        // Apply executive control activation - **FIXED: Use SpecializedModule::sigmoid**
        output_buffer_[i] = SpecializedModule::sigmoid(control_signal * 1.3f);
    }
    
    return output_buffer_;
}

// ============================================================================
// SPECIALIZED MODULE - WORKING MEMORY PROCESSING
// ============================================================================

std::vector<float> SpecializedModule::process_working_memory(const std::vector<float>& memory_input) {
    if (memory_input.empty()) {
        return std::vector<float>(output_buffer_.size(), 0.0f);
    }
    
    // Working memory maintains temporary linguistic information
    for (size_t i = 0; i < internal_state_.size() && i < memory_input.size(); ++i) {
        // Working memory uses rapid updates with decay
        float decay_factor = 0.9f * attention_weight_;
        internal_state_[i] = internal_state_[i] * decay_factor + memory_input[i] * (1.0f - decay_factor);
    }
    
    // Generate working memory outputs
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        size_t state_index = (i * internal_state_.size()) / output_buffer_.size();
        float memory_output = internal_state_[state_index];
        
        // Apply working memory characteristics - bounded activation
        output_buffer_[i] = std::tanh(memory_output * 1.2f);
    }
    
    return output_buffer_;
}

// ============================================================================
// SPECIALIZED MODULE - REWARD SYSTEM PROCESSING
// ============================================================================

std::vector<float> SpecializedModule::process_reward_system(const std::vector<float>& reward_input) {
    if (reward_input.empty()) {
        return std::vector<float>(output_buffer_.size(), 0.0f);
    }
    
    // Reward system processes reinforcement signals for language learning
    for (size_t i = 0; i < internal_state_.size() && i < reward_input.size(); ++i) {
        // Reward system integrates positive and negative signals
        float reward_integration = reward_input[i] * attention_weight_;
        internal_state_[i] = 0.8f * internal_state_[i] + 0.2f * reward_integration;
    }
    
    // Generate reward-modulated outputs
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        size_t state_index = (i * internal_state_.size()) / output_buffer_.size();
        float reward_signal = internal_state_[state_index];
        
        // Apply reward system characteristics - bipolar activation
        output_buffer_[i] = std::tanh(reward_signal * 2.0f);
    }
    
    return output_buffer_;
}

// ============================================================================
// SPECIALIZED MODULE - ATTENTION SYSTEM PROCESSING
// ============================================================================

std::vector<float> SpecializedModule::process_attention_system(const std::vector<float>& attention_input) {
    if (attention_input.empty()) {
        return std::vector<float>(output_buffer_.size(), 0.0f);
    }
    
    // Attention system allocates processing resources for language
    for (size_t i = 0; i < internal_state_.size() && i < attention_input.size(); ++i) {
        // Attention uses competitive dynamics
        float competition_factor = 1.0f + 0.5f * std::sin(static_cast<float>(i) * 0.1f);
        internal_state_[i] = attention_input[i] * competition_factor * attention_weight_;
    }
    
    // Apply softmax-like normalization for attention weights
    float sum_exp = 0.0f;
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        size_t state_index = (i * internal_state_.size()) / output_buffer_.size();
        float exp_val = std::exp(internal_state_[state_index]);
        output_buffer_[i] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize attention weights
    if (sum_exp > 0.0f) {
        for (float& weight : output_buffer_) {
            weight /= sum_exp;
        }
    }
    
    return output_buffer_;
}

// ============================================================================
// SPECIALIZED MODULE - MOTOR CORTEX PROCESSING (ADAPTED FOR TEXT OUTPUT)
// ============================================================================

std::vector<float> SpecializedModule::process_motor_cortex(const std::vector<float>& motor_input) {
    if (motor_input.empty()) {
        return std::vector<float>(output_buffer_.size(), 0.0f);
    }
    
    // Motor cortex adapted for text generation control
    for (size_t i = 0; i < internal_state_.size() && i < motor_input.size(); ++i) {
        // Motor control uses smooth, coordinated activation
        float smoothing_factor = 0.85f * attention_weight_;
        internal_state_[i] = internal_state_[i] * smoothing_factor + motor_input[i] * (1.0f - smoothing_factor);
    }
    
    // Generate motor control outputs for text generation
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        size_t state_index = (i * internal_state_.size()) / output_buffer_.size();
        float motor_signal = internal_state_[state_index];
        
        // Apply motor control characteristics - smooth, bounded activation
        output_buffer_[i] = SpecializedModule::sigmoid(motor_signal * 1.5f);
    }
    
    return output_buffer_;
}

// ============================================================================
// AUTONOMOUS LEARNING AGENT - DECISION AND ACTION METHODS
// ============================================================================

void AutonomousLearningAgent::make_autonomous_decision() {
    if (!modules_.count("executive_function")) return;
    
    // Gather decision context from language processing modules
    std::vector<float> decision_context;
    decision_context.reserve(1024);
    
    // Add language comprehension state
    if (modules_.count("language_comprehension")) {
        auto comp_state = modules_["language_comprehension"]->get_output();
        size_t comp_size = std::min(comp_state.size(), size_t(256));
        decision_context.insert(decision_context.end(), 
                              comp_state.begin(), 
                              comp_state.begin() + comp_size);
    }
    
    // Add semantic memory context
    if (modules_.count("semantic_memory")) {
        auto sem_state = modules_["semantic_memory"]->get_output();
        size_t sem_size = std::min(sem_state.size(), size_t(256));
        decision_context.insert(decision_context.end(),
                              sem_state.begin(),
                              sem_state.begin() + sem_size);
    }
    
    // Add working memory - **FIXED: Use available method**
    if (memory_system_) {
        auto working_memory = memory_system_->getWorkingMemory();
        size_t wm_size = std::min(working_memory.size(), size_t(256));
        decision_context.insert(decision_context.end(),
                              working_memory.begin(),
                              working_memory.begin() + wm_size);
    }
    
    // Add current goals and constraints
    size_t goal_size = std::min(current_goals_.size(), size_t(128));
    decision_context.insert(decision_context.end(),
                          current_goals_.begin(),
                          current_goals_.begin() + goal_size);
    
    // Add global context
    decision_context.push_back(global_reward_signal_);
    decision_context.push_back(simulation_time_ / 1000.0f);
    
    // Process through executive function module
    float exec_attention = attention_controller_->get_attention_weight("executive_function");
    
    // Apply attention weighting
    for (size_t i = 0; i < decision_context.size(); ++i) {
        decision_context[i] *= exec_attention;
    }
    
    auto decision_output = modules_["executive_function"]->process(decision_context);
    
    // Extract decision from neural output
    if (!decision_output.empty()) {
        // Simple decision extraction - could be more sophisticated
        float decision_value = decision_output[0];
        
        if (decision_value > 0.7f) {
            current_decision_ = "generate_response";
        } else if (decision_value > 0.4f) {
            current_decision_ = "seek_more_information";
        } else if (decision_value > 0.1f) {
            current_decision_ = "consolidate_memory";
        } else {
            current_decision_ = "wait_and_observe";
        }
        
        // Update decision confidence
        decision_confidence_ = std::abs(decision_value);
    }
}

void AutonomousLearningAgent::execute_action() {
    // Execute action based on current decision (language-focused)
    
    if (current_decision_ == "generate_response") {
        process_language_generation();
        
        // Output response if available
        if (!current_language_response_.empty() && language_interface_) {
            language_interface_->outputResponse(current_language_response_);
            std::cout << "Generated Response: " << current_language_response_ << std::endl;
        }
        
    } else if (current_decision_ == "seek_more_information") {
        // Request more input or context
        if (language_interface_) {
            language_interface_->requestMoreInput();
        }
        
    } else if (current_decision_ == "consolidate_memory") {
        // Trigger memory consolidation
        consolidate_learning();
        
    } else if (current_decision_ == "wait_and_observe") {
        // Continue processing current context
        // No immediate action needed
    }
    
    // Log action for learning
    if (safety_manager_) {
        std::cout << "Action Decision: " << current_decision_ 
                  << " (confidence: " << decision_confidence_ << ")" << std::endl;
    }
}

void AutonomousLearningAgent::consolidate_learning() {
    // **FIXED: Use available memory system methods**
    if (memory_system_) {
        try {
            // Use available consolidation method
            memory_system_->consolidateMemories();
            std::cout << "Memory consolidation completed" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Memory consolidation failed: " << e.what() << std::endl;
        }
    }
    
    // **FIXED: Use available attention controller methods**
    if (attention_controller_) {
        // Simple attention pattern consolidation
        std::cout << "Consolidating attention patterns for language processing..." << std::endl;
        // Could implement custom consolidation logic here
    }
}

// ============================================================================
// HELPER METHODS FOR LINGUISTIC PROCESSING
// ============================================================================

// **REMOVED: sigmoid redefinition to avoid conflict with header inline definition**

std::vector<float> AutonomousLearningAgent::extractLanguageFeatures(const std::string& text) {
    // Simplified feature extraction - would be much more sophisticated in practice
    std::vector<float> features(768, 0.0f); // Standard embedding size
    
    // Basic word count and length features
    features[0] = static_cast<float>(text.length()) / 1000.0f; // Normalized text length
    
    // Count common words (simplified)
    size_t word_count = std::count(text.begin(), text.end(), ' ') + 1;
    features[1] = static_cast<float>(word_count) / 100.0f; // Normalized word count
    
    // Character frequency analysis (basic linguistic features)
    std::map<char, int> char_freq;
    for (char c : text) {
        if (std::isalpha(c)) {
            char_freq[std::tolower(c)]++;
        }
    }
    
    // Map character frequencies to features
    for (const auto& [ch, freq] : char_freq) {
        size_t index = static_cast<size_t>(ch - 'a') + 10;
        if (index < features.size()) {
            features[index] = static_cast<float>(freq) / static_cast<float>(text.length());
        }
    }
    
    // Simple syntactic features (count punctuation, sentence structures)
    features[50] = static_cast<float>(std::count(text.begin(), text.end(), '.')) / static_cast<float>(word_count);
    features[51] = static_cast<float>(std::count(text.begin(), text.end(), '?')) / static_cast<float>(word_count);
    features[52] = static_cast<float>(std::count(text.begin(), text.end(), '!')) / static_cast<float>(word_count);
    
    return features;
}

void AutonomousLearningAgent::update_global_state() {
    // Update global cognitive state based on language processing modules
    
    // Aggregate language comprehension state
    if (modules_.count("language_comprehension")) {
        auto comp_output = modules_["language_comprehension"]->get_output();
        size_t comp_size = std::min(comp_output.size(), global_state_.size() / 6);
        for (size_t i = 0; i < comp_size; ++i) {
            global_state_[i] = comp_output[i];
        }
    }
    
    // Aggregate semantic memory state
    if (modules_.count("semantic_memory")) {
        auto sem_output = modules_["semantic_memory"]->get_output();
        size_t sem_offset = global_state_.size() / 6;
        size_t sem_size = std::min(sem_output.size(), global_state_.size() / 6);
        for (size_t i = 0; i < sem_size; ++i) {
            global_state_[sem_offset + i] = sem_output[i];
        }
    }
    
    // Aggregate working memory state
    if (modules_.count("working_memory")) {
        auto wm_output = modules_["working_memory"]->get_output();
        size_t wm_offset = 2 * global_state_.size() / 6;
        size_t wm_size = std::min(wm_output.size(), global_state_.size() / 6);
        for (size_t i = 0; i < wm_size; ++i) {
            global_state_[wm_offset + i] = wm_output[i];
        }
    }
    
    // Aggregate executive function state
    if (modules_.count("executive_function")) {
        auto exec_output = modules_["executive_function"]->get_output();
        size_t exec_offset = 3 * global_state_.size() / 6;
        size_t exec_size = std::min(exec_output.size(), global_state_.size() / 6);
        for (size_t i = 0; i < exec_size; ++i) {
            global_state_[exec_offset + i] = exec_output[i];
        }
    }
    
    // Add attention and reward system states
    if (modules_.count("attention_system")) {
        auto att_output = modules_["attention_system"]->get_output();
        size_t att_offset = 4 * global_state_.size() / 6;
        size_t att_size = std::min(att_output.size(), global_state_.size() / 12);
        for (size_t i = 0; i < att_size; ++i) {
            global_state_[att_offset + i] = att_output[i];
        }
    }
    
    if (modules_.count("reward_system")) {
        auto rew_output = modules_["reward_system"]->get_output();
        size_t rew_offset = 4 * global_state_.size() / 6 + global_state_.size() / 12;
        size_t rew_size = std::min(rew_output.size(), global_state_.size() / 12);
        for (size_t i = 0; i < rew_size; ++i) {
            global_state_[rew_offset + i] = rew_output[i];
        }
    }
    
    // Update global metrics
    global_reward_signal_ = calculateCurrentReward();
    
    // Update simulation time
    simulation_time_ += 0.1f; // Increment by typical time step
}

float AutonomousLearningAgent::calculateCurrentReward() {
    float total_reward = 0.0f;
    int reward_sources = 0;
    
    // Language processing quality rewards
    if (modules_.count("language_comprehension")) {
        auto comp_output = modules_["language_comprehension"]->get_output();
        float comp_activity = 0.0f;
        for (float val : comp_output) {
            comp_activity += std::abs(val);
        }
        comp_activity /= comp_output.size();
        total_reward += comp_activity * 0.3f; // 30% weight for comprehension
        reward_sources++;
    }
    
    if (modules_.count("language_production")) {
        auto prod_output = modules_["language_production"]->get_output();
        float prod_activity = 0.0f;
        for (float val : prod_output) {
            prod_activity += std::abs(val);
        }
        prod_activity /= prod_output.size();
        total_reward += prod_activity * 0.3f; // 30% weight for production
        reward_sources++;
    }
    
    // Decision making reward
    total_reward += decision_confidence_ * 0.2f; // 20% weight for decision confidence
    reward_sources++;
    
    // Learning objective rewards
    for (const auto& [objective, target] : language_learning_objectives_) {
        // Simple objective achievement reward
        float achievement = std::min(1.0f, target);
        total_reward += achievement * 0.2f; // 20% weight for objectives
    }
    if (!language_learning_objectives_.empty()) {
        reward_sources++;
    }
    
    return (reward_sources > 0) ? total_reward / reward_sources : 0.0f;
}