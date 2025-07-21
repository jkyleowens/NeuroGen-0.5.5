// ============================================================================
// DECISION MAKING AND ACTION EXECUTION SYSTEMS - LANGUAGE-FOCUSED
// File: src/DecisionAndActionSystems.cpp
// ============================================================================

#include <NeuroGen/SpecializedModule.h>
#include <NeuroGen/AutonomousLearningAgent.h>
#include <NeuroGen/SafetyManager.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>

// ============================================================================
// SPECIALIZED MODULE PROCESSING METHODS - LANGUAGE-FOCUSED
// ============================================================================

std::vector<float> SpecializedModule::process_language_comprehension(const std::vector<float>& language_input) {
    // Language comprehension: Deep semantic and syntactic understanding
    size_t input_size = std::min(language_input.size(), internal_state_.size());
    
    // Update comprehension state with attention weighting
    for (size_t i = 0; i < input_size; ++i) {
        // Language comprehension uses deeper integration for semantic understanding
        internal_state_[i] = internal_state_[i] * 0.7f + language_input[i] * attention_weight_ * 0.3f;
    }
    
    // Apply linguistic processing with hierarchical feature extraction
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float semantic_activation = 0.0f;
        
        // Integrate from corresponding internal state region with semantic clustering
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            // Apply semantic weighting based on position in comprehension hierarchy
            float semantic_weight = 1.0f + 0.5f * std::sin(static_cast<float>(j) * 0.1f);
            semantic_activation += internal_state_[j] * semantic_weight;
        }
        
        semantic_activation /= (state_end - state_start);
        
        // Apply linguistic activation function (more sensitive to language patterns)
        output_buffer_[i] = std::tanh(semantic_activation * 1.5f) * attention_weight_;
    }
    
    return output_buffer_;
}

std::vector<float> SpecializedModule::process_language_production(const std::vector<float>& generation_input) {
    // Language production: Generate coherent and contextually appropriate text
    size_t input_size = std::min(generation_input.size(), internal_state_.size());
    
    // Update generation state with creative and coherent patterns
    for (size_t i = 0; i < input_size; ++i) {
        // Language production uses more dynamic integration for creativity
        internal_state_[i] = internal_state_[i] * 0.6f + generation_input[i] * attention_weight_ * 0.4f;
    }
    
    // Generate language output with syntactic and semantic constraints
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float generation_strength = 0.0f;
        
        // Integrate from corresponding internal state with language flow
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            // Apply syntactic flow weighting for natural language generation
            float flow_weight = 1.0f + 0.3f * std::cos(static_cast<float>(j) * 0.15f);
            generation_strength += internal_state_[j] * flow_weight;
        }
        
        generation_strength /= (state_end - state_start);
        
        // Apply generation activation with creativity modulation
        float creativity_factor = 0.8f + 0.4f * attention_weight_;
        output_buffer_[i] = std::tanh(generation_strength * creativity_factor);
    }
    
    return output_buffer_;
}

std::vector<float> SpecializedModule::process_semantic_memory(const std::vector<float>& semantic_input) {
    // Semantic memory: Store and retrieve conceptual knowledge
    size_t input_size = std::min(semantic_input.size(), internal_state_.size());
    
    // Update semantic memory with long-term pattern storage
    for (size_t i = 0; i < input_size; ++i) {
        // Semantic memory uses gradual integration to preserve knowledge
        internal_state_[i] = internal_state_[i] * 0.95f + semantic_input[i] * attention_weight_ * 0.05f;
    }
    
    // Retrieve semantic associations and concepts
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float semantic_retrieval = 0.0f;
        
        // Access semantic networks with associative activation
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            // Apply associative weighting for semantic network activation
            float association_strength = 1.0f;
            if (j > 0 && j < internal_state_.size() - 1) {
                // Lateral connections in semantic network
                association_strength += 0.2f * (internal_state_[j-1] + internal_state_[j+1]) / 2.0f;
            }
            semantic_retrieval += internal_state_[j] * association_strength;
        }
        
        semantic_retrieval /= (state_end - state_start);
        
        // Apply semantic activation with concept strengthening
        output_buffer_[i] = semantic_retrieval * attention_weight_;
    }
    
    return output_buffer_;
}

std::vector<float> SpecializedModule::process_syntactic_processor(const std::vector<float>& syntax_input) {
    // Syntactic processor: Handle grammar, sentence structure, and linguistic rules
    size_t input_size = std::min(syntax_input.size(), internal_state_.size());
    
    // Update syntactic processing state
    for (size_t i = 0; i < input_size; ++i) {
        // Syntactic processing maintains structural patterns
        internal_state_[i] = internal_state_[i] * 0.8f + syntax_input[i] * attention_weight_ * 0.2f;
    }
    
    // Process grammatical structures and syntactic patterns
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float syntactic_pattern = 0.0f;
        
        // Analyze syntactic structure with hierarchical processing
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            // Apply hierarchical syntactic weighting
            float hierarchy_level = static_cast<float>(j) / internal_state_.size();
            float syntactic_weight = 1.0f + 0.5f * std::pow(hierarchy_level, 0.3f);
            syntactic_pattern += internal_state_[j] * syntactic_weight;
        }
        
        syntactic_pattern /= (state_end - state_start);
        
        // Apply grammatical constraints
        output_buffer_[i] = std::tanh(syntactic_pattern * 1.2f) * attention_weight_;
    }
    
    return output_buffer_;
}

std::vector<float> SpecializedModule::process_working_memory(const std::vector<float>& memory_input) {
    // Working memory: Maintain and manipulate linguistic information temporarily
    size_t input_size = std::min(memory_input.size(), internal_state_.size());
    
    // Update working memory with gating mechanisms
    for (size_t i = 0; i < input_size; ++i) {
        // Working memory uses selective gating for relevance
        float relevance_gate = attention_weight_ > 0.5f ? 1.0f : 0.3f;
        float update_rate = relevance_gate * 0.4f;
        internal_state_[i] = internal_state_[i] * (1.0f - update_rate) + 
                            memory_input[i] * update_rate;
    }
    
    // Working memory decay (forgetting) - slightly faster for language processing
    for (float& state : internal_state_) {
        state *= 0.97f; // Faster decay for linguistic working memory
    }
    
    // Output maintained linguistic information
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float maintained_info = 0.0f;
        
        // Average over corresponding internal state with recency weighting
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            // More recent items (higher indices) have stronger weights
            float recency_weight = 1.0f + 0.3f * (static_cast<float>(j) / internal_state_.size());
            maintained_info += internal_state_[j] * recency_weight;
        }
        
        output_buffer_[i] = maintained_info / (state_end - state_start);
    }
    
    return output_buffer_;
}

std::vector<float> SpecializedModule::process_executive_function(const std::vector<float>& executive_input) {
    // Executive function: High-level language control and decision making
    size_t input_size = std::min(executive_input.size(), internal_state_.size());
    
    // Update executive state with goal-directed processing
    for (size_t i = 0; i < input_size; ++i) {
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
        
        // Apply executive control activation
        output_buffer_[i] = std::sigmoid(control_signal * 1.3f);
    }
    
    return output_buffer_;
}

// ============================================================================
// AUTONOMOUS LEARNING AGENT - LANGUAGE PROCESSING METHODS
// ============================================================================

void AutonomousLearningAgent::update_global_state() {
    // Update global cognitive state based on language processing modules
    
    // Aggregate language comprehension state
    if (modules_.count("language_comprehension")) {
        auto comp_output = modules_["language_comprehension"]->get_output();
        size_t comp_size = std::min(comp_output.size(), global_state_.size() / 6);
        for (size_t i = 0; i < comp_size; ++i) {
            global_state_[i] = global_state_[i] * 0.9f + comp_output[i] * 0.1f;
        }
    }
    
    // Aggregate semantic memory state
    if (modules_.count("semantic_memory")) {
        auto sem_output = modules_["semantic_memory"]->get_output();
        size_t sem_offset = global_state_.size() / 6;
        size_t sem_size = std::min(sem_output.size(), global_state_.size() / 6);
        for (size_t i = 0; i < sem_size && (sem_offset + i) < global_state_.size(); ++i) {
            global_state_[sem_offset + i] = global_state_[sem_offset + i] * 0.95f + sem_output[i] * 0.05f;
        }
    }
    
    // Aggregate working memory state
    if (modules_.count("working_memory")) {
        auto wm_output = modules_["working_memory"]->get_output();
        size_t wm_offset = 2 * global_state_.size() / 6;
        size_t wm_size = std::min(wm_output.size(), global_state_.size() / 6);
        for (size_t i = 0; i < wm_size && (wm_offset + i) < global_state_.size(); ++i) {
            global_state_[wm_offset + i] = global_state_[wm_offset + i] * 0.85f + wm_output[i] * 0.15f;
        }
    }
    
    // Add attention distribution to global state
    auto attention_weights = getAttentionWeights();
    std::vector<float> attention_vector;
    for (const auto& [module, weight] : attention_weights) {
        attention_vector.push_back(weight);
    }
    
    size_t att_offset = 3 * global_state_.size() / 6;
    for (size_t i = 0; i < std::min(attention_vector.size(), global_state_.size() / 8); ++i) {
        if (att_offset + i < global_state_.size()) {
            global_state_[att_offset + i] = attention_vector[i];
        }
    }
    
    // Add current language processing state
    size_t lang_offset = 4 * global_state_.size() / 6;
    if (!environmental_context_.empty()) {
        size_t lang_size = std::min(environmental_context_.size() / 4, global_state_.size() / 8);
        for (size_t i = 0; i < lang_size && (lang_offset + i) < global_state_.size(); ++i) {
            global_state_[lang_offset + i] = environmental_context_[i];
        }
    }
    
    // Add reward history
    size_t reward_offset = 5 * global_state_.size() / 6;
    if (reward_offset < global_state_.size()) {
        global_state_[reward_offset] = global_reward_signal_;
    }
    
    // Bound global state values
    for (float& state : global_state_) {
        state = std::max(-2.0f, std::min(state, 2.0f));
    }
}

void AutonomousLearningAgent::consolidate_learning() {
    // Language-focused learning consolidation
    
    std::cout << "Learning consolidation: Integrating language experiences..." << std::endl;
    
    // Consolidate language memories
    if (memory_system_) {
        memory_system_->consolidateMemories();
        
        // Strengthen language patterns that led to positive outcomes
        if (global_reward_signal_ > 0.3f) {
            memory_system_->strengthenRecentMemories(0.1f);
        }
    }
    
    // Transfer knowledge between language modules
    transfer_knowledge_between_modules();
    
    // Consolidate attention patterns for language processing
    if (attention_controller_) {
        attention_controller_->consolidateAttentionPatterns();
    }
    
    std::cout << "Language learning consolidation complete." << std::endl;
}

// ============================================================================
// LANGUAGE-SPECIFIC UTILITY FUNCTIONS
// ============================================================================

float SpecializedModule::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<float> AutonomousLearningAgent::extract_linguistic_context() {
    // Extract current linguistic context for processing
    std::vector<float> context;
    context.reserve(512);
    
    // Add language comprehension context
    if (modules_.count("language_comprehension")) {
        auto comp_state = modules_["language_comprehension"]->get_output();
        size_t comp_size = std::min(comp_state.size(), size_t(128));
        context.insert(context.end(), comp_state.begin(), comp_state.begin() + comp_size);
    }
    
    // Add semantic memory context
    if (modules_.count("semantic_memory")) {
        auto sem_state = modules_["semantic_memory"]->get_output();
        size_t sem_size = std::min(sem_state.size(), size_t(128));
        context.insert(context.end(), sem_state.begin(), sem_state.begin() + sem_size);
    }
    
    // Add working memory context
    auto working_memory = memory_system_->get_working_memory();
    size_t wm_size = std::min(working_memory.size(), size_t(128));
    context.insert(context.end(), working_memory.begin(), working_memory.begin() + wm_size);
    
    // Add goal context
    size_t goal_size = std::min(current_goals_.size(), size_t(64));
    context.insert(context.end(), current_goals_.begin(), current_goals_.begin() + goal_size);
    
    // Add temporal context
    context.push_back(simulation_time_ / 1000.0f);
    context.push_back(global_reward_signal_);
    
    return context;
}

void AutonomousLearningAgent::update_language_environment() {
    // Update environmental context with current language processing state
    
    // Reset context periodically to prevent staleness
    static int update_counter = 0;
    if (++update_counter > 100) {
        std::fill(environmental_context_.begin(), environmental_context_.end(), 0.0f);
        update_counter = 0;
    }
    
    // Update with current language processing outputs
    if (modules_.count("language_comprehension")) {
        auto comp_output = modules_["language_comprehension"]->get_output();
        size_t comp_size = std::min(comp_output.size(), environmental_context_.size() / 3);
        for (size_t i = 0; i < comp_size; ++i) {
            environmental_context_[i] = comp_output[i];
        }
    }
    
    if (modules_.count("semantic_memory")) {
        auto sem_output = modules_["semantic_memory"]->get_output();
        size_t sem_offset = environmental_context_.size() / 3;
        size_t sem_size = std::min(sem_output.size(), environmental_context_.size() / 3);
        for (size_t i = 0; i < sem_size && (sem_offset + i) < environmental_context_.size(); ++i) {
            environmental_context_[sem_offset + i] = sem_output[i];
        }
    }
    
    if (modules_.count("syntactic_processor")) {
        auto synt_output = modules_["syntactic_processor"]->get_output();
        size_t synt_offset = 2 * environmental_context_.size() / 3;
        size_t synt_size = std::min(synt_output.size(), environmental_context_.size() / 6);
        for (size_t i = 0; i < synt_size && (synt_offset + i) < environmental_context_.size(); ++i) {
            environmental_context_[synt_offset + i] = synt_output[i];
        }
    }
}