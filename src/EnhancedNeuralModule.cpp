// ============================================================================
// ENHANCED NEURAL MODULE IMPLEMENTATION
// File: src/EnhancedNeuralModule.cpp
// ============================================================================

#include <NeuroGen/EnhancedNeuralModule.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>

// ============================================================================
// CORE LIFECYCLE METHODS
// ============================================================================

bool EnhancedNeuralModule::initialize() {
    std::cout << "🧠 Initializing Enhanced Neural Module: " << module_name_ << std::endl;
    
    // Call parent initialization first
    if (!NeuralModule::initialize()) {
        std::cerr << "❌ Failed to initialize base neural module" << std::endl;
        return false;
    }
    
    // Initialize enhanced features
    attention_weight_ = 1.0f;
    is_active_ = true;
    developmental_stage_ = 0;
    feedback_strength_ = 0.5f;
    last_feedback_update_ = std::chrono::steady_clock::now();
    
    // Initialize neuromodulator levels
    neuromodulator_levels_["dopamine"] = 0.5f;
    neuromodulator_levels_["serotonin"] = 0.5f;
    neuromodulator_levels_["norepinephrine"] = 0.5f;
    neuromodulator_levels_["acetylcholine"] = 0.5f;
    
    // Initialize feedback state
    feedback_state_.resize(config_.num_neurons, 0.0f);
    
    std::cout << "✅ Enhanced Neural Module initialized successfully" << std::endl;
    return true;
}

void EnhancedNeuralModule::update(float dt, const std::vector<float>& inputs, float reward) {
    if (!is_active_) {
        return;
    }
    
    // Call parent update first
    NeuralModule::update(dt, inputs, reward);
    
    // Update enhanced biological processes
    updateBiologicalProcesses(dt);
    
    // Process feedback loops
    processFeedbackLoops(dt);
    
    // Update developmental state
    updateDevelopmentalState(dt);
    
    // Process inter-module communication
    processInterModuleCommunication();
}

std::map<std::string, float> EnhancedNeuralModule::getPerformanceMetrics() const {
    // Get base metrics from parent class
    auto metrics = NeuralModule::getPerformanceMetrics();
    
    // Add enhanced metrics
    metrics["attention_weight"] = attention_weight_;
    metrics["developmental_stage"] = static_cast<float>(developmental_stage_);
    metrics["feedback_strength"] = feedback_strength_;
    metrics["is_active"] = is_active_ ? 1.0f : 0.0f;
    
    // Add neuromodulator levels
    for (const auto& [modulator, level] : neuromodulator_levels_) {
        metrics["neuromodulator_" + modulator] = level;
    }
    
    // Compute enhanced connectivity metrics
    float connection_diversity = static_cast<float>(connections_.size());
    metrics["inter_module_connections"] = connection_diversity;
    
    return metrics;
}

std::vector<float> EnhancedNeuralModule::getOutputs() const {
    return applyAttentionWeighting(neuron_outputs_);
}

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

EnhancedNeuralModule::ModuleState EnhancedNeuralModule::saveState() const {
    ModuleState state;
    state.module_name = module_name_;
    state.module_attention_weight = attention_weight_;
    state.developmental_stage = developmental_stage_;
    state.is_active = is_active_;
    state.timestamp = std::chrono::steady_clock::now();
    
    // Save neuron states
    state.neuron_states = neuron_outputs_;
    
    // Save synapse weights
    state.synapse_weights = synaptic_weights_;
    
    // Save neuromodulator levels
    state.neuromodulator_levels.clear();
    for (const auto& [modulator, level] : neuromodulator_levels_) {
        state.neuromodulator_levels.push_back(level);
    }
    
    // Save performance metrics
    state.performance_metrics = getPerformanceMetrics();
    
    return state;
}

void EnhancedNeuralModule::loadState(const ModuleState& state) {
    module_name_ = state.module_name;
    attention_weight_ = state.module_attention_weight;
    developmental_stage_ = state.developmental_stage;
    is_active_ = state.is_active;
    
    // Load neuron states
    if (!state.neuron_states.empty()) {
        neuron_outputs_ = state.neuron_states;
    }
    
    // Load synapse weights
    if (!state.synapse_weights.empty()) {
        synaptic_weights_ = state.synapse_weights;
    }
    
    // Load neuromodulator levels
    if (state.neuromodulator_levels.size() >= 4) {
        auto it = neuromodulator_levels_.begin();
        for (size_t i = 0; i < state.neuromodulator_levels.size() && it != neuromodulator_levels_.end(); ++i, ++it) {
            it->second = state.neuromodulator_levels[i];
        }
    }
    
    std::cout << "✅ Module state loaded for: " << module_name_ << std::endl;
}

bool EnhancedNeuralModule::save_state(const std::string& filename) const {
    try {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "❌ Cannot open file for writing: " << filename << std::endl;
            return false;
        }
        
        // Save enhanced state
        ModuleState state = saveState();
        
        // Write state data (simplified binary format)
        file.write(reinterpret_cast<const char*>(&state.module_attention_weight), sizeof(float));
        file.write(reinterpret_cast<const char*>(&state.developmental_stage), sizeof(int));
        file.write(reinterpret_cast<const char*>(&state.is_active), sizeof(bool));
        
        // Write vector sizes and data
        size_t neuron_count = state.neuron_states.size();
        file.write(reinterpret_cast<const char*>(&neuron_count), sizeof(size_t));
        if (neuron_count > 0) {
            file.write(reinterpret_cast<const char*>(state.neuron_states.data()), 
                      neuron_count * sizeof(float));
        }
        
        size_t synapse_count = state.synapse_weights.size();
        file.write(reinterpret_cast<const char*>(&synapse_count), sizeof(size_t));
        if (synapse_count > 0) {
            file.write(reinterpret_cast<const char*>(state.synapse_weights.data()),
                      synapse_count * sizeof(float));
        }
        
        size_t neuromod_count = state.neuromodulator_levels.size();
        file.write(reinterpret_cast<const char*>(&neuromod_count), sizeof(size_t));
        if (neuromod_count > 0) {
            file.write(reinterpret_cast<const char*>(state.neuromodulator_levels.data()),
                      neuromod_count * sizeof(float));
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error saving module state: " << e.what() << std::endl;
        return false;
    }
}

bool EnhancedNeuralModule::load_state(const std::string& filename) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "❌ Cannot open file for reading: " << filename << std::endl;
            return false;
        }
        
        ModuleState state;
        
        // Read state data
        file.read(reinterpret_cast<char*>(&state.module_attention_weight), sizeof(float));
        file.read(reinterpret_cast<char*>(&state.developmental_stage), sizeof(int));
        file.read(reinterpret_cast<char*>(&state.is_active), sizeof(bool));
        
        // Read vector data
        size_t neuron_count;
        file.read(reinterpret_cast<char*>(&neuron_count), sizeof(size_t));
        if (neuron_count > 0) {
            state.neuron_states.resize(neuron_count);
            file.read(reinterpret_cast<char*>(state.neuron_states.data()),
                     neuron_count * sizeof(float));
        }
        
        size_t synapse_count;
        file.read(reinterpret_cast<char*>(&synapse_count), sizeof(size_t));
        if (synapse_count > 0) {
            state.synapse_weights.resize(synapse_count);
            file.read(reinterpret_cast<char*>(state.synapse_weights.data()),
                     synapse_count * sizeof(float));
        }
        
        size_t neuromod_count;
        file.read(reinterpret_cast<char*>(&neuromod_count), sizeof(size_t));
        if (neuromod_count > 0) {
            state.neuromodulator_levels.resize(neuromod_count);
            file.read(reinterpret_cast<char*>(state.neuromodulator_levels.data()),
                     neuromod_count * sizeof(float));
        }
        
        // Load the state
        loadState(state);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error loading module state: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// ATTENTION MECHANISM
// ============================================================================

void EnhancedNeuralModule::updateAttention(const std::vector<float>& context_vector) {
    if (context_vector.empty()) {
        return;
    }
    
    // Simple attention computation based on context similarity
    float context_match = 0.0f;
    float norm_context = 0.0f;
    float norm_internal = 0.0f;
    
    // Use current neuron outputs as internal state representation
    size_t min_size = std::min(context_vector.size(), neuron_outputs_.size());
    
    for (size_t i = 0; i < min_size; ++i) {
        context_match += context_vector[i] * neuron_outputs_[i];
        norm_context += context_vector[i] * context_vector[i];
        norm_internal += neuron_outputs_[i] * neuron_outputs_[i];
    }
    
    if (norm_context > 0 && norm_internal > 0) {
        float similarity = context_match / (std::sqrt(norm_context) * std::sqrt(norm_internal));
        
        // Update attention weight based on similarity
        float target_attention = std::max(0.1f, std::min(1.0f, similarity + 0.5f));
        attention_weight_ = 0.9f * attention_weight_ + 0.1f * target_attention;
    }
}

// ============================================================================
// BIOLOGICAL FEATURES
// ============================================================================

void EnhancedNeuralModule::applyNeuromodulation(const std::string& modulator_type, float level) {
    // Call parent implementation first
    NeuralModule::applyNeuromodulation(modulator_type, level);
    
    // Update neuromodulator levels
    neuromodulator_levels_[modulator_type] = std::max(0.0f, std::min(1.0f, level));
    
    // Apply modulator-specific effects
    if (modulator_type == "dopamine") {
        // Dopamine enhances learning and exploration
        for (auto& weight : synaptic_weights_) {
            weight *= (1.0f + 0.1f * level);
        }
    } else if (modulator_type == "serotonin") {
        // Serotonin affects mood and attention
        attention_weight_ *= (1.0f + 0.05f * level);
        attention_weight_ = std::min(1.0f, attention_weight_);
    } else if (modulator_type == "acetylcholine") {
        // Acetylcholine enhances attention and learning
        attention_weight_ *= (1.0f + 0.1f * level);
        attention_weight_ = std::min(1.0f, attention_weight_);
    }
}

void EnhancedNeuralModule::processFeedbackLoops(float dt) {
    auto current_time = std::chrono::steady_clock::now();
    auto time_since_last = std::chrono::duration<float>(current_time - last_feedback_update_).count();
    
    if (time_since_last < 0.01f) { // Update at most every 10ms
        return;
    }
    
    // Update feedback state based on current outputs
    for (size_t i = 0; i < feedback_state_.size() && i < neuron_outputs_.size(); ++i) {
        float feedback_input = feedback_strength_ * neuron_outputs_[i];
        feedback_state_[i] = 0.9f * feedback_state_[i] + 0.1f * feedback_input;
    }
    
    // Apply feedback to modify current processing
    for (size_t i = 0; i < neuron_outputs_.size() && i < feedback_state_.size(); ++i) {
        neuron_outputs_[i] = std::tanh(neuron_outputs_[i] + 0.1f * feedback_state_[i]);
    }
    
    last_feedback_update_ = current_time;
}

void EnhancedNeuralModule::addInterModuleConnection(const InterModuleConnection& connection) {
    connections_.push_back(connection);
    
    // Initialize input buffer for this connection if needed
    if (input_buffers_.find(connection.source_port) == input_buffers_.end()) {
        input_buffers_[connection.source_port] = std::queue<std::vector<float>>();
    }
}

void EnhancedNeuralModule::sendInterModuleSignal(const std::vector<float>& signal_data, 
                                                 const std::string& target_port) {
    // Store in output buffer (simplified implementation)
    output_buffers_[target_port] = signal_data;
}

void EnhancedNeuralModule::receiveInterModuleSignal(const std::vector<float>& signal_data,
                                                   const std::string& source_port) {
    // Add to input buffer
    if (input_buffers_.find(source_port) != input_buffers_.end()) {
        input_buffers_[source_port].push(signal_data);
        
        // Limit buffer size
        while (input_buffers_[source_port].size() > 10) {
            input_buffers_[source_port].pop();
        }
    }
}

// ============================================================================
// PROTECTED HELPER METHODS
// ============================================================================

void EnhancedNeuralModule::updateBiologicalProcesses(float dt) {
    // Update neuromodulator decay
    for (auto& [modulator, level] : neuromodulator_levels_) {
        level *= (1.0f - 0.01f * dt); // Slow decay
        level = std::max(0.1f, level); // Maintain baseline level
    }
    
    // Update feedback strength based on activity
    float avg_activity = 0.0f;
    for (float output : neuron_outputs_) {
        avg_activity += std::abs(output);
    }
    avg_activity /= neuron_outputs_.size();
    
    // Adapt feedback strength to maintain optimal activity
    float target_activity = 0.3f;
    if (avg_activity > target_activity) {
        feedback_strength_ *= 0.99f; // Reduce feedback if too active
    } else {
        feedback_strength_ *= 1.01f; // Increase feedback if too quiet
    }
    feedback_strength_ = std::max(0.1f, std::min(1.0f, feedback_strength_));
}

std::vector<float> EnhancedNeuralModule::applyAttentionWeighting(const std::vector<float>& raw_output) const {
    std::vector<float> weighted_output = raw_output;
    
    // Apply attention weight to all outputs
    for (float& output : weighted_output) {
        output *= attention_weight_;
    }
    
    return weighted_output;
}

void EnhancedNeuralModule::updateDevelopmentalState(float dt) {
    // Simple developmental progression based on activity and time
    static float accumulated_time = 0.0f;
    accumulated_time += dt;
    
    float avg_activity = 0.0f;
    for (float output : neuron_outputs_) {
        avg_activity += std::abs(output);
    }
    avg_activity /= neuron_outputs_.size();
    
    // Progress development based on activity and time
    float development_rate = 0.001f * avg_activity;
    if (accumulated_time > 10.0f && developmental_stage_ < 5) { // Every 10 seconds of simulation
        developmental_stage_++;
        accumulated_time = 0.0f;
        std::cout << "🌱 Module " << module_name_ << " progressed to developmental stage " 
                  << developmental_stage_ << std::endl;
    }
}

void EnhancedNeuralModule::processInterModuleCommunication() {
    // Process incoming signals from input buffers
    for (auto& [source_port, buffer] : input_buffers_) {
        if (!buffer.empty()) {
            auto signal = buffer.front();
            buffer.pop();
            
            // Simple integration: add signal to current outputs
            for (size_t i = 0; i < signal.size() && i < neuron_outputs_.size(); ++i) {
                neuron_outputs_[i] += 0.1f * signal[i];
                neuron_outputs_[i] = std::tanh(neuron_outputs_[i]); // Keep bounded
            }
        }
    }
}