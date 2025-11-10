// ============================================================================
// SIMPLIFIED BRAIN MODULE ARCHITECTURE FOR NLP - FIXED
// File: src/BrainModuleArchitecture.cpp
// ============================================================================

#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/ModularNeuralNetwork.h"
#include "NeuroGen/LearningStateManager.h"
#include "NeuroGen/cuda/NetworkCUDA.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <numeric>

// ============================================================================
// CONSTRUCTOR AND INITIALIZATION - FIXED
// ============================================================================

BrainModuleArchitecture::BrainModuleArchitecture() 
    : last_update_time_(std::chrono::steady_clock::now()),  // FIXED: Correct order
      creation_time_(std::chrono::steady_clock::now()) {   // FIXED: Correct order
    
    // Initialize simplified NLP architecture configuration
    architecture_config_.max_modules = 5; // Only 5 modules for NLP
    architecture_config_.enable_inter_module_learning = true;
    architecture_config_.enable_attention_mechanism = true;
    architecture_config_.enable_memory_consolidation = true;
    architecture_config_.enable_structural_plasticity = false; // Simplified for NLP
    architecture_config_.global_inhibition_strength = 0.1f;
    architecture_config_.attention_update_rate = 0.01f;
    architecture_config_.memory_consolidation_rate = 0.005f;
    
    // Initialize neuromodulator levels for language processing
    global_dopamine_level_ = 0.2f;      // Higher for language reward
    global_acetylcholine_level_ = 0.3f; // Higher for attention in language
    global_norepinephrine_level_ = 0.15f;
    global_serotonin_level_ = 0.1f;
    
    std::cout << "🧠 Simplified Brain Module Architecture created for NLP processing" << std::endl;
}

BrainModuleArchitecture::~BrainModuleArchitecture() {
    shutdown();
}

bool BrainModuleArchitecture::initialize(int input_width, int input_height) {
    // Standard initialization (kept for compatibility)
    static_cast<void>(input_width);  // Suppress unused parameter warning
    static_cast<void>(input_height); // Suppress unused parameter warning
    return initializeForNLP();
}

bool BrainModuleArchitecture::initializeForNLP() {
    std::lock_guard<std::mutex> lock(learning_state_mutex_);
    
    std::cout << "🔧 Initializing Brain Module Architecture for NLP processing..." << std::endl;
    
    try {
        // Create simplified modular network for NLP
        modular_network_ = std::make_unique<ModularNeuralNetwork>();
        
        // Initialize learning state manager with proper parameters - FIXED
        learning_state_manager_ = std::make_shared<LearningStateManager>(
            shared_from_this(), "nlp_learning_state");
        
        // Create the five core NLP modules
        createNLPModules();
        
        // Set up inter-module connections
        setupNLPConnections();
        
        // Initialize attention system for language processing
        initializeNLPAttentionSystem();
        
        // Initialize context vector for language understanding
        global_context_vector_.resize(512, 0.0f);

        // Setup CUDA network for biologically inspired processing
        NetworkCUDA::CUDAConfig cuda_cfg;
        cuda_network_ = std::make_shared<NetworkCUDA>(cuda_cfg);

        size_t total_neurons = 0;
        for (const auto& [name, cfg] : module_configs_) {
            total_neurons += cfg.num_neurons;
        }

        NetworkConfig net_cfg;
        net_cfg.num_neurons = static_cast<int>(total_neurons);
        auto [success, err] = cuda_network_->initialize(net_cfg);
        if (!success) {
            std::cerr << "❌ Failed to initialize CUDA network: " << err << std::endl;
        } else {
            cuda_network_->setBrainArchitecture(shared_from_this());
            cuda_network_->setLearningStateManager(learning_state_manager_);
        }

        std::cout << "✅ Brain Module Architecture initialized successfully for NLP" << std::endl;
        std::cout << "📊 Architecture: " << modules_.size() << " modules, "
                  << connections_.size() << " connections" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to initialize Brain Module Architecture: " << e.what() << std::endl;
        return false;
    }
}

void BrainModuleArchitecture::createNLPModules() {
    // Module 1: Neuromodulation Module (adaptive control)
    ModuleConfig neuromod_config;
    neuromod_config.module_name = "neuromodulation";
    neuromod_config.module_type = "adaptive_control";
    neuromod_config.num_neurons = 2048;
    neuromod_config.input_size = 512;
    neuromod_config.output_size = 512;
    neuromod_config.learning_rate = 0.005f;
    neuromod_config.attention_weight = 1.0f;
    neuromod_config.enable_plasticity = true;
    
    NetworkConfig neuromod_net_config;
    neuromod_net_config.num_neurons = neuromod_config.num_neurons;
    neuromod_net_config.input_size = neuromod_config.input_size;
    neuromod_net_config.output_size = neuromod_config.output_size;
    
    auto neuromod_module = std::make_shared<EnhancedNeuralModule>(
        neuromod_config.module_name, neuromod_net_config);
    neuromod_module->initialize();

    modules_["neuromodulation"] = neuromod_module;
    module_configs_["neuromodulation"] = neuromod_config;
    
    // Module 2: Language Perception Module
    ModuleConfig perception_config;
    perception_config.module_name = "language_perception";
    perception_config.module_type = "tokenization";
    perception_config.num_neurons = 1024;
    perception_config.input_size = 1024; // tokenized characters
    perception_config.output_size = 512;
    perception_config.learning_rate = 0.01f;
    perception_config.attention_weight = 0.7f;
    perception_config.enable_plasticity = true;
    
    NetworkConfig perception_net_config;
    perception_net_config.num_neurons = perception_config.num_neurons;
    perception_net_config.input_size = perception_config.input_size;
    perception_net_config.output_size = perception_config.output_size;
    
    auto perception_module = std::make_shared<EnhancedNeuralModule>(
        perception_config.module_name, perception_net_config);
    perception_module->initialize();

    modules_["language_perception"] = perception_module;
    module_configs_["language_perception"] = perception_config;
    
    // Module 3: Comprehension Module
    ModuleConfig comprehension_config;
    comprehension_config.module_name = "comprehension";
    comprehension_config.module_type = "semantic_integration";
    comprehension_config.num_neurons = 4096; // complex language understanding
    comprehension_config.input_size = 512;
    comprehension_config.output_size = 1024;
    comprehension_config.learning_rate = 0.008f;
    comprehension_config.attention_weight = 0.9f;
    comprehension_config.enable_plasticity = true;
    
    NetworkConfig comprehension_net_config;
    comprehension_net_config.num_neurons = comprehension_config.num_neurons;
    comprehension_net_config.input_size = comprehension_config.input_size;
    comprehension_net_config.output_size = comprehension_config.output_size;
    
    auto comprehension_module = std::make_shared<EnhancedNeuralModule>(
        comprehension_config.module_name, comprehension_net_config);
    comprehension_module->initialize();

    modules_["comprehension"] = comprehension_module;
    module_configs_["comprehension"] = comprehension_config;
    
    // Module 4: Reasoning Module
    ModuleConfig reasoning_config;
    reasoning_config.module_name = "reasoning";
    reasoning_config.module_type = "logical_reasoning";
    reasoning_config.num_neurons = 2048;
    reasoning_config.input_size = 1024;
    reasoning_config.output_size = 512;
    reasoning_config.learning_rate = 0.006f;
    reasoning_config.attention_weight = 0.8f;
    reasoning_config.enable_plasticity = true;
    
    NetworkConfig reasoning_net_config;
    reasoning_net_config.num_neurons = reasoning_config.num_neurons;
    reasoning_net_config.input_size = reasoning_config.input_size;
    reasoning_net_config.output_size = reasoning_config.output_size;
    
    auto reasoning_module = std::make_shared<EnhancedNeuralModule>(
        reasoning_config.module_name, reasoning_net_config);
    reasoning_module->initialize();
    
    modules_["reasoning"] = reasoning_module;
    module_configs_["reasoning"] = reasoning_config;
    
    // Module 5: Output Generation Module
    ModuleConfig output_config;
    output_config.module_name = "output_generation";
    output_config.module_type = "language_production";
    output_config.num_neurons = 1024;
    output_config.input_size = 512;
    output_config.output_size = 256;
    output_config.learning_rate = 0.01f;
    output_config.attention_weight = 0.6f;
    output_config.enable_plasticity = true;
    
    NetworkConfig output_net_config;
    output_net_config.num_neurons = output_config.num_neurons;
    output_net_config.input_size = output_config.input_size;
    output_net_config.output_size = output_config.output_size;
    
    auto output_module = std::make_shared<EnhancedNeuralModule>(
        output_config.module_name, output_net_config);
    output_module->initialize();
    
    modules_["output_generation"] = output_module;
    module_configs_["output_generation"] = output_config;
    
    std::cout << "✅ Created 5 NLP modules: Neuromodulation, Perception, Comprehension, Reasoning, Output" << std::endl;
}

void BrainModuleArchitecture::setupNLPConnections() {
    // Create forward processing pipeline
    connections_.clear();
    
    // Perception -> Comprehension (primary path)
    InterModuleConnection percept_to_comp;
    percept_to_comp.source_module = "language_perception";
    percept_to_comp.target_module = "comprehension";
    percept_to_comp.connection_strength = 0.8f;
    percept_to_comp.connection_type = "excitatory";
    percept_to_comp.is_active = true;
    connections_.push_back(percept_to_comp);
    
    // Comprehension -> Reasoning (reasoning path)
    InterModuleConnection comp_to_reason;
    comp_to_reason.source_module = "comprehension";
    comp_to_reason.target_module = "reasoning";
    comp_to_reason.connection_strength = 0.7f;
    comp_to_reason.connection_type = "excitatory";
    comp_to_reason.is_active = true;
    connections_.push_back(comp_to_reason);
    
    // Reasoning -> Output (output path)
    InterModuleConnection reason_to_output;
    reason_to_output.source_module = "reasoning";
    reason_to_output.target_module = "output_generation";
    reason_to_output.connection_strength = 0.9f;
    reason_to_output.connection_type = "excitatory";
    reason_to_output.is_active = true;
    connections_.push_back(reason_to_output);
    
    // Central Controller connections (neuromodulatory)
    std::vector<std::string> target_modules = {
        "language_perception", "comprehension", "reasoning", "output_generation"
    };
    
    for (const auto& target : target_modules) {
        InterModuleConnection control_conn;
        control_conn.source_module = "neuromodulation";
        control_conn.target_module = target;
        control_conn.connection_strength = 0.5f;
        control_conn.connection_type = "modulatory";
        control_conn.is_active = true;
        connections_.push_back(control_conn);
    }
    
    // Feedback connections
    InterModuleConnection reason_feedback;
    reason_feedback.source_module = "reasoning";
    reason_feedback.target_module = "comprehension";
    reason_feedback.connection_strength = 0.3f;
    reason_feedback.connection_type = "inhibitory";
    reason_feedback.is_active = true;
    connections_.push_back(reason_feedback);
    
    InterModuleConnection output_feedback;
    output_feedback.source_module = "output_generation";
    output_feedback.target_module = "neuromodulation";
    output_feedback.connection_strength = 0.5f;
    output_feedback.connection_type = "excitatory";
    output_feedback.is_active = true;
    connections_.push_back(output_feedback);
    
    std::cout << "✅ Established " << connections_.size() << " inter-module connections for NLP pipeline" << std::endl;
}

void BrainModuleArchitecture::initializeNLPAttentionSystem() {
    // Initialize attention weights for NLP modules
    attention_weights_["neuromodulation"] = 1.0f;
    attention_weights_["language_perception"] = 0.7f;
    attention_weights_["comprehension"] = 0.9f;
    attention_weights_["reasoning"] = 0.8f;
    attention_weights_["output_generation"] = 0.6f;
    
    // Initialize attention history
    for (const auto& [module_name, _] : modules_) {
        attention_history_[module_name] = attention_weights_[module_name];
    }
    
    std::cout << "✅ NLP attention system initialized" << std::endl;
}

// ============================================================================
// CORE PROCESSING INTERFACE
// ============================================================================

std::map<std::string, std::vector<float>> BrainModuleArchitecture::processNLPInput(
    const std::string& text_input) {
    
    std::map<std::string, std::vector<float>> module_outputs;
    
    // Tokenize input for neural processing
    std::vector<float> tokenized_input = tokenizeText(text_input);
    
    // STEP 1: Perception module processes tokenized text
    if (modules_.count("language_perception")) {
        auto percept_output = modules_["language_perception"]->process(tokenized_input);
        module_outputs["language_perception"] = percept_output;

        // STEP 2: Neuromodulation module provides control signals
        if (modules_.count("neuromodulation")) {
            auto control_output = modules_["neuromodulation"]->process(percept_output);
            module_outputs["neuromodulation"] = control_output;

            // Apply neuromodulation to comprehension
            auto modulated = applyNeuromodulation(percept_output, control_output);

            // STEP 3: Comprehension module
            if (modules_.count("comprehension")) {
                auto comp_output = modules_["comprehension"]->process(modulated);
                module_outputs["comprehension"] = comp_output;

                // STEP 4: Reasoning module
                if (modules_.count("reasoning")) {
                    auto reason_output = modules_["reasoning"]->process(comp_output);
                    module_outputs["reasoning"] = reason_output;

                    // STEP 5: Output generation
                    if (modules_.count("output_generation")) {
                        auto final_output = modules_["output_generation"]->process(reason_output);
                        module_outputs["output_generation"] = final_output;
                    }
                }
            }
        }
    }
    
    // Update attention weights based on processing results
    updateAttentionWeights(module_outputs);
    
    return module_outputs;
}

std::vector<float> BrainModuleArchitecture::tokenizeText(const std::string& text) {
    std::vector<float> tokens(1024, 0.0f);
    
    // Simple character-level tokenization
    for (size_t i = 0; i < text.length() && i < 512; ++i) {
        tokens[i] = static_cast<float>(text[i]) / 255.0f;
    }
    
    // Add positional encoding
    for (size_t i = 0; i < text.length() && i < 512; ++i) {
        tokens[i + 512] = static_cast<float>(i) / 512.0f;
    }
    
    return tokens;
}

std::vector<float> BrainModuleArchitecture::applyNeuromodulation(
    const std::vector<float>& input, 
    const std::vector<float>& control_signals) {
    
    std::vector<float> modulated(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        float control_weight = control_signals.size() > i ? 
            control_signals[i] : 0.5f;
        
        // Apply neuromodulatory scaling
        modulated[i] = input[i] * (0.5f + 0.5f * std::tanh(control_weight));
    }
    
    return modulated;
}

void BrainModuleArchitecture::updateAttentionWeights(
    const std::map<std::string, std::vector<float>>& module_outputs) {
    
    // Update attention based on module activation levels
    for (const auto& [module_name, output] : module_outputs) {
        if (attention_weights_.count(module_name)) {
            float avg_activation = std::accumulate(output.begin(), output.end(), 0.0f) / output.size();
            
            // Gradually adjust attention weight based on activation
            float target_weight = 0.1f + 0.9f * std::tanh(avg_activation);
            attention_weights_[module_name] = 
                0.95f * attention_weights_[module_name] + 0.05f * target_weight;
        }
    }
}

// ============================================================================
// MODULE MANAGEMENT
// ============================================================================

std::vector<std::string> BrainModuleArchitecture::getModuleNames() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : modules_) {
        names.push_back(name);
    }
    return names;
}

size_t BrainModuleArchitecture::getModuleCount() const {
    return modules_.size();
}

bool BrainModuleArchitecture::hasModule(const std::string& module_name) const {
    return modules_.count(module_name) > 0;
}

std::shared_ptr<EnhancedNeuralModule> BrainModuleArchitecture::getModule(
    const std::string& module_name) const {
    auto it = modules_.find(module_name);
    return (it != modules_.end()) ? it->second : nullptr;
}

BrainModuleArchitecture::ModuleConfig BrainModuleArchitecture::getModuleConfig(
    const std::string& module_name) const {
    auto it = module_configs_.find(module_name);
    return (it != module_configs_.end()) ? it->second : ModuleConfig{};
}

std::vector<float> BrainModuleArchitecture::getModuleOutput(const std::string& module_name) const {
    auto it = modules_.find(module_name);
    if (it != modules_.end() && it->second) {
        return it->second->process(std::vector<float>(it->second->getName().length(), 0.1f));
    }
    return std::vector<float>();
}

// ============================================================================
// CONNECTION MANAGEMENT - FIXED
// ============================================================================

bool BrainModuleArchitecture::createConnection(
    const std::string& source_module, 
    const std::string& target_module, 
    float strength) {
    
    if (!hasModule(source_module) || !hasModule(target_module)) {
        std::cerr << "❌ Cannot create connection: module not found" << std::endl;
        return false;
    }
    
    InterModuleConnection connection;
    connection.source_module = source_module;
    connection.target_module = target_module;
    connection.connection_strength = strength;
    connection.connection_type = "excitatory";
    connection.is_active = true;
    
    connections_.push_back(connection);
    
    std::cout << "✅ Created connection: " << source_module 
              << " -> " << target_module << " (strength: " << strength << ")" << std::endl;
    
    return true;
}

std::vector<BrainModuleArchitecture::InterModuleConnection> BrainModuleArchitecture::getConnections() const {
    return connections_;  // FIXED: Return correct type
}

bool BrainModuleArchitecture::hasConnection(
    const std::string& source_module, 
    const std::string& target_module) const {
    
    return std::any_of(connections_.begin(), connections_.end(),
        [&](const InterModuleConnection& conn) {
            return conn.source_module == source_module && 
                   conn.target_module == target_module;
        });
}

std::vector<BrainModuleArchitecture::InterModuleConnection> BrainModuleArchitecture::getModuleConnections(
    const std::string& module_name, bool incoming) const {
    
    std::vector<InterModuleConnection> result;
    
    for (const auto& conn : connections_) {
        if (incoming && conn.target_module == module_name) {
            result.push_back(conn);
        } else if (!incoming && conn.source_module == module_name) {
            result.push_back(conn);
        }
    }
    
    return result;
}

// ============================================================================
// ATTENTION AND CONTROL
// ============================================================================

float BrainModuleArchitecture::getAttentionWeight(const std::string& module_name) const {
    auto it = attention_weights_.find(module_name);
    return (it != attention_weights_.end()) ? it->second : 0.5f;
}

void BrainModuleArchitecture::setAttentionWeight(const std::string& module_name, float weight) {
    attention_weights_[module_name] = std::max(0.0f, std::min(1.0f, weight));
}

std::vector<float> BrainModuleArchitecture::getGlobalContext() const {
    return global_context_vector_;
}

void BrainModuleArchitecture::updateGlobalContext(const std::vector<float>& new_context) {
    if (new_context.size() == global_context_vector_.size()) {
        global_context_vector_ = new_context;
    }
}

std::map<std::string, float> BrainModuleArchitecture::getNeuromodulatorLevels() const {
    std::map<std::string, float> levels;
    levels["dopamine"] = global_dopamine_level_;
    levels["acetylcholine"] = global_acetylcholine_level_;
    levels["norepinephrine"] = global_norepinephrine_level_;
    levels["serotonin"] = global_serotonin_level_;
    return levels;
}

// ============================================================================
// UPDATE AND LEARNING
// ============================================================================

void BrainModuleArchitecture::update(float dt, float global_reward) {
    std::lock_guard<std::mutex> lock(learning_state_mutex_);
    
    // Update global learning steps
    global_learning_steps_++;
    global_reward_accumulator_ += global_reward;
    
    // Update neuromodulator levels based on reward
    updateNeuromodulatorLevels(global_reward, dt);
    
    // Update all modules
    for (auto& [name, module] : modules_) {
        if (module) {
            std::vector<float> empty_input;
            module->update(dt, empty_input, global_reward);
        }
    }
    
    // Update attention system
    updateGlobalAttention(dt);
    
    last_update_time_ = std::chrono::steady_clock::now();
}

void BrainModuleArchitecture::applyLearningUpdates(float reward_signal, float dt) {
    // Apply learning updates to all modules
    for (auto& [name, module] : modules_) {
        if (module) {
            std::vector<float> empty_input;
            module->update(dt, empty_input, reward_signal);
        }
    }
}

BrainModuleArchitecture::GlobalLearningState BrainModuleArchitecture::getGlobalLearningState() const {
    std::lock_guard<std::mutex> lock(learning_state_mutex_);
    
    GlobalLearningState state;
    state.total_learning_steps = global_learning_steps_;
    state.cumulative_reward = global_reward_accumulator_;
    state.average_module_performance = 0.5f; // Placeholder
    state.last_update = last_update_time_;
    
    return state;
}

void BrainModuleArchitecture::updateNeuromodulatorLevels(float reward, float dt) {
    // Update dopamine based on reward prediction error
    float dopamine_target = 0.1f + 0.4f * std::tanh(reward);
    global_dopamine_level_ = 0.95f * global_dopamine_level_ + 0.05f * dopamine_target;
    
    // Update acetylcholine for attention
    float attention_demand = std::accumulate(attention_weights_.begin(), attention_weights_.end(), 0.0f,
        [](float sum, const auto& pair) { return sum + pair.second; }) / attention_weights_.size();
    global_acetylcholine_level_ = 0.98f * global_acetylcholine_level_ + 0.02f * attention_demand;
    
    // Maintain norepinephrine and serotonin levels
    global_norepinephrine_level_ = std::max(0.1f, global_norepinephrine_level_ * 0.999f);
    global_serotonin_level_ = std::max(0.05f, global_serotonin_level_ * 0.999f);
    
    static_cast<void>(dt); // Suppress unused parameter warning
}

void BrainModuleArchitecture::updateGlobalAttention(float dt) {
    // Update attention weights with biological dynamics
    for (auto& [module_name, weight] : attention_weights_) {
        // Apply attention decay
        weight *= (1.0f - architecture_config_.attention_update_rate * dt);
        
        // Maintain minimum attention
        weight = std::max(0.1f, weight);
        
        // Update history
        attention_history_[module_name] = 0.9f * attention_history_[module_name] + 0.1f * weight;
    }
}

// ============================================================================
// STATE PERSISTENCE
// ============================================================================

bool BrainModuleArchitecture::saveLearningState(const std::string& save_directory, 
                                               const std::string& checkpoint_name) {
    try {
        // Create directory if it doesn't exist
        std::filesystem::create_directories(save_directory);
        
        // Save basic state information
        std::ofstream state_file(save_directory + "/" + checkpoint_name + "_brain_state.txt");
        if (state_file.is_open()) {
            state_file << "learning_steps: " << global_learning_steps_ << std::endl;
            state_file << "reward_accumulator: " << global_reward_accumulator_ << std::endl;
            state_file << "module_count: " << modules_.size() << std::endl;
            state_file << "connection_count: " << connections_.size() << std::endl;
            state_file.close();
        }
        
        std::cout << "💾 Brain architecture state saved to: " << save_directory << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to save brain architecture state: " << e.what() << std::endl;
        return false;
    }
}

bool BrainModuleArchitecture::loadLearningState(const std::string& save_directory, 
                                               const std::string& checkpoint_name) {
    try {
        std::ifstream state_file(save_directory + "/" + checkpoint_name + "_brain_state.txt");
        if (state_file.is_open()) {
            std::string line;
            while (std::getline(state_file, line)) {
                // Parse basic state information
                if (line.find("learning_steps:") != std::string::npos) {
                    global_learning_steps_ = std::stoull(line.substr(line.find(":") + 1));
                } else if (line.find("reward_accumulator:") != std::string::npos) {
                    global_reward_accumulator_ = std::stof(line.substr(line.find(":") + 1));
                }
            }
            state_file.close();
        }
        
        std::cout << "📂 Brain architecture state loaded from: " << save_directory << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to load brain architecture state: " << e.what() << std::endl;
        return false;
    }
}

bool BrainModuleArchitecture::saveModuleLearningState(const std::string& module_name, 
                                                     const std::string& save_directory) {
    // Simple module state saving
    try {
        std::ofstream module_file(save_directory + "/" + module_name + "_module.state");
        if (module_file.is_open()) {
            module_file << "module_name: " << module_name << std::endl;
            module_file << "timestamp: " << std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count() << std::endl;
            module_file.close();
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to save module state for " << module_name << ": " << e.what() << std::endl;
        return false;
    }
}

bool BrainModuleArchitecture::loadModuleLearningState(const std::string& module_name, 
                                                     const std::string& save_directory) {
    // Simple module state loading
    try {
        std::ifstream module_file(save_directory + "/" + module_name + "_module.state");
        if (module_file.is_open()) {
            std::string line;
            while (std::getline(module_file, line)) {
                // Parse module state information
                // Add actual state loading logic here
            }
            module_file.close();
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to load module state for " << module_name << ": " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

BrainModuleArchitecture::ArchitectureConfig BrainModuleArchitecture::getArchitectureConfig() const {
    return architecture_config_;
}

bool BrainModuleArchitecture::updateArchitectureConfig(const ArchitectureConfig& config) {
    architecture_config_ = config;
    return true;
}

// ============================================================================
// SHUTDOWN
// ============================================================================

void BrainModuleArchitecture::shutdown() {
    std::lock_guard<std::mutex> lock(learning_state_mutex_);
    
    modules_.clear();
    module_configs_.clear();
    connections_.clear();
    attention_weights_.clear();
    attention_history_.clear();
    
    if (learning_state_manager_) {
        learning_state_manager_.reset();
    }
    
    std::cout << "🔌 Brain Module Architecture shutdown complete" << std::endl;
}