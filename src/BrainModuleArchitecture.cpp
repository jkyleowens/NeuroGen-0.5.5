// ============================================================================
// BRAIN MODULE ARCHITECTURE IMPLEMENTATION
// File: src/BrainModuleArchitecture.cpp
// ============================================================================

#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/SpecializedModule.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <filesystem>

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

BrainModuleArchitecture::BrainModuleArchitecture() 
    : visual_input_width_(1920), visual_input_height_(1080),
      visual_feature_size_(0), context_size_(512), goal_size_(64), action_size_(32),
      global_reward_signal_(0.0f), is_initialized_(false), total_activity_(0.0f),
      update_count_(0) {
    
    modular_network_ = std::make_unique<ModularNeuralNetwork>();
    global_context_.resize(context_size_, 0.0f);
    
    std::cout << "BrainModuleArchitecture created" << std::endl;
}

BrainModuleArchitecture::~BrainModuleArchitecture() {
    shutdown();
}

bool BrainModuleArchitecture::initialize(int screen_width, int screen_height) {
    if (is_initialized_) {
        std::cout << "Architecture already initialized" << std::endl;
        return true;
    }
    
    visual_input_width_ = screen_width;
    visual_input_height_ = screen_height;
    
    // Calculate dynamic sizes based on input dimensions
    calculateDynamicSizes();
    
    // Initialize the modular network
    if (!modular_network_->initialize()) {
        std::cerr << "Failed to initialize modular network" << std::endl;
        return false;
    }
    
    // Create default module configurations
    createDefaultConfigurations();
    
    // Create all brain modules
    if (!createVisualCortex()) return false;
    if (!createComprehensionModule()) return false;
    if (!createExecutiveFunction()) return false;
    if (!createMemoryModule()) return false;
    if (!createCentralController()) return false;
    if (!createOutputModule()) return false;
    if (!createMotorCortex()) return false;
    if (!createRewardSystem()) return false;
    if (!createAttentionSystem()) return false;
    
    // Initialize inter-module connections
    initializeConnections();
    
    is_initialized_ = true;
    std::cout << "BrainModuleArchitecture initialized successfully" << std::endl;
    std::cout << "Visual input: " << visual_input_width_ << "x" << visual_input_height_ << std::endl;
    std::cout << "Visual features: " << visual_feature_size_ << std::endl;
    std::cout << "Context size: " << context_size_ << std::endl;
    std::cout << "Total modules: " << modular_network_->get_module_count() << std::endl;
    
    return true;
}

void BrainModuleArchitecture::shutdown() {
    if (!is_initialized_) return;
    
    if (modular_network_) {
        modular_network_->shutdown();
    }
    
    module_configs_.clear();
    connections_.clear();
    attention_weights_.clear();
    
    is_initialized_ = false;
    std::cout << "BrainModuleArchitecture shutdown complete" << std::endl;
}

// ============================================================================
// DYNAMIC SIZE CALCULATION
// ============================================================================

void BrainModuleArchitecture::calculateDynamicSizes() {
    // Calculate visual feature size based on input dimensions
    // Using a hierarchical reduction approach similar to CNNs
    int reduced_width = visual_input_width_ / 8;  // 8x reduction
    int reduced_height = visual_input_height_ / 8;
    visual_feature_size_ = reduced_width * reduced_height / 4; // Further compression
    
    // Ensure minimum and maximum sizes
    visual_feature_size_ = std::max(static_cast<size_t>(256), visual_feature_size_);
    visual_feature_size_ = std::min(static_cast<size_t>(2048), visual_feature_size_);
    
    // Adjust other sizes based on visual features
    context_size_ = visual_feature_size_ + 256; // Visual + additional context
    goal_size_ = 64; // Fixed goal representation size
    action_size_ = 32; // Fixed action space size
    
    std::cout << "Dynamic sizes calculated:" << std::endl;
    std::cout << "  Visual features: " << visual_feature_size_ << std::endl;
    std::cout << "  Context: " << context_size_ << std::endl;
    std::cout << "  Goals: " << goal_size_ << std::endl;
    std::cout << "  Actions: " << action_size_ << std::endl;
}

// ============================================================================
// MODULE CREATION METHODS
// ============================================================================

bool BrainModuleArchitecture::createVisualCortex() {
    ModuleConfig config;
    config.type = ModuleType::VISUAL_CORTEX;
    config.name = "visual_cortex";
    config.input_size = visual_input_width_ * visual_input_height_ / 16; // Downsampled input
    config.output_size = visual_feature_size_;
    config.internal_neurons = 2048;
    config.cortical_columns = 64;
    config.learning_rate = 0.001f;
    config.plasticity_strength = 0.8f;
    config.enable_stdp = true;
    config.enable_homeostasis = true;
    config.output_connections = {"comprehension_module", "executive_function", "attention_system"};
    
    return createModule(config);
}

bool BrainModuleArchitecture::createComprehensionModule() {
    ModuleConfig config;
    config.type = ModuleType::COMPREHENSION_MODULE;
    config.name = "comprehension_module";
    config.input_size = visual_feature_size_ + 256; // Visual + text features
    config.output_size = 512;
    config.internal_neurons = 1024;
    config.cortical_columns = 32;
    config.learning_rate = 0.001f;
    config.plasticity_strength = 0.7f;
    config.enable_stdp = true;
    config.enable_homeostasis = true;
    config.input_connections = {"visual_cortex"};
    config.output_connections = {"executive_function", "memory_module"};
    
    return createModule(config);
}

bool BrainModuleArchitecture::createExecutiveFunction() {
    ModuleConfig config;
    config.type = ModuleType::EXECUTIVE_FUNCTION;
    config.name = "executive_function";
    config.input_size = visual_feature_size_ + 512 + goal_size_; // Visual + comprehension + goals
    config.output_size = 256;
    config.internal_neurons = 1536;
    config.cortical_columns = 48;
    config.learning_rate = 0.002f;
    config.plasticity_strength = 0.9f;
    config.enable_stdp = true;
    config.enable_homeostasis = true;
    config.input_connections = {"visual_cortex", "comprehension_module", "memory_module"};
    config.output_connections = {"motor_cortex", "memory_module", "attention_system"};
    
    return createModule(config);
}

bool BrainModuleArchitecture::createMemoryModule() {
    ModuleConfig config;
    config.type = ModuleType::MEMORY_MODULE;
    config.name = "memory_module";
    config.input_size = 512 + 256; // Comprehension + executive
    config.output_size = 384;
    config.internal_neurons = 2048;
    config.cortical_columns = 64;
    config.learning_rate = 0.0005f;
    config.plasticity_strength = 0.6f;
    config.enable_stdp = true;
    config.enable_homeostasis = true;
    config.input_connections = {"comprehension_module", "executive_function"};
    config.output_connections = {"executive_function", "central_controller"};
    
    return createModule(config);
}

bool BrainModuleArchitecture::createCentralController() {
    ModuleConfig config;
    config.type = ModuleType::CENTRAL_CONTROLLER;
    config.name = "central_controller";
    config.input_size = 384 + 128; // Memory + reward
    config.output_size = 128;
    config.internal_neurons = 512;
    config.cortical_columns = 16;
    config.learning_rate = 0.003f;
    config.plasticity_strength = 1.0f;
    config.enable_stdp = true;
    config.enable_homeostasis = true;
    config.input_connections = {"memory_module", "reward_system"};
    config.output_connections = {"attention_system", "reward_system"};
    
    return createModule(config);
}

bool BrainModuleArchitecture::createOutputModule() {
    ModuleConfig config;
    config.type = ModuleType::OUTPUT_MODULE;
    config.name = "output_module";
    config.input_size = action_size_;
    config.output_size = action_size_;
    config.internal_neurons = 256;
    config.cortical_columns = 8;
    config.learning_rate = 0.001f;
    config.plasticity_strength = 0.5f;
    config.enable_stdp = false; // Output module uses different learning
    config.enable_homeostasis = false;
    config.input_connections = {"motor_cortex"};
    
    return createModule(config);
}

bool BrainModuleArchitecture::createMotorCortex() {
    ModuleConfig config;
    config.type = ModuleType::MOTOR_CORTEX;
    config.name = "motor_cortex";
    config.input_size = 256; // From executive function
    config.output_size = action_size_;
    config.internal_neurons = 512;
    config.cortical_columns = 16;
    config.learning_rate = 0.002f;
    config.plasticity_strength = 0.8f;
    config.enable_stdp = true;
    config.enable_homeostasis = true;
    config.input_connections = {"executive_function"};
    config.output_connections = {"output_module"};
    
    return createModule(config);
}

bool BrainModuleArchitecture::createRewardSystem() {
    ModuleConfig config;
    config.type = ModuleType::REWARD_SYSTEM;
    config.name = "reward_system";
    config.input_size = 256 + action_size_; // Executive + action feedback
    config.output_size = 128;
    config.internal_neurons = 256;
    config.cortical_columns = 8;
    config.learning_rate = 0.005f;
    config.plasticity_strength = 1.2f;
    config.enable_stdp = true;
    config.enable_homeostasis = false; // Reward system has different dynamics
    config.input_connections = {"executive_function", "output_module"};
    config.output_connections = {"central_controller"};
    
    return createModule(config);
}

bool BrainModuleArchitecture::createAttentionSystem() {
    ModuleConfig config;
    config.type = ModuleType::ATTENTION_SYSTEM;
    config.name = "attention_system";
    config.input_size = visual_feature_size_ + 256 + 128; // Visual + executive + controller
    config.output_size = 9; // Attention weights for 9 modules
    config.internal_neurons = 256;
    config.cortical_columns = 8;
    config.learning_rate = 0.003f;
    config.plasticity_strength = 0.7f;
    config.enable_stdp = true;
    config.enable_homeostasis = true;
    config.input_connections = {"visual_cortex", "executive_function", "central_controller"};
    
    return createModule(config);
}

// ============================================================================
// MODULE MANAGEMENT
// ============================================================================

bool BrainModuleArchitecture::createModule(const ModuleConfig& config) {
    if (!validateModuleConfig(config)) {
        std::cerr << "Invalid module configuration for: " << config.name << std::endl;
        return false;
    }
    
    // Create network configuration for this module
    NetworkConfig net_config;
    
    // Set basic network parameters
    net_config.input_size = static_cast<int>(config.input_size);
    net_config.output_size = static_cast<int>(config.output_size);
    net_config.hidden_size = static_cast<int>(config.internal_neurons);
    net_config.num_neurons = config.internal_neurons;
    
    // Set cortical column organization
    net_config.numColumns = static_cast<int>(config.cortical_columns);
    net_config.neuronsPerColumn = static_cast<int>(config.internal_neurons / config.cortical_columns);
    
    // Set learning parameters
    net_config.enable_stdp = config.enable_stdp;
    net_config.stdp_learning_rate = config.learning_rate;
    net_config.reward_learning_rate = config.learning_rate;
    net_config.A_plus = config.plasticity_strength * 0.01f;
    net_config.A_minus = config.plasticity_strength * 0.012f;
    
    // Set homeostatic parameters
    if (config.enable_homeostasis) {
        net_config.homeostatic_strength = 0.001f;
    } else {
        net_config.homeostatic_strength = 0.0f;
    }
    
    // Set connectivity parameters based on module type
    switch (config.type) {
        case ModuleType::VISUAL_CORTEX:
            net_config.input_hidden_prob = 0.9f;
            net_config.hidden_hidden_prob = 0.2f;
            net_config.hidden_output_prob = 0.8f;
            net_config.localFanOut = 50;
            net_config.localFanIn = 50;
            break;
        case ModuleType::COMPREHENSION_MODULE:
            net_config.input_hidden_prob = 0.8f;
            net_config.hidden_hidden_prob = 0.3f;
            net_config.hidden_output_prob = 0.9f;
            net_config.localFanOut = 40;
            net_config.localFanIn = 40;
            break;
        case ModuleType::EXECUTIVE_FUNCTION:
            net_config.input_hidden_prob = 0.7f;
            net_config.hidden_hidden_prob = 0.4f;
            net_config.hidden_output_prob = 0.9f;
            net_config.localFanOut = 60;
            net_config.localFanIn = 60;
            break;
        case ModuleType::MEMORY_MODULE:
            net_config.input_hidden_prob = 0.6f;
            net_config.hidden_hidden_prob = 0.5f;
            net_config.hidden_output_prob = 0.7f;
            net_config.localFanOut = 80;
            net_config.localFanIn = 80;
            break;
        case ModuleType::CENTRAL_CONTROLLER:
            net_config.input_hidden_prob = 0.9f;
            net_config.hidden_hidden_prob = 0.6f;
            net_config.hidden_output_prob = 1.0f;
            net_config.localFanOut = 30;
            net_config.localFanIn = 30;
            break;
        case ModuleType::OUTPUT_MODULE:
            net_config.input_hidden_prob = 1.0f;
            net_config.hidden_hidden_prob = 0.1f;
            net_config.hidden_output_prob = 1.0f;
            net_config.localFanOut = 20;
            net_config.localFanIn = 20;
            break;
        case ModuleType::MOTOR_CORTEX:
            net_config.input_hidden_prob = 0.8f;
            net_config.hidden_hidden_prob = 0.3f;
            net_config.hidden_output_prob = 0.9f;
            net_config.localFanOut = 35;
            net_config.localFanIn = 35;
            break;
        case ModuleType::REWARD_SYSTEM:
            net_config.input_hidden_prob = 0.9f;
            net_config.hidden_hidden_prob = 0.2f;
            net_config.hidden_output_prob = 0.8f;
            net_config.localFanOut = 25;
            net_config.localFanIn = 25;
            break;
        case ModuleType::ATTENTION_SYSTEM:
            net_config.input_hidden_prob = 0.7f;
            net_config.hidden_hidden_prob = 0.4f;
            net_config.hidden_output_prob = 1.0f;
            net_config.localFanOut = 30;
            net_config.localFanIn = 30;
            break;
    }
    
    // Finalize configuration
    net_config.finalizeConfig();
    
    // Validate configuration
    if (!net_config.validate()) {
        std::cerr << "Invalid network configuration for module: " << config.name << std::endl;
        return false;
    }
    
    // Create the neural module
    auto module = std::make_unique<EnhancedNeuralModule>(config.name, net_config);
    
    if (!module->initialize()) {
        std::cerr << "Failed to initialize module: " << config.name << std::endl;
        return false;
    }
    
    // Register input and output ports
    std::vector<size_t> input_neurons, output_neurons;
    
    // Create input port
    for (size_t i = 0; i < config.input_size && i < config.internal_neurons; ++i) {
        input_neurons.push_back(i);
    }
    module->register_neuron_port("input", input_neurons);
    
    // Create output port
    size_t output_start = config.internal_neurons - config.output_size;
    for (size_t i = output_start; i < config.internal_neurons; ++i) {
        output_neurons.push_back(i);
    }
    module->register_neuron_port("output", output_neurons);
    
    // Store configuration
    module_configs_[config.name] = config;
    
    // Add to modular network
    modular_network_->add_module(std::move(module));
    
    // Initialize attention weight
    attention_weights_[config.name] = 1.0f / 9.0f; // Equal initial attention
    
    std::cout << "Created module: " << config.name 
              << " (neurons: " << config.internal_neurons 
              << ", input: " << config.input_size 
              << ", output: " << config.output_size << ")" << std::endl;
    
    return true;
}

bool BrainModuleArchitecture::connectModules(const InterModuleConnection& connection) {
    auto source_module = modular_network_->get_module(connection.source_module);
    auto target_module = modular_network_->get_module(connection.target_module);
    
    if (!source_module || !target_module) {
        std::cerr << "Cannot connect modules: " << connection.source_module 
                  << " -> " << connection.target_module << " (module not found)" << std::endl;
        return false;
    }
    
    // Store connection for later processing
    connections_.push_back(connection);
    
    std::cout << "Connected: " << connection.source_module 
              << ":" << connection.source_port << " -> " 
              << connection.target_module << ":" << connection.target_port << std::endl;
    
    return true;
}

// ============================================================================
// CONNECTION INITIALIZATION
// ============================================================================

void BrainModuleArchitecture::initializeConnections() {
    std::cout << "Initializing inter-module connections..." << std::endl;
    
    // Visual Cortex -> Comprehension Module
    InterModuleConnection conn1;
    conn1.source_module = "visual_cortex";
    conn1.source_port = "output";
    conn1.target_module = "comprehension_module";
    conn1.target_port = "input";
    conn1.connection_strength = 0.8f;
    conn1.plastic = true;
    conn1.connection_size = visual_feature_size_;
    connectModules(conn1);
    
    // Visual Cortex -> Executive Function
    InterModuleConnection conn2;
    conn2.source_module = "visual_cortex";
    conn2.source_port = "output";
    conn2.target_module = "executive_function";
    conn2.target_port = "input";
    conn2.connection_strength = 0.7f;
    conn2.plastic = true;
    conn2.connection_size = visual_feature_size_;
    connectModules(conn2);
    
    // Comprehension -> Executive Function
    InterModuleConnection conn3;
    conn3.source_module = "comprehension_module";
    conn3.source_port = "output";
    conn3.target_module = "executive_function";
    conn3.target_port = "input";
    conn3.connection_strength = 0.9f;
    conn3.plastic = true;
    conn3.connection_size = 512;
    connectModules(conn3);
    
    // Executive Function -> Motor Cortex
    InterModuleConnection conn4;
    conn4.source_module = "executive_function";
    conn4.source_port = "output";
    conn4.target_module = "motor_cortex";
    conn4.target_port = "input";
    conn4.connection_strength = 0.9f;
    conn4.plastic = true;
    conn4.connection_size = 256;
    connectModules(conn4);
    
    // Motor Cortex -> Output Module
    InterModuleConnection conn5;
    conn5.source_module = "motor_cortex";
    conn5.source_port = "output";
    conn5.target_module = "output_module";
    conn5.target_port = "input";
    conn5.connection_strength = 1.0f;
    conn5.plastic = false; // Direct motor output
    conn5.connection_size = action_size_;
    connectModules(conn5);
    
    // Memory connections
    InterModuleConnection conn6;
    conn6.source_module = "comprehension_module";
    conn6.source_port = "output";
    conn6.target_module = "memory_module";
    conn6.target_port = "input";
    conn6.connection_strength = 0.6f;
    conn6.plastic = true;
    conn6.connection_size = 512;
    connectModules(conn6);
    
    InterModuleConnection conn7;
    conn7.source_module = "memory_module";
    conn7.source_port = "output";
    conn7.target_module = "executive_function";
    conn7.target_port = "input";
    conn7.connection_strength = 0.7f;
    conn7.plastic = true;
    conn7.connection_size = 384;
    connectModules(conn7);
    
    // Reward system connections
    InterModuleConnection conn8;
    conn8.source_module = "executive_function";
    conn8.source_port = "output";
    conn8.target_module = "reward_system";
    conn8.target_port = "input";
    conn8.connection_strength = 0.8f;
    conn8.plastic = true;
    conn8.connection_size = 256;
    connectModules(conn8);
    
    InterModuleConnection conn9;
    conn9.source_module = "reward_system";
    conn9.source_port = "output";
    conn9.target_module = "central_controller";
    conn9.target_port = "input";
    conn9.connection_strength = 1.0f;
    conn9.plastic = false; // Direct reward signal
    conn9.connection_size = 128;
    connectModules(conn9);
    
    // Attention system connections
    InterModuleConnection conn10;
    conn10.source_module = "visual_cortex";
    conn10.source_port = "output";
    conn10.target_module = "attention_system";
    conn10.target_port = "input";
    conn10.connection_strength = 0.5f;
    conn10.plastic = true;
    conn10.connection_size = visual_feature_size_;
    connectModules(conn10);
    
    std::cout << "Initialized " << connections_.size() << " inter-module connections" << std::endl;
}

// ============================================================================
// PROCESSING PIPELINE
// ============================================================================

std::vector<float> BrainModuleArchitecture::processVisualInput(const std::vector<float>& visual_input) {
    if (!is_initialized_) return {};
    
    auto visual_cortex = modular_network_->get_module("visual_cortex");
    if (!visual_cortex) return {};
    
    // Process through visual cortex
    auto visual_features = visual_cortex->process(visual_input);
    
    // Update global context with visual information
    size_t copy_size = std::min(visual_features.size(), global_context_.size() / 2);
    std::copy(visual_features.begin(), visual_features.begin() + copy_size, 
              global_context_.begin());
    
    return visual_features;
}

std::vector<float> BrainModuleArchitecture::processTextInput(const std::vector<float>& text_input) {
    if (!is_initialized_) return {};
    
    auto comprehension = modular_network_->get_module("comprehension_module");
    if (!comprehension) return {};
    
    return comprehension->process(text_input);
}

std::vector<float> BrainModuleArchitecture::executeDecisionMaking(const std::vector<float>& context_input,
                                                                 const std::vector<float>& goals) {
    if (!is_initialized_) return {};
    
    auto executive = modular_network_->get_module("executive_function");
    if (!executive) return {};
    
    // Combine context and goals
    std::vector<float> combined_input = context_input;
    combined_input.insert(combined_input.end(), goals.begin(), goals.end());
    
    return executive->process(combined_input);
}

std::vector<float> BrainModuleArchitecture::generateMotorOutput(const std::vector<float>& decision_input) {
    if (!is_initialized_) return {};
    
    auto motor_cortex = modular_network_->get_module("motor_cortex");
    auto output_module = modular_network_->get_module("output_module");
    
    if (!motor_cortex || !output_module) return {};
    
    // Process through motor cortex then output module
    auto motor_output = motor_cortex->process(decision_input);
    return output_module->process(motor_output);
}

void BrainModuleArchitecture::update(float dt) {
    if (!is_initialized_) return;
    
    // Update all modules
    modular_network_->update(dt);
    
    // Process inter-module signals
    processInterModuleSignals();
    
    // Update attention
    updateAttention(global_context_);
    
    // Update performance metrics
    updatePerformanceMetrics();
    
    update_count_++;
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

void BrainModuleArchitecture::createDefaultConfigurations() {
    // Default configurations are created in individual create methods
    std::cout << "Default module configurations created" << std::endl;
}

bool BrainModuleArchitecture::validateModuleConfig(const ModuleConfig& config) const {
    if (config.name.empty()) return false;
    if (config.input_size == 0 || config.output_size == 0) return false;
    if (config.internal_neurons < config.output_size) return false;
    if (config.cortical_columns == 0) return false;
    if (config.learning_rate <= 0.0f || config.learning_rate > 1.0f) return false;
    return true;
}

void BrainModuleArchitecture::updatePerformanceMetrics() {
    if (!modular_network_) return;
    
    total_activity_ = modular_network_->get_total_activity();
}

void BrainModuleArchitecture::processInterModuleSignals() {
    // Process signals between connected modules
    for (const auto& connection : connections_) {
        auto source = modular_network_->get_module(connection.source_module);
        auto target = modular_network_->get_module(connection.target_module);
        
        if (source && target) {
            auto output = source->get_output();
            if (!output.empty()) {
                // Apply connection strength
                for (auto& val : output) {
                    val *= connection.connection_strength;
                }
                
                // Send signal to target
                target->receive_signal(output, connection.source_module, connection.source_port);
            }
        }
    }
}

// ============================================================================
// INTERFACE METHODS
// ============================================================================

NeuralModule* BrainModuleArchitecture::getModule(const std::string& name) {
    if (!modular_network_) return nullptr;
    return modular_network_->get_module(name);
}

const NeuralModule* BrainModuleArchitecture::getModule(const std::string& name) const {
    if (!modular_network_) return nullptr;
    return modular_network_->get_module(name);
}

std::vector<std::string> BrainModuleArchitecture::getModuleNames() const {
    if (!modular_network_) return {};
    return modular_network_->get_module_names();
}

void BrainModuleArchitecture::updateAttention(const std::vector<float>& context) {
    auto attention_system = modular_network_->get_module("attention_system");
    if (!attention_system) return;
    
    auto attention_output = attention_system->process(context);
    
    // Update attention weights for all modules
    auto module_names = getModuleNames();
    for (size_t i = 0; i < module_names.size() && i < attention_output.size(); ++i) {
        attention_weights_[module_names[i]] = std::max(0.1f, attention_output[i]);
    }
}

void BrainModuleArchitecture::applyNeuromodulation(float reward_signal, 
                                                  const std::vector<float>& attention_signal) {
    global_reward_signal_ = reward_signal;
    
    // Apply neuromodulation to all modules
    for (const auto& module_name : getModuleNames()) {
        auto module = getModule(module_name);
        if (module) {
            module->applyNeuromodulation("dopamine", reward_signal);
            
            // Apply attention modulation
            auto it = attention_weights_.find(module_name);
            if (it != attention_weights_.end()) {
                module->applyNeuromodulation("attention", it->second);
            }
        }
    }
}

std::map<std::string, float> BrainModuleArchitecture::getAttentionWeights() const {
    return attention_weights_;
}

void BrainModuleArchitecture::applyLearning(float reward, float prediction_error) {
    // Apply learning signals to all modules
    for (const auto& module_name : getModuleNames()) {
        auto module = getModule(module_name);
        if (module) {
            // Update with reward and prediction error
            std::vector<float> learning_signal = {reward, prediction_error};
            module->update(0.1f, learning_signal, reward);
        }
    }
}

std::map<std::string, float> BrainModuleArchitecture::getPerformanceMetrics() const {
    std::map<std::string, float> metrics;
    metrics["total_activity"] = total_activity_;
    metrics["update_count"] = static_cast<float>(update_count_);
    metrics["global_reward"] = global_reward_signal_;
    metrics["num_modules"] = static_cast<float>(getModuleNames().size());
    metrics["num_connections"] = static_cast<float>(connections_.size());
    return metrics;
}

bool BrainModuleArchitecture::isStable() const {
    if (!modular_network_) return false;
    return modular_network_->is_stable();
}

float BrainModuleArchitecture::getTotalActivity() const {
    return total_activity_;
}

// ============================================================================
// MEMORY OPERATIONS (Simplified Implementation)
// ============================================================================

void BrainModuleArchitecture::storeExperience(const std::vector<float>& experience, 
                                             const std::string& context) {
    auto memory_module = getModule("memory_module");
    if (memory_module) {
        memory_module->process(experience);
    }
}

std::vector<std::vector<float>> BrainModuleArchitecture::retrieveExperiences(const std::vector<float>& query, 
                                                                            size_t max_results) {
    auto memory_module = getModule("memory_module");
    if (memory_module) {
        auto result = memory_module->process(query);
        return {result}; // Simplified - return single result
    }
    return {};
}

void BrainModuleArchitecture::updateInterModuleConnections() {
    // Update plastic connections based on activity
    for (auto& connection : connections_) {
        if (connection.plastic) {
            // Simple Hebbian-like update
            auto source = getModule(connection.source_module);
            auto target = getModule(connection.target_module);
            
            if (source && target) {
                auto source_output = source->get_output();
                auto target_output = target->get_output();
                
                if (!source_output.empty() && !target_output.empty()) {
                    float correlation = 0.0f;
                    size_t min_size = std::min(source_output.size(), target_output.size());
                    
                    for (size_t i = 0; i < min_size; ++i) {
                        correlation += source_output[i] * target_output[i];
                    }
                    correlation /= min_size;
                    
                    // Update connection strength
                    float learning_rate = 0.001f;
                    connection.connection_strength += learning_rate * correlation;
                    connection.connection_strength = std::max(0.1f, 
                        std::min(2.0f, connection.connection_strength));
                }
            }
        }
    }
}

std::map<std::string, std::map<std::string, float>> BrainModuleArchitecture::getLearningStats() const {
    std::map<std::string, std::map<std::string, float>> stats;
    
    for (const auto& module_name : getModuleNames()) {
        auto module = getModule(module_name);
        if (module) {
            stats[module_name] = module->getPerformanceMetrics();
        }
    }
    
    return stats;
}

bool BrainModuleArchitecture::saveState(const std::string& directory) const {
    if (!modular_network_) return false;

    std::filesystem::create_directories(directory);

    // Save each module state
    for (const auto& name : modular_network_->get_module_names()) {
        auto mod = modular_network_->get_module(name);
        if (mod) {
            mod->save_state(directory + "/" + name + ".bin");
        }
    }

    // Save architecture metadata
    std::ofstream meta(directory + "/architecture.meta", std::ios::binary);
    if (!meta.is_open()) return false;

    meta.write(reinterpret_cast<const char*>(&visual_input_width_), sizeof(visual_input_width_));
    meta.write(reinterpret_cast<const char*>(&visual_input_height_), sizeof(visual_input_height_));

    size_t map_size = attention_weights_.size();
    meta.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));
    for (const auto& [name, weight] : attention_weights_) {
        size_t len = name.size();
        meta.write(reinterpret_cast<const char*>(&len), sizeof(len));
        meta.write(name.c_str(), len);
        meta.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
    }

    return true;
}

bool BrainModuleArchitecture::loadState(const std::string& directory) {
    if (!modular_network_) return false;

    // Load architecture metadata
    std::ifstream meta(directory + "/architecture.meta", std::ios::binary);
    if (!meta.is_open()) return false;

    meta.read(reinterpret_cast<char*>(&visual_input_width_), sizeof(visual_input_width_));
    meta.read(reinterpret_cast<char*>(&visual_input_height_), sizeof(visual_input_height_));

    size_t map_size = 0;
    meta.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
    attention_weights_.clear();
    for (size_t i = 0; i < map_size; ++i) {
        size_t len = 0;
        meta.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string name(len, '\0');
        meta.read(name.data(), len);
        float weight = 0.0f;
        meta.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        attention_weights_[name] = weight;
    }

    // Load each module
    for (const auto& name : modular_network_->get_module_names()) {
        auto mod = modular_network_->get_module(name);
        if (mod) {
            mod->load_state(directory + "/" + name + ".bin");
        }
    }

    return true;
}
