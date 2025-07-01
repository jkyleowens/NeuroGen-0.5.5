// ============================================================================
// CENTRAL CONTROLLER IMPLEMENTATION
// File: src/CentralController.cpp
// ============================================================================

#include <NeuroGen/VisualInterface.h>
#include <NeuroGen/CentralController.h>
#include <NeuroGen/ControllerModule.h>
#include <NeuroGen/TaskAutomationModules.h>
#include <NeuroGen/AutonomousLearningAgent.h>
#include <NeuroGen/NeuralModule.h>
#include <NeuroGen/NetworkConfig.h>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <memory>
#include <vector>
#include <unordered_map>
#include <iomanip>

// Function to create a default configuration for a neural module
NetworkConfig create_default_config() {
    NetworkConfig config;
    config.num_neurons = 256; // Enhanced neuron count for version 0.5.5
    config.enable_neurogenesis = true;
    config.enable_stdp = true;
    config.enable_pruning = true;
    config.enable_structural_plasticity = true; // Enable dynamic synaptogenesis
    
    // Enhanced connectivity parameters for version 0.5.5
    config.input_hidden_prob = 0.6f;
    config.hidden_hidden_prob = 0.3f;
    config.hidden_output_prob = 0.8f;
    config.exc_ratio = 0.8f;
    
    // Synaptic parameters
    config.min_weight = 0.01f;
    config.max_weight = 1.5f;
    config.weight_init_std = 0.3f;
    
    // Topology parameters
    config.numColumns = 4;
    config.neuronsPerColumn = 64;
    config.localFanOut = 15;
    config.localFanIn = 15;
    
    // Enhanced timing
    config.dt = 0.1;
    config.simulation_time = 1000.0f; // 1 second simulation
    
    config.finalizeConfig();
    return config;
}

CentralController::CentralController() 
    : is_initialized_(false), cycle_count_(0) {
    std::cout << "CentralController: Initializing..." << std::endl;
}

CentralController::~CentralController() {
    shutdown();
}

bool CentralController::initialize() {
    std::cout << "CentralController: Starting system initialization..." << std::endl;
    
    try {
        // Initialize neural modules first
        initialize_neural_modules();
        
        // Initialize neuromodulatory controller
        ControllerConfig controller_config;
        controller_config.initial_dopamine_level = 0.4f;
        controller_config.initial_serotonin_level = 0.5f;
        controller_config.curiosity_drive_strength = 0.4f;
        controller_config.enable_adaptive_baselines = true;
        controller_config.enable_stress_response = true;
        
        neuro_controller_ = std::make_unique<ControllerModule>(controller_config);
        
        // Register modules with the controller
        neuro_controller_->register_module("PerceptionNet", perception_module_);
        neuro_controller_->register_module("PlanningNet", planning_module_);
        neuro_controller_->register_module("MotorControlNet", motor_module_);
        
        std::cout << "✅ Neuromodulatory controller initialized with 3 modules" << std::endl;
        
        // Initialize task-level modules
        initialize_task_modules();
        
        // Initialize visual interface
        visual_interface_ = std::make_unique<VisualInterface>(1920, 1080);
        if (!visual_interface_->initialize_capture()) {
            std::cerr << "Warning: Visual interface initialization failed, using simulation mode" << std::endl;
        }
        
        // Initialize autonomous learning agent
        NetworkConfig agent_config = create_default_config();
        learning_agent_ = std::make_unique<AutonomousLearningAgent>(agent_config);
        if (!learning_agent_->initialize()) {
            std::cerr << "Warning: Autonomous learning agent initialization failed" << std::endl;
        }
        
        is_initialized_ = true;
        std::cout << "✅ CentralController initialization complete!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "CentralController: Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void CentralController::initialize_neural_modules() {
    std::cout << "CentralController: Initializing neural modules..." << std::endl;
    
    // Create configurations
    auto cognitive_config = create_default_config();
    cognitive_config.neurogenesis_rate = 0.002; // Higher plasticity for cognitive tasks
    
    auto motor_config = create_default_config();
    motor_config.stdp_learning_rate = 0.005; // Lower learning rate for stable motor control
    
    // Create neural modules
    perception_module_ = std::make_shared<NeuralModule>("PerceptionNet", cognitive_config);
    planning_module_ = std::make_shared<NeuralModule>("PlanningNet", cognitive_config);
    motor_module_ = std::make_shared<NeuralModule>("MotorControlNet", motor_config);
    
    std::cout << "✅ Neural modules created" << std::endl;
}

void CentralController::initialize_task_modules() {
    std::cout << "CentralController: Initializing task modules..." << std::endl;
    
    // Create task-level modules
    cognitive_module_ = std::make_shared<CognitiveModule>(perception_module_, planning_module_);
    motor_task_module_ = std::make_shared<MotorModule>(motor_module_);
    
    // Initialize them
    cognitive_module_->initialize();
    motor_task_module_->initialize();
    
    std::cout << "✅ Task modules initialized" << std::endl;
}

void CentralController::simulateNewScreenData(const std::vector<ScreenElement>& screen_elements) {
    if (!is_initialized_) {
        std::cerr << "CentralController: Cannot process screen data - not initialized!" << std::endl;
        return;
    }
    
    current_screen_elements_ = screen_elements;
    
    std::cout << "CentralController: Processing " << screen_elements.size() << " screen elements:" << std::endl;
    for (const auto& element : screen_elements) {
        std::cout << "  - " << element.type << " at (" << element.x << ", " << element.y 
                  << ") size (" << element.width << "x" << element.height << ")";
        if (!element.text.empty()) {
            std::cout << " text: \"" << element.text << "\"";
        }
        std::cout << " clickable: " << (element.is_clickable ? "yes" : "no") << std::endl;
    }
    
    process_screen_elements();
}

void CentralController::process_screen_elements() {
    if (current_screen_elements_.empty()) {
        return;
    }
    
    // Convert screen elements to neural input patterns
    std::vector<float> visual_input(256, 0.0f);
    
    for (size_t i = 0; i < std::min(current_screen_elements_.size(), size_t(8)); ++i) {
        const auto& element = current_screen_elements_[i];
        
        // Encode element properties into neural input
        size_t base_idx = i * 32;
        if (base_idx + 31 < visual_input.size()) {
            // Position encoding (normalized to 0-1)
            visual_input[base_idx + 0] = element.x / 1920.0f;
            visual_input[base_idx + 1] = element.y / 1080.0f;
            visual_input[base_idx + 2] = element.width / 1920.0f;
            visual_input[base_idx + 3] = element.height / 1080.0f;
            
            // Type encoding
            if (element.type == "button") visual_input[base_idx + 4] = 1.0f;
            else if (element.type == "textbox") visual_input[base_idx + 5] = 1.0f;
            else if (element.type == "link") visual_input[base_idx + 6] = 1.0f;
            else if (element.type == "image") visual_input[base_idx + 7] = 1.0f;
            
            // Properties
            visual_input[base_idx + 8] = element.is_clickable ? 1.0f : 0.0f;
            visual_input[base_idx + 9] = element.confidence;
        }
    }
    
    // Send to perception module
    float dt = 0.1f;
    perception_module_->update(dt, visual_input, 0.0f);
    
    // Generate reward based on element detection quality
    float reward = 0.0f;
    for (const auto& element : current_screen_elements_) {
        if (element.is_clickable && element.confidence > 0.7f) {
            reward += 0.1f; // Reward for detecting interactive elements
        }
    }
    
    if (reward > 0.0f) {
        RewardSignal signal;
        signal.type = RewardSignalType::NOVELTY_DETECTION;
        signal.magnitude = reward;
        signal.confidence = 0.8f;
        signal.target_module = "PerceptionNet";
        neuro_controller_->process_reward_signal(signal);
    }
}

void CentralController::run(int cycles) {
    if (!is_initialized_) {
        std::cerr << "CentralController: Cannot run - not initialized!" << std::endl;
        return;
    }
    
    std::cout << "CentralController: Running " << cycles << " cognitive cycle(s)..." << std::endl;
    
    for (int i = 0; i < cycles; ++i) {
        cycle_count_++;
        std::cout << "\n--- Cognitive Cycle " << cycle_count_ << " ---" << std::endl;
        
        execute_cognitive_cycle();
        update_performance_metrics();
        
        // Update neuromodulatory controller
        float dt = 100.0f; // 100ms per cycle
        neuro_controller_->update(dt);
        
        // Brief pause between cycles
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    std::cout << "\n✅ Cognitive cycles complete!" << std::endl;
}

void CentralController::execute_cognitive_cycle() {
    float dt = 0.1f;
    
    // Get perception output
    auto perception_stats = perception_module_->get_stats();
    std::vector<float> perception_output(64, 0.1f); // Simplified output representation
    
    // Planning processes perception output
    planning_module_->update(dt, perception_output, 0.0f);
    auto planning_stats = planning_module_->get_stats();
    std::vector<float> planning_output(32, 0.05f);
    
    // Motor module executes planned actions
    motor_module_->update(dt, planning_output, 0.0f);
    auto motor_stats = motor_module_->get_stats();
    
    std::cout << "  🧠 Perception: " << perception_stats.active_neuron_count << " active neurons" << std::endl;
    std::cout << "  🎯 Planning: " << planning_stats.active_neuron_count << " active neurons" << std::endl;
    std::cout << "  🏃 Motor: " << motor_stats.active_neuron_count << " active neurons" << std::endl;
    
    // Attention allocation based on activity
    float total_activity = perception_stats.active_neuron_count + 
                          planning_stats.active_neuron_count + 
                          motor_stats.active_neuron_count;
    
    if (total_activity > 0) {
        std::unordered_map<std::string, float> attention_weights;
        attention_weights["PerceptionNet"] = perception_stats.active_neuron_count / total_activity;
        attention_weights["PlanningNet"] = planning_stats.active_neuron_count / total_activity;
        attention_weights["MotorControlNet"] = motor_stats.active_neuron_count / total_activity;
        
        neuro_controller_->allocate_attention(attention_weights);
    }
}

void CentralController::update_performance_metrics() {
    // Calculate system performance metrics
    float system_performance = neuro_controller_->calculate_overall_system_performance();
    
    std::cout << "  📈 System Performance: " << std::fixed << std::setprecision(1) 
              << system_performance * 100 << "%" << std::endl;
    
    // Display neuromodulator concentrations
    auto concentrations = neuro_controller_->get_all_concentrations();
    std::cout << "  🧬 Dopamine: " << std::setprecision(2) 
              << concentrations[NeuromodulatorType::DOPAMINE] 
              << " | Serotonin: " << concentrations[NeuromodulatorType::SEROTONIN] << std::endl;
}

void CentralController::shutdown() {
    if (is_initialized_) {
        std::cout << "CentralController: Shutting down..." << std::endl;
        
        if (visual_interface_) {
            visual_interface_->stop_capture();
        }
        
        if (learning_agent_) {
            learning_agent_->shutdown();
        }
        
        if (neuro_controller_) {
            neuro_controller_->emergency_stop();
        }
        
        is_initialized_ = false;
        std::cout << "✅ CentralController shutdown complete" << std::endl;
    }
}

std::string CentralController::getSystemStatus() const {
    if (!is_initialized_) {
        return "System not initialized";
    }
    
    std::ostringstream status;
    status << "CentralController Status:\n";
    status << "  Cycles completed: " << cycle_count_ << "\n";
    status << "  Screen elements: " << current_screen_elements_.size() << "\n";
    status << "  System performance: " << std::fixed << std::setprecision(1) 
           << getSystemPerformance() * 100 << "%\n";
    
    if (neuro_controller_) {
        status << "  Registered modules: " << neuro_controller_->get_registered_modules().size() << "\n";
    }
    
    return status.str();
}

float CentralController::getSystemPerformance() const {
    if (!is_initialized_ || !neuro_controller_) {
        return 0.0f;
    }
    
    return neuro_controller_->calculate_overall_system_performance();
}
