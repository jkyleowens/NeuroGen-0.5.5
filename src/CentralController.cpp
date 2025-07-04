// ============================================================================
// CENTRAL CONTROLLER IMPLEMENTATION - FIXED
// File: src/CentralController.cpp
// ============================================================================

#include "NeuroGen/CentralController.h"
#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/VisualInterface.h"
#include "NeuroGen/TaskAutomationModules.h"
#include "NeuroGen/NeuralModule.h"
#include "NeuroGen/Network.h"
#include "NeuroGen/NetworkConfig.h"
#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>

// ============================================================================
// CONSTRUCTOR AND DESTRUCTOR
// ============================================================================

CentralController::CentralController() 
    : is_initialized_(false), cycle_count_(0) {
    std::cout << "CentralController: Initializing central coordination system..." << std::endl;
}

CentralController::~CentralController() {
    shutdown();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

bool CentralController::initialize() {
    if (is_initialized_) {
        std::cout << "CentralController: Already initialized" << std::endl;
        return true;
    }
    
    try {
        std::cout << "CentralController: Starting initialization sequence..." << std::endl;
        
        // Step 1: Initialize neural modules
        initialize_neural_modules();
        
        // Step 2: Initialize controller module with proper configuration
        ControllerConfig controller_config;
        // **FIXED: Use correct member names that exist in ControllerConfig**
        controller_config.initial_dopamine_level = 0.4f;
        controller_config.reward_learning_rate = 0.02f;  // Use correct member name
        controller_config.enable_detailed_logging = true;
        
        neuro_controller_ = std::make_unique<ControllerModule>(controller_config);
        
        // Register neural modules with controller
        neuro_controller_->register_module("PerceptionNet", perception_module_);
        neuro_controller_->register_module("PlanningNet", planning_module_);
        neuro_controller_->register_module("MotorControlNet", motor_module_);
        
        // Step 3: Initialize task modules
        initialize_task_modules();
        
        // Step 4: Initialize autonomous learning agent
        NetworkConfig agent_config;
        agent_config.hidden_size = 512;
        // **FIXED: NetworkConfig doesn't have learning_rate - removed this line**
        learning_agent_ = std::make_unique<AutonomousLearningAgent>(agent_config);
        learning_agent_->initialize();
        
        // Step 5: Initialize visual interface
        visual_interface_ = std::make_unique<VisualInterface>(1920, 1080);
        visual_interface_->initialize_capture();
        
        is_initialized_ = true;
        std::cout << "✅ CentralController: Initialization complete!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "CentralController: Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void CentralController::initialize_neural_modules() {
    std::cout << "CentralController: Initializing neural modules..." << std::endl;
    
    // Create configurations
    NetworkConfig cognitive_config;
    cognitive_config.hidden_size = 512;
    // **FIXED: NetworkConfig doesn't have learning_rate - removed this line**
    
    NetworkConfig motor_config;
    motor_config.hidden_size = 256;
    // **FIXED: NetworkConfig doesn't have learning_rate - removed this line**
    
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

// ============================================================================
// MAIN CONTROL INTERFACE
// ============================================================================

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
        // **FIXED: Use proper RewardSignal structure with existing members**
        RewardSignal signal;
        signal.signal_type = RewardSignalType::NOVELTY_DETECTION;
        signal.magnitude = reward;
        signal.source_module = "PerceptionNet";  // Use source_module instead of target_module
        // **FIXED: Use correct method name that exists in ControllerModule**
        neuro_controller_->apply_reward("PerceptionNet", reward, RewardSignalType::NOVELTY_DETECTION);
    }
}

void CentralController::run(int cycles) {
    if (!is_initialized_) {
        std::cerr << "CentralController: Cannot run - not initialized!" << std::endl;
        return;
    }
    
    std::cout << "CentralController: Running " << cycles << " cognitive cycles..." << std::endl;
    
    for (int i = 0; i < cycles; ++i) {
        cycle_count_++;
        std::cout << "\n--- Cycle " << cycle_count_ << " ---" << std::endl;
        
        execute_cognitive_cycle();
        update_performance_metrics();
    }
    
    std::cout << "CentralController: Completed " << cycles << " cycles" << std::endl;
}

void CentralController::execute_cognitive_cycle() {
    const float dt = 0.1f; // 100ms per cycle
    
    // Update controller module
    neuro_controller_->update(dt);
    
    // Update learning agent
    learning_agent_->update(dt);
    
    // Generate attention allocation
    std::unordered_map<std::string, float> attention_weights;
    attention_weights["PerceptionNet"] = 0.4f;
    attention_weights["PlanningNet"] = 0.4f;
    attention_weights["MotorControlNet"] = 0.2f;
    
    // **FIXED: Use available method for attention allocation**
    // Apply attention through individual module updates
    for (const auto& [module_name, weight] : attention_weights) {
        auto module = neuro_controller_->get_module(module_name);
        if (module) {
            // Apply attention weight through neuromodulator release
            neuro_controller_->release_neuromodulator(NeuromodulatorType::NOREPINEPHRINE, weight, module_name);
        }
    }
    
    // Coordinate neural activity
    neuro_controller_->coordinate_module_activities();
    
    std::cout << "CentralController: Cognitive cycle completed" << std::endl;
}

void CentralController::update_performance_metrics() {
    // **FIXED: Use available methods to get neuromodulator information**
    std::cout << "CentralController: Performance Metrics:" << std::endl;
    std::cout << "  - Dopamine: " << neuro_controller_->get_concentration(NeuromodulatorType::DOPAMINE) << std::endl;
    std::cout << "  - Serotonin: " << neuro_controller_->get_concentration(NeuromodulatorType::SEROTONIN) << std::endl;
    std::cout << "  - Norepinephrine: " << neuro_controller_->get_concentration(NeuromodulatorType::NOREPINEPHRINE) << std::endl;
    std::cout << "  - System Coherence: " << neuro_controller_->get_system_coherence() << std::endl;
}

// ============================================================================
// SYSTEM STATUS AND SHUTDOWN
// ============================================================================

void CentralController::shutdown() {
    if (!is_initialized_) return;
    
    std::cout << "CentralController: Shutting down..." << std::endl;
    
    if (learning_agent_) {
        learning_agent_->shutdown();
    }
    
    if (visual_interface_) {
        visual_interface_->stop_capture();
    }
    
    if (neuro_controller_) {
        neuro_controller_->emergency_stop();
    }
    
    is_initialized_ = false;
    std::cout << "CentralController: Shutdown complete" << std::endl;
}

std::string CentralController::getSystemStatus() const {
    if (!is_initialized_) {
        return "System not initialized";
    }
    
    std::stringstream ss;
    ss << "Central Controller Status:\n";
    ss << "  Initialized: " << (is_initialized_ ? "Yes" : "No") << "\n";
    ss << "  Cycles completed: " << cycle_count_ << "\n";
    ss << "  Screen elements: " << current_screen_elements_.size() << "\n";
    
    if (neuro_controller_) {
        ss << "  Controller active: Yes\n";
        ss << "  Registered modules: " << neuro_controller_->get_registered_modules().size() << "\n";
    }
    
    return ss.str();
}

float CentralController::getSystemPerformance() const {
    if (!is_initialized_ || !neuro_controller_) {
        return 0.0f;
    }
    
    // Calculate performance based on system coherence and activity
    float performance = neuro_controller_->get_system_coherence();
    
    // Factor in number of processed elements
    if (!current_screen_elements_.empty()) {
        performance += 0.1f * std::min(5.0f, static_cast<float>(current_screen_elements_.size()));
    }
    
    return std::min(1.0f, performance);
}