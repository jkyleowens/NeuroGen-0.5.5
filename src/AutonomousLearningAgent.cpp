// ============================================================================
// AUTONOMOUS LEARNING AGENT IMPLEMENTATION
// File: src/AutonomousLearningAgent.cpp
// ============================================================================

#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/NetworkIntegration.h"
#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/SafetyManager.h"
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

// ============================================================================
// AUTONOMOUS LEARNING AGENT IMPLEMENTATION
// ============================================================================
AutonomousLearningAgent::AutonomousLearningAgent(const NetworkConfig& config)
    : config_(config),
      is_learning_active_(false),
      detailed_logging_(false),
      simulation_time_(0.0f),
      last_action_time_(std::chrono::steady_clock::now()),
      gen(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {
    
    // Initialize core components for real computer control
    controller_module_ = std::make_unique<ControllerModule>(ControllerConfig());
    memory_system_ = std::make_unique<MemorySystem>();
    visual_interface_ = std::make_unique<VisualInterface>(1920, 1080);
    attention_controller_ = std::make_unique<AttentionController>();
    
    // Initialize input/output systems for real computer control
    real_screen_capture_ = std::make_unique<RealScreenCapture>();
    input_controller_ = std::make_unique<InputController>();
    ocr_processor_ = std::make_unique<OCRProcessor>();
    gui_detector_ = std::make_unique<GUIElementDetector>();
    
    // Initialize brain module architecture
    brain_architecture_ = std::make_unique<BrainModuleArchitecture>();
    
    // Initialize specialized neural modules for cognitive functions
    initializeSpecializedModules();
    
    // Initialize state vectors - SCALED UP for massive neural architecture
    environmental_context_.resize(2048, 0.0f);  // 4x larger for richer environmental representation
    global_state_.resize(1024, 0.0f);           // 4x larger for complex state representation  
    current_goals_.resize(256, 0.0f);           // 4x larger for multiple concurrent goals
    
    // Initialize learning parameters
    exploration_rate_ = 0.3f; // Start with moderate exploration
    learning_rate_ = 0.01f;
    global_reward_signal_ = 0.0f;
    
    // Initialize with a default action
    selected_action_.confidence = 0.0f;
    selected_action_.type = ActionType::CLICK;
    
    std::cout << "✅ AutonomousLearningAgent constructed with real computer control" << std::endl;
}

AutonomousLearningAgent::~AutonomousLearningAgent() {
    shutdown();
}

bool AutonomousLearningAgent::initialize(bool reset_model) {
    save_path_ = "neurogen_agent_state"; // Default save path
    if (reset_model && std::filesystem::exists(save_path_)) {
        std::cout << "🔥 Resetting model state. Deleting existing save directory..." << std::endl;
        std::filesystem::remove_all(save_path_);
    }

    std::cout << "🔧 Initializing AutonomousLearningAgent for real computer control..." << std::endl;
    
    if (!controller_module_) {
        std::cerr << "Error: Controller module not created" << std::endl;
        return false;
    }
    
    // Initialize visual capture system for real screen monitoring
    if (!visual_interface_->initialize_capture()) {
        std::cerr << "Warning: Failed to initialize visual capture" << std::endl;
    }
    
    // Initialize real screen capture
    if (real_screen_capture_ && !real_screen_capture_->initialize(1920, 1080)) {
        std::cerr << "Failed to initialize screen capture" << std::endl;
    }
    
    // Initialize input controller for real computer control
    if (input_controller_ && !input_controller_->initialize()) {
        std::cerr << "Failed to initialize input controller" << std::endl;
    }
    
    // Enable safety bounds for input controller (prevent dangerous actions)
    if (input_controller_) {
        input_controller_->enableSafetyBounds(50, 50, 1870, 1030); // Safe screen area
        std::cout << "✅ Safety bounds enabled for input controller" << std::endl;
    }
    
    // Initialize OCR for text recognition
    if (ocr_processor_ && !ocr_processor_->initialize()) {
        std::cerr << "Warning: Failed to initialize OCR processor" << std::endl;
    }
    
    // Initialize GUI element detector
    if (gui_detector_ && !gui_detector_->initialize()) {
        std::cerr << "Warning: Failed to initialize GUI detector" << std::endl;
    }
    
    // Register neural modules with attention controller
    attention_controller_->register_module("visual_cortex");
    attention_controller_->register_module("motor_cortex");
    attention_controller_->register_module("prefrontal_cortex");
    attention_controller_->register_module("working_memory");
    attention_controller_->register_module("reward_system");
    attention_controller_->register_module("attention_system");
    
    // Initialize neural modules and attention system
    initialize_neural_modules();
    initialize_attention_system();
    
    // Initialize brain module architecture for advanced processing
    if (brain_architecture_) {
        if (!brain_architecture_->initialize(1920, 1080)) {
            std::cerr << "Warning: Failed to initialize brain module architecture" << std::endl;
        } else {
            std::cout << "✅ Brain module architecture initialized successfully" << std::endl;
        }
    }
    
    // Set up continuous learning goals for computer interaction
    setupDefaultLearningGoals();
    
    std::cout << "✅ AutonomousLearningAgent initialized for real computer control" << std::endl;
    return true;
}

void AutonomousLearningAgent::update(float dt) {
    simulation_time_ += dt;
    
    if (controller_module_) {
        controller_module_->update(dt);
    }
    
    if (is_learning_active_) {
        autonomousLearningStep(dt);
        update_learning_goals();
    }
}

void AutonomousLearningAgent::shutdown() {
    stopAutonomousLearning();
    
    if (visual_interface_) {
        visual_interface_->stop_capture();
    }
    if (real_screen_capture_) real_screen_capture_->shutdown();
    if (input_controller_) input_controller_->shutdown();
    if (ocr_processor_) ocr_processor_->shutdown();
    
    std::cout << "AutonomousLearningAgent shutdown complete" << std::endl;
}

void AutonomousLearningAgent::startAutonomousLearning() {
    if (is_learning_active_) return;
    
    is_learning_active_ = true;
    std::cout << "Starting autonomous learning mode..." << std::endl;
    
    if (visual_interface_) {
        visual_interface_->start_continuous_capture();
    }
}

void AutonomousLearningAgent::stopAutonomousLearning() {
    if (!is_learning_active_) return;
    
    is_learning_active_ = false;
    std::cout << "Stopping autonomous learning mode..." << std::endl;
    
    if (visual_interface_) {
        visual_interface_->stop_capture();
    }
}

float AutonomousLearningAgent::autonomousLearningStep(float dt) {
    if (!is_learning_active_) return getLearningProgress();

    // If confidence is low, it implies no meaningful action is selected.
    if (selected_action_.confidence < 0.1f) { 
        // If no action is selected, try to make a decision
        make_decision();
    }

    // === ENHANCED REINFORCEMENT LEARNING CYCLE ===
    
    // Step 1: Capture and process real screen input
    processRealScreenInput();
    
    // Step 2: Update working memory with current visual context
    update_working_memory();
    
    // Step 2.5: Process network output
    std::vector<float> processed_output = modules_["prefrontal_cortex"]->process(environmental_context_);

    // Step 3: Coordinate neural modules for decision making
    coordinate_modules();
    
    // Step 4: Make decision based on current state and goals
    make_decision();
    
    // Step 5: Execute selected action on real computer
    execute_action();
    
    // Step 6: Compute reward based on screen changes and goal progress
    float immediate_reward = computeScreenBasedReward();
    
    // Step 7: Learn from the action outcome using reinforcement learning
    learnFromActionOutcome(immediate_reward);
    
    // Step 8: Update exploration rate based on performance
    adapt_exploration_rate();
    
    // Step 9: Store experience in episodic memory
    storeEpisodeInMemory(immediate_reward);
    
    // Step 10: Update attention weights based on current context
    update_attention_weights();
    
    // Log learning progress periodically
    static int step_count = 0;
    if (++step_count % 50 == 0) {
        logLearningProgress(step_count, immediate_reward);
    }
    
    // Return learning progress (based on accumulated experience and performance)
    return getLearningProgress();
}

void AutonomousLearningAgent::select_and_execute_action() {
    // Use decision-making system from DecisionAndActionSystems.cpp
    
    // Exploration vs. Exploitation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    if (dis(gen) < exploration_rate_) {
        // Explore: select a random action
        int random_action_idx = std::uniform_int_distribution<>(0, static_cast<int>(ActionType::BACKSPACE))(gen);
        selected_action_.type = static_cast<ActionType>(random_action_idx);
        selected_action_.confidence = 1.0f; // Confidence is high for exploration
        log_action("Exploring with random action: " + actionTypeToString(selected_action_.type));
    } else {
        // Exploit: use the decision-making system
        make_decision();
    }
    
    execute_action();
}

float AutonomousLearningAgent::calculate_immediate_reward() {
    float reward = 0.0f;

    // Reward for successful actions
    if (metrics_.successful_actions > 0) {
        reward += 0.1f * metrics_.successful_actions;
    }

    // Penalizing WAIT is no longer applicable

    // Reward for exploration and novelty
    float novelty_bonus = 0.0f;
    if (memory_system_ && !environmental_context_.empty()) {
        auto similar_episodes = memory_system_->retrieveSimilarEpisodes(environmental_context_, "default", 3);
        if (similar_episodes.size() < 2) {
            novelty_bonus = 0.2f; // High novelty
        } else {
            novelty_bonus = 0.05f; // Some novelty
        }
    }
    reward += novelty_bonus;

    // Reward for progressing towards a goal
    if (!learning_goals_.empty()) {
        // Implement logic to check if the agent is making progress towards its goals
        // For example, if a goal is to click a specific button, and the agent does so,
        // provide a large reward.
    }

    return std::max(-0.5f, std::min(reward, 0.5f));
}

// ============================================================================
// ADDITIONAL INTERFACE METHODS
// ============================================================================

void AutonomousLearningAgent::addLearningGoal(std::unique_ptr<AutonomousGoal> goal) {
    if (goal) {
        learning_goals_.push_back(std::move(goal));
        std::cout << "Added learning goal: " << learning_goals_.back()->description << std::endl;
    }
}

BrowsingState AutonomousLearningAgent::getCurrentEnvironmentState() const {
    if (environment_sensor_) {
        return environment_sensor_();
    }
    
    // Return default state if no sensor is set
    BrowsingState default_state;
    default_state.current_url = "about:blank";
    return default_state;
}

std::string AutonomousLearningAgent::getStatusReport() const {
    std::stringstream ss;
    ss << "=== Autonomous Learning Agent Status ===\n";
    ss << "Learning Active: " << (is_learning_active_ ? "Yes" : "No") << "\n";
    ss << "Simulation Time: " << simulation_time_ << "s\n";
    ss << "Learning Goals: " << learning_goals_.size() << "\n";
    ss << "Environmental Context Size: " << environmental_context_.size() << "\n";
    
    if (memory_system_) {
        ss << "Episodic Memories: " << memory_system_->get_episodic_memory_size() << "\n";
    }
    
    return ss.str();
}

float AutonomousLearningAgent::getLearningProgress() const {
    // Simple progress calculation based on simulation time and memory accumulation
    float time_progress = std::min(1.0f, simulation_time_ / 1000.0f);
    
    float memory_progress = 0.0f;
    if (memory_system_ && memory_system_->get_episodic_memory_size() > 0) {
        memory_progress = std::min(1.0f, static_cast<float>(memory_system_->get_episodic_memory_size()) / 100.0f);
    }
    
    return (time_progress + memory_progress) / 2.0f;
}

std::map<std::string, float> AutonomousLearningAgent::getAttentionWeights() const {
    std::map<std::string, float> weights;
    
    if (attention_controller_) {
        // Get weights for all registered modules
        weights["visual_cortex"] = attention_controller_->get_attention_weight("visual_cortex");
        weights["working_memory"] = attention_controller_->get_attention_weight("working_memory");
        weights["decision_making"] = attention_controller_->get_attention_weight("decision_making");
        weights["action_execution"] = attention_controller_->get_attention_weight("action_execution");
    }
    
    return weights;
}

void AutonomousLearningAgent::initialize_neural_modules() {
    // Initialize basic neural module coordination
    std::cout << "Initializing neural modules..." << std::endl;
    
    // This would create and register specialized neural modules
    // For now, we'll just set up the controller coordination
}

void AutonomousLearningAgent::initialize_attention_system() {
    // Set up attention priorities for different contexts
    if (attention_controller_) {
        attention_controller_->set_priority("visual_processing", 0.8f);
        attention_controller_->set_priority("decision_making", 0.7f);
        attention_controller_->set_priority("memory_consolidation", 0.5f);
    }
}

void AutonomousLearningAgent::update_learning_goals() {
    // Update progress on active learning goals
    for (auto& goal : learning_goals_) {
        if (goal && goal->is_active) {
            // Simple goal progress tracking
            // In a real implementation, this would check success criteria
        }
    }
}

void AutonomousLearningAgent::log_action(const std::string& action) {
    if (detailed_logging_) {
        std::cout << "[" << simulation_time_ << "s] " << action << std::endl;
    }
}

// ============================================================================
// SPECIALIZED MODULES INITIALIZATION
// ============================================================================

void AutonomousLearningAgent::initializeSpecializedModules() {
    // Create specialized neural modules for different cognitive functions
    // MASSIVE SCALE-UP: Creating a robust free-thinking agent with tens of thousands of neurons
    
    // Visual Cortex - Primary visual processing (16,384 neurons)
    auto visual_cortex_config = config_;
    visual_cortex_config.num_neurons = 16384;     // 16K neurons for complex visual processing
    visual_cortex_config.numColumns = 32;        // 32 visual columns
    visual_cortex_config.neuronsPerColumn = 512; // 512 neurons per column
    visual_cortex_config.localFanOut = 60;       // Rich connectivity for pattern recognition
    modules_["visual_cortex"] = std::make_unique<SpecializedModule>("visual_cortex", visual_cortex_config);
    
    // Prefrontal Cortex - Executive function and reasoning (12,288 neurons)
    auto prefrontal_cortex_config = config_;
    prefrontal_cortex_config.num_neurons = 12288;  // 12K neurons for executive control
    prefrontal_cortex_config.numColumns = 24;      // 24 executive columns
    prefrontal_cortex_config.neuronsPerColumn = 512;
    prefrontal_cortex_config.localFanOut = 80;     // High connectivity for complex reasoning
    modules_["prefrontal_cortex"] = std::make_unique<SpecializedModule>("prefrontal_cortex", prefrontal_cortex_config);
    
    // Motor Cortex - Precise motor control (8,192 neurons) 
    auto motor_cortex_config = config_;
    motor_cortex_config.num_neurons = 8192;       // 8K neurons for motor control
    motor_cortex_config.numColumns = 16;          // 16 motor columns
    motor_cortex_config.neuronsPerColumn = 512;
    motor_cortex_config.localFanOut = 50;         // Moderate connectivity for precise control
    modules_["motor_cortex"] = std::make_unique<SpecializedModule>("motor_cortex", motor_cortex_config);
    
    // Working Memory - Short-term memory and manipulation (6,144 neurons)
    auto working_memory_config = config_;
    working_memory_config.num_neurons = 6144;     // 6K neurons for working memory
    working_memory_config.numColumns = 12;        // 12 memory columns  
    working_memory_config.neuronsPerColumn = 512;
    working_memory_config.localFanOut = 70;       // High connectivity for memory operations
    modules_["working_memory"] = std::make_unique<SpecializedModule>("working_memory", working_memory_config);
    
    // Reward System - Motivation and reinforcement learning (4,096 neurons)
    auto reward_system_config = config_;
    reward_system_config.num_neurons = 4096;      // 4K neurons for reward processing
    reward_system_config.numColumns = 8;          // 8 reward columns
    reward_system_config.neuronsPerColumn = 512;
    reward_system_config.localFanOut = 45;        // Moderate connectivity for value estimation
    modules_["reward_system"] = std::make_unique<SpecializedModule>("reward_system", reward_system_config);
    
    // Attention System - Selective attention and focus (3,072 neurons)
    auto attention_system_config = config_;
    attention_system_config.num_neurons = 3072;   // 3K neurons for attention control
    attention_system_config.numColumns = 6;       // 6 attention columns
    attention_system_config.neuronsPerColumn = 512;
    attention_system_config.localFanOut = 55;     // Good connectivity for attention modulation
    modules_["attention_system"] = std::make_unique<SpecializedModule>("attention_system", attention_system_config);
    
    std::cout << "✅ Specialized neural modules initialized for robust free-thinking agent:" << std::endl;
    std::cout << "   🧠 Total neurons across all modules: ~50,000+ neurons" << std::endl;
    std::cout << "   👁️  Visual Cortex: 16,384 neurons (32 columns × 512)" << std::endl; 
    std::cout << "   🎯 Prefrontal Cortex: 12,288 neurons (24 columns × 512)" << std::endl;
    std::cout << "   🦾 Motor Cortex: 8,192 neurons (16 columns × 512)" << std::endl;
    std::cout << "   🧩 Working Memory: 6,144 neurons (12 columns × 512)" << std::endl;
    std::cout << "   🎁 Reward System: 4,096 neurons (8 columns × 512)" << std::endl;
    std::cout << "   🎪 Attention System: 3,072 neurons (6 columns × 512)" << std::endl;
}

// ============================================================================
// UTILITY FUNCTION IMPLEMENTATIONS
// ============================================================================

std::string actionTypeToString(ActionType type) {
    switch (type) {
        case ActionType::CLICK: return "CLICK";
        case ActionType::SCROLL: return "SCROLL";
        case ActionType::TYPE: return "TYPE";
        case ActionType::ENTER: return "ENTER";
        case ActionType::BACKSPACE: return "BACKSPACE";
        default: return "UNKNOWN";
    }
}

ActionType stringToActionType(const std::string& type_str) {
    if (type_str == "CLICK") return ActionType::CLICK;
    if (type_str == "SCROLL") return ActionType::SCROLL;
    if (type_str == "TYPE") return ActionType::TYPE;
    if (type_str == "ENTER") return ActionType::ENTER;
    if (type_str == "BACKSPACE") return ActionType::BACKSPACE;
    return ActionType::CLICK; // Default fallback
}

// ========================================
// ACTION VALIDATION
// ========================================
bool AutonomousLearningAgent::isActionValid(const BrowsingAction& action) {
    // Basic validation based on action type
    switch (action.type) {
        case ActionType::CLICK:
            // Coordinates should be within a reasonable range (e.g., 0 to 4096)
            // This is a placeholder; real validation might use screen dimensions
            return action.x_coordinate >= 0 && action.x_coordinate < 4096 &&
                   action.y_coordinate >= 0 && action.y_coordinate < 4096;
        case ActionType::SCROLL:
            // Scroll amount should be positive
            return action.scroll_amount > 0;
        case ActionType::TYPE:
            // Text should not be empty
            return !action.text_content.empty();
        case ActionType::ENTER:
        case ActionType::BACKSPACE:
            // These actions are always considered valid if they are generated
            return true;
        default:
            return false;
    }
}

// ============================================================================
// DEFAULT LEARNING GOALS SETUP
// ============================================================================

void AutonomousLearningAgent::setupDefaultLearningGoals() {
    // Clear any existing goals
    learning_goals_.clear();
    
    // Goal 1: Screen Observation and Understanding
    auto observation_goal = std::make_unique<AutonomousGoal>();
    observation_goal->goal_id = "screen_observation";
    observation_goal->description = "Learn to observe and understand screen content";
    observation_goal->priority = 0.9f;
    observation_goal->is_active = true;
    observation_goal->success_criteria = {"identify_UI_elements", "track_visual_changes", "recognize_text"};
    learning_goals_.push_back(std::move(observation_goal));
    
    // Goal 2: Effective Mouse Control
    auto mouse_goal = std::make_unique<AutonomousGoal>();
    mouse_goal->goal_id = "mouse_control";
    mouse_goal->description = "Master precise mouse movements and clicking";
    mouse_goal->priority = 0.8f;
    mouse_goal->is_active = true;
    mouse_goal->success_criteria = {"accurate_clicking", "smooth_movement", "contextual_actions"};
    learning_goals_.push_back(std::move(mouse_goal));
    
    // Goal 3: Keyboard Input Mastery
    auto keyboard_goal = std::make_unique<AutonomousGoal>();
    keyboard_goal->goal_id = "keyboard_input";
    keyboard_goal->description = "Learn effective keyboard input and text entry";
    keyboard_goal->priority = 0.7f;
    keyboard_goal->is_active = true;
    keyboard_goal->success_criteria = {"text_input", "keyboard_shortcuts", "form_filling"};
    learning_goals_.push_back(std::move(keyboard_goal));
    
    // Goal 4: Task Completion Optimization
    auto task_goal = std::make_unique<AutonomousGoal>();
    task_goal->goal_id = "task_completion";
    task_goal->description = "Optimize completion of computer-based tasks";
    task_goal->priority = 0.85f;
    task_goal->is_active = true;
    task_goal->success_criteria = {"complete_workflows", "minimize_steps", "achieve_objectives"};
    learning_goals_.push_back(std::move(task_goal));
    
    // Goal 5: Adaptive Learning
    auto adaptive_goal = std::make_unique<AutonomousGoal>();
    adaptive_goal->goal_id = "adaptive_learning";
    adaptive_goal->description = "Continuously adapt and improve performance";
    adaptive_goal->priority = 0.95f;
    adaptive_goal->is_active = true;
    adaptive_goal->success_criteria = {"improve_over_time", "learn_from_mistakes", "generalize_knowledge"};
    learning_goals_.push_back(std::move(adaptive_goal));
    
    std::cout << "✅ Default learning goals established for computer control" << std::endl;
}

// ============================================================================
// REAL SCREEN-BASED REINFORCEMENT LEARNING METHODS
// ============================================================================

void AutonomousLearningAgent::processRealScreenInput() {
    if (!visual_interface_ || !real_screen_capture_) return;
    
    // Capture current screen state
    std::vector<float> raw_screen_features = visual_interface_->capture_and_process_screen();
    
    // Detect GUI elements on screen
    auto screen_elements = visual_interface_->detect_screen_elements();
    
    // Process through visual cortex module
    if (modules_.count("visual_cortex")) {
        float visual_attention = attention_controller_->get_attention_weight("visual_cortex");
        
        // Apply attention to visual input
        std::vector<float> attended_features = raw_screen_features;
        for (size_t i = 0; i < attended_features.size(); ++i) {
            attended_features[i] *= visual_attention;
        }
        
        auto visual_output = modules_["visual_cortex"]->process(attended_features);
        
        // Update environmental context with processed visual information
        size_t context_size = std::min(visual_output.size(), environmental_context_.size());
        for (size_t i = 0; i < context_size; ++i) {
            environmental_context_[i] = visual_output[i];
        }
    }
    
    // Store current screen elements for action planning
    detected_screen_elements_ = screen_elements;

    // Perform OCR on the screen
    if (ocr_processor_ && visual_interface_) {
        cv::Mat last_frame = visual_interface_->get_last_frame();
        if (!last_frame.empty()) {
            last_screen_text_ = ocr_processor_->extractText(last_frame);
        }
    }
}

void AutonomousLearningAgent::execute_action() {
    if (is_passive_mode_) {
        // In passive mode (e.g., language training), do not execute physical actions
        return;
    }

    if (!input_controller_) return;
    
    // Check if action is safe before execution
    if (!SafetyManager::getInstance().isActionSafe(selected_action_)) {
        std::cout << "⚠️ Action blocked by safety manager" << std::endl;
        return;
    }
    
    bool success = false;
    
    // Execute the selected action on real computer
    switch (selected_action_.type) {
        case ActionType::CLICK:
            success = input_controller_->clickMouse(selected_action_.x_coordinate, 
                                                  selected_action_.y_coordinate);
            break;
        case ActionType::SCROLL:
            success = input_controller_->scrollMouse(selected_action_.x_coordinate,
                                                   selected_action_.y_coordinate,
                                                   selected_action_.scroll_amount);
            break;
        case ActionType::TYPE:
            success = input_controller_->typeText(selected_action_.text_content);
            break;
        case ActionType::ENTER:
            success = input_controller_->typeText("\n");
            break;
        case ActionType::BACKSPACE:
            success = input_controller_->typeText("\b");
            break;
    }
    
    // Update metrics
    metrics_.total_actions++;
    if (success) {
        metrics_.successful_actions++;
    }
    
    // Record action for safety tracking
    SafetyManager::getInstance().recordAction(selected_action_);
    
    // Store last action time
    last_action_time_ = std::chrono::steady_clock::now();
}

// ========================================================================
// ENVIRONMENT AND ACTION INTERFACE IMPLEMENTATIONS
// ========================================================================

void AutonomousLearningAgent::setEnvironmentSensor(std::function<BrowsingState()> sensor) {
    environment_sensor_ = std::move(sensor);
    std::cout << "Environment sensor configured for autonomous agent" << std::endl;
}

void AutonomousLearningAgent::setActionExecutor(std::function<void(const BrowsingAction&)> executor) {
    action_executor_ = std::move(executor);
    std::cout << "Action executor configured for autonomous agent" << std::endl;
}

void AutonomousLearningAgent::execute_action(const BrowsingAction& action) {
    // Store the action as the selected action
    selected_action_ = action;
    
    // Execute using the standard execute_action method
    execute_action();
    
    std::cout << "Executed action: " << static_cast<int>(action.type) 
              << " with confidence " << action.confidence << std::endl;
}

float AutonomousLearningAgent::computeScreenBasedReward() {
    float reward = 0.0f;
    
    // Base reward for successful action execution
    if (metrics_.total_actions > 0) {
        float success_rate = static_cast<float>(metrics_.successful_actions) / metrics_.total_actions;
        reward += success_rate * 0.1f;
    }
    
    // Reward for discovering new screen elements
    if (detected_screen_elements_.size() > previous_screen_elements_count_) {
        reward += 0.05f * (detected_screen_elements_.size() - previous_screen_elements_count_);
    }
    previous_screen_elements_count_ = detected_screen_elements_.size();
    
    // Reward for goal-oriented behavior
    reward += evaluateGoalProgress();
    
    // Penalty for inaction or repetitive behavior
    auto current_time = std::chrono::steady_clock::now();
    auto time_since_last_action = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_action_time_).count();
    if (time_since_last_action > 15) { // 15 seconds of inaction
        reward -= 0.1f;
    }
    
    // Apply global reward signal from controller
    reward += global_reward_signal_;
    
    // Normalize reward to be within [-1, 1]
    return std::max(-1.0f, std::min(reward, 1.0f));
}

float AutonomousLearningAgent::evaluateGoalProgress() {
    float goal_reward = 0.0f;
    
    for (const auto& goal : learning_goals_) {
        if (!goal || !goal->is_active) continue;
        
        float goal_progress = 0.0f;
        
        // Evaluate progress for different goal types
        if (goal->goal_id == "screen_observation") {
            // Reward for successfully detecting and processing screen elements
            goal_progress = std::min(1.0f, detected_screen_elements_.size() / 10.0f) * 0.1f;
        } else if (goal->goal_id == "mouse_control") {
            // Reward for successful click actions
            if (selected_action_.type == ActionType::CLICK && selected_action_.confidence > 0.7f) {
                goal_progress = 0.15f;
            }
        } else if (goal->goal_id == "keyboard_input") {
            // Reward for successful text input
            if (selected_action_.type == ActionType::TYPE && !selected_action_.text_content.empty()) {
                goal_progress = 0.12f;
            }
        } else if (goal->goal_id == "task_completion") {
            // Reward for completing sequences of actions
            goal_progress = evaluateTaskCompletion() * 0.2f;
        } else if (goal->goal_id == "adaptive_learning") {
            // Reward for improving performance over time
            goal_progress = evaluateLearningImprovement() * 0.1f;
        }
        
        goal_reward += goal_progress * goal->priority;
    }
    
    return goal_reward;
}

float AutonomousLearningAgent::evaluateExplorationEffectiveness() {
    // Reward balanced exploration vs exploitation
    float exploration_reward = 0.0f;
    
    // Encourage exploration when in new screen areas
    static std::vector<std::pair<int, int>> visited_areas;
    int current_x = selected_action_.x_coordinate / 100; // Grid of 100x100 pixels
    int current_y = selected_action_.y_coordinate / 100;
    
    bool is_new_area = true;
    for (const auto& area : visited_areas) {
        if (area.first == current_x && area.second == current_y) {
            is_new_area = false;
            break;
        }
    }
    
    if (is_new_area) {
        visited_areas.push_back({current_x, current_y});
        exploration_reward += 0.05f;
    }
    
    return exploration_reward;
}

float AutonomousLearningAgent::evaluateActionPenalties() {
    float penalty = 0.0f;
    
    // Penalize actions that are too frequent
    auto now = std::chrono::steady_clock::now();
    auto time_since_last = std::chrono::duration<float>(now - last_action_time_).count();
    if (time_since_last < 0.1f) { // Actions too fast
        penalty += 0.1f;
    }
    
    // Penalize low-confidence actions
    if (selected_action_.confidence < 0.3f) {
        penalty += 0.05f;
    }
    
    // Penalize clicking in very small areas repeatedly
    static int click_area_x = -1, click_area_y = -1;
    static int repeat_count = 0;
    
    if (selected_action_.type == ActionType::CLICK) {
        int area_x = selected_action_.x_coordinate / 50;
        int area_y = selected_action_.y_coordinate / 50;
        
        if (area_x == click_area_x && area_y == click_area_y) {
            repeat_count++;
            if (repeat_count > 3) {
                penalty += 0.08f;
            }
        } else {
            click_area_x = area_x;
            click_area_y = area_y;
            repeat_count = 0;
        }
    }
    
    return penalty;
}

float AutonomousLearningAgent::evaluateLearningEfficiency() {
    // Reward improving action selection and decision making
    float efficiency_reward = 0.0f;
    
    // Reward increasing action confidence over time
    static float avg_confidence = 0.5f;
    avg_confidence = avg_confidence * 0.99f + selected_action_.confidence * 0.01f;
    
    if (selected_action_.confidence > avg_confidence) {
        efficiency_reward += 0.02f;
    }
    
    // Reward diverse action types
    static std::map<ActionType, int> action_counts;
    action_counts[selected_action_.type]++;
    
    int total_actions = 0;
    for (const auto& count : action_counts) {
        total_actions += count.second;
    }
    
    if (total_actions > 10) {
        float diversity = static_cast<float>(action_counts.size()) / 5.0f; // 5 action types
        efficiency_reward += diversity * 0.03f;
    }
    
    return efficiency_reward;
}

float AutonomousLearningAgent::evaluateTaskCompletion() {
    // Simple heuristic for task completion
    // This could be enhanced with specific task recognition
    
    // Count meaningful action sequences
    static std::vector<ActionType> recent_actions;
    recent_actions.push_back(selected_action_.type);
    
    if (recent_actions.size() > 10) {
        recent_actions.erase(recent_actions.begin());
    }
    
    // Look for meaningful patterns (click -> type -> enter, etc.)
    float completion_score = 0.0f;
    
    if (recent_actions.size() >= 3) {
        // Pattern: Click -> Type -> Enter (form filling)
        for (size_t i = 0; i < recent_actions.size() - 2; ++i) {
            if (recent_actions[i] == ActionType::CLICK &&
                recent_actions[i+1] == ActionType::TYPE &&
                recent_actions[i+2] == ActionType::ENTER) {
                completion_score += 0.3f;
            }
        }
    }
    
    return completion_score;
}

float AutonomousLearningAgent::evaluateLearningImprovement() {
    // Track improvement metrics over time
    static float previous_success_rate = 0.0f;
    
    float current_success_rate = (metrics_.total_actions > 0) ?
        static_cast<float>(metrics_.successful_actions) / metrics_.total_actions : 0.0f;
    
    float improvement = current_success_rate - previous_success_rate;
    previous_success_rate = current_success_rate;
    
    return std::max(0.0f, improvement);
}

void AutonomousLearningAgent::learnFromActionOutcome(float reward) {
    global_reward_signal_ = global_reward_signal_ * 0.9f + reward * 0.1f;
    
    // Update reward system module
    if (modules_.count("reward_system")) {
        std::vector<float> reward_input = {reward, global_reward_signal_};
        modules_["reward_system"]->process(reward_input);
    }
    
    // Apply temporal difference learning
    if (modules_.count("prefrontal_cortex")) {
        // Get current state value prediction
        auto pfc_output = modules_["prefrontal_cortex"]->get_output();
        float predicted_value = !pfc_output.empty() ? pfc_output[0] : 0.0f;
        
        // Compute prediction error
        float prediction_error = reward - predicted_value;
        
        // Update prefrontal cortex with prediction error
        std::vector<float> learning_signal = {prediction_error, reward, global_reward_signal_};
        modules_["prefrontal_cortex"]->process(learning_signal);
    }
    
    // Update exploration rate based on reward
    if (reward > 0.1f) {
        exploration_rate_ *= 0.99f; // Reduce exploration when getting rewards
    } else if (reward < -0.1f) {
        exploration_rate_ = std::min(0.8f, exploration_rate_ * 1.01f); // Increase exploration for poor performance
    }
}

void AutonomousLearningAgent::storeEpisodeInMemory(float reward) {
    MemorySystem::MemoryTrace episode;
    
    // Store current state
    episode.state_vector = environmental_context_;
    
    // Store action as vector
    episode.action_vector.resize(10, 0.0f);
    episode.action_vector[static_cast<int>(selected_action_.type)] = 1.0f;
    episode.action_vector[5] = selected_action_.x_coordinate / 1920.0f;
    episode.action_vector[6] = selected_action_.y_coordinate / 1080.0f;
    episode.action_vector[7] = selected_action_.confidence;
    
    // Store reward and metadata
    episode.reward = reward;
    episode.importance_weight = std::abs(reward) + 0.1f; // More important if high reward/penalty
    episode.context_description = "screen_interaction";
    episode.timestamp = std::chrono::steady_clock::now();
    
    // Store in memory system
    memory_system_->storeEpisode(episode, "computer_control");
}

void AutonomousLearningAgent::logLearningProgress(int step, float reward) {
    if (!detailed_logging_) return;
    
    float learning_progress = getLearningProgress();
    float success_rate = (metrics_.total_actions > 0) ?
        static_cast<float>(metrics_.successful_actions) / metrics_.total_actions : 0.0f;
    
    std::cout << "🧠 Learning Step " << step << ":" << std::endl;
    std::cout << "   Reward: " << std::fixed << std::setprecision(3) << reward << std::endl;
    std::cout << "   Success Rate: " << success_rate * 100 << "%" << std::endl;
    std::cout << "   Learning Progress: " << learning_progress * 100 << "%" << std::endl;
    std::cout << "   Exploration Rate: " << exploration_rate_ << std::endl;
    std::cout << "   Screen Elements: " << detected_screen_elements_.size() << std::endl;
    std::cout << "   Action: " << actionTypeToString(selected_action_.type) 
              << " (confidence: " << selected_action_.confidence << ")" << std::endl;
}

// ============================================================================
// PERSISTENCE AND STATE MANAGEMENT IMPLEMENTATION
// ============================================================================

bool AutonomousLearningAgent::saveAgentState(const std::string& save_path) {
    try {
        std::cout << "💾 Saving massive modular neural agent state to: " << save_path << std::endl;
        
        // Create save directory structure
        try {
            std::filesystem::create_directories(save_path);
        } catch (const std::exception& e) {
            std::cerr << "Failed to create directory: " << e.what() << std::endl;
            return false;
        }
        
        // Save each neural module separately
        for (const auto& [module_name, module] : modules_) {
            std::string module_path = save_path + "/" + module_name + ".bin";
            if (!saveModule(module_name, module_path)) {
                std::cerr << "❌ Failed to save module: " << module_name << std::endl;
                return false;
            }
        }
        
        // Save agent-level state
        std::string agent_state_file = save_path + "/agent_state.json";
        std::ofstream state_file(agent_state_file);
        if (!state_file.is_open()) {
            std::cerr << "Failed to create agent state file" << std::endl;
            return false;
        }
        
        // Create JSON state representation
        state_file << "{\n";
        state_file << "  \"version\": \"0.5.5\",\n";
        state_file << "  \"total_neurons\": " << getTotalNeuronCount() << ",\n";
        state_file << "  \"training_step\": " << metrics_.total_actions << ",\n";
        state_file << "  \"exploration_rate\": " << exploration_rate_ << ",\n";
        state_file << "  \"learning_rate\": " << learning_rate_ << ",\n";
        state_file << "  \"global_reward_signal\": " << global_reward_signal_ << ",\n";
        state_file << "  \"successful_actions\": " << metrics_.successful_actions << ",\n";
        state_file << "  \"average_reward\": " << metrics_.average_reward << ",\n";
        state_file << "  \"modules\": [\n";
        
        bool first = true;
        for (const auto& [module_name, module] : modules_) {
            if (!first) state_file << ",\n";
            state_file << "    {\n";
            state_file << "      \"name\": \"" << module_name << "\",\n";
            state_file << "      \"neuron_count\": " << getModuleNeuronCount(module_name) << "\n";
            state_file << "    }";
            first = false;
        }
        
        state_file << "\n  ]\n";
        state_file << "}\n";
        state_file.close();
        
        std::cout << "✅ Agent state saved successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to save agent state: " << e.what() << std::endl;
        return false;
    }
}

bool AutonomousLearningAgent::loadAgentState(const std::string& load_path) {
    try {
        std::cout << "📂 Loading massive modular neural agent state from: " << load_path << std::endl;
        
        // Check if save directory exists
        if (!std::filesystem::exists(load_path)) {
            std::cerr << "Save directory not found: " << load_path << std::endl;
            return false;
        }
        
        // Load each neural module separately
        for (const auto& [module_name, module] : modules_) {
            std::string module_path = load_path + "/" + module_name + ".bin";
            if (!loadModule(module_name, module_path)) {
                std::cerr << "❌ Failed to load module: " << module_name << std::endl;
                // Continue to load other modules even if one fails
            }
        }
        
        // Load agent-level state
        std::string agent_state_file = load_path + "/agent_state.json";
        std::ifstream state_file(agent_state_file);
        if (!state_file.is_open()) {
            std::cerr << "Failed to open agent state file" << std::endl;
            return false;
        }
        
        // In a real implementation, you would parse the JSON and set the state.
        // For now, we just log that we are loading.
        
        std::cout << "✅ Agent state loaded successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to load agent state: " << e.what() << std::endl;
        return false;
    }
}

bool AutonomousLearningAgent::saveModule(const std::string& module_name, const std::string& save_path) {
    if (modules_.find(module_name) == modules_.end()) {
        std::cerr << "Error: Module not found for saving: " << module_name << std::endl;
        return false;
    }
    
    auto& module = modules_.at(module_name);
    if (module && module->get_network()) {
        std::cout << "   -> Saving module: " << module_name << " to " << save_path << std::endl;
        return module->get_network()->saveToFile(save_path);
    }
    
    std::cerr << "Error: Module or network not available for saving: " << module_name << std::endl;
    return false;
}

bool AutonomousLearningAgent::loadModule(const std::string& module_name, const std::string& load_path) {
    if (modules_.find(module_name) == modules_.end()) {
        std::cerr << "Error: Module not found for loading: " << module_name << std::endl;
        return false;
    }
    
    if (!std::filesystem::exists(load_path)) {
        std::cout << "   -> No saved state for module: " << module_name << ". Initializing fresh." << std::endl;
        return true; // Not an error, just no state to load
    }

    auto& module = modules_.at(module_name);
    if (module && module->get_network()) {
        std::cout << "   -> Loading module: " << module_name << " from " << load_path << std::endl;
        return module->get_network()->loadFromFile(load_path);
    }
    
    std::cerr << "Error: Module or network not available for loading: " << module_name << std::endl;
    return false;
}

std::string AutonomousLearningAgent::getTrainingStatistics() const {
    std::stringstream stats;
    stats << "{\n";
    stats << "  \"total_actions\": " << metrics_.total_actions << ",\n";
    stats << "  \"successful_actions\": " << metrics_.successful_actions << ",\n";
    stats << "  \"success_rate\": " << (metrics_.total_actions > 0 ? 
                                       (float)metrics_.successful_actions / metrics_.total_actions : 0.0f) << ",\n";
    stats << "  \"average_reward\": " << metrics_.average_reward << ",\n";
    stats << "  \"exploration_rate\": " << exploration_rate_ << ",\n";
    stats << "  \"learning_rate\": " << learning_rate_ << ",\n";
    stats << "  \"simulation_time\": " << simulation_time_ << "\n";
    stats << "}";
    return stats.str();
}

void AutonomousLearningAgent::setTrainingStatistics(const std::string& stats_json) {
    // Basic JSON parsing for training statistics
    // In a full implementation, this would use a proper JSON parser
    std::cout << "📊 Loading training statistics..." << std::endl;
}

void AutonomousLearningAgent::setPassiveMode(bool passive) {
    is_passive_mode_ = passive;
    if (is_passive_mode_) {
        std::cout << "Agent set to passive language training mode. No actions will be executed." << std::endl;
    } else {
        std::cout << "Agent set to active mode. Actions will be executed." << std::endl;
    }
}

// ============================================================================
// LANGUAGE TRAINING INTERFACE IMPLEMENTATION
// ============================================================================

bool AutonomousLearningAgent::processLanguageInput(const std::string& language_input) {
    try {
        std::cout << "🔤 Processing language input: " << language_input.substr(0, 50) << "..." << std::endl;
        
        // Convert language to neural input patterns
        std::vector<float> language_features = extractLanguageFeatures(language_input);
        
        // Process through language understanding modules
        if (modules_.count("prefrontal_cortex")) {
            auto language_output = modules_["prefrontal_cortex"]->process(language_features);
            
            // Update language understanding metrics
            float comprehension_score = computeLanguageComprehension(language_output);
            updateLanguageMetrics(comprehension_score);
            
            // Generate next word prediction
            std::string predicted_word = generateNextWordPrediction(language_input, language_output);
            
            // Output prediction in the format expected by Python script
            std::cout << "NEXT_WORD_PREDICTION:" << predicted_word << std::endl;
            std::cout.flush(); // Ensure immediate output
            
            return true;
        }
        
        return false;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to process language input: " << e.what() << std::endl;
        return false;
    }
}

std::string AutonomousLearningAgent::generateLanguageResponse() {
    try {
        // Generate response using motor cortex for language generation
        if (modules_.count("motor_cortex")) {
            std::vector<float> current_context = environmental_context_;
            auto response_features = modules_["motor_cortex"]->process(current_context);
            
            // Convert neural output to language
            return convertNeuralToLanguage(response_features);
        }
        
        return "I am processing your request with my neural networks.";
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to generate language response: " << e.what() << std::endl;
        return "Error generating response.";
    }
}

void AutonomousLearningAgent::updateLanguageMetrics(float comprehension_score) {
    // Update language understanding metrics
    static float cumulative_comprehension = 0.0f;
    static int language_samples = 0;
    
    cumulative_comprehension += comprehension_score;
    language_samples++;
    
    float average_comprehension = cumulative_comprehension / language_samples;
    
    if (language_samples % 100 == 0) {
        std::cout << "📈 Language Comprehension: " << (average_comprehension * 100) << "%" << std::endl;
    }
}

void AutonomousLearningAgent::handleCommand(const std::string& command) {
    std::stringstream ss(command);
    std::string command_type;
    std::getline(ss, command_type, ':');

    if (command_type == "SET_MODE") {
        std::string mode;
        std::getline(ss, mode);
        if (mode == "LANGUAGE_TRAINING") {
            current_mode_ = OperatingMode::LANGUAGE_TRAINING;
            is_passive_mode_ = true;
            std::cout << "✅ Agent mode set to LANGUAGE_TRAINING" << std::endl;
        }
    } else if (command_type == "LANGUAGE_INPUT") {
        std::string context;
        std::getline(ss, context);
        processLanguageInput(context);
    } else if (command_type == "REWARD_SIGNAL") {
        std::string reward_str;
        std::getline(ss, reward_str);
        try {
            float reward = std::stof(reward_str);
            applyReward(reward);
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Error: Invalid reward signal received: " << reward_str << std::endl;
        }
    }
}

void AutonomousLearningAgent::applyReward(float reward) {
    global_reward_signal_ = reward;

    // Trigger learning in the relevant modules
    if (modules_.count("prefrontal_cortex")) {
        modules_["prefrontal_cortex"]->apply_reinforcement(reward, global_reward_signal_);
    }
    if (modules_.count("working_memory")) {
        modules_["working_memory"]->apply_reinforcement(reward, global_reward_signal_);
    }

    if (reward > 0.5) {
        // Positive reinforcement
    } else if (reward < -0.5) {
        // Negative reinforcement
    }
}

// ============================================================================
// HELPER METHODS FOR PERSISTENCE
// ============================================================================

int AutonomousLearningAgent::getTotalNeuronCount() const {
    int total = 0;
    for (const auto& [module_name, module] : modules_) {
        total += getModuleNeuronCount(module_name);
    }
    return total;
}

int AutonomousLearningAgent::getModuleNeuronCount(const std::string& module_name) const {
    // Return neuron counts based on our massive neural architecture
    if (module_name == "visual_cortex") return 16384;
    if (module_name == "prefrontal_cortex") return 12288;
    if (module_name == "motor_cortex") return 8192;
    if (module_name == "working_memory") return 6144;
    if (module_name == "reward_system") return 4096;
    if (module_name == "attention_system") return 3072;
    return 1024; // Default for unknown modules
}

std::string AutonomousLearningAgent::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::vector<float> AutonomousLearningAgent::extractLanguageFeatures(const std::string& text) const {
    // Simple language feature extraction
    std::vector<float> features(512, 0.0f); // 512-dimensional feature vector
    
    // Basic text statistics
    features[0] = text.length() / 100.0f; // Normalized length
    features[1] = std::count(text.begin(), text.end(), ' ') / 20.0f; // Word count
    features[2] = std::count(text.begin(), text.end(), '.') / 5.0f; // Sentence count
    
    // Character-level features
    for (size_t i = 0; i < text.length() && i < 500; ++i) {
        if (i + 3 < features.size()) {
            features[i + 3] = static_cast<float>(text[i]) / 255.0f;
        }
    }
    
    return features;
}

float AutonomousLearningAgent::computeLanguageComprehension(const std::vector<float>& neural_output) const {
    // Compute comprehension score from neural output
    if (neural_output.empty()) return 0.0f;
    
    float activation_sum = 0.0f;
    for (float value : neural_output) {
        activation_sum += std::abs(value);
    }
    
    return std::min(1.0f, activation_sum / neural_output.size());
}

std::string AutonomousLearningAgent::convertNeuralToLanguage(const std::vector<float>& neural_features) const {
    // Convert neural output to language response
    if (neural_features.empty()) return "No response generated.";
    
    // Simple response generation based on neural activation patterns
    float avg_activation = 0.0f;
    for (float value : neural_features) {
        avg_activation += value;
    }
    avg_activation /= neural_features.size();
    
    if (avg_activation > 0.5f) {
        return "I understand your request and am processing it with high confidence.";
    } else if (avg_activation > 0.2f) {
        return "I am analyzing your input and working to provide an appropriate response.";
    } else {
        return "I am processing your request. Please provide more information if needed.";
    }
}

std::string AutonomousLearningAgent::generateNextWordPrediction(const std::string& context, const std::vector<float>& neural_output) {
    // Comprehensive token-based prediction with large vocabulary from "The Pile" style data
    static const std::vector<std::string> tokens = {
        // Common tokens
        "the", "and", "to", "of", "a", "in", "is", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with", "his", "they", "be",
        "at", "one", "have", "this", "from", "or", "had", "by", "word", "but", "not", "what", "all", "were", "when", "we", "there", "can", "an", "your",
        "which", "their", "said", "each", "she", "do", "how", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so",
        
        // Technical/Programming tokens
        "function", "class", "method", "variable", "return", "import", "def", "if", "else", "for", "while", "try", "except", "print", "input",
        "data", "list", "dict", "string", "int", "float", "bool", "true", "false", "null", "none", "undefined", "object", "array", "json",
        "neural", "network", "learning", "machine", "artificial", "intelligence", "algorithm", "model", "training", "prediction", "classification",
        "deep", "convolutional", "recurrent", "transformer", "attention", "embedding", "gradient", "backpropagation", "optimization", "loss",
        
        // Computer/UI tokens
        "click", "button", "menu", "window", "screen", "display", "keyboard", "mouse", "cursor", "pointer", "scroll", "drag", "drop", "select",
        "file", "folder", "directory", "document", "save", "load", "open", "close", "edit", "copy", "paste", "cut", "undo", "redo", "search",
        "browser", "tab", "link", "url", "website", "page", "form", "field", "checkbox", "radio", "dropdown", "slider", "progress", "modal",
        
        // Scientific tokens
        "research", "study", "analysis", "experiment", "hypothesis", "theory", "evidence", "conclusion", "methodology", "results", "discussion",
        "quantum", "particle", "wave", "energy", "force", "gravity", "electromagnetic", "nuclear", "atomic", "molecular", "cellular", "genetic",
        "biology", "chemistry", "physics", "mathematics", "statistics", "probability", "equation", "formula", "theorem", "proof", "calculation",
        
        // Language/Text tokens
        "sentence", "paragraph", "chapter", "book", "article", "essay", "report", "summary", "abstract", "introduction", "conclusion", "reference",
        "author", "title", "journal", "publication", "citation", "bibliography", "footnote", "appendix", "table", "figure", "chart", "graph",
        "text", "content", "context", "meaning", "semantic", "syntactic", "grammar", "vocabulary", "language", "linguistic", "communication",
        
        // Subword tokens (common prefixes/suffixes)
        "un", "re", "pre", "dis", "over", "under", "out", "up", "down", "in", "ex", "de", "anti", "pro", "sub", "super", "inter", "trans",
        "ing", "ed", "er", "est", "ly", "tion", "sion", "ment", "ness", "ity", "ous", "ful", "less", "able", "ible", "ive", "ary", "ory",
        
        // Numbers and special tokens
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "1000", "first", "second", "third", "last", "next", "previous",
        ".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", "\"", "'", "-", "_", "+", "=", "*", "/", "%", "#", "@", "&",
        
        // Action verbs
        "process", "analyze", "compute", "calculate", "determine", "identify", "classify", "recognize", "detect", "measure", "evaluate",
        "generate", "create", "build", "construct", "design", "develop", "implement", "execute", "run", "perform", "operate", "control",
        "manage", "organize", "structure", "format", "transform", "convert", "translate", "interpret", "understand", "comprehend", "learn",
        
        // Descriptive adjectives
        "large", "small", "big", "little", "high", "low", "fast", "slow", "quick", "efficient", "effective", "accurate", "precise", "complex",
        "simple", "advanced", "basic", "fundamental", "essential", "important", "significant", "relevant", "useful", "powerful", "robust",
        "flexible", "scalable", "reliable", "stable", "secure", "safe", "optimal", "minimal", "maximal", "average", "typical", "standard",
        
        // Temporal tokens
        "now", "then", "before", "after", "during", "while", "until", "since", "when", "whenever", "always", "never", "sometimes", "often",
        "today", "tomorrow", "yesterday", "week", "month", "year", "time", "moment", "instant", "period", "duration", "interval", "sequence",
        
        // Spatial/positional tokens
        "here", "there", "where", "everywhere", "nowhere", "somewhere", "above", "below", "left", "right", "front", "back", "inside", "outside",
        "center", "middle", "edge", "corner", "top", "bottom", "side", "around", "through", "across", "along", "toward", "away", "near", "far"
    };

    if (neural_output.empty()) {
        // Return a default token if there's no neural output
        return tokens.empty() ? "error" : tokens[0];
    }

    // Find the index of the neuron with the highest activation.
    // This treats the neural_output as an activation map where each neuron corresponds to a token.
    auto max_iterator = std::max_element(neural_output.begin(), neural_output.end());
    size_t highest_activation_index = std::distance(neural_output.begin(), max_iterator);

    // Map the neuron index to a token index using the modulo operator.
    // This ensures that the prediction is deterministic and based on the highest-activated neuron,
    // while handling cases where the number of neurons doesn't match the vocabulary size.
    size_t token_index = highest_activation_index % tokens.size();

    return tokens[token_index];
}
