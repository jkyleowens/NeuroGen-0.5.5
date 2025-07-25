// ============================================================================
// AUTONOMOUS LEARNING AGENT IMPLEMENTATION
// File: src/AutonomousLearningAgent.cpp
// ============================================================================

#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/SafetyManager.h"
#include "NeuroGen/AttentionController.h"
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
    : config_(config), gen(std::random_device{}()), save_path_("neural_agent_saves") {
    controller_module_ = std::make_unique<ControllerModule>();
    memory_system_ = std::make_unique<MemorySystem>(10000, 512);
    attention_controller_ = std::make_unique<AttentionController>();
    input_controller_ = std::make_unique<InputController>();
    brain_architecture_ = std::make_unique<BrainModuleArchitecture>();

    // Initialize environmental context and global state
    environmental_context_.resize(1024, 0.0f);
    global_state_.resize(2048, 0.0f);
    global_reward_signal_ = 0.0f;
    exploration_rate_ = 0.9f; // Start with high exploration
    is_learning_active_ = true;
    detailed_logging_ = false;
    is_passive_mode_ = false;
    simulation_time_ = 0.0f;
    last_action_time_ = std::chrono::steady_clock::now();
}

AutonomousLearningAgent::~AutonomousLearningAgent() {
    // Destructor logic
}

void AutonomousLearningAgent::initializeSpecializedModules() {
    // Remove visual modules, add language modules
    
    // Language Encoder Module (text → features)
    auto language_encoder = std::make_shared<EnhancedNeuralModule>(
        "language_encoder", 1024, 512  // vocab_size → hidden_size
    );
    modules_["language_encoder"] = language_encoder;
    
    // Language Processor Module (feature processing)
    auto language_processor = std::make_shared<EnhancedNeuralModule>(
        "language_processor", 512, 512  // hidden processing
    );
    modules_["language_processor"] = language_processor;
    
    // Language Decoder Module (features → output)
    auto language_decoder = std::make_shared<EnhancedNeuralModule>(
        "language_decoder", 512, 1024  // hidden_size → vocab_size
    );
    modules_["language_decoder"] = language_decoder;
    
    // Working Memory Module (for context)
    auto working_memory = std::make_shared<EnhancedNeuralModule>(
        "working_memory", 512, 256
    );
    modules_["working_memory"] = working_memory;
    
    std::cout << "✅ Initialized " << modules_.size() << " language processing modules" << std::endl;
}

bool AutonomousLearningAgent::initialize(bool reset_model) {
    std::cout << "🤖 Initializing NLP Learning Agent..." << std::endl;
    
    // Remove input_controller initialization
    // input_controller_ = std::make_unique<InputController>(); // REMOVED
    
    // Initialize language dataset reader
    dataset_reader_ = std::make_unique<LanguageDatasetReader>();
    
    // Initialize text tokenizer
    text_tokenizer_ = std::make_unique<TextTokenizer>(vocab_size_);
    
    // Initialize language modules
    initializeSpecializedModules();
    
    // Initialize brain architecture for language processing
    if (brain_architecture_) {
        brain_architecture_->initialize(vocab_size_, max_sequence_length_);
    }
    
    std::cout << "✅ NLP Agent initialization complete" << std::endl;
    return true;
}

void AutonomousLearningAgent::shutdown() {
    std::cout << "🤖 Shutting down Autonomous Learning Agent..." << std::endl;
    // Shutdown logic here
    std::cout << "✅ Agent shutdown complete." << std::endl;
}

void AutonomousLearningAgent::start() {
    // Implementation for starting the agent
    current_mode_ = OperatingMode::AUTONOMOUS;
    std::cout << "Agent started." << std::endl;
}

void AutonomousLearningAgent::stop() {
    // Implementation for stopping the agent
    current_mode_ = OperatingMode::IDLE;
    std::cout << "Agent stopped." << std::endl;
}

void AutonomousLearningAgent::run() {
    // Main loop for the agent
    while (current_mode_ == OperatingMode::AUTONOMOUS) {
        autonomousLearningStep(0.1f); // Example time step
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
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

void AutonomousLearningAgent::startAutonomousLearning() {
    if (is_learning_active_) return;

    is_learning_active_ = true;
    std::cout << "Starting autonomous learning mode..." << std::endl;
}

void AutonomousLearningAgent::stopAutonomousLearning() {
    if (!is_learning_active_) return;

    is_learning_active_ = false;
    std::cout << "Stopping autonomous learning mode..." << std::endl;
}

float AutonomousLearningAgent::autonomousLearningStep(float dt) {
    if (!is_learning_active_) return getLearningProgress();

    // === NLP-FOCUSED LEARNING CYCLE ===
    
    // Step 1: Process current text batch from dataset
    processTextBatch();
    
    // Step 2: Language understanding through neural modules
    std::vector<float> language_output;
    if (modules_.count("language_processor") > 0) {
        language_output = modules_["language_processor"]->process(current_text_features_);
    }
    
    // Step 3: Coordinate language processing modules
    coordinate_language_modules();
    
    // Step 4: Compute language learning reward
    float language_reward = computeLanguageLearningReward(language_output);
    
    // Step 5: Store language learning experience
    if (memory_system_) {
        storeLanguageEpisode(language_reward);
    }
    
    // Step 6: Update attention for next text processing
    update_language_attention_weights();

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
    // Not yet implemented
}

void AutonomousLearningAgent::set_learning_goal(const std::string& goal) {
    learning_goals_.push_back(goal);
}

void AutonomousLearningAgent::execute_action() {
    // NLP Agent: No physical actions, only text processing
    metrics_.total_actions++;
    last_action_time_ = std::chrono::steady_clock::now();
    
    // Log language processing step instead
    if (detailed_logging_) {
        std::cout << "[NLP Agent] Processed text sequence, length: " 
                  << current_text_input_.length() << " chars" << std::endl;
    }
}

void AutonomousLearningAgent::setEnvironmentSensor(std::function<BrowsingState()> sensor) {
    environment_sensor_ = sensor;
}

void AutonomousLearningAgent::setActionExecutor(std::function<void(const BrowsingAction&)> executor) {
    action_executor_ = executor;
}

bool AutonomousLearningAgent::isActionValid(const BrowsingAction& action) {
    // Basic validation, can be expanded
    if (action.type == ActionType::CLICK) {
        // Check if coordinates are within reasonable bounds
        // This requires knowledge of the screen/window size, which should be in the state
    }
    return true; // Placeholder
}

// ============================================================================
// PRIVATE HELPER METHODS
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

// ============================================================================
// DEFAULT LEARNING GOALS SETUP
// ============================================================================

void AutonomousLearningAgent::setupDefaultLearningGoals() {
    // Not implemented in this version
}

// ============================================================================
// REAL SCREEN-BASED REINFORCEMENT LEARNING METHODS
// ============================================================================

void AutonomousLearningAgent::processRealScreenInput() {
    // This method is disabled for NLP focus - no visual processing required
    return;
}

void AutonomousLearningAgent::processTextBatch() {
    if (!dataset_reader_ || !dataset_reader_->hasNextBatch()) {
        return;
    }
    
    // Get next text batch from dataset
    auto text_batch = dataset_reader_->getNextBatch(batch_size_);
    
    for (const auto& text_sample : text_batch) {
        current_text_input_ = text_sample.text;
        current_text_features_ = extractLanguageFeatures(text_sample.text);
        
        // Set target for supervised learning if available
        if (!text_sample.target.empty()) {
            current_text_target_ = extractLanguageFeatures(text_sample.target);
        }
    }
}

float AutonomousLearningAgent::computeScreenBasedReward() {
    float reward = 0.0f;

    // Reward for successful actions
    if (metrics_.total_actions > 0) {
        float success_rate = static_cast<float>(metrics_.successful_actions) / metrics_.total_actions;
        reward += success_rate * 0.1f;
    }

    // Reward for exploration and discovery (simplified)
    if (exploration_rate_ > 0.5f) {
        reward += 0.02f;
    }

    // Penalty for inaction
    auto current_time = std::chrono::steady_clock::now();
    auto time_since_last_action = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_action_time_).count();
    if (time_since_last_action > 15) {
        reward -= 0.1f;
    }

    return reward;
}

// Placeholder implementations for new private methods
float AutonomousLearningAgent::evaluateGoalProgress() { return 0.0f; }
float AutonomousLearningAgent::evaluateExplorationEffectiveness() { return 0.0f; }
float AutonomousLearningAgent::evaluateActionPenalties() { return 0.0f; }
float AutonomousLearningAgent::evaluateLearningEfficiency() { return 0.0f; }
float AutonomousLearningAgent::evaluateTaskCompletion() { return 0.0f; }
float AutonomousLearningAgent::evaluateLearningImprovement() { return 0.0f; }
void AutonomousLearningAgent::updateLanguageMetrics(float comprehension_score) {}
void AutonomousLearningAgent::applyReward(float reward) {}
int AutonomousLearningAgent::getTotalNeuronCount() const { return 0; }
int AutonomousLearningAgent::getModuleNeuronCount(const std::string& module_name) const { return 0; }
std::string AutonomousLearningAgent::getCurrentTimestamp() const { return ""; }
std::vector<float> AutonomousLearningAgent::extractLanguageFeatures(const std::string& text) const { return {}; }
float AutonomousLearningAgent::computeLanguageComprehension(const std::vector<float>& neural_output) const { return 0.0f; }
std::string AutonomousLearningAgent::convertNeuralToLanguage(const std::vector<float>& neural_features) const { return ""; }
std::string AutonomousLearningAgent::generateNextWordPrediction(const std::string& context, const std::vector<float>& neural_output) { return ""; }
bool AutonomousLearningAgent::saveAgentState(const std::string& save_path) { return true; }
bool AutonomousLearningAgent::loadAgentState(const std::string& load_path) { return true; }
bool AutonomousLearningAgent::saveModule(const std::string& module_name, const std::string& save_path) { return true; }
bool AutonomousLearningAgent::loadModule(const std::string& module_name, const std::string& load_path) { return true; }
std::string AutonomousLearningAgent::getTrainingStatistics() const { return ""; }
void AutonomousLearningAgent::setTrainingStatistics(const std::string& stats_json) {}
void AutonomousLearningAgent::setPassiveMode(bool passive) { is_passive_mode_ = passive; }

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

void AutonomousLearningAgent::execute_action() {
    // This method is disabled for NLP focus - no actions to execute
    // Just update metrics for compatibility
    metrics_.total_actions++;
    last_action_time_ = std::chrono::steady_clock::now();
    
    if (detailed_logging_) {
        std::cout << "[NLP Agent] Action execution disabled (NLP-only mode)" << std::endl;
    }
}

void AutonomousLearningAgent::coordinate_language_modules() {
    // Inter-module coordination for language processing
    
    // Language encoder → Language processor
    if (modules_.count("language_encoder") && modules_.count("language_processor")) {
        auto encoded_features = modules_["language_encoder"]->getOutput();
        modules_["language_processor"]->receiveInput(encoded_features);
    }
    
    // Language processor → Language decoder
    if (modules_.count("language_processor") && modules_.count("language_decoder")) {
        auto processed_features = modules_["language_processor"]->getOutput();
        modules_["language_decoder"]->receiveInput(processed_features);
    }
    
    // Apply attention weighting
    float attention_weight = attention_weights_.count("language_processor") > 0 ? 
                           attention_weights_["language_processor"] : 1.0f;
    
    for (auto& [name, module] : modules_) {
        if (name.find("language") != std::string::npos && module) {
            auto output = module->getOutput();
            for (float& val : output) {
                val *= attention_weight;
            }
        }
    }
}

float AutonomousLearningAgent::computeLanguageLearningReward(const std::vector<float>& output) {
    if (output.empty() || current_text_target_.empty()) {
        return 0.1f; // Small reward for processing
    }
    
    // Compute similarity between output and target
    float similarity = 0.0f;
    size_t min_size = std::min(output.size(), current_text_target_.size());
    
    for (size_t i = 0; i < min_size; ++i) {
        similarity += 1.0f - std::abs(output[i] - current_text_target_[i]);
    }
    
    return similarity / min_size;
}

void AutonomousLearningAgent::storeLanguageEpisode(float reward) {
    if (!memory_system_) return;
    
    MemorySystem::MemoryTrace trace;
    trace.episode_id = episode_counter_++;
    trace.timestamp = std::chrono::system_clock::now();
    trace.reward = reward;
    trace.environmental_state = current_text_features_;
    trace.action_type = 0; // Text processing action
    
    // Store text-specific data
    trace.text_content = current_text_input_;
    trace.language_features = current_text_features_;
    
    memory_system_->store(trace);
    
    // Update language metrics
    metrics_.total_actions++;
    metrics_.total_reward += reward;
    if (reward > 0.5f) {
        metrics_.successful_actions++;
    }
}

void AutonomousLearningAgent::update_language_attention_weights() {
    if (!attention_controller_) return;
    
    std::lock_guard<std::mutex> lock(attention_mutex_);
    
    // Compute attention based on language processing performance
    std::map<std::string, std::vector<float>> module_outputs;
    for (const auto& [name, module] : modules_) {
        if (name.find("language") != std::string::npos && module && module->isInitialized()) {
            module_outputs[name] = module->getOutput();
        }
    }
    
    // Update attention weights for language modules
    attention_weights_ = attention_controller_->computeLanguageAttentionWeights(
        module_outputs, current_text_features_
    );
}