// ============================================================================
// ENHANCED AUTONOMOUS LEARNING AGENT - NLP-FOCUSED IMPLEMENTATION (FIXED)
// File: src/AutonomousLearningAgent.cpp
// ============================================================================

#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/NetworkIntegration.h"
#include "NeuroGen/ControllerModule.h"
// SafetyManager disabled - not needed for NLP-only mode
// #include "NeuroGen/SafetyManager.h"
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

// ============================================================================
// AUTONOMOUS LEARNING AGENT IMPLEMENTATION - NLP FOCUSED (FIXED)
// ============================================================================
AutonomousLearningAgent::AutonomousLearningAgent(const NetworkConfig& config)
    : config_(config),
      save_path_("neurogen_nlp_agent_state"),
      is_learning_active_(false),
      detailed_logging_(false),
      simulation_time_(0.0f),
      last_action_time_(std::chrono::steady_clock::now()),
      autonomous_control_disabled_(true),  // NEW: Disable autonomous control by default
      nlp_mode_active_(true),              // NEW: Enable NLP mode
      is_passive_mode_(true),              // Always passive in NLP mode
      gen(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {
    
    // Initialize core NLP processing architecture
    controller_module_ = std::make_unique<ControllerModule>(ControllerConfig());
    memory_system_ = std::make_unique<MemorySystem>();
    attention_controller_ = std::make_unique<AttentionController>();
    
    // DISABLED: Remove autonomous computer control systems
    // No instantiation of visual/input components to avoid incomplete type errors
    
    // Initialize simplified brain module architecture for NLP (as shared_ptr)
    brain_architecture_ = std::make_shared<BrainModuleArchitecture>();
    
    // Initialize specialized NLP modules
    initializeNLPModules();
    
    // Initialize state vectors for language processing
    environmental_context_.resize(1024, 0.0f);  // Context for language understanding
    global_state_.resize(512, 0.0f);            // Global processing state
    current_goals_.resize(128, 0.0f);            // Language processing goals
    
    // Initialize learning parameters for NLP
    exploration_rate_ = 0.1f; // Lower exploration for language tasks
    learning_rate_ = 0.005f;  // Refined learning rate for language
    global_reward_signal_ = 0.0f;
    
    // Initialize language metrics
    language_metrics_.comprehension_score = 0.0f;
    language_metrics_.reasoning_score = 0.0f;
    language_metrics_.response_quality = 0.0f;
    language_metrics_.learning_efficiency = 0.0f;
    language_metrics_.processed_inputs = 0;
    language_metrics_.successful_responses = 0;
    language_metrics_.last_update = std::chrono::steady_clock::now();

    // NEW - Initialize vocabulary for text generation
    initializeVocabulary();

    std::cout << "✅ AutonomousLearningAgent constructed with NLP-focused architecture" << std::endl;
    std::cout << "🚫 Autonomous computer control DISABLED" << std::endl;
    std::cout << "🔤 Natural Language Processing mode ENABLED" << std::endl;
    std::cout << "📚 Vocabulary initialized with " << vocabulary_.size() << " words" << std::endl;
}

AutonomousLearningAgent::~AutonomousLearningAgent() {
    shutdown();
}

bool AutonomousLearningAgent::initialize(bool reset_model) {
    if (reset_model && std::filesystem::exists(save_path_)) {
        std::cout << "🔥 Resetting NLP model state. Deleting existing save directory..." << std::endl;
        std::filesystem::remove_all(save_path_);
    }

    std::cout << "🔧 Initializing AutonomousLearningAgent for Natural Language Processing..." << std::endl;
    
    if (!controller_module_) {
        std::cerr << "Error: Controller module not created" << std::endl;
        return false;
    }
    
    // DISABLED: No visual or input initialization for NLP mode
    // All autonomous control systems are permanently disabled
    
    // Register NLP-specific modules with attention controller
    attention_controller_->register_module("central_controller");
    attention_controller_->register_module("input_module");
    attention_controller_->register_module("language_processing");
    attention_controller_->register_module("reasoning_module");
    attention_controller_->register_module("output_module");
    
    // Initialize NLP modules and attention system
    initialize_nlp_modules();
    initialize_nlp_attention_system();
    
    // Initialize brain module architecture for NLP processing
    if (brain_architecture_) {
        if (!brain_architecture_->initializeForNLP()) {
            std::cerr << "Warning: Failed to initialize brain module architecture for NLP" << std::endl;
        } else {
            std::cout << "✅ Brain module architecture initialized for NLP successfully" << std::endl;
        }
    }
    
    // Set up NLP learning goals
    setupNLPLearningGoals();
    
    std::cout << "✅ AutonomousLearningAgent initialized for Natural Language Processing" << std::endl;
    return true;
}

void AutonomousLearningAgent::update(float dt) {
    simulation_time_ += dt;
    
    if (controller_module_) {
        controller_module_->update(dt);
    }
    
    if (is_learning_active_ && nlp_mode_active_) {
        nlpLearningStep(dt);
        update_nlp_learning_goals();
    }
}

void AutonomousLearningAgent::shutdown() {
    stopAutonomousLearning();
    
    // DISABLED: No visual or input cleanup needed
    // All autonomous control systems are permanently disabled
    
    std::cout << "AutonomousLearningAgent shutdown complete" << std::endl;
}

void AutonomousLearningAgent::startAutonomousLearning() {
    if (is_learning_active_) return;
    
    is_learning_active_ = true;
    std::cout << "🚀 Starting NLP learning mode..." << std::endl;
}

void AutonomousLearningAgent::stopAutonomousLearning() {
    if (!is_learning_active_) return;
    
    is_learning_active_ = false;
    std::cout << "⏹️  Stopping NLP learning mode..." << std::endl;
}

// ============================================================================
// NLP-SPECIFIC INITIALIZATION METHODS
// ============================================================================

void AutonomousLearningAgent::initializeNLPModules() {
    std::cout << "🔧 Initializing NLP-specific neural modules..." << std::endl;
    
    // Central Controller Module (Neuromodulatory Control)
    NetworkConfig central_config = config_;
    central_config.num_neurons = 2048;
    central_config.input_size = 512;
    central_config.output_size = 512;
    modules_["central_controller"] = std::make_unique<SpecializedModule>(
        "central_controller", central_config, "neuromodulatory_control");
    
    // Input Module (Text Input Processing)
    NetworkConfig input_config = config_;
    input_config.num_neurons = 1024;
    input_config.input_size = 1024;  // Large input for tokenized text
    input_config.output_size = 512;
    modules_["input_module"] = std::make_unique<SpecializedModule>(
        "input_module", input_config, "text_input_processing");
    
    // Language Processing Module
    NetworkConfig language_config = config_;
    language_config.num_neurons = 4096; // Largest module for complex language understanding
    language_config.input_size = 512;
    language_config.output_size = 1024;
    modules_["language_processing"] = std::make_unique<SpecializedModule>(
        "language_processing", language_config, "language_understanding");
    
    // Reasoning Module
    NetworkConfig reasoning_config = config_;
    reasoning_config.num_neurons = 2048;
    reasoning_config.input_size = 1024;
    reasoning_config.output_size = 512;
    modules_["reasoning_module"] = std::make_unique<SpecializedModule>(
        "reasoning_module", reasoning_config, "logical_reasoning");
    
    // Output Module (Spike to Action Conversion)
    NetworkConfig output_config = config_;
    output_config.num_neurons = 1024;
    output_config.input_size = 512;
    output_config.output_size = 256;
    modules_["output_module"] = std::make_unique<SpecializedModule>(
        "output_module", output_config, "spike_to_action");
    
    // Initialize all modules with validation
    bool all_modules_initialized = true;
    for (auto& [name, module] : modules_) {
        if (!module) {
            std::cerr << "❌ CRITICAL: Module '" << name << "' is null!" << std::endl;
            all_modules_initialized = false;
            continue;
        }

        if (!module->initialize()) {
            std::cerr << "❌ CRITICAL: Failed to initialize module: " << name << std::endl;
            all_modules_initialized = false;
        } else {
            std::cout << "✅ Initialized " << name << " module" << std::endl;
        }
    }

    if (!all_modules_initialized) {
        std::cerr << "❌ WARNING: Not all modules initialized successfully!" << std::endl;
        std::cerr << "   This will cause blank or incorrect outputs!" << std::endl;
    }
}

void AutonomousLearningAgent::initialize_nlp_modules() {
    // Set up inter-module connections for NLP processing pipeline
    setupNLPModuleConnections();
    
    // Initialize module-specific parameters
    if (modules_["central_controller"]) {
        modules_["central_controller"]->set_attention_weight(1.0f);
        modules_["central_controller"]->set_specialization_type("neuromodulatory_control");
    }
    
    if (modules_["language_processing"]) {
        modules_["language_processing"]->set_attention_weight(0.9f);
        modules_["language_processing"]->set_specialization_type("language_understanding");
    }
    
    if (modules_["reasoning_module"]) {
        modules_["reasoning_module"]->set_attention_weight(0.8f);
        modules_["reasoning_module"]->set_specialization_type("logical_reasoning");
    }
    
    std::cout << "✅ NLP module initialization complete" << std::endl;
}

void AutonomousLearningAgent::initialize_nlp_attention_system() {
    // Set initial attention weights for NLP processing
    attention_controller_->set_attention_weight("central_controller", 1.0f);
    attention_controller_->set_attention_weight("input_module", 0.7f);
    attention_controller_->set_attention_weight("language_processing", 0.9f);
    attention_controller_->set_attention_weight("reasoning_module", 0.8f);
    attention_controller_->set_attention_weight("output_module", 0.6f);
    
    std::cout << "✅ NLP attention system initialized" << std::endl;
}

void AutonomousLearningAgent::setupNLPModuleConnections() {
    // Create processing pipeline: Input -> Language -> Reasoning -> Output
    // Central Controller oversees all modules with neuromodulatory control
    
    if (brain_architecture_) {
        // Input to Language Processing
        brain_architecture_->createConnection("input_module", "language_processing", 0.8f);
        
        // Language Processing to Reasoning
        brain_architecture_->createConnection("language_processing", "reasoning_module", 0.7f);
        
        // Reasoning to Output
        brain_architecture_->createConnection("reasoning_module", "output_module", 0.9f);
        
        // Central Controller connections (neuromodulatory)
        brain_architecture_->createConnection("central_controller", "input_module", 0.5f);
        brain_architecture_->createConnection("central_controller", "language_processing", 0.6f);
        brain_architecture_->createConnection("central_controller", "reasoning_module", 0.7f);
        brain_architecture_->createConnection("central_controller", "output_module", 0.4f);
        
        // Feedback connections
        brain_architecture_->createConnection("reasoning_module", "language_processing", 0.3f);
        brain_architecture_->createConnection("output_module", "central_controller", 0.5f);
        
        std::cout << "✅ NLP module connections established" << std::endl;
    }
}

// ============================================================================
// NLP LEARNING AND PROCESSING METHODS
// ============================================================================

float AutonomousLearningAgent::nlpLearningStep(float dt) {
    if (!is_learning_active_ || !nlp_mode_active_) return 0.0f;

    // Process any pending language input
    if (!pending_language_input_.empty()) {
        processLanguageInputPipeline(pending_language_input_);
        pending_language_input_.clear();
    }
    
    // Update all NLP modules with neuromodulatory control
    updateNLPModules(dt);
    
    // Compute learning reward based on language understanding performance
    float language_reward = computeLanguageUnderstandingReward();
    
    // Apply learning updates
    applyNLPLearningUpdates(language_reward, dt);
    
    // Update learning metrics
    updateNLPMetrics(language_reward);
    
    return language_reward;
}

bool AutonomousLearningAgent::processLanguageInput(const std::string& language_input) {
    if (!nlp_mode_active_) {
        std::cerr << "❌ NLP mode not active" << std::endl;
        return false;
    }
    
    try {
        std::cout << "🔤 Processing language input: " << language_input.substr(0, 50) << "..." << std::endl;
        
        // Store input for processing in next update cycle
        pending_language_input_ = language_input;
        last_language_input_ = language_input;
        
        // Immediately process the input
        processLanguageInputPipeline(language_input);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to queue language input: " << e.what() << std::endl;
        return false;
    }
}

void AutonomousLearningAgent::processLanguageInputPipeline(const std::string& input) {
    // STEP 1: Input Module - Convert text to neural representation
    std::vector<float> tokenized_input = tokenizeTextInput(input);
    if (tokenized_input.empty()) {
        std::cerr << "❌ ERROR: Tokenization produced empty input!" << std::endl;
        current_language_response_ = "[ERROR: Tokenization failed]";
        return;
    }

    auto input_output = modules_["input_module"]->process(tokenized_input);
    if (input_output.empty()) {
        std::cerr << "❌ ERROR: Input module produced empty output!" << std::endl;
    }

    // STEP 2: Central Controller - Apply neuromodulatory control
    auto control_signals = modules_["central_controller"]->process(input_output);
    if (control_signals.empty()) {
        std::cerr << "❌ ERROR: Central controller produced empty output!" << std::endl;
    }

    // STEP 3: Language Processing - Deep language understanding
    std::vector<float> modulated_input = modulateWithControl(input_output, control_signals);
    auto language_output = modules_["language_processing"]->process(modulated_input);
    if (language_output.empty()) {
        std::cerr << "❌ ERROR: Language processing module produced empty output!" << std::endl;
    }

    // STEP 4: Reasoning Module - Logical reasoning and inference
    auto reasoning_output = modules_["reasoning_module"]->process(language_output);
    if (reasoning_output.empty()) {
        std::cerr << "❌ ERROR: Reasoning module produced empty output!" << std::endl;
    }

    // STEP 5: Output Module - Convert to actionable response
    auto final_output = modules_["output_module"]->process(reasoning_output);
    if (final_output.empty()) {
        std::cerr << "❌ ERROR: Output module produced empty output!" << std::endl;
        current_language_response_ = "[ERROR: No neural output generated]";
        return;
    }

    // Generate and cache language response
    current_language_response_ = generateLanguageResponseFromSpikes(final_output);

    // Validate generated response
    if (current_language_response_.empty()) {
        std::cerr << "⚠️  WARNING: Generated response is empty despite valid neural output!" << std::endl;
        current_language_response_ = "[WARNING: Response generation incomplete]";
    }

    // Update environmental context with processed information
    updateContextFromLanguageProcessing(language_output, reasoning_output);

    std::cout << "✅ Language processing pipeline complete" << std::endl;
}

std::string AutonomousLearningAgent::generateLanguageResponse() {
    if (!current_language_response_.empty()) {
        std::string response = current_language_response_;
        // Don't clear the response immediately, allow multiple calls
        return response;
    }
    
    // Generate default response if no cached response available
    return "I am processing your request through my neural language system.";
}

// ============================================================================
// NLP UTILITY METHODS
// ============================================================================

std::vector<float> AutonomousLearningAgent::tokenizeTextInput(const std::string& text) {
    std::vector<float> tokens(1024, 0.0f); // Fixed size tokenization
    
    // Simple character-level tokenization with position encoding
    for (size_t i = 0; i < text.length() && i < 512; ++i) {
        if (i < tokens.size()) {
            tokens[i] = static_cast<float>(text[i]) / 255.0f; // Character value
            tokens[i + 512] = static_cast<float>(i) / 512.0f; // Position encoding
        }
    }
    
    // Add special tokens
    if (tokens.size() > 0) tokens[0] = 1.0f; // Start token
    if (tokens.size() > 1 && text.length() < 511) tokens[text.length() + 1] = -1.0f; // End token
    
    return tokens;
}

std::vector<float> AutonomousLearningAgent::modulateWithControl(
    const std::vector<float>& input, 
    const std::vector<float>& control_signals) {
    
    std::vector<float> modulated(input.size());
    size_t control_size = control_signals.size();
    
    for (size_t i = 0; i < input.size(); ++i) {
        float control_weight = control_size > 0 ? 
            control_signals[i % control_size] : 1.0f;
        modulated[i] = input[i] * (0.5f + 0.5f * std::tanh(control_weight));
    }
    
    return modulated;
}

std::string AutonomousLearningAgent::generateLanguageResponseFromSpikes(
    const std::vector<float>& spike_data) {

    if (spike_data.empty()) {
        return "Neural processing incomplete.";
    }

    // NEW FIX: Actually decode neural output to text
    std::string decoded_text = decodeNeuralOutputToText(spike_data, 8);

    if (!decoded_text.empty()) {
        // Successfully generated text from neural output
        return decoded_text;
    }

    // Fallback: Use template responses if decoding fails
    float avg_activation = std::accumulate(spike_data.begin(), spike_data.end(), 0.0f) / spike_data.size();
    float response_confidence = std::min(1.0f, std::abs(avg_activation) * 2.0f);

    // Response generation based on spike patterns
    if (response_confidence > 0.8f) {
        return "I understand your request with high confidence and am generating a comprehensive response.";
    } else if (response_confidence > 0.5f) {
        return "I am processing your input and working to provide an appropriate response.";
    } else if (response_confidence > 0.2f) {
        return "I am analyzing your request. Could you provide more context or clarify your question?";
    } else {
        return "I need more information to properly process your request.";
    }
}

void AutonomousLearningAgent::updateContextFromLanguageProcessing(
    const std::vector<float>& language_output,
    const std::vector<float>& reasoning_output) {
    
    // Update environmental context with language understanding
    size_t lang_size = std::min(language_output.size(), environmental_context_.size() / 2);
    for (size_t i = 0; i < lang_size; ++i) {
        environmental_context_[i] = language_output[i];
    }
    
    // Update global state with reasoning output
    size_t reason_size = std::min(reasoning_output.size(), global_state_.size());
    for (size_t i = 0; i < reason_size; ++i) {
        global_state_[i] = reasoning_output[i];
    }
}

// ============================================================================
// NLP LEARNING GOALS AND METRICS
// ============================================================================

void AutonomousLearningAgent::setupNLPLearningGoals() {
    learning_goals_.clear();
    
    // Goal 1: Language Understanding
    auto understanding_goal = std::make_unique<AutonomousGoal>();
    understanding_goal->goal_id = "language_understanding";
    understanding_goal->description = "Develop deep language comprehension capabilities";
    understanding_goal->priority = 0.95f;
    understanding_goal->is_active = true;
    understanding_goal->success_criteria = {"semantic_understanding", "context_awareness", "inference_ability"};
    learning_goals_.push_back(std::move(understanding_goal));
    
    // Goal 2: Reasoning and Logic
    auto reasoning_goal = std::make_unique<AutonomousGoal>();
    reasoning_goal->goal_id = "logical_reasoning";
    reasoning_goal->description = "Master logical reasoning and inference";
    reasoning_goal->priority = 0.9f;
    reasoning_goal->is_active = true;
    reasoning_goal->success_criteria = {"logical_consistency", "inference_chains", "problem_solving"};
    learning_goals_.push_back(std::move(reasoning_goal));
    
    // Goal 3: Response Generation
    auto response_goal = std::make_unique<AutonomousGoal>();
    response_goal->goal_id = "response_generation";
    response_goal->description = "Generate coherent and contextual responses";
    response_goal->priority = 0.85f;
    response_goal->is_active = true;
    response_goal->success_criteria = {"coherent_responses", "contextual_relevance", "helpfulness"};
    learning_goals_.push_back(std::move(response_goal));
    
    std::cout << "✅ NLP learning goals established" << std::endl;
}

void AutonomousLearningAgent::update_nlp_learning_goals() {
    for (auto& goal : learning_goals_) {
        if (goal->is_active) {
            // Update goal progress based on recent performance
            float progress = evaluateNLPGoalProgress(goal->goal_id);
            goal->current_progress = std::min(1.0f, goal->current_progress + progress * 0.01f);
            
            // Log significant progress
            if (goal->current_progress > goal->last_logged_progress + 0.1f) {
                std::cout << "📈 Goal '" << goal->goal_id 
                         << "' progress: " << (goal->current_progress * 100.0f) << "%" << std::endl;
                goal->last_logged_progress = goal->current_progress;
            }
        }
    }
}

float AutonomousLearningAgent::evaluateNLPGoalProgress(const std::string& goal_id) {
    if (goal_id == "language_understanding") {
        return computeLanguageUnderstandingScore();
    } else if (goal_id == "logical_reasoning") {
        return computeReasoningScore();
    } else if (goal_id == "response_generation") {
        return computeResponseQualityScore();
    }
    return 0.0f;
}

// ============================================================================
// MISSING METHOD IMPLEMENTATIONS
// ============================================================================

void AutonomousLearningAgent::updateNLPModules(float dt) {
    for (auto& [name, module] : modules_) {
        if (module) {
            std::vector<float> empty_input; // Placeholder input
            module->update(dt, empty_input, global_reward_signal_);
        }
    }
}

float AutonomousLearningAgent::computeLanguageUnderstandingReward() {
    // Simple reward computation based on successful processing
    if (!current_language_response_.empty() && 
        current_language_response_ != "Neural processing incomplete.") {
        return 0.8f + 0.2f * static_cast<float>(gen()) / gen.max();
    }
    return 0.2f;
}

void AutonomousLearningAgent::applyNLPLearningUpdates(float reward, float dt) {
    global_reward_signal_ = reward;
    
    // Apply reward-based learning to all modules
    for (auto& [name, module] : modules_) {
        if (module) {
            module->apply_reinforcement_signal(reward);
        }
    }
}

void AutonomousLearningAgent::updateNLPMetrics(float reward) {
    language_metrics_.processed_inputs++;
    if (reward > 0.5f) {
        language_metrics_.successful_responses++;
    }
    
    // Update metrics with exponential moving average
    float alpha = 0.1f;
    language_metrics_.comprehension_score = (1.0f - alpha) * language_metrics_.comprehension_score + 
                                           alpha * computeLanguageUnderstandingScore();
    language_metrics_.reasoning_score = (1.0f - alpha) * language_metrics_.reasoning_score + 
                                       alpha * computeReasoningScore();
    language_metrics_.response_quality = (1.0f - alpha) * language_metrics_.response_quality + 
                                        alpha * computeResponseQualityScore();
    language_metrics_.learning_efficiency = (1.0f - alpha) * language_metrics_.learning_efficiency + 
                                           alpha * reward;
    
    language_metrics_.last_update = std::chrono::steady_clock::now();
}

float AutonomousLearningAgent::computeLanguageUnderstandingScore() {
    // Compute based on language processing module activity
    if (modules_.count("language_processing")) {
        auto output = modules_["language_processing"]->get_output();
        if (!output.empty()) {
            float avg_activation = std::accumulate(output.begin(), output.end(), 0.0f) / output.size();
            return std::min(1.0f, std::abs(avg_activation) * 2.0f);
        }
    }
    return 0.0f;
}

float AutonomousLearningAgent::computeReasoningScore() {
    // Compute based on reasoning module activity
    if (modules_.count("reasoning_module")) {
        auto output = modules_["reasoning_module"]->get_output();
        if (!output.empty()) {
            float avg_activation = std::accumulate(output.begin(), output.end(), 0.0f) / output.size();
            return std::min(1.0f, std::abs(avg_activation) * 1.5f);
        }
    }
    return 0.0f;
}

float AutonomousLearningAgent::computeResponseQualityScore() {
    // Simple quality score based on response characteristics
    if (current_language_response_.empty()) return 0.0f;
    
    float length_score = std::min(1.0f, current_language_response_.length() / 100.0f);
    float content_score = current_language_response_.find("confidence") != std::string::npos ? 0.8f : 0.5f;
    
    return 0.6f * length_score + 0.4f * content_score;
}

AutonomousLearningAgent::LanguageProcessingMetrics AutonomousLearningAgent::getLanguageMetrics() const {
    return language_metrics_;
}

void AutonomousLearningAgent::setProcessingMode(ProcessingMode mode) {
    // Only NLP mode is supported
    if (mode == ProcessingMode::NLP_ONLY) {
        nlp_mode_active_ = true;
        std::cout << "🔤 Processing mode set to NLP_ONLY" << std::endl;
    } else {
        std::cout << "⚠️ Only NLP_ONLY mode is supported in this build" << std::endl;
    }
}

float AutonomousLearningAgent::getLearningProgress() const {
    // Return average progress across all goals
    if (learning_goals_.empty()) return 0.0f;
    
    float total_progress = 0.0f;
    for (const auto& goal : learning_goals_) {
        total_progress += goal->current_progress;
    }
    return total_progress / learning_goals_.size();
}

int AutonomousLearningAgent::getModuleNeuronCount(const std::string& module_name) const {
    // Return neuron counts based on our neural architecture
    if (module_name == "central_controller") return 2048;
    if (module_name == "input_module") return 1024;
    if (module_name == "language_processing") return 4096;
    if (module_name == "reasoning_module") return 2048;
    if (module_name == "output_module") return 1024;
    return 0; // Unknown module
}

std::vector<std::string> AutonomousLearningAgent::getActiveModuleNames() const {
    std::vector<std::string> names;
    for (const auto& [name, module] : modules_) {
        if (module) {
            names.push_back(name);
        }
    }
    return names;
}

std::vector<float> AutonomousLearningAgent::getModuleOutput(const std::string& module_name) const {
    auto it = modules_.find(module_name);
    if (it != modules_.end() && it->second) {
        return it->second->get_output();
    }
    return std::vector<float>();
}

bool AutonomousLearningAgent::saveLearningState(const std::string& save_path) {
    try {
        // Create directory if it doesn't exist
        std::filesystem::create_directories(save_path);
        
        // Save basic state information
        std::ofstream state_file(save_path + "/agent_state.txt");
        if (state_file.is_open()) {
            state_file << "simulation_time: " << simulation_time_ << std::endl;
            state_file << "learning_active: " << is_learning_active_ << std::endl;
            state_file << "processed_inputs: " << language_metrics_.processed_inputs << std::endl;
            state_file << "successful_responses: " << language_metrics_.successful_responses << std::endl;
            state_file.close();
        }
        
        std::cout << "💾 Learning state saved to: " << save_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to save learning state: " << e.what() << std::endl;
        return false;
    }
}

bool AutonomousLearningAgent::loadLearningState(const std::string& save_path) {
    try {
        std::ifstream state_file(save_path + "/agent_state.txt");
        if (state_file.is_open()) {
            std::string line;
            while (std::getline(state_file, line)) {
                // Parse basic state information
                if (line.find("simulation_time:") != std::string::npos) {
                    simulation_time_ = std::stof(line.substr(line.find(":") + 1));
                }
                // Add more state loading as needed
            }
            state_file.close();
        }
        
        std::cout << "📂 Learning state loaded from: " << save_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to load learning state: " << e.what() << std::endl;
        return false;
    }
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
    std::vector<float> features(512, 0.0f);
    
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
    if (neural_output.empty()) return 0.0f;
    
    float activation_sum = 0.0f;
    for (float value : neural_output) {
        activation_sum += std::abs(value);
    }
    
    return std::min(1.0f, activation_sum / neural_output.size());
}

std::string AutonomousLearningAgent::convertNeuralToLanguage(const std::vector<float>& neural_features) const {
    if (neural_features.empty()) return "No response generated.";
    
    float avg_activation = std::accumulate(neural_features.begin(), neural_features.end(), 0.0f) / neural_features.size();
    
    if (avg_activation > 0.5f) {
        return "I understand your request and am processing it with high confidence.";
    } else if (avg_activation > 0.2f) {
        return "I am analyzing your input and working to provide an appropriate response.";
    } else {
        return "I am processing your request. Please provide more information if needed.";
    }
}

// ============================================================================
// VOCABULARY AND TEXT GENERATION METHODS (NEW - FIX FOR BLANK OUTPUTS)
// ============================================================================

void AutonomousLearningAgent::initializeVocabulary() {
    // Build a basic vocabulary with common words
    vocabulary_ = {
        // Common verbs
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must",
        "understand", "process", "analyze", "learn", "recognize", "know", "think",
        "see", "look", "find", "search", "explore", "discover",
        "create", "make", "build", "generate", "produce",
        "help", "assist", "support", "guide", "teach",

        // Common nouns
        "information", "data", "knowledge", "pattern", "concept", "idea", "thought",
        "language", "text", "word", "sentence", "phrase", "message", "response",
        "question", "answer", "solution", "problem", "task", "goal",
        "system", "network", "module", "agent", "model", "algorithm",
        "input", "output", "process", "result", "outcome",
        "computer", "machine", "program", "code", "software",
        "neural", "brain", "neuron", "synapse", "connection",

        // Common adjectives
        "good", "better", "best", "bad", "worse", "worst",
        "big", "small", "large", "huge", "tiny",
        "high", "low", "deep", "shallow",
        "fast", "slow", "quick", "rapid",
        "complex", "simple", "difficult", "easy",
        "important", "significant", "relevant", "useful",
        "correct", "accurate", "precise", "exact",
        "new", "old", "recent", "current", "latest",

        // Pronouns and articles
        "I", "you", "we", "they", "it", "this", "that", "these", "those",
        "the", "a", "an", "some", "any", "all", "each", "every",
        "my", "your", "our", "their", "its",

        // Prepositions
        "in", "on", "at", "to", "from", "with", "without", "by", "for",
        "about", "through", "during", "before", "after", "between", "among",
        "over", "under", "above", "below",

        // Conjunctions and common words
        "and", "or", "but", "if", "then", "because", "so", "when", "where",
        "what", "which", "who", "how", "why",
        "not", "no", "yes", "maybe", "perhaps", "possibly",

        // NLP specific
        "processing", "understanding", "comprehension", "reasoning", "inference",
        "semantic", "syntactic", "linguistic", "grammar", "vocabulary",
        "token", "embedding", "representation", "feature", "activation",
        "attention", "context", "memory", "learning", "training",

        // Response-building words
        "currently", "based", "using", "through", "via",
        "indicates", "suggests", "shows", "demonstrates",
        "appears", "seems", "looks", "sounds",
        "analyzing", "processing", "working", "computing",
        "ready", "able", "capable", "designed", "intended",

        // Filler/connecting words
        "very", "quite", "rather", "somewhat", "fairly",
        "more", "most", "less", "least", "much", "many",
        "just", "only", "even", "also", "too", "as", "well",
        "now", "then", "here", "there",

        // Punctuation as words (for sentence structure)
        ".", ",", "?", "!", ";", ":"
    };

    // Build word-to-index and index-to-word mappings
    word_to_index_.clear();
    index_to_word_.clear();

    for (size_t i = 0; i < vocabulary_.size(); ++i) {
        word_to_index_[vocabulary_[i]] = static_cast<int>(i);
        index_to_word_[static_cast<int>(i)] = vocabulary_[i];
    }

    std::cout << "📚 Vocabulary initialized with " << vocabulary_.size() << " words" << std::endl;
}

int AutonomousLearningAgent::getWordIndexFromActivation(float activation) const {
    // Map activation (0.0 to 1.0) to vocabulary index
    if (vocabulary_.empty()) return 0;

    // Use absolute value and scale to vocabulary size
    float normalized = std::abs(activation);
    int index = static_cast<int>(normalized * vocabulary_.size()) % vocabulary_.size();

    return index;
}

std::vector<std::string> AutonomousLearningAgent::selectWordsFromActivations(
    const std::vector<float>& activations, int num_words) {

    std::vector<std::string> selected_words;
    if (activations.empty() || vocabulary_.empty()) {
        return selected_words;
    }

    // Select words based on strongest activations
    std::vector<std::pair<float, int>> activation_indices;
    for (size_t i = 0; i < activations.size(); ++i) {
        activation_indices.push_back({std::abs(activations[i]), static_cast<int>(i)});
    }

    // Sort by activation strength (descending)
    std::sort(activation_indices.begin(), activation_indices.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Select top words
    for (int i = 0; i < num_words && i < static_cast<int>(activation_indices.size()); ++i) {
        int word_idx = getWordIndexFromActivation(activation_indices[i].first);
        if (word_idx >= 0 && word_idx < static_cast<int>(vocabulary_.size())) {
            selected_words.push_back(vocabulary_[word_idx]);
        }
    }

    return selected_words;
}

std::string AutonomousLearningAgent::decodeNeuralOutputToText(
    const std::vector<float>& neural_output, int max_words) {

    if (neural_output.empty() || vocabulary_.empty()) {
        return "";
    }

    // Select words from neural activations
    std::vector<std::string> words = selectWordsFromActivations(neural_output, max_words);

    if (words.empty()) {
        return "";
    }

    // Join words into a sentence
    std::string result = words[0];
    for (size_t i = 1; i < words.size(); ++i) {
        // Don't add space before punctuation
        if (words[i] == "." || words[i] == "," || words[i] == "?" ||
            words[i] == "!" || words[i] == ";" || words[i] == ":") {
            result += words[i];
        } else {
            result += " " + words[i];
        }
    }

    return result;
}

// ============================================================================
// DISABLED AUTONOMOUS CONTROL METHODS
// ============================================================================

void AutonomousLearningAgent::setPassiveMode(bool passive) {
    is_passive_mode_ = passive;
    if (passive) {
        std::cout << "🔒 Agent set to passive mode. Autonomous control remains DISABLED." << std::endl;
    } else {
        std::cout << "⚠️ Note: Autonomous computer control is permanently DISABLED in NLP mode." << std::endl;
    }
}

// DISABLED METHODS - These methods are neutered to prevent autonomous control
void AutonomousLearningAgent::processRealScreenInput() {
    // DISABLED: No screen processing in NLP mode
    std::cout << "⚠️ Screen processing disabled in NLP mode" << std::endl;
}

void AutonomousLearningAgent::execute_action() {
    // DISABLED: No action execution in NLP mode
    std::cout << "⚠️ Action execution disabled in NLP mode" << std::endl;
}

void AutonomousLearningAgent::executeRealAction() {
    // DISABLED: No real action execution disabled in NLP mode
    std::cout << "⚠️ Real action execution disabled in NLP mode" << std::endl;
}