// ============================================================================
// AUTONOMOUS LEARNING AGENT HEADER - UPDATED VERSION
// File: include/NeuroGen/AutonomousLearningAgent.h
// ============================================================================

#ifndef AUTONOMOUS_LEARNING_AGENT_H
#define AUTONOMOUS_LEARNING_AGENT_H

#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/Network.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/SpecializedModule.h"
#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/InputController.h"
#include "NeuroGen/MemorySystem.h"
#include "NeuroGen/AttentionController.h"
#include "ModularNeuralNetwork.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <thread>
#include <memory>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <map>
#include <filesystem>
#include <mutex>

// Forward declarations
class LanguageDatasetReader;
class TextTokenizer;

/**
 * @brief Defines the operating modes for the agent.
 */
enum class OperatingMode {
    IDLE,
    AUTONOMOUS,
    MANUAL_CONTROL
};

/**
 * @brief Action types for autonomous browsing
 */
enum class ActionType {
    CLICK = 0,
    SCROLL = 1,
    TYPE = 2,
    NAVIGATE = 3,
    WAIT = 4
};

/**
 * @brief Browsing action structure
 */
struct BrowsingAction {
    ActionType type;
    float confidence;
    int x_coordinate;
    int y_coordinate;
    int scroll_amount;
    std::string scroll_direction;
    std::string text_input;
    std::string url;
    
    BrowsingAction() : type(ActionType::WAIT), confidence(0.5f), 
                      x_coordinate(0), y_coordinate(0), scroll_amount(0) {}
};

/**
 * @brief Autonomous goal structure for learning objectives
 */
struct AutonomousGoal {
    std::string goal_id;
    std::string description;
    float priority;
    std::vector<std::string> success_criteria;
    bool is_active;
    
    AutonomousGoal() : priority(0.5f), is_active(false) {}
};

// ============================================================================
// CORE STRUCTURES
// ============================================================================

/**
 * @brief Learning task structure for tracking current processing
 */
struct LearningTask {
    std::string task_id;
    std::string description;
    std::string current_text;
    float progress;
    bool is_active;
    
    LearningTask() : progress(0.0f), is_active(false) {}
};

// ============================================================================
// AUTONOMOUS LEARNING AGENT CLASS DECLARATION
// ============================================================================

/**
 * @brief Autonomous Learning Agent for Language Processing
 * 
 * This class implements an autonomous agent capable of:
 * - Natural language processing and understanding
 * - Continuous learning from text data
 * - Modular neural network coordination
 * - Adaptive language model training
 * - Memory-based learning and inference
 */
class AutonomousLearningAgent {
public:
    // Constructor and Destructor
    AutonomousLearningAgent(const NetworkConfig& config);
    ~AutonomousLearningAgent();

    // Core Lifecycle Methods
    bool initialize(bool reset_model = false);
    void update(float dt);
    void shutdown();
    
    // Learning Control Interface
    void startAutonomousLearning();
    void stopAutonomousLearning();
    bool isLearningActive() const;
    
    // Public Learning Interface - ADDED FOR CENTRAL CONTROLLER ACCESS
    float performLearningStep(float dt);  // Public wrapper for autonomous learning
    
    // Decision System Interface - ADDED FOR CENTRAL CONTROLLER ACCESS
    std::string getCurrentDecision() const;
    float getDecisionConfidence() const;
    
    // State Query Methods
    OperatingMode getCurrentMode() const;
    float getLearningProgress() const;
    std::string getSystemStatus() const;
    
    // Language Processing Interface
    void processLanguageInput(const std::string& text_input);
    std::string generateLanguageOutput(const std::string& prompt);
    void setLanguageDataset(const std::string& dataset_path);
    
    // Training and Configuration
    void setLearningRate(float learning_rate);
    void setExplorationRate(float exploration_rate);
    void setBatchSize(size_t batch_size);
    void setMaxSequenceLength(size_t max_length);
    void setPassiveMode(bool passive);
    
    // Memory and Experience Management
    void saveExperience(const std::string& filepath);
    void loadExperience(const std::string& filepath);
    void clearMemory();
    
    // Text Generation and Analysis
    std::string predictNextWords(const std::string& context, int num_words = 5);
    float evaluateLanguageModel(const std::string& test_text);
    std::vector<std::string> getVocabulary() const;
    
    // Metrics and Statistics
    std::string getTrainingStatistics() const;
    void setTrainingStatistics(const std::string& stats_json);
    std::string getLearningMetrics() const;
    
    // Content Handling Methods
    void handleContextSwitch(const std::string& new_context);
    
    // Callback registration for external systems
    void registerStatusCallback(std::function<void(const std::string&)> callback);

private:
    // ========================================================================
    // CORE STATE VARIABLES
    // ========================================================================
    
    // Basic state
    bool is_initialized_;
    bool is_learning_active_;
    bool is_passive_mode_;
    float simulation_time_;
    OperatingMode current_mode_;
    
    // Learning parameters
    float learning_rate_;
    float exploration_rate_;
    float learning_progress_;
    float global_reward_signal_;
    
    // Current decision state - ADDED FOR DECISION TRACKING
    BrowsingAction selected_action_;
    std::string current_decision_description_;
    float current_decision_confidence_;
    
    // Language processing state
    std::vector<float> global_state_;
    std::vector<float> environmental_context_;
    std::vector<float> current_goals_;
    
    // Module architecture
    std::unique_ptr<MemorySystem> memory_system_;
    std::unique_ptr<AttentionController> attention_controller_;
    std::unique_ptr<BrainModuleArchitecture> brain_architecture_;
    std::unordered_map<std::string, std::unique_ptr<SpecializedModule>> modules_;

    // Language processing components
    std::unique_ptr<LanguageDatasetReader> dataset_reader_;
    std::unique_ptr<TextTokenizer> text_tokenizer_;
    
    // Language processing parameters
    size_t vocab_size_;
    size_t max_sequence_length_;
    size_t batch_size_;
    
    // Language processing state
    std::string current_text_input_;
    std::vector<float> current_text_features_;
    std::vector<float> current_text_target_;
    
    // Learning state
    int episode_counter_;
    
    // Thread safety
    mutable std::mutex attention_mutex_;
    mutable std::mutex decision_mutex_;  // ADDED FOR DECISION THREAD SAFETY

    struct AgentMetrics {
        int total_actions = 0;
        int successful_actions = 0;
        float average_reward = 0.0f;
    };

    AgentMetrics metrics_;

    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================
    
    void initialize_neural_modules();
    void initialize_attention_system();
    void update_learning_goals();
    void log_action(const std::string& action);
    void execute_learning_step();
    void coordinate_modules();
    void update_attention_weights();
    
    // Language processing methods
    void processTextBatch();
    float computeLanguageLearningReward(const std::vector<float>& output);
    void storeLanguageEpisode(float reward);
    
    // Decision and action methods - MOVED TO PRIVATE FOR INTERNAL USE
    void make_decision();
    void execute_action();
    void update_decision_state();  // ADDED FOR DECISION TRACKING
    
    // Core learning step - MADE PRIVATE, ACCESSED VIA PUBLIC WRAPPER
    float autonomousLearningStep(float dt);
};

#endif // AUTONOMOUS_LEARNING_AGENT_H