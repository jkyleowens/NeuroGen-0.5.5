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

    // Autonomous Learning Control
    void startAutonomousLearning();
    void stopAutonomousLearning();

    // Command Handling
    void handleCommand(const std::string& command);

    // State Management
    bool saveAgentState(const std::string& save_path);
    bool loadAgentState(const std::string& load_path);

    // Module Management
    bool saveModule(const std::string& module_name, const std::string& save_path);
    bool loadModule(const std::string& module_name, const std::string& load_path);

    // Configuration
    void setPassiveMode(bool passive);
    bool isPassiveMode() const;

    // Detailed logging configuration
    void setDetailedLogging(bool detailed_logging) { detailed_logging_ = detailed_logging; }

    // Status query methods
    bool isLearningActive() const { return is_learning_active_; }
    OperatingMode getCurrentMode() const { return current_mode_; }
    std::string getCurrentTask() const { return current_task_; }
    float getLearningProgress() const { return learning_progress_; }
    
    // Learning and goal management
    void addLearningGoal(std::unique_ptr<AutonomousGoal> goal);
    void set_learning_goal(const std::string& goal);
    
    // Text processing interface
    std::vector<float> processText(const std::string& text);
    std::string generateResponse(const std::vector<float>& context);
    void trainOnText(const std::string& text, const std::string& target = "");

    // Statistics and metrics
    std::string getTrainingStatistics() const;
    void setTrainingStatistics(const std::string& stats_json);
    int getModuleNeuronCount(const std::string& module_name) const;

    // Language processing methods
    std::vector<float> extractLanguageFeatures(const std::string& text) const;
    float computeLanguageComprehension(const std::vector<float>& neural_output) const;
    std::string convertNeuralToLanguage(const std::vector<float>& neural_features) const;
    std::string generateNextWordPrediction(const std::string& context, const std::vector<float>& neural_output);

    // Learning methods
    void updateLanguageMetrics(float comprehension_score);
    void applyReward(float reward);

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    // Configuration
    NetworkConfig config_;
    OperatingMode current_mode_;
    bool is_learning_active_;
    bool detailed_logging_;
    bool is_passive_mode_;
    float simulation_time_;
    std::chrono::steady_clock::time_point last_action_time_;
    std::mt19937 gen;
    std::string save_path_;

    // Learning state variables
    std::vector<std::string> learning_goals_;
    std::vector<float> environmental_context_;
    std::vector<float> current_goals_;
    float exploration_rate_;
    std::vector<float> global_state_;
    float global_reward_signal_;
    std::string current_task_;
    float learning_progress_;

    // Core Components
    std::unique_ptr<ControllerModule> controller_module_;
    std::unique_ptr<MemorySystem> memory_system_;
    std::unique_ptr<AttentionController> attention_controller_;
    std::unique_ptr<InputController> input_controller_;
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
    
    // Core learning step
    float autonomousLearningStep(float dt);
};

#endif // AUTONOMOUS_LEARNING_AGENT_H