// ============================================================================
// AUTONOMOUS LEARNING AGENT HEADER - NLP-FOCUSED ARCHITECTURE (FIXED)
// File: include/NeuroGen/AutonomousLearningAgent.h
// ============================================================================

#ifndef AUTONOMOUS_LEARNING_AGENT_H
#define AUTONOMOUS_LEARNING_AGENT_H

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <random>
#include <mutex>
#include <atomic>
#include <functional>
#include <numeric>

// NeuroGen Framework includes
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/SpecializedModule.h"
#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/MemorySystem.h"
#include "NeuroGen/AttentionController.h"

// Forward declarations for disabled systems (no longer used)
// These are kept for compatibility but not instantiated

/**
 * @brief Natural Language Processing focused Autonomous Learning Agent
 * 
 * This agent has been simplified to focus on language processing while
 * disabling autonomous computer control capabilities. The architecture
 * consists of five core modules:
 * 
 * 1. Central Controller - Neuromodulatory control system
 * 2. Input Module - Text input processing and tokenization
 * 3. Language Processing Module - Deep language understanding
 * 4. Reasoning Module - Logical reasoning and inference
 * 5. Output Module - Converts spike data to actionable responses
 * 
 * Key Changes:
 * - DISABLED: Screen capture, mouse/keyboard control, visual processing
 * - ENABLED: Text processing, language understanding, reasoning
 * - Simplified architecture with 5 specialized modules
 * - Focus on continuous learning through language interaction
 */
class AutonomousLearningAgent {
public:
    // ========================================================================
    // CORE ARCHITECTURE ENUMS
    // ========================================================================
    
    enum class ProcessingMode {
        NLP_ONLY,           // Only natural language processing
        DISABLED_AUTONOMOUS // Autonomous control permanently disabled
    };
    
    enum class LearningPhase {
        INPUT_PROCESSING,
        LANGUAGE_UNDERSTANDING, 
        REASONING,
        RESPONSE_GENERATION,
        LEARNING_UPDATE
    };
    
    // ========================================================================
    // GOAL AND ACTION STRUCTURES
    // ========================================================================
    
    struct AutonomousGoal {
        std::string goal_id;
        std::string description;
        float priority = 0.5f;
        float current_progress = 0.0f;
        float last_logged_progress = 0.0f;
        bool is_active = true;
        std::vector<std::string> success_criteria;
        std::chrono::steady_clock::time_point creation_time;
        std::chrono::steady_clock::time_point last_update_time;
    };
    
    struct LanguageProcessingMetrics {
        float comprehension_score = 0.0f;
        float reasoning_score = 0.0f;
        float response_quality = 0.0f;
        float learning_efficiency = 0.0f;
        int processed_inputs = 0;
        int successful_responses = 0;
        std::chrono::steady_clock::time_point last_update;
    };
    
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Construct NLP-focused learning agent
     * @param config Neural network configuration
     */
    explicit AutonomousLearningAgent(const NetworkConfig& config);
    
    /**
     * @brief Destructor
     */
    ~AutonomousLearningAgent();
    
    /**
     * @brief Initialize agent for language processing
     * @param reset_model Whether to reset existing model state
     * @return Success status
     */
    bool initialize(bool reset_model = false);
    
    /**
     * @brief Update agent processing
     * @param dt Time step in seconds
     */
    void update(float dt);
    
    /**
     * @brief Shutdown agent and cleanup resources
     */
    void shutdown();
    
    // ========================================================================
    // LANGUAGE PROCESSING INTERFACE
    // ========================================================================
    
    /**
     * @brief Process natural language input
     * @param language_input Text input to process
     * @return Success status
     */
    bool processLanguageInput(const std::string& language_input);
    
    /**
     * @brief Generate language response from current state
     * @return Generated response text
     */
    std::string generateLanguageResponse();
    
    /**
     * @brief Get current language processing metrics
     * @return Metrics structure
     */
    LanguageProcessingMetrics getLanguageMetrics() const;
    
    /**
     * @brief Set processing mode (NLP only)
     * @param mode Processing mode to set
     */
    void setProcessingMode(ProcessingMode mode);
    
    /**
     * @brief Check if NLP mode is active
     * @return True if in NLP mode
     */
    bool isNLPModeActive() const { return nlp_mode_active_; }
    
    // ========================================================================
    // LEARNING CONTROL
    // ========================================================================
    
    /**
     * @brief Start continuous learning process
     */
    void startAutonomousLearning();
    
    /**
     * @brief Stop learning process
     */
    void stopAutonomousLearning();
    
    /**
     * @brief Check if learning is active
     * @return True if learning is active
     */
    bool isLearningActive() const { return is_learning_active_; }
    
    /**
     * @brief Get overall learning progress
     * @return Progress value between 0.0 and 1.0
     */
    float getLearningProgress() const;
    
    /**
     * @brief Set learning rate for language processing
     * @param rate New learning rate
     */
    void setLearningRate(float rate) { learning_rate_ = rate; }
    
    // ========================================================================
    // DISABLED AUTONOMOUS CONTROL (LEGACY INTERFACE)
    // ========================================================================
    
    /**
     * @brief DISABLED: Set passive mode (autonomous control always disabled)
     * @param passive Ignored - always passive in NLP mode
     */
    void setPassiveMode(bool passive);
    
    /**
     * @brief DISABLED: Process screen input (no-op in NLP mode)
     */
    void processRealScreenInput();
    
    /**
     * @brief DISABLED: Execute actions (no-op in NLP mode)
     */
    void execute_action();
    
    /**
     * @brief DISABLED: Execute real actions (no-op in NLP mode)
     */
    void executeRealAction();
    
    // ========================================================================
    // STATE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Save current learning state
     * @param save_path Directory path for saving
     * @return Success status
     */
    bool saveLearningState(const std::string& save_path);
    
    /**
     * @brief Load learning state from file
     * @param save_path Directory path for loading
     * @return Success status
     */
    bool loadLearningState(const std::string& save_path);
    
    /**
     * @brief Get current simulation time
     * @return Simulation time in seconds
     */
    float getSimulationTime() const { return simulation_time_; }
    
    /**
     * @brief Get brain module architecture
     * @return Shared pointer to brain architecture
     */
    std::shared_ptr<BrainModuleArchitecture> getBrainArchitecture() const {
        return brain_architecture_;
    }
    
    // ========================================================================
    // MODULE INTERFACE
    // ========================================================================
    
    /**
     * @brief Get neuron count for specific module
     * @param module_name Name of the module
     * @return Number of neurons in module
     */
    int getModuleNeuronCount(const std::string& module_name) const;
    
    /**
     * @brief Get list of active module names
     * @return Vector of module names
     */
    std::vector<std::string> getActiveModuleNames() const;
    
    /**
     * @brief Get module output
     * @param module_name Name of the module
     * @return Module output vector
     */
    std::vector<float> getModuleOutput(const std::string& module_name) const;

private:
    // ========================================================================
    // CORE CONFIGURATION
    // ========================================================================
    
    NetworkConfig config_;
    std::string save_path_;
    
    // Processing state (in correct initialization order)
    std::atomic<bool> is_learning_active_;
    std::atomic<bool> detailed_logging_;
    float simulation_time_;
    std::chrono::steady_clock::time_point last_action_time_;
    std::atomic<bool> autonomous_control_disabled_; // NEW: Always true
    std::atomic<bool> nlp_mode_active_;             // NEW: NLP mode flag
    std::atomic<bool> is_passive_mode_;
    mutable std::mt19937 gen; // FIXED: Correct mutable placement
    
    // ========================================================================
    // NLP PROCESSING ARCHITECTURE
    // ========================================================================
    
    // Core NLP modules (5 modules total)
    std::unordered_map<std::string, std::unique_ptr<SpecializedModule>> modules_;
    
    // Control and coordination systems
    std::unique_ptr<ControllerModule> controller_module_;
    std::unique_ptr<MemorySystem> memory_system_;
    std::unique_ptr<AttentionController> attention_controller_;
    std::shared_ptr<BrainModuleArchitecture> brain_architecture_; // FIXED: Changed to shared_ptr
    
    // ========================================================================
    // LEARNING AND STATE
    // ========================================================================
    
    // Learning parameters
    float exploration_rate_;
    float learning_rate_;
    float global_reward_signal_;
    
    // State vectors for language processing
    std::vector<float> environmental_context_;
    std::vector<float> global_state_;
    std::vector<float> current_goals_;
    
    // Language processing state
    std::string pending_language_input_;
    std::string last_language_input_;
    std::string current_language_response_;
    LanguageProcessingMetrics language_metrics_;

    // Vocabulary and text generation (NEW - FIX FOR BLANK OUTPUTS)
    std::vector<std::string> vocabulary_;
    std::unordered_map<std::string, int> word_to_index_;
    std::unordered_map<int, std::string> index_to_word_;
    static constexpr size_t VOCABULARY_SIZE = 1000;

    // Goals and learning objectives
    std::vector<std::unique_ptr<AutonomousGoal>> learning_goals_;
    
    // ========================================================================
    // INTERNAL NLP METHODS
    // ========================================================================
    
    // Initialization
    void initializeNLPModules();
    void initialize_nlp_modules();
    void initialize_nlp_attention_system();
    void setupNLPModuleConnections();
    void setupNLPLearningGoals();
    void initializeVocabulary(); // NEW - Initialize word vocabulary
    
    // Core processing pipeline
    float nlpLearningStep(float dt);
    void processLanguageInputPipeline(const std::string& input);
    void updateNLPModules(float dt);
    
    // Language processing utilities
    std::vector<float> tokenizeTextInput(const std::string& text);
    std::vector<float> modulateWithControl(const std::vector<float>& input,
                                          const std::vector<float>& control_signals);
    std::string generateLanguageResponseFromSpikes(const std::vector<float>& spike_data);
    void updateContextFromLanguageProcessing(const std::vector<float>& language_output,
                                           const std::vector<float>& reasoning_output);

    // NEW - Text generation from neural outputs (FIX FOR BLANK OUTPUTS)
    std::string decodeNeuralOutputToText(const std::vector<float>& neural_output, int max_words = 10);
    std::vector<std::string> selectWordsFromActivations(const std::vector<float>& activations, int num_words);
    int getWordIndexFromActivation(float activation) const;
    
    // Learning and metrics
    float computeLanguageUnderstandingReward();
    void applyNLPLearningUpdates(float reward, float dt);
    void updateNLPMetrics(float reward);
    void update_nlp_learning_goals();
    
    // Goal evaluation
    float evaluateNLPGoalProgress(const std::string& goal_id);
    float computeLanguageUnderstandingScore();
    float computeReasoningScore();
    float computeResponseQualityScore();
    
    // Utility methods
    std::string getCurrentTimestamp() const;
    std::vector<float> extractLanguageFeatures(const std::string& text) const;
    float computeLanguageComprehension(const std::vector<float>& neural_output) const;
    std::string convertNeuralToLanguage(const std::vector<float>& neural_features) const;
};

#endif // AUTONOMOUS_LEARNING_AGENT_H