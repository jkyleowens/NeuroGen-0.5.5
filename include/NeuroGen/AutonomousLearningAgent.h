// ============================================================================
// AUTONOMOUS LEARNING AGENT - NATURAL LANGUAGE PROCESSING FOCUSED
// File: include/NeuroGen/AutonomousLearningAgent.h
// ============================================================================

#ifndef AUTONOMOUS_LEARNING_AGENT_H
#define AUTONOMOUS_LEARNING_AGENT_H

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <atomic>
#include <mutex>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <functional>

// NeuroGen Framework Includes
#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/LanguageInterface.h"
#include "NeuroGen/MemorySystem.h"
#include "NeuroGen/AttentionController.h"
#include "NeuroGen/SafetyManager.h"
#include "NeuroGen/LearningGoal.h"

// Forward declarations
class LanguageInterface;
class MemorySystem;
class AttentionController;
class SafetyManager;
class AutonomousGoal;
class LearningGoal;

/**
 * @brief Autonomous Learning Agent for Natural Language Processing
 * 
 * This agent implements a brain-inspired architecture focused on natural language
 * understanding, generation, and continuous learning from textual interactions.
 * It uses a modular neural network approach with specialized language processing
 * modules that work together to achieve human-like language comprehension and production.
 * 
 * Key Capabilities:
 * - Real-time language understanding and generation
 * - Continuous learning from text interactions
 * - Memory consolidation for language patterns
 * - Attention-based processing for complex linguistic tasks
 * - Multi-modal language processing (future expansion ready)
 * - Safety-conscious language generation
 * - Autonomous goal-directed language learning
 */
class AutonomousLearningAgent {
public:
    // ========================================================================
    // OPERATING MODES FOR DIFFERENT LEARNING CONTEXTS
    // ========================================================================
    
    enum class OperatingMode {
        LANGUAGE_TRAINING,      // Focused language model training mode
        CONVERSATION,           // Interactive conversation mode
        TEXT_ANALYSIS,          // Document analysis and summarization
        KNOWLEDGE_ACQUISITION,  // Learning from reading materials
        CREATIVE_WRITING,       // Creative text generation mode
        RESEARCH_ASSISTANCE,    // Research and information gathering
        AUTONOMOUS_EXPLORATION  // Self-directed learning and exploration
    };
    
    enum class LearningPhase {
        INITIALIZATION,         // Setting up neural modules
        BASIC_LANGUAGE,        // Learning fundamental language patterns
        ADVANCED_COMPREHENSION, // Complex understanding development
        GENERATION_REFINEMENT,  // Improving text generation quality
        SPECIALIZED_DOMAINS,    // Learning domain-specific knowledge
        CONTINUOUS_ADAPTATION   // Ongoing learning and adaptation
    };

    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Constructor with configuration
     * @param config Initial configuration for the agent
     */
    explicit AutonomousLearningAgent(const BrainModuleArchitecture::ArchitectureConfig& config = {});
    
    /**
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~AutonomousLearningAgent();
    
    /**
     * @brief Initialize the autonomous learning agent
     * @param vocabulary_size Size of vocabulary for language processing
     * @param max_sequence_length Maximum sequence length for processing
     * @return Success status
     */
    bool initialize(size_t vocabulary_size = 50000, size_t max_sequence_length = 512);
    
    /**
     * @brief Initialize with custom language modules
     * @param module_configs Custom module configurations
     * @return Success status with error details
     */
    std::pair<bool, std::string> initializeWithCustomModules(
        const std::vector<BrainModuleArchitecture::ModuleConfig>& module_configs);
    
    /**
     * @brief Shutdown the agent and cleanup resources
     */
    void shutdown();

    // ========================================================================
    // CORE LANGUAGE PROCESSING INTERFACE
    // ========================================================================
    
    /**
     * @brief Process natural language input
     * @param text Input text to process
     * @param context Optional context for processing
     * @param learning_enabled Whether to learn from this input
     * @return Processing success status
     */
    bool processLanguageInput(const std::string& text, 
                            const std::string& context = "",
                            bool learning_enabled = true);
    
    /**
     * @brief Generate language response
     * @param context Context for generation
     * @param max_length Maximum response length
     * @param creativity Creativity level (0.0 = conservative, 1.0 = creative)
     * @return Generated text response
     */
    std::string generateLanguageResponse(const std::string& context = "",
                                       size_t max_length = 200,
                                       float creativity = 0.7f);
    
    /**
     * @brief Process conversational input with turn-taking
     * @param user_input Current user input
     * @param conversation_history Previous conversation turns
     * @return Agent's conversational response
     */
    std::string processConversation(const std::string& user_input,
                                  const std::vector<std::string>& conversation_history = {});
    
    /**
     * @brief Analyze and understand text content
     * @param text Text to analyze
     * @param analysis_type Type of analysis ("sentiment", "topics", "summary", etc.)
     * @return Analysis results
     */
    std::map<std::string, float> analyzeText(const std::string& text,
                                           const std::string& analysis_type = "comprehensive");
    
    /**
     * @brief Learn from text documents
     * @param documents Vector of text documents to learn from
     * @param learning_rate Learning rate for this session
     * @return Learning progress indicator
     */
    float learnFromDocuments(const std::vector<std::string>& documents,
                           float learning_rate = 0.001f);

    // ========================================================================
    // AUTONOMOUS LEARNING INTERFACE
    // ========================================================================
    
    /**
     * @brief Perform one step of autonomous learning
     * @param dt Time step
     * @return Learning progress indicator [0-1]
     */
    float autonomousLearningStep(float dt);
    
    /**
     * @brief Set operating mode for the agent
     * @param mode New operating mode
     * @param mode_specific_config Optional mode-specific configuration
     */
    void setOperatingMode(OperatingMode mode, 
                         const std::map<std::string, float>& mode_specific_config = {});
    
    /**
     * @brief Add a learning goal for the agent
     * @param goal Autonomous goal to pursue
     */
    void addLearningGoal(std::unique_ptr<AutonomousGoal> goal);
    
    /**
     * @brief Set learning objectives for language development
     * @param objectives Map of objective names to target scores
     */
    void setLanguageLearningObjectives(const std::map<std::string, float>& objectives);
    
    /**
     * @brief Start autonomous learning loop
     * @param duration_seconds Duration to run autonomous learning (0 = indefinite)
     */
    void startAutonomousLearning(float duration_seconds = 0.0f);
    
    /**
     * @brief Stop autonomous learning loop
     */
    void stopAutonomousLearning();

    // ========================================================================
    // CORE LANGUAGE PROCESSING METHODS
    // ========================================================================
    
    /**
     * @brief Process language input from environment
     */
    void process_language_input();
    
    /**
     * @brief Generate language output based on current context
     */
    void process_language_generation();
    
    /**
     * @brief Update working memory with current linguistic context
     */
    void update_working_memory();
    
    /**
     * @brief Update attention weights based on language processing demands
     */
    void update_attention_weights();
    
    /**
     * @brief Make decisions based on language understanding and goals
     */
    void make_decision();
    
    /**
     * @brief Learn from recent language experiences
     */
    void learn_from_experience();
    
    /**
     * @brief Transfer knowledge between language processing modules
     */
    void transfer_knowledge_between_modules();

    // ========================================================================
    // STATE MANAGEMENT AND PERSISTENCE
    // ========================================================================
    
    /**
     * @brief Save agent state including language knowledge
     * @param save_path Path to save location
     * @param include_language_model Whether to save learned language patterns
     * @return Success status
     */
    bool saveAgentState(const std::string& save_path, bool include_language_model = true);
    
    /**
     * @brief Load agent state from file
     * @param save_path Path to saved state
     * @param merge_language_knowledge Whether to merge with existing knowledge
     * @return Success status
     */
    bool loadAgentState(const std::string& save_path, bool merge_language_knowledge = true);
    
    /**
     * @brief Export learned language model
     * @param export_path Export location
     * @param format Export format ("onnx", "huggingface", "custom")
     * @return Success status with export details
     */
    std::pair<bool, std::string> exportLanguageModel(const std::string& export_path,
                                                   const std::string& format = "custom");

    // ========================================================================
    // MONITORING AND DIAGNOSTICS
    // ========================================================================
    
    /**
     * @brief Get comprehensive status report
     * @return Human-readable status report
     */
    std::string getStatusReport() const;
    
    /**
     * @brief Get learning progress metrics
     * @return Learning progress [0-1]
     */
    float getLearningProgress() const;
    
    /**
     * @brief Get current attention distribution
     * @return Map of module names to attention weights
     */
    std::map<std::string, float> getAttentionWeights() const;
    
    /**
     * @brief Get language processing statistics
     * @return Map of metric names to values
     */
    std::map<std::string, float> getLanguageProcessingStats() const;
    
    /**
     * @brief Get current language model performance
     * @return Performance metrics
     */
    std::map<std::string, float> getLanguageModelPerformance() const;
    
    /**
     * @brief Get detailed training statistics
     * @return Training statistics in JSON format
     */
    std::string getTrainingStatistics() const;
    
    /**
     * @brief Set training statistics from external source
     * @param stats_json Training statistics in JSON format
     */
    void setTrainingStatistics(const std::string& stats_json);

    // ========================================================================
    // CONFIGURATION AND CONTROL
    // ========================================================================
    
    /**
     * @brief Enable or disable detailed logging
     * @param detailed_logging Whether to enable detailed logging
     */
    void setDetailedLogging(bool detailed_logging) { detailed_logging_ = detailed_logging; }
    
    /**
     * @brief Set language processing parameters
     * @param parameters Map of parameter names to values
     */
    void setLanguageParameters(const std::map<std::string, float>& parameters);
    
    /**
     * @brief Handle external commands
     * @param command Command string to process
     */
    void handleCommand(const std::string& command);
    
    /**
     * @brief Apply reward signal for reinforcement learning
     * @param reward Reward value [-1.0, 1.0]
     * @param context Context that generated the reward
     */
    void applyReward(float reward, const std::string& context = "");

    // ========================================================================
    // LANGUAGE-SPECIFIC UTILITIES
    // ========================================================================
    
    /**
     * @brief Extract linguistic features from text
     * @param text Input text
     * @return Feature vector representing the text
     */
    std::vector<float> extractLanguageFeatures(const std::string& text) const;
    
    /**
     * @brief Convert neural output to natural language
     * @param neural_features Neural network output
     * @return Human-readable text
     */
    std::string convertNeuralToLanguage(const std::vector<float>& neural_features) const;
    
    /**
     * @brief Update language learning metrics
     * @param comprehension_score Language comprehension score
     * @param generation_quality Text generation quality score
     */
    void updateLanguageMetrics(float comprehension_score, float generation_quality = 0.0f);
    
    /**
     * @brief Generate next word predictions
     * @param context Input context
     * @param neural_output Neural network output
     * @param num_predictions Number of predictions to generate
     * @return Vector of predicted words with confidence scores
     */
    std::vector<std::pair<std::string, float>> generateWordPredictions(
        const std::string& context,
        const std::vector<float>& neural_output,
        size_t num_predictions = 5);
    
    /**
     * @brief Compute language comprehension score
     * @param neural_output Neural network output from comprehension module
     * @return Comprehension score [0.0, 1.0]
     */
    float computeLanguageComprehension(const std::vector<float>& neural_output) const;

        /**
    * @brief Check if learning is currently enabled
    * @return True if learning is enabled, false otherwise
    */
    bool isLearningEnabled() const { return is_learning_enabled_.load(); }

    /**
    * @brief Enable or disable learning
    * @param enabled Learning state to set
    */
    void setLearningEnabled(bool enabled) { is_learning_enabled_.store(enabled); }

    /**
    * @brief Check if the agent is currently running
    * @return True if running, false otherwise
    */
    bool isRunning() const { return is_running_.load(); }

    /**
    * @brief Check if the agent is in passive mode
    * @return True if in passive mode, false otherwise
    */
    bool isPassiveMode() const { return is_passive_mode_.load(); }

    /**
    * @brief Set passive mode
    * @param passive Passive mode state to set
    */
    void setPassiveMode(bool passive) { is_passive_mode_.store(passive); }

    /**
     * @brief Make autonomous decision based on current state
     */
    void make_autonomous_decision();
    
    /**
     * @brief Execute action based on current decision
     */
    void execute_action();
    
    /**
     * @brief Extract linguistic features from text
     * @param text Input text to analyze
     * @return Vector of linguistic features
     */
    std::vector<float> extractLanguageFeatures(const std::string& text);
    
    /**
     * @brief Calculate current reward signal
     * @return Current reward value
     */
    float calculateCurrentReward();

// ============================================================================
// ADDITIONAL METHODS TO ADD TO PUBLIC SECTION (if needed for external access)
// ============================================================================

    /**
     * @brief Get current decision
     * @return Current decision string
     */
    const std::string& getCurrentDecision() const { return current_decision_; }
    
    /**
     * @brief Get decision confidence
     * @return Decision confidence [0.0-1.0]
     */
    float getDecisionConfidence() const { return decision_confidence_; }
    
    /**
     * @brief Get global reward signal
     * @return Current global reward
     */
    float getGlobalReward() const { return global_reward_signal_; }
    
    /**
     * @brief Get simulation time
     * @return Current simulation time
     */
    float getSimulationTime() const { return simulation_time_; }

private:
    // ========================================================================
    // INTERNAL STATE AND COMPONENTS
    // ========================================================================
    
    // Core architecture
    std::unique_ptr<BrainModuleArchitecture> brain_architecture_;
    BrainModuleArchitecture::ArchitectureConfig config_;
    
    // Language processing components
    std::unique_ptr<LanguageInterface> language_interface_;
    std::unique_ptr<MemorySystem> memory_system_;
    std::unique_ptr<AttentionController> attention_controller_;
    std::unique_ptr<SafetyManager> safety_manager_;
    
    // Neural modules (language-focused)
    std::map<std::string, std::shared_ptr<EnhancedNeuralModule>> modules_;
    
    // Agent state
    std::atomic<bool> is_running_;
    std::atomic<bool> is_learning_enabled_;
    std::atomic<bool> is_passive_mode_;
    std::atomic<bool> detailed_logging_;
    
    OperatingMode current_mode_;
    LearningPhase current_learning_phase_;
    
    // Language processing state
    std::string current_language_response_;
    std::string current_decision_;
    float decision_confidence_;
    std::vector<float> environmental_context_;  // Now contains language context
    std::vector<float> current_goals_;
    std::vector<float> global_state_;
    
    // Learning state
    float global_reward_signal_;
    float simulation_time_;
    std::vector<std::unique_ptr<AutonomousGoal>> learning_goals_;
    std::map<std::string, float> language_learning_objectives_;
    
    // Performance tracking
    std::map<std::string, float> language_metrics_;
    std::chrono::high_resolution_clock::time_point last_update_time_;
    
    // Threading and synchronization
    std::unique_ptr<std::thread> autonomous_learning_thread_;
    std::mutex state_mutex_;
    std::condition_variable learning_cv_;
    std::atomic<bool> stop_learning_flag_;

    // ========================================================================
    // INTERNAL HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Initialize language processing modules
     */
    void initialize_neural_modules();
    
    /**
     * @brief Initialize attention system for language processing
     */
    void initialize_attention_system();
    
    /**
     * @brief Initialize language interface
     * @return Success status
     */
    bool initializeLanguageInterface();
    
    /**
     * @brief Shutdown language interface
     */
    void shutdownLanguageInterface();
    
    /**
     * @brief Update learning goals based on progress
     */
    void update_learning_goals();
    
    /**
     * @brief Consolidate learning across modules
     */
    void consolidate_learning();
    
    /**
     * @brief Update global cognitive state
     */
    void update_global_state();
    
    /**
     * @brief Gather input for specific neural module
     * @param target_module Target module name
     * @return Combined input vector
     */
    std::vector<float> gather_module_input(const std::string& target_module);
    
    /**
     * @brief Distribute module output to connected modules
     * @param source_module Source module name
     * @param output Output vector to distribute
     */
    void distribute_module_output(const std::string& source_module, 
                                const std::vector<float>& output);
    
    /**
     * @brief Get neuron count for specific module
     * @param module_name Module name
     * @return Number of neurons in the module
     */
    int getModuleNeuronCount(const std::string& module_name) const;
    
    /**
     * @brief Get total neuron count across all modules
     * @return Total number of neurons
     */
    int getTotalNeuronCount() const;
    
    /**
     * @brief Get current timestamp string
     * @return Formatted timestamp
     */
    std::string getCurrentTimestamp() const;
    
    /**
     * @brief Autonomous learning main loop
     */
    void autonomousLearningLoop();
    
    /**
     * @brief Process language-specific attention updates
     */
    void processLanguageAttention();
    
    /**
     * @brief Validate current language processing state
     * @return Validation success status
     */
    bool validateLanguageProcessingState() const;
    
    /**
     * @brief Update attention weights for language modules
     * @param weights Vector of attention weights
     */
    void updateAttentionWeights(const std::vector<float>& weights);
    
    /**
     * @brief Consolidate attention patterns for better language processing
     */
    void consolidateAttentionPatterns();
};

#endif // AUTONOMOUS_LEARNING_AGENT_H