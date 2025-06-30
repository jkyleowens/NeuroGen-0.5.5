// AutonomousLearningAgent.h
// Continuous Reinforcement Learning Agent with Dynamic Neural Network Expansion
// Version 0.5.5 - Advanced Autonomous Learning Framework

#ifndef AUTONOMOUS_LEARNING_AGENT_H
#define AUTONOMOUS_LEARNING_AGENT_H

#include "NeuroGen/EnhancedLearningSystem.h"
#include "NeuroGen/AdvancedReinforcementLearning.h"
#include "NeuroGen/NetworkIntegration.h"
#include "NeuroGen/ModularNeuralNetwork.h"
#include "NeuroGen/DynamicNeurogenesisFramework.h"
#include "NeuroGen/DynamicSynaptogenesisFramework.h"
#include <vector>
#include <memory>
#include <functional>
#include <random>
#include <chrono>
#include <queue>
#include <map>

/**
 * Advanced Autonomous Learning Agent with Continuous RL
 * 
 * This system implements a sophisticated autonomous agent that:
 * - Continuously learns from environmental interactions
 * - Dynamically expands its neural network based on complexity
 * - Explores autonomously using intrinsic motivation
 * - Adapts its learning strategies in real-time
 * - Manages multiple concurrent learning objectives
 */

// ============================================================================
// AGENT CONFIGURATION PARAMETERS
// ============================================================================

struct AutonomousAgentConfig {
    // Core learning parameters
    float base_learning_rate = 0.001f;
    float exploration_rate = 0.1f;
    float curiosity_weight = 0.2f;
    float intrinsic_motivation_strength = 0.15f;
    
    // Network expansion parameters
    int initial_neuron_count = 512;
    int max_neuron_count = 4096;
    float expansion_threshold = 0.8f;        // Learning complexity threshold
    float pruning_threshold = 0.1f;          // Network sparsity threshold
    int expansion_batch_size = 64;           // Neurons added per expansion
    
    // Exploration parameters
    float exploration_decay = 0.995f;
    float novelty_bonus_weight = 0.1f;
    float competence_threshold = 0.7f;
    int max_exploration_steps = 1000;
    
    // Adaptation parameters
    float adaptation_speed = 0.01f;
    float meta_learning_rate = 0.0001f;
    int memory_capacity = 10000;
    float experience_replay_probability = 0.3f;
    
    // Performance monitoring
    int evaluation_interval = 100;
    float performance_window = 1000.0f;
    float complexity_growth_limit = 2.0f;
};

// ============================================================================
// LEARNING EXPERIENCE STRUCTURES
// ============================================================================

struct LearningExperience {
    std::vector<float> state;               // Environmental state
    std::vector<float> action;              // Action taken
    float reward;                           // Received reward
    std::vector<float> next_state;          // Resulting state
    float intrinsic_reward;                 // Internal curiosity reward
    float surprise_level;                   // Prediction error magnitude
    float learning_progress;                // Competence improvement
    double timestamp;                       // When experience occurred
    
    // Meta-learning information
    float difficulty_estimate;              // How challenging this experience was
    float novelty_score;                    // How novel this experience was
    std::vector<float> neural_activation;   // Neural activity pattern
};

struct AutonomousGoal {
    std::string description;                // Goal description
    std::function<float(const std::vector<float>&)> evaluation_fn; // Goal evaluation
    float current_competence = 0.0f;        // Current achievement level
    float target_competence = 0.8f;         // Target achievement level
    int priority = 1;                       // Goal priority (1-10)
    bool is_active = true;                  // Whether goal is being pursued
    
    // Learning history for this goal
    std::vector<float> performance_history;
    std::vector<float> learning_curve;
    float last_reward = 0.0f;
    int attempts = 0;
    int successes = 0;
};

// ============================================================================
// AUTONOMOUS LEARNING AGENT CLASS
// ============================================================================

class AutonomousLearningAgent {
private:
    // Core components
    std::unique_ptr<ModularNeuralNetwork> neural_network_;
    std::unique_ptr<EnhancedNetworkManager> network_manager_;
    AutonomousAgentConfig config_;
    
    // Learning and adaptation
    std::unique_ptr<EnhancedLearningSystem> learning_system_;
    std::vector<std::unique_ptr<AutonomousGoal>> goals_;
    std::queue<LearningExperience> experience_buffer_;
    std::map<std::string, float> skill_competencies_;
    
    // Exploration and curiosity
    std::mt19937 random_generator_;
    std::uniform_real_distribution<float> uniform_dist_;
    std::normal_distribution<float> exploration_noise_;
    
    // Performance tracking
    struct PerformanceMetrics {
        float average_reward = 0.0f;
        float learning_efficiency = 0.0f;
        float exploration_effectiveness = 0.0f;
        float network_complexity = 0.0f;
        float adaptation_speed = 0.0f;
        int total_experiences = 0;
        int successful_adaptations = 0;
        int network_expansions = 0;
        float cumulative_intrinsic_reward = 0.0f;
        
        // Real-time metrics
        std::vector<float> reward_history;
        std::vector<float> complexity_history;
        std::vector<float> learning_progress_history;
    } metrics_;
    
    // Environment interaction
    std::function<std::vector<float>()> environment_sensor_;
    std::function<float(const std::vector<float>&)> environment_actuator_;
    std::function<float()> environment_reward_signal_;
    
    // Internal state
    std::vector<float> current_state_;
    std::vector<float> current_action_;
    float current_exploration_rate_;
    bool is_learning_enabled_;
    bool is_autonomous_mode_;
    int simulation_step_;
    double current_time_;
    
    // Network expansion tracking
    int current_neuron_count_;
    float last_expansion_time_;
    std::vector<int> expansion_history_;
    
public:
    /**
     * Constructor: Initialize autonomous learning agent
     */
    explicit AutonomousLearningAgent(const AutonomousAgentConfig& config = AutonomousAgentConfig{});
    
    /**
     * Destructor: Cleanup resources
     */
    ~AutonomousLearningAgent();
    
    // ========================================
    // CORE AUTONOMOUS LEARNING INTERFACE
    // ========================================
    
    /**
     * Start autonomous learning mode
     * The agent will continuously interact with environment and learn
     */
    void startAutonomousLearning();
    
    /**
     * Stop autonomous learning mode
     */
    void stopAutonomousLearning();
    
    /**
     * Single step of autonomous learning
     * Returns: learning progress indicator (0.0 = no progress, 1.0 = significant progress)
     */
    float autonomousLearningStep(float dt = 1.0f);
    
    /**
     * Continuous learning loop (runs in separate thread if needed)
     */
    void continuousLearningLoop(int max_steps = -1);
    
    // ========================================
    // ENVIRONMENT INTERACTION
    // ========================================
    
    /**
     * Set environment interaction functions
     */
    void setEnvironmentSensor(std::function<std::vector<float>()> sensor);
    void setEnvironmentActuator(std::function<float(const std::vector<float>&)> actuator);
    void setEnvironmentRewardSignal(std::function<float()> reward_signal);
    
    /**
     * Manual environment interaction
     */
    std::vector<float> perceiveEnvironment();
    float executeAction(const std::vector<float>& action);
    float getEnvironmentalReward();
    
    // ========================================
    // GOAL MANAGEMENT
    // ========================================
    
    /**
     * Add learning goal
     */
    void addLearningGoal(std::unique_ptr<AutonomousGoal> goal);
    
    /**
     * Remove learning goal
     */
    void removeLearningGoal(const std::string& goal_description);
    
    /**
     * Get current goal competencies
     */
    std::map<std::string, float> getGoalCompetencies() const;
    
    /**
     * Set goal priority
     */
    void setGoalPriority(const std::string& goal_description, int priority);
    
    // ========================================
    // NEURAL NETWORK EXPANSION
    // ========================================
    
    /**
     * Check if network expansion is needed based on learning complexity
     */
    bool shouldExpandNetwork() const;
    
    /**
     * Dynamically expand neural network
     */
    bool expandNeuralNetwork(int additional_neurons = -1);
    
    /**
     * Prune unnecessary connections
     */
    void pruneNeuralNetwork();
    
    /**
     * Get current network statistics
     */
    struct NetworkStatistics {
        int neuron_count;
        int synapse_count;
        float network_density;
        float average_activity;
        float learning_capacity;
        float computational_complexity;
    };
    NetworkStatistics getNetworkStatistics() const;
    
    // ========================================
    // LEARNING ANALYSIS AND MONITORING
    // ========================================
    
    /**
     * Get performance metrics
     */
    const PerformanceMetrics& getPerformanceMetrics() const { return metrics_; }
    
    /**
     * Get learning progress for specific skill
     */
    float getLearningProgress(const std::string& skill_name) const;
    
    /**
     * Get exploration effectiveness
     */
    float getExplorationEffectiveness() const;
    
    /**
     * Get recent experiences for analysis
     */
    std::vector<LearningExperience> getRecentExperiences(int count = 100) const;
    
    /**
     * Generate learning report
     */
    std::string generateLearningReport() const;
    
    // ========================================
    // CONFIGURATION AND CONTROL
    // ========================================
    
    /**
     * Update agent configuration
     */
    void updateConfiguration(const AutonomousAgentConfig& new_config);
    
    /**
     * Enable/disable specific learning mechanisms
     */
    void setLearningEnabled(bool enabled);
    void setExplorationEnabled(bool enabled);
    void setNetworkExpansionEnabled(bool enabled);
    
    /**
     * Get current configuration
     */
    const AutonomousAgentConfig& getConfiguration() const { return config_; }
    
    // ========================================
    // SAVE/LOAD FUNCTIONALITY
    // ========================================
    
    /**
     * Save agent state including network and learning history
     */
    bool saveAgentState(const std::string& filepath) const;
    
    /**
     * Load agent state
     */
    bool loadAgentState(const std::string& filepath);
    
    /**
     * Export learning data for analysis
     */
    bool exportLearningData(const std::string& filepath) const;

private:
    // ========================================
    // INTERNAL LEARNING MECHANISMS
    // ========================================
    
    /**
     * Generate intrinsic reward based on curiosity and novelty
     */
    float computeIntrinsicReward(const LearningExperience& experience);
    
    /**
     * Update exploration strategy based on performance
     */
    void updateExplorationStrategy();
    
    /**
     * Select action using current policy with exploration
     */
    std::vector<float> selectAction(const std::vector<float>& state);
    
    /**
     * Process and store learning experience
     */
    void processLearningExperience(const LearningExperience& experience);
    
    /**
     * Update neural network based on experiences
     */
    void updateNeuralNetwork(float dt);
    
    /**
     * Evaluate goal progress and update competencies
     */
    void updateGoalProgress();
    
    /**
     * Meta-learning: adapt learning parameters based on performance
     */
    void performMetaLearning();
    
    /**
     * Experience replay for enhanced learning
     */
    void performExperienceReplay();
    
    /**
     * Analyze learning patterns and adapt strategies
     */
    void analyzeLearningPatterns();
    
    /**
     * Update performance metrics
     */
    void updatePerformanceMetrics();
    
    /**
     * Initialize neural network with base configuration
     */
    void initializeNeuralNetwork();
    
    /**
     * Setup default learning goals
     */
    void setupDefaultGoals();
    
    // ========================================
    // UTILITY FUNCTIONS
    // ========================================
    
    /**
     * Calculate complexity of current learning task
     */
    float calculateLearningComplexity() const;
    
    /**
     * Calculate novelty score for state
     */
    float calculateNoveltyScore(const std::vector<float>& state) const;
    
    /**
     * Check if experience is worth storing
     */
    bool isExperienceWorthStoring(const LearningExperience& experience) const;
    
    /**
     * Generate exploration action
     */
    std::vector<float> generateExplorationAction(const std::vector<float>& state);
    
    /**
     * Log learning event
     */
    void logLearningEvent(const std::string& event, const std::map<std::string, float>& data = {});
};

// ============================================================================
// UTILITY FUNCTIONS FOR CREATING COMMON GOALS
// ============================================================================

/**
 * Create goal for learning basic navigation
 */
std::unique_ptr<AutonomousGoal> createNavigationGoal();

/**
 * Create goal for pattern recognition
 */
std::unique_ptr<AutonomousGoal> createPatternRecognitionGoal();

/**
 * Create goal for exploration and mapping
 */
std::unique_ptr<AutonomousGoal> createExplorationGoal();

/**
 * Create goal for skill transfer learning
 */
std::unique_ptr<AutonomousGoal> createSkillTransferGoal();

/**
 * Create goal for adaptive behavior learning
 */
std::unique_ptr<AutonomousGoal> createAdaptiveBehaviorGoal();

#endif // AUTONOMOUS_LEARNING_AGENT_H
