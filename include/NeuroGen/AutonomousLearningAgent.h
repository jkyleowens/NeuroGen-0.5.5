#ifndef AUTONOMOUS_LEARNING_AGENT_H
#define AUTONOMOUS_LEARNING_AGENT_H

#include <NeuroGen/ControllerModule.h>
#include <NeuroGen/Network.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/EnhancedNeuralModule.h>
#include <NeuroGen/VisualInterface.h>
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
#include <functional>

// Forward declarations
class AttentionController;
class EnhancedLearningSystem;
class SpecializedModule;

// ============================================================================
// CORE ENUMERATIONS AND STRUCTURES
// ============================================================================

/**
 * @brief Action types that the autonomous agent can perform
 */
enum class ActionType {
    CLICK,
    SCROLL,
    TYPE,
    NAVIGATE,
    WAIT,
    OBSERVE,
    BACK,
    FORWARD,
    REFRESH
};

/**
 * @brief Browsing state representation
 */
struct BrowsingState {
    std::string current_url;
    std::vector<std::string> page_elements;
    int scroll_position;
    int window_width;
    int window_height;
    bool page_loading;
    std::vector<float> visual_features;
    
    BrowsingState() : scroll_position(0), window_width(1920), window_height(1080), page_loading(false) {}
};

/**
 * @brief Memory trace structure for episodic memory
 */
struct MemoryTrace {
    std::vector<float> state_vector;        // State representation
    std::vector<float> action_vector;       // Action taken
    float reward_received = 0.0f;           // Reward received
    float importance_weight = 1.0f;         // Importance weight
    float confidence = 1.0f;                // Confidence in memory
    std::chrono::steady_clock::time_point timestamp;
    std::string context_type;               // Context classification
    
    MemoryTrace() : timestamp(std::chrono::steady_clock::now()) {}
};

/**
 * @brief Action representation for browsing tasks
 */
struct BrowsingAction {
    ActionType type;
    std::string url;           // For NAVIGATE actions
    std::string text;          // For TYPE actions
    int x_coordinate;          // For CLICK actions
    int y_coordinate;          // For CLICK actions
    int scroll_amount;         // For SCROLL actions
    float confidence;          // Action confidence [0-1]
    float expected_reward;     // Expected reward from this action
    
    // Additional fields for action feature extraction
    struct {
        int x, y;              // Position parameters
        int scroll_amount;     // Scroll parameters
        float wait_duration;   // Wait duration
        std::string text;      // Text input
    } parameters;
    
    int priority = 5;          // Action priority
    bool requires_confirmation = false;  // Whether action needs confirmation
    std::vector<BrowsingAction> followup_actions;  // Follow-up actions
    
    BrowsingAction() : type(ActionType::WAIT), x_coordinate(0), y_coordinate(0), 
                      scroll_amount(0), confidence(0.5f), expected_reward(0.0f) {
        parameters.x = 0;
        parameters.y = 0;
        parameters.scroll_amount = 0;
        parameters.wait_duration = 0.0f;
    }
};

/**
 * @brief Autonomous agent configuration
 */
struct AutonomousAgentConfig {
    int initial_neuron_count = 256;
    int max_neuron_count = 1024;
    float exploration_rate = 0.1f;
    float curiosity_weight = 0.2f;
    float intrinsic_motivation_strength = 0.2f;
    float expansion_threshold = 0.8f;
    int max_exploration_steps = 1000;
};

/**
 * @brief Learning goal for autonomous behavior
 */
struct AutonomousGoal {
    std::string description;
    float target_competence;
    int priority;
    std::function<float(const std::vector<float>&)> evaluation_fn;
    
    AutonomousGoal() : target_competence(0.8f), priority(5) {}
};

/**
 * @brief Performance statistics structure
 */
struct PerformanceStats {
    float total_actions_taken;
    float successful_actions;
    float average_reward;
    float exploration_rate;
    std::map<ActionType, int> action_counts;
    std::map<std::string, float> context_performance;
    
    PerformanceStats() : 
        total_actions_taken(0.0f),
        successful_actions(0.0f), 
        average_reward(0.0f),
        exploration_rate(0.1f) {}
};

// ============================================================================
// UTILITY FUNCTION DECLARATIONS
// ============================================================================

/**
 * @brief Converts ActionType enum to string representation
 */
std::string actionTypeToString(ActionType type);

/**
 * @brief Converts string to ActionType enum
 */
ActionType stringToActionType(const std::string& type_str);

/**
 * @brief Computes similarity between two browsing states
 */
float computeBrowsingStateSimilarity(const BrowsingState& state1, const BrowsingState& state2);

// ============================================================================
// AUTONOMOUS LEARNING AGENT CLASS DECLARATION
// ============================================================================

/**
 * @brief Autonomous Learning Agent for Complex Decision Making
 * 
 * This class implements an autonomous agent capable of:
 * - Multi-modal perception and processing
 * - Dynamic decision making and action execution  
 * - Continuous learning from environmental feedback
 * - Modular neural network coordination
 * - Adaptive exploration and exploitation strategies
 */
class AutonomousLearningAgent {
public:
    // Type aliases for convenience
    using ActionCandidates = std::vector<BrowsingAction>;
    using ActionValues = std::vector<float>;
    using ModuleMap = std::unordered_map<std::string, std::shared_ptr<EnhancedNeuralModule>>;
    
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Construct autonomous learning agent
     * @param config Network configuration for internal modules
     */
    explicit AutonomousLearningAgent(const NetworkConfig& config);
    
    /**
     * @brief Virtual destructor
     */
    virtual ~AutonomousLearningAgent() = default;
    
    /**
     * @brief Initialize the autonomous learning agent
     * @return Success status of initialization
     */
    bool initialize();
    
    /**
     * @brief Update agent state and processing
     * @param dt Time step for update
     */
    void update(float dt);
    
    /**
     * @brief Shutdown the agent and cleanup resources
     */
    void shutdown();
    
    // ========================================================================
    // AUTONOMOUS LEARNING INTERFACE
    // ========================================================================
    
    /**
     * @brief Start autonomous learning mode
     */
    void startAutonomousLearning();
    
    /**
     * @brief Stop autonomous learning mode
     */
    void stopAutonomousLearning();
    
    /**
     * @brief Perform one step of autonomous learning
     * @param dt Time step
     * @return Learning progress indicator [0-1]
     */
    float autonomousLearningStep(float dt);
    
    /**
     * @brief Add a learning goal for the agent
     * @param goal Autonomous goal to pursue
     */
    void addLearningGoal(std::unique_ptr<AutonomousGoal> goal);
    
    // ========================================================================
    // ENVIRONMENT INTERACTION
    // ========================================================================
    
    /**
     * @brief Set environment sensor function
     * @param sensor Function that returns environmental state
     */
    void setEnvironmentSensor(std::function<BrowsingState()> sensor);
    
    /**
     * @brief Set action executor function
     * @param executor Function that executes actions in environment
     */
    void setActionExecutor(std::function<float(const BrowsingAction&)> executor);
    
    /**
     * @brief Process current environmental state
     * @param state Current browsing state
     * @return Processed state vector
     */
    std::vector<float> processEnvironmentalState(const BrowsingState& state);
    
    /**
     * @brief Generate action candidates based on current state
     * @return Vector of potential actions
     */
    std::vector<BrowsingAction> generate_action_candidates();
    
    /**
     * @brief Select best action from candidates
     * @param candidates Available action candidates
     * @return Selected action
     */
    BrowsingAction selectBestAction(const std::vector<BrowsingAction>& candidates);
    
    // ========================================================================
    // INTER-MODULE COMMUNICATION
    // ========================================================================
    
    /**
     * @brief Collect signals from modules for target module
     * @param target_module Name of target module
     * @return Combined signal vector
     */
    std::vector<float> collect_inter_module_signals(const std::string& target_module);
    
    /**
     * @brief Distribute output from source module to connected modules
     * @param source_module Name of source module
     * @param output Output vector to distribute
     */
    void distribute_module_output(const std::string& source_module, const std::vector<float>& output);
    
    // ========================================================================
    // PERFORMANCE AND MONITORING
    // ========================================================================
    
    /**
     * @brief Get current performance metrics
     * @return Performance statistics
     */
    PerformanceStats getPerformanceMetrics() const;
    
    /**
     * @brief Get network statistics
     * @return Network statistics
     */
    std::map<std::string, float> getNetworkStatistics() const;
    
    /**
     * @brief Get goal competencies
     * @return Map of goal descriptions to competency levels
     */
    std::map<std::string, float> getGoalCompetencies() const;
    
    /**
     * @brief Get exploration effectiveness
     * @return Exploration effectiveness measure
     */
    float getExplorationEffectiveness() const;
    
    /**
     * @brief Generate comprehensive learning report
     * @return Learning report string
     */
    std::string generateLearningReport() const;

private:
    // ========================================================================
    // PRIVATE MEMBER VARIABLES
    // ========================================================================
    
    NetworkConfig config_;
    ModuleMap specialized_modules_;
    std::vector<std::unique_ptr<AutonomousGoal>> learning_goals_;
    
    // Environment interaction
    std::function<BrowsingState()> environment_sensor_;
    std::function<float(const BrowsingAction&)> action_executor_;
    
    // Learning state
    bool autonomous_learning_active_;
    float exploration_rate_;
    float learning_progress_;
    
    // Performance tracking
    PerformanceStats performance_stats_;
    std::vector<MemoryTrace> recent_experiences_;
    
    // ========================================================================
    // PRIVATE HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Initialize specialized neural modules
     * @return Success status
     */
    bool initializeSpecializedModules();
    
    /**
     * @brief Update learning progress based on recent performance
     */
    void updateLearningProgress();
    
    /**
     * @brief Store experience in memory
     * @param state Current state
     * @param action Action taken
     * @param reward Reward received
     */
    void storeExperience(const BrowsingState& state, const BrowsingAction& action, float reward);
};

#endif // AUTONOMOUS_LEARNING_AGENT_H