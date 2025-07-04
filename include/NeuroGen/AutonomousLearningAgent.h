// ============================================================================
// AUTONOMOUS LEARNING AGENT HEADER
// File: include/NeuroGen/AutonomousLearningAgent.h
// ============================================================================

#ifndef AUTONOMOUS_LEARNING_AGENT_H
#define AUTONOMOUS_LEARNING_AGENT_H

#include "NeuroGen/ControllerModule.h"
#include "NeuroGen/ScreenElement.h"
#include "NeuroGen/Network.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/VisualInterface.h"
#include "NeuroGen/SpecializedModule.h"
#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/RealScreenCapture.h"
#include "NeuroGen/InputController.h"
#include "NeuroGen/OCRProcessor.h"
#include "NeuroGen/GUIElementDetector.h"
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

// Forward declarations for remaining classes
// class EnhancedLearningSystem; // Removed to avoid CUDA dependencies

/**
 * @brief Memory system for autonomous learning agent
 */
class MemorySystem {
public:
    /**
     * @brief Memory trace structure for episodic memory
     */
    struct MemoryTrace {
        std::vector<float> state_vector;
        std::vector<float> action_vector;
        float reward_received = 0.0f;
        float reward = 0.0f;  // Alias for backward compatibility
        float importance_weight = 0.5f;
        float temporal_discount = 0.99f;
        bool is_consolidated = false;
        std::chrono::steady_clock::time_point timestamp;
        std::string episode_context;
        std::string context_description;  // Alias for backward compatibility
        int episode_id = -1;
        
        MemoryTrace() {
            timestamp = std::chrono::steady_clock::now();
        }
    };
    
    /**
     * @brief Episodic memory cluster structure
     */
    struct EpisodicCluster {
        std::vector<MemoryTrace> episodes;
        std::vector<float> prototype_state;
        std::string context_type;
        float cluster_coherence = 1.0f;
        uint32_t access_count = 0;
        std::chrono::steady_clock::time_point last_accessed;
    };
    
    // Constructor
    MemorySystem(size_t max_episodes = 10000, size_t working_capacity = 100);
    
    // Core memory operations
    void storeEpisode(const MemoryTrace& episode, const std::string& context = "default");
    std::vector<MemoryTrace> retrieveSimilarEpisodes(const std::vector<float>& current_state, 
                                                      const std::string& context = "",
                                                      size_t max_results = 10);
    void consolidateMemories();
    void updateWorkingMemory(const MemoryTrace& trace);
    
    // Working memory operations
    std::vector<float> get_working_memory() const;
    void update_working_memory(const std::vector<float>& new_memory);
    
    // Memory management
    void forgetOldEpisodes();
    void strengthenMemory(const std::string& context, int episode_id, float strength_boost);
    
    // Skill and procedural memory
    float getSkillLevel(const std::string& skill_name) const;
    void updateSkillLevel(const std::string& skill_name, float performance);
    
    // Memory statistics and introspection
    std::vector<std::string> getKnownContexts() const;
    size_t getEpisodeCount(const std::string& context = "") const;
    
    // Persistence
    bool saveMemoryState(const std::string& filename) const;
    bool loadMemoryState(const std::string& filename);
    
    // Memory similarity functions
    std::vector<MemoryTrace> retrieve_similar_episodes(const std::vector<float>& state, size_t max_results);
    void store_episode(const std::vector<float>& state, const std::vector<float>& action, 
                      float reward, float confidence);
    
    // Public access to episodic memory for debugging/introspection
    const std::unordered_map<std::string, EpisodicCluster>& get_episodic_memory() const { return episodic_memory_; }
    size_t get_episodic_memory_size() const { return episodic_memory_.size(); }

private:
    // Memory storage
    std::unordered_map<std::string, EpisodicCluster> episodic_memory_;
    std::unordered_map<std::string, float> skill_memory_;
    std::vector<MemoryTrace> working_memory_;
    std::vector<float> current_working_memory_;
    
    // Configuration
    size_t max_episodes_per_cluster_;
    size_t max_total_episodes_;
    size_t working_memory_capacity_;
    float consolidation_threshold_;
    float forgetting_rate_;
    
    // Helper methods
    void searchClusterForSimilarEpisodes(const EpisodicCluster& cluster,
                                       const std::vector<float>& current_state,
                                       std::vector<MemoryTrace>& results);
    void updateClusterCoherence(EpisodicCluster& cluster);
    void organizeMemoryStructure();
    float computeCosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
};

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
 * @brief Scroll directions for scrolling actions
 */
enum class ScrollDirection {
    UP,
    DOWN,
    LEFT,
    RIGHT
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
 * @brief Action representation for browsing tasks
 */
struct BrowsingAction {
    ActionType type;
    std::string target_url;         // For NAVIGATE actions
    std::string text_content;       // For TYPE actions
    int x_coordinate;               // For CLICK actions
    int y_coordinate;               // For CLICK actions
    int scroll_amount;              // For SCROLL actions
    ScrollDirection scroll_direction; // For SCROLL actions
    float confidence;               // Action confidence [0-1]
    float expected_reward;          // Expected reward for this action
    std::string description;        // Human-readable description
    
    BrowsingAction() : type(ActionType::WAIT), x_coordinate(0), y_coordinate(0),
                      scroll_amount(0), scroll_direction(ScrollDirection::DOWN),
                      confidence(0.5f), expected_reward(0.0f) {}
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
    virtual ~AutonomousLearningAgent();
    
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
    // CORE PROCESSING METHODS
    // ========================================================================
    
    /**
     * @brief Process visual input from environment
     */
    void process_visual_input();
    
    /**
     * @brief Update working memory with current context
     */
    void update_working_memory();
    
    /**
     * @brief Select and execute action based on current state
     */
    void select_and_execute_action();
    
    /**
     * @brief Learn from recent experience
     */
    void learn_from_experience();
    
    // ========================================================================
    // ACTION GENERATION AND EXECUTION
    // ========================================================================
    
    /**
     * @brief Generate possible actions based on current context
     * @return Vector of possible actions  
     */
    std::vector<BrowsingAction> generate_action_candidates();
    
    /**
     * @brief Execute a specific action
     * @param action Action to execute
     */
    void execute_action(const BrowsingAction& action);
    
    /**
     * @brief Calculate reward for immediate action
     * @return Reward value
     */
    float calculate_immediate_reward();
    
    // ========================================================================
    // ENVIRONMENT INTERACTION
    // ========================================================================
    
    /**
     * @brief Set environment sensor function
     * @param sensor Function that returns environmental state
     */
    void setEnvironmentSensor(std::function<BrowsingState()> sensor) {
        environment_sensor_ = sensor;
    }
    
    /**
     * @brief Set action executor function
     * @param executor Function that executes actions in environment
     */
    void setActionExecutor(std::function<void(const BrowsingAction&)> executor) {
        action_executor_ = executor;
    }
    
    /**
     * @brief Get current environment state
     * @return Current browsing state
     */
    BrowsingState getCurrentEnvironmentState() const;
    
    // ========================================================================
    // MONITORING AND DIAGNOSTICS
    // ========================================================================
    
    /**
     * @brief Get agent status report
     * @return Status string
     */
    std::string getStatusReport() const;
    
    /**
     * @brief Get learning progress
     * @return Progress value [0-1]
     */
    float getLearningProgress() const;
    
    /**
     * @brief Get current attention weights
     * @return Map of module attention weights
     */
    std::map<std::string, float> getAttentionWeights() const;

    /**
     * @brief Save agent neural state and memory
     * @param directory Base directory to store state files
     */
    bool saveAgentState(const std::string& directory) const;

    /**
     * @brief Load agent neural state and memory
     * @param directory Base directory with saved state files
     */
    bool loadAgentState(const std::string& directory);
    
    /**
     * @brief Enable/disable detailed logging
     * @param enable Enable flag
     */
    void enableDetailedLogging(bool enable) { detailed_logging_ = enable; }

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    // Configuration
    NetworkConfig config_;
    bool is_learning_active_;
    bool detailed_logging_;
    float simulation_time_;
    
    // Core components
    std::unique_ptr<ControllerModule> controller_module_;
    std::unique_ptr<MemorySystem> memory_system_;
    std::unique_ptr<VisualInterface> visual_interface_;
    std::unique_ptr<AttentionController> attention_controller_;
    std::unique_ptr<BrainModuleArchitecture> brain_architecture_;
    std::unique_ptr<RealScreenCapture> real_screen_capture_;
    std::unique_ptr<InputController> input_controller_;
    std::unique_ptr<OCRProcessor> ocr_processor_;
    std::unique_ptr<GUIElementDetector> gui_detector_;
    
    // Environment interaction
    std::function<BrowsingState()> environment_sensor_;
    std::function<void(const BrowsingAction&)> action_executor_;
    
    // State tracking
    std::vector<float> environmental_context_;
    BrowsingAction last_action_;
    std::vector<std::unique_ptr<AutonomousGoal>> learning_goals_;
    
    // Missing member variables needed by DecisionAndActionSystems.cpp
    std::unordered_map<std::string, std::unique_ptr<SpecializedModule>> modules_;
    std::vector<float> current_goals_;
    float exploration_rate_;
    // std::unique_ptr<EnhancedLearningSystem> learning_system_; // Removed to avoid CUDA dependencies
    std::vector<float> global_state_;
    BrowsingAction selected_action_;
    
    // Performance metrics
    struct {
        int total_actions = 0;
        int successful_actions = 0;
        float average_reward = 0.0f;
        int network_expansions = 0;
    } metrics_;
    
    float global_reward_signal_;
    float learning_rate_;
    
    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================
    
    void initialize_neural_modules();
    void initialize_attention_system();
    void update_learning_goals();
    void log_action(const std::string& action);
    
    // Missing method declarations needed by DecisionAndActionSystems.cpp
    void update_attention_weights();
    void coordinate_modules();
    std::vector<float> collect_inter_module_signals(const std::string& target_module);
    void distribute_module_output(const std::string& source_module, const std::vector<float>& output);
    void make_decision();
    std::vector<float> evaluate_action_candidates(const std::vector<BrowsingAction>& candidates, 
                                                   const std::vector<MemorySystem::MemoryTrace>& similar_episodes);
    BrowsingAction select_action_with_exploration(const std::vector<BrowsingAction>& candidates, 
                                                   const std::vector<float>& values);
    void execute_action();
    void execute_click_action();
    void execute_scroll_action();
    void execute_type_action();
    void execute_navigate_action();
    void execute_wait_action();
    std::vector<float> convert_action_to_motor_command(const BrowsingAction& action);
    void learn_from_feedback();
    float compute_action_reward();
    void adapt_exploration_rate();
    void apply_modular_learning(float reward);
    void update_global_state();
    void consolidate_learning();
    void transfer_knowledge_between_modules();
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Convert ActionType to string
 */
std::string actionTypeToString(ActionType type);

/**
 * @brief Convert string to ActionType
 */
ActionType stringToActionType(const std::string& type_str);

/**
 * @brief Compute similarity between two browsing states
 */
float computeBrowsingStateSimilarity(const BrowsingState& state1, const BrowsingState& state2);

/**
 * @brief Compute action value using simple heuristics
 */
float computeActionValue(const BrowsingAction& action, const BrowsingState& state);

/**
 * @brief Update exploration rate based on performance
 */
float updateExplorationRate(float current_rate, float recent_performance, 
                           float target_performance = 0.8f);

#endif // AUTONOMOUS_LEARNING_AGENT_H
