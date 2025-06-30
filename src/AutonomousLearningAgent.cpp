#ifndef AUTONOMOUS_LEARNING_AGENT_H
#define AUTONOMOUS_LEARNING_AGENT_H

// ============================================================================
// NEUREGEN FRAMEWORK INCLUDES
// ============================================================================
#include <NeuroGen/ControllerModule.h>
#include <NeuroGen/Network.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/EnhancedNeuralModule.h>

// ============================================================================
// SYSTEM INCLUDES
// ============================================================================
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <chrono>

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================
class ControllerModule;
class MemorySystem;

/**
 * @brief Memory system for the autonomous learning agent
 * 
 * This class manages episodic and semantic memories for the agent,
 * enabling learning from past experiences and knowledge consolidation.
 */
class MemorySystem {
public:
    // ========================================================================
    // PUBLIC MEMORY STRUCTURES
    // ========================================================================
    
    /**
     * @brief Individual memory trace representing a learning episode
     */
    struct MemoryTrace {
        std::vector<float> state_vector;     // Environmental state at time t
        std::vector<float> action_vector;    // Action taken in the state
        float reward_received;               // Immediate reward from action
        std::vector<float> next_state;       // Resulting state at t+1
        float temporal_discount;             // Discount factor for future rewards
        float importance_weight;             // Memory consolidation weight
        std::chrono::steady_clock::time_point timestamp;  // When episode occurred
        bool is_consolidated;                // Whether memory is consolidated
        
        // Contextual information
        std::string episode_context;         // Semantic context description
        int episode_id;                      // Unique episode identifier
        float prediction_error;              // TD error when episode was stored
        std::vector<std::string> involved_modules; // Which modules were active
    };

    /**
     * @brief Episodic memory cluster for related experiences
     */
    struct EpisodicCluster {
        std::string context_type;            // Type of context (e.g., "login", "navigation")
        std::vector<MemoryTrace> episodes;   // Related memory traces
        float cluster_coherence;             // How well episodes fit together
        std::vector<float> prototype_state;  // Prototypical state for this cluster
        int access_count;                    // How often this cluster is accessed
        std::chrono::steady_clock::time_point last_accessed;
    };

private:
    // Memory storage
    std::map<std::string, EpisodicCluster> episodic_memory_;
    std::map<std::string, std::vector<float>> semantic_memory_;
    std::map<std::string, float> skill_memory_;
    
    // Memory parameters
    size_t max_episodes_per_cluster_;
    size_t max_total_episodes_;
    float consolidation_threshold_;
    float forgetting_rate_;
    
    // Working memory
    std::vector<MemoryTrace> working_memory_;
    size_t working_memory_capacity_;

public:
    MemorySystem(size_t max_episodes = 10000, size_t working_capacity = 7);
    ~MemorySystem() = default;
    
    // Core memory operations
    void storeEpisode(const MemoryTrace& episode, const std::string& context);
    std::vector<MemoryTrace> retrieveSimilarEpisodes(const std::vector<float>& current_state, 
                                                    const std::string& context = "",
                                                    size_t max_results = 10);
    void consolidateMemories();
    void updateWorkingMemory(const MemoryTrace& trace);
    
    // Memory management
    void forgetOldEpisodes();
    void strengthenMemory(const std::string& context, int episode_id, float strength_boost);
    void organizeMemoryStructure();
    
    // Query interface
    float getSkillLevel(const std::string& skill_name) const;
    void updateSkillLevel(const std::string& skill_name, float performance);
    std::vector<std::string> getKnownContexts() const;
    size_t getEpisodeCount(const std::string& context = "") const;
    
    // State management
    bool saveMemoryState(const std::string& filename) const;
    bool loadMemoryState(const std::string& filename);
    
    // Access to memory structures for other classes
    const std::map<std::string, EpisodicCluster>& getEpisodicMemory() const { 
        return episodic_memory_; 
    }
};

/**
 * @brief Autonomous learning agent for web browsing and interaction
 * 
 * This class implements a complete autonomous agent capable of learning
 * to browse the internet, interact with web pages, and continuously
 * improve its performance through reinforcement learning.
 */
class AutonomousLearningAgent {
public:
    // ========================================================================
    // TYPE DEFINITIONS AND STRUCTURES
    // ========================================================================
    
    /**
     * @brief Action types the agent can perform
     */
    enum class ActionType {
        CLICK = 0,
        SCROLL = 1,
        TYPE = 2,
        NAVIGATE = 3,
        WAIT = 4,
        OBSERVE = 5,
        BACK = 6,
        FORWARD = 7,
        REFRESH = 8
    };
    
    /**
     * @brief Complete browsing action specification
     */
    struct BrowsingAction {
        ActionType type;                     // Type of action to perform
        
        // Action parameters
        struct Parameters {
            int x, y;                        // Screen coordinates for clicks
            std::string text;                // Text to type
            std::string url;                 // URL for navigation
            float scroll_amount;             // Scroll distance (pixels)
            float wait_duration;             // Wait time (seconds)
            int element_id;                  // Target element identifier
            std::string element_selector;    // CSS/XPath selector
        } parameters;
        
        // Action metadata
        float confidence;                    // Confidence in action choice [0-1]
        float expected_reward;               // Expected reward from action
        std::string reasoning;               // Human-readable action reasoning
        int priority;                        // Action priority (higher = more urgent)
        std::vector<std::string> preconditions; // Required conditions
        
        // Execution context
        std::chrono::steady_clock::time_point planned_time;
        bool requires_confirmation;          // Whether action needs confirmation
        std::vector<BrowsingAction> followup_actions; // Subsequent actions
    };
    
    /**
     * @brief Current state of the browsing environment
     */
    struct BrowsingState {
        std::string current_url;             // Current page URL
        std::string page_title;              // Current page title
        std::vector<float> visual_features;  // Extracted visual features
        std::vector<float> text_features;    // Extracted text features
        std::vector<float> interaction_features; // Available interactions
        
        // Page elements
        struct Element {
            int id;
            std::string tag;                 // HTML tag type
            std::string text;                // Element text content
            int x, y, width, height;         // Element bounds
            bool is_clickable;               // Whether element is interactive
            bool is_visible;                 // Whether element is visible
            std::string css_selector;        // CSS selector for element
        };
        std::vector<Element> page_elements;
        
        // Interaction state
        bool page_loading;                   // Whether page is still loading
        float load_progress;                 // Loading progress [0-1]
        std::chrono::steady_clock::time_point state_timestamp;
        std::string previous_action;         // Last action performed
    };

private:
    // ========================================================================
    // CORE COMPONENTS
    // ========================================================================
    
    std::unique_ptr<ControllerModule> controller_module_;
    std::unique_ptr<MemorySystem> memory_system_;
    
    // Action and decision state
    BrowsingAction selected_action_;
    std::vector<BrowsingAction> action_candidates_;
    std::map<std::string, float> action_values_;
    BrowsingState current_state_;
    
    // Learning parameters
    float learning_rate_;
    float exploration_rate_;
    float discount_factor_;
    float eligibility_decay_;
    
    // Performance tracking
    struct PerformanceStats {
        int total_actions;
        int successful_actions;
        int failed_actions;
        float cumulative_reward;
        float average_confidence;
        std::chrono::steady_clock::time_point session_start;
        std::map<ActionType, int> action_counts;
        std::map<std::string, float> context_performance;
    } performance_stats_;
    
    // Configuration
    NetworkConfig agent_config_;
    bool learning_enabled_;
    bool exploration_enabled_;
    bool verbose_logging_;

public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    explicit AutonomousLearningAgent(const NetworkConfig& config);
    ~AutonomousLearningAgent();
    
    // Core lifecycle
    bool initialize();
    void update(double dt);
    void shutdown();
    
    // Configuration
    void setLearningEnabled(bool enabled) { learning_enabled_ = enabled; }
    void setExplorationEnabled(bool enabled) { exploration_enabled_ = enabled; }
    void setVerboseLogging(bool enabled) { verbose_logging_ = enabled; }
    
    // ========================================================================
    // DECISION AND ACTION SYSTEMS (Required by DecisionAndActionSystems.cpp)
    // ========================================================================
    
    // Inter-module communication
    std::vector<float> collect_inter_module_signals(const std::string& target_module);
    void distribute_module_output(const std::string& source_module, 
                                 const std::vector<float>& output_data);
    
    // Action generation and evaluation
    std::vector<BrowsingAction> generate_action_candidates();
    std::vector<float> evaluate_action_candidates(
        const std::vector<BrowsingAction>& candidates,
        const std::vector<MemorySystem::MemoryTrace>& similar_episodes);
    
    // Action selection and execution
    BrowsingAction select_action_with_exploration(
        const std::vector<BrowsingAction>& candidates,
        const std::vector<float>& action_values);
    void execute_action();
    
    // Specific action executors
    void execute_click_action();
    void execute_scroll_action();
    void execute_type_action();
    void execute_navigate_action();
    void execute_wait_action();
    void execute_back_action();
    void execute_forward_action();
    void execute_refresh_action();
    
    // Motor command interface
    std::vector<float> convert_action_to_motor_command(const BrowsingAction& action);
    void sendMotorCommand(const std::vector<float>& motor_command);
    
    // ========================================================================
    // LEARNING AND MEMORY INTERFACE
    // ========================================================================
    
    // Experience management
    void recordExperience(const BrowsingState& state, const BrowsingAction& action, 
                         float reward, const BrowsingState& next_state);
    void updateActionValues(const BrowsingAction& action, float reward);
    void updateExplorationStrategy();
    
    // Memory operations
    std::vector<MemorySystem::MemoryTrace> getSimilarEpisodes(
        const BrowsingState& current_state, size_t max_results = 10);
    void consolidateMemories();
    void strengthenMemory(const std::string& context, int episode_id, float strength);
    
    // ========================================================================
    // PERCEPTION AND STATE MANAGEMENT
    // ========================================================================
    
    // State updates
    void updateBrowsingState(const BrowsingState& new_state);
    void processVisualInput(const std::vector<float>& visual_features);
    void processTextInput(const std::vector<float>& text_features);
    void updatePageElements(const std::vector<BrowsingState::Element>& elements);
    
    // State queries
    const BrowsingState& getCurrentState() const { return current_state_; }
    const BrowsingAction& getSelectedAction() const { return selected_action_; }
    bool isPageReady() const { return !current_state_.page_loading; }
    
    // ========================================================================
    // PERFORMANCE AND MONITORING
    // ========================================================================
    
    // Performance metrics
    const PerformanceStats& getPerformanceStats() const { return performance_stats_; }
    float getSuccessRate() const;
    float getAverageReward() const;
    std::map<ActionType, float> getActionSuccessRates() const;
    
    // Logging and debugging
    void logAction(const BrowsingAction& action, float reward = 0.0f);
    void logError(const std::string& error_message);
    std::string generatePerformanceReport() const;
    void exportLearningData(const std::string& filename) const;
    
    // ========================================================================
    // STATE PERSISTENCE
    // ========================================================================
    
    // Save/load agent state
    bool saveAgentState(const std::string& filename) const;
    bool loadAgentState(const std::string& filename);
    
    // Save/load individual components
    bool saveMemoryState(const std::string& filename) const;
    bool loadMemoryState(const std::string& filename);
    bool saveActionValues(const std::string& filename) const;
    bool loadActionValues(const std::string& filename);
    
    // ========================================================================
    // COMPONENT ACCESS (for integration with existing code)
    // ========================================================================
    
    // Component getters
    ControllerModule* getControllerModule() const { return controller_module_.get(); }
    MemorySystem* getMemorySystem() const { return memory_system_.get(); }
    
    // Direct access to internal state (use carefully)
    std::map<std::string, float>& getActionValues() { return action_values_; }
    const std::map<std::string, float>& getActionValues() const { return action_values_; }
    
    // Configuration access
    const NetworkConfig& getConfig() const { return agent_config_; }
    void updateConfig(const NetworkConfig& new_config);
    
protected:
    // ========================================================================
    // INTERNAL HELPER METHODS
    // ========================================================================
    
    // Initialization helpers
    void initializeComponents();
    void setupDefaultActionValues();
    void configureLearningParameters();
    
    // Decision-making helpers
    float computeActionValue(const BrowsingAction& action, 
                           const std::vector<MemorySystem::MemoryTrace>& episodes);
    float computeExplorationBonus(const BrowsingAction& action);
    bool validateAction(const BrowsingAction& action) const;
    
    // Learning helpers
    float computeReward(const BrowsingState& prev_state, const BrowsingAction& action, 
                       const BrowsingState& new_state);
    float computeTemporalDifferenceError(const BrowsingAction& action, float reward, 
                                       const BrowsingState& next_state);
    void updateEligibilityTraces(const BrowsingAction& action, float td_error);
    
    // State processing helpers
    std::vector<float> extractStateFeatures(const BrowsingState& state) const;
    std::vector<BrowsingState::Element> filterRelevantElements(
        const std::vector<BrowsingState::Element>& all_elements) const;
    float computeStateSimilarity(const BrowsingState& state1, 
                               const BrowsingState& state2) const;
    
    // Performance helpers
    void updatePerformanceStats(const BrowsingAction& action, float reward, bool success);
    void resetPerformanceStats();
    void analyzePerformanceTrends();
};

// ============================================================================
// HELPER FUNCTIONS AND UTILITIES
// ============================================================================

/**
 * @brief Converts ActionType enum to string representation
 */
std::string actionTypeToString(AutonomousLearningAgent::ActionType type);

/**
 * @brief Converts string to ActionType enum
 */
AutonomousLearningAgent::ActionType stringToActionType(const std::string& type_str);

/**
 * @brief Computes similarity between two browsing states
 */
float computeBrowsingStateSimilarity(const AutonomousLearningAgent::BrowsingState& state1,
                                   const AutonomousLearningAgent::BrowsingState& state2);

/**
 * @brief Extracts features from a browsing action for learning
 */
std::vector<float> extractActionFeatures(const AutonomousLearningAgent::BrowsingAction& action);

#endif // AUTONOMOUS_LEARNING_AGENT_H