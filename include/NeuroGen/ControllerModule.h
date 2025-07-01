// ============================================================================
// NEUROMODULATORY CONTROLLER MODULE
// File: include/NeuroGen/ControllerModule.h
// ============================================================================

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <chrono>
#include <mutex>
#include <queue>

#include "NeuroGen/NeuralModule.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/NetworkStats.h"

// ============================================================================
// NEUROTRANSMITTER AND NEUROMODULATOR DEFINITIONS
// ============================================================================

enum class NeuromodulatorType {
    DOPAMINE,           // Reward prediction, motivation, learning
    SEROTONIN,          // Mood regulation, plasticity modulation
    NOREPINEPHRINE,     // Attention, arousal, stress response
    ACETYLCHOLINE,      // Attention, learning, memory consolidation
    GABA,               // Inhibition, anxiety regulation
    GLUTAMATE,          // Excitation, synaptic plasticity
    OXYTOCIN,           // Social bonding, trust
    ENDORPHINS,         // Pain relief, reward enhancement
    CORTISOL,           // Stress response, memory modulation
    ADENOSINE           // Sleep pressure, metabolic regulation
};

enum class RewardSignalType {
    INTRINSIC_CURIOSITY,    // Internal exploration reward
    EXTRINSIC_TASK,         // External task completion reward
    SOCIAL_COOPERATION,     // Multi-module coordination reward
    EFFICIENCY_BONUS,       // Energy-efficient operation reward
    NOVELTY_DETECTION,      // New pattern discovery reward
    PREDICTION_ACCURACY,    // Successful prediction reward
    HOMEOSTATIC_BALANCE,    // Maintaining optimal states reward
    CREATIVITY_BURST        // Novel solution generation reward
};

struct NeuromodulatorState {
    NeuromodulatorType type;
    float concentration;        // Current concentration (0.0 - 1.0)
    float baseline_level;       // Homeostatic baseline
    float production_rate;      // Rate of synthesis
    float degradation_rate;     // Rate of breakdown
    float half_life;           // Time to half concentration
    float target_modules_sensitivity; // How sensitive target modules are
    
    // Temporal dynamics
    float peak_concentration;   // Maximum achieved
    float time_since_peak;     // Time since last peak
    std::vector<float> concentration_history; // Recent history for analysis
    
    NeuromodulatorState(NeuromodulatorType t) : type(t) {
        reset_to_defaults();
    }
    
    void reset_to_defaults();
    void update_dynamics(float dt);
    void apply_stimulus(float intensity, float duration = 1.0f);
};

struct RewardSignal {
    RewardSignalType type;
    float magnitude;           // Strength of reward signal
    float confidence;          // Confidence in reward assessment
    float temporal_delay;      // Delay between action and reward
    std::string source_module; // Which module generated this
    std::string target_module; // Which module should receive this
    std::chrono::high_resolution_clock::time_point timestamp;
    
    // Context information
    std::vector<float> context_state;    // Environmental context
    std::unordered_map<std::string, float> module_contributions; // How each module contributed
};

struct NeuromodulationCommand {
    std::string target_module;
    NeuromodulatorType modulator_type;
    float intensity;           // How much to release
    float duration;            // How long the effect should last
    float urgency;            // Priority of this command
    std::string reasoning;     // Why this command was issued
};

// ============================================================================
// CONTROLLER MODULE CONFIGURATION
// ============================================================================

struct ControllerConfig {
    // Neuromodulator pool settings
    float initial_dopamine_level = 0.3f;
    float initial_serotonin_level = 0.4f;
    float initial_norepinephrine_level = 0.2f;
    float initial_acetylcholine_level = 0.3f;
    float initial_gaba_level = 0.5f;
    float initial_glutamate_level = 0.4f;
    
    // Production and regulation
    float base_production_rate = 0.1f;
    float stress_production_multiplier = 2.0f;
    float reward_production_multiplier = 1.5f;
    float homeostatic_regulation_strength = 0.8f;
    
    // Reward processing
    float reward_integration_window = 10.0f;  // seconds
    float reward_prediction_learning_rate = 0.01f;
    float curiosity_drive_strength = 0.3f;
    
    // Module coordination
    float inter_module_communication_strength = 0.7f;
    float attention_allocation_sensitivity = 0.5f;
    float global_inhibition_threshold = 0.8f;
    
    // Adaptive behavior
    bool enable_adaptive_baselines = true;
    bool enable_circadian_modulation = true;
    bool enable_stress_response = true;
    bool enable_social_learning = true;
    
    // Performance monitoring
    float performance_assessment_window = 5.0f;  // seconds
    float module_efficiency_threshold = 0.6f;
    float coordination_bonus_threshold = 0.7f;
};

// ============================================================================
// CONTROLLER MODULE CLASS
// ============================================================================

class ControllerModule {
public:
    // Constructor and initialization
    explicit ControllerModule(const ControllerConfig& config = ControllerConfig());
    ~ControllerModule();
    
    // Module management
    void register_module(const std::string& name, std::shared_ptr<NeuralModule> module);
    void unregister_module(const std::string& name);
    std::shared_ptr<NeuralModule> get_module(const std::string& name) const;
    std::vector<std::string> get_registered_modules() const;
    
    // Core control functions
    void update(float dt);
    void reset();
    void emergency_stop();
    
    // Neuromodulator management
    void release_neuromodulator(NeuromodulatorType type, float intensity, 
                               const std::string& target_module = "");
    void set_baseline_level(NeuromodulatorType type, float level);
    float get_concentration(NeuromodulatorType type) const;
    std::unordered_map<NeuromodulatorType, float> get_all_concentrations() const;
    
    // Reward system
    void process_reward_signal(const RewardSignal& signal);
    void generate_intrinsic_reward(const std::string& module_name, float curiosity_level);
    void distribute_global_reward(float reward_magnitude, RewardSignalType type);
    float calculate_module_performance(const std::string& module_name);
    
    // Attention and coordination
    void allocate_attention(const std::unordered_map<std::string, float>& attention_weights);
    void coordinate_module_activities();
    void apply_global_inhibition(float strength);
    void promote_inter_module_cooperation();
    
    // Learning and adaptation
    void adapt_to_performance();
    void update_reward_predictions();
    void learn_optimal_modulation_patterns();
    void adjust_homeostatic_setpoints();
    
    // Assessment and monitoring
    std::unordered_map<std::string, float> assess_module_health();
    std::unordered_map<std::string, float> measure_coordination_efficiency();
    float calculate_overall_system_performance();
    std::string generate_status_report();
    
    // Advanced features
    void enable_creative_mode(float intensity = 0.7f);
    void enable_focus_mode(const std::string& target_module, float intensity = 0.8f);
    void enable_exploration_mode(float curiosity_boost = 0.5f);
    void enable_consolidation_mode(float memory_strength = 0.6f);
    
    // Configuration and tuning
    void update_config(const ControllerConfig& new_config);
    ControllerConfig get_config() const;
    void save_state(const std::string& filename) const;
    void load_state(const std::string& filename);
    
    // Diagnostics and debugging
    std::vector<std::string> get_recent_actions() const;
    std::unordered_map<std::string, std::vector<float>> get_modulator_histories() const;
    void enable_detailed_logging(bool enable = true);
    
private:
    // Configuration
    ControllerConfig config_;
    
    // Module registry
    std::unordered_map<std::string, std::shared_ptr<NeuralModule>> registered_modules_;
    mutable std::mutex modules_mutex_;
    
    // Neuromodulator state
    std::unordered_map<NeuromodulatorType, std::unique_ptr<NeuromodulatorState>> neuromodulators_;
    
    // Reward processing
    std::queue<RewardSignal> pending_rewards_;
    std::unordered_map<std::string, float> module_performance_history_;
    std::unordered_map<RewardSignalType, float> reward_prediction_errors_;
    
    // Command queue
    std::queue<NeuromodulationCommand> pending_commands_;
    std::vector<std::string> action_history_;
    
    // Temporal tracking
    float simulation_time_;
    std::chrono::high_resolution_clock::time_point last_update_time_;
    
    // Performance metrics
    std::unordered_map<std::string, NetworkStats> module_stats_history_;
    float global_performance_trend_;
    float system_coherence_level_;
    
    // State flags
    bool is_running_;
    bool detailed_logging_enabled_;
    
    // Internal methods
    void initialize_neuromodulators();
    void update_neuromodulator_dynamics(float dt);
    void process_pending_rewards();
    void execute_pending_commands();
    void assess_system_state();
    void generate_automatic_responses();
    void update_performance_metrics();
    void log_action(const std::string& action);
    
    // Helper functions
    float calculate_stress_level() const;
    float calculate_attention_demand() const;
    float calculate_learning_opportunity() const;
    bool should_trigger_homeostatic_response() const;
    std::vector<NeuromodulationCommand> generate_coordinated_response();
    
    // Specialized control algorithms
    void dopamine_reward_prediction_update(const RewardSignal& signal);
    void serotonin_mood_regulation();
    void norepinephrine_attention_modulation();
    void acetylcholine_learning_enhancement();
    void gaba_inhibitory_balance();
    void glutamate_excitatory_drive();
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Convert enum to string for logging and debugging
std::string to_string(NeuromodulatorType type);
std::string to_string(RewardSignalType type);

// Factory function for creating pre-configured controllers
std::unique_ptr<ControllerModule> create_learning_focused_controller();
std::unique_ptr<ControllerModule> create_exploration_focused_controller();
std::unique_ptr<ControllerModule> create_balanced_controller();
std::unique_ptr<ControllerModule> create_performance_focused_controller();

// Helper for module interconnection
void setup_standard_module_connections(ControllerModule& controller,
                                      std::shared_ptr<NeuralModule> perception,
                                      std::shared_ptr<NeuralModule> planning,
                                      std::shared_ptr<NeuralModule> motor);
