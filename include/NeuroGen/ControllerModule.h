#ifndef CONTROLLER_MODULE_H
#define CONTROLLER_MODULE_H

// ============================================================================
// NEUREGEN FRAMEWORK INCLUDES - Modular Architecture
// ============================================================================
#include <NeuroGen/EnhancedNeuralModule.h>
#include <NeuroGen/Network.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/ModularNeuralNetwork.h>
#include <NeuroGen/cuda/NetworkCUDA.cuh>
#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>

// ============================================================================
// SYSTEM INCLUDES
// ============================================================================
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <map>
#include <queue>
#include <functional>
#include <chrono>
#include <atomic>
#include <mutex>

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================
class MemorySystem;
class NeuralModule;

/**
 * @brief Neuromodulatory controller subnetwork for executive control
 * 
 * This specialized neural subnetwork implements central executive functions
 * through neuromodulatory mechanisms, providing top-down control and
 * attention allocation across the modular neural network.
 */
class NeuromodulatorController {
public:
    enum ModulatorType {
        DOPAMINE = 0,     // Reward/motivation
        SEROTONIN = 1,    // Mood/exploration
        NOREPINEPHRINE = 2, // Attention/arousal
        ACETYLCHOLINE = 3,  // Learning/plasticity
        GABA = 4,          // Inhibition/stability
        NUM_MODULATORS = 5
    };

    struct ModulatorState {
        float concentration;
        float release_rate;
        float reuptake_rate;
        float baseline_level;
        float decay_constant;
        std::vector<std::string> target_modules;
        std::vector<float> modulation_weights;
    };

private:
    std::array<ModulatorState, NUM_MODULATORS> modulators_;
    std::unique_ptr<Network> controller_network_;
    std::map<std::string, float> module_activation_levels_;
    std::vector<float> attention_weights_;
    float global_arousal_;
    float cognitive_load_;

public:
    NeuromodulatorController(const NetworkConfig& config);
    ~NeuromodulatorController() = default;

    // Core neuromodulation interface
    void initialize();
    void update(float dt);
    void setTargetModulation(ModulatorType type, float target_level);
    float getModulatorLevel(ModulatorType type) const;
    
    // Module orchestration
    void setModuleActivation(const std::string& module_name, float activation);
    float getModuleActivation(const std::string& module_name) const;
    void updateAttentionWeights(const std::vector<float>& context_input);
    
    // Executive control
    void processControlSignals(const std::vector<float>& control_input);
    std::vector<float> generateModulationOutput() const;
    void applyGlobalInhibition(float inhibition_strength);
    
    // State management
    void saveControllerState(const std::string& filename) const;
    bool loadControllerState(const std::string& filename);
};

/**
 * @brief Central controller module managing the entire modular neural network
 * 
 * This class implements a biologically-inspired central control system that
 * orchestrates specialized neural modules through attention mechanisms,
 * neuromodulation, and inter-module communication pathways.
 */
class ControllerModule : public EnhancedNeuralModule {
public:
    // ========================================================================
    // MODULE COMMUNICATION STRUCTURES
    // ========================================================================
    
    struct InterModuleSignal {
        std::string source_module;
        std::string target_module;
        std::string signal_type;
        std::vector<float> data;
        float timestamp;
        float strength;
        int priority;
    };

    struct ModuleState {
        std::string module_name;
        bool is_active;
        float activation_level;
        float attention_weight;
        std::vector<float> output_signals;
        std::vector<float> feedback_signals;
        float processing_load;
        float specialization_index;
    };

    struct BrowsingAction {
        enum Type {
            CLICK = 0,
            SCROLL = 1,
            TYPE = 2,
            NAVIGATE = 3,
            WAIT = 4,
            OBSERVE = 5
        } type;
        
        struct {
            int x, y;                    // Screen coordinates
            std::string text;            // Text to type
            std::string url;             // URL to navigate
            float scroll_amount;         // Scroll distance
            float wait_duration;         // Wait time in seconds
        } parameters;
        
        float confidence;                // Action confidence [0-1]
        float expected_reward;           // Expected outcome value
        int element_id;                  // Target screen element ID
        std::string reasoning;           // Action reasoning text
    };

    // ========================================================================
    // MEMORY AND LEARNING STRUCTURES
    // ========================================================================
    
    struct MemoryTrace {
        std::vector<float> state_vector;
        BrowsingAction action_taken;
        float reward_received;
        std::vector<float> next_state;
        float temporal_discount;
        float importance_weight;
        std::chrono::steady_clock::time_point timestamp;
        bool is_consolidated;
    };

    struct ContextualMemory {
        std::map<std::string, std::vector<MemoryTrace>> episodic_memories;
        std::map<std::string, std::vector<float>> semantic_knowledge;
        std::map<std::string, float> skill_competencies;
        std::queue<MemoryTrace> working_memory;
        size_t working_memory_capacity;
    };

private:
    // ========================================================================
    // CORE CONTROLLER COMPONENTS
    // ========================================================================
    
    std::unique_ptr<NeuromodulatorController> neuromodulator_controller_;
    std::unique_ptr<ModularNeuralNetwork> modular_network_;
    std::unique_ptr<NetworkCUDA> gpu_accelerator_;
    
    // Module management
    std::unordered_map<std::string, std::unique_ptr<NeuralModule>> registered_modules_;
    std::unordered_map<std::string, ModuleState> module_states_;
    std::vector<InterModuleSignal> inter_module_signals_;
    std::queue<InterModuleSignal> signal_queue_;
    
    // Decision and action systems
    BrowsingAction selected_action_;
    std::vector<BrowsingAction> action_candidates_;
    std::map<std::string, float> action_values_;
    std::unique_ptr<ContextualMemory> memory_system_;
    
    // Control and coordination
    std::vector<float> global_context_;
    std::vector<float> attention_allocation_;
    std::map<std::string, float> module_specializations_;
    float cognitive_load_threshold_;
    bool adaptive_processing_enabled_;
    
    // Performance monitoring
    struct PerformanceMetrics {
        float processing_efficiency;
        float decision_accuracy;
        float learning_progress;
        float module_coordination_quality;
        float memory_utilization;
        int successful_actions;
        int total_actions;
        std::chrono::steady_clock::time_point last_update;
    } performance_metrics_;
    
    // Synchronization and threading
    mutable std::mutex state_mutex_;
    std::atomic<bool> is_processing_;
    std::atomic<bool> shutdown_requested_;

public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    explicit ControllerModule(const std::string& name, const NetworkConfig& config);
    virtual ~ControllerModule();
    
    // Core module interface
    void initialize() override;
    void update(double dt) override;
    void shutdown();
    
    // ========================================================================
    // MODULE MANAGEMENT AND ORCHESTRATION
    // ========================================================================
    
    // Module registration and management
    void registerModule(std::unique_ptr<NeuralModule> module);
    void unregisterModule(const std::string& module_name);
    NeuralModule* getModule(const std::string& module_name) const;
    std::vector<std::string> getRegisteredModules() const;
    
    // Module activation and control
    void activateModule(const std::string& module_name, float activation_level = 1.0f);
    void deactivateModule(const std::string& module_name);
    void setModuleSpecialization(const std::string& module_name, float specialization);
    float getModuleActivation(const std::string& module_name) const;
    
    // Inter-module communication
    std::vector<float> collect_inter_module_signals(const std::string& target_module);
    void distribute_module_output(const std::string& source_module, 
                                 const std::vector<float>& output_data);
    void processInterModuleCommunication();
    void establishModuleConnection(const std::string& source_module,
                                  const std::string& target_module,
                                  float connection_strength = 1.0f);
    
    // ========================================================================
    // DECISION AND ACTION SYSTEMS
    // ========================================================================
    
    // Action generation and selection
    std::vector<BrowsingAction> generate_action_candidates();
    std::vector<float> evaluate_action_candidates(
        const std::vector<BrowsingAction>& candidates,
        const std::vector<MemoryTrace>& similar_episodes);
    BrowsingAction select_action_with_exploration(
        const std::vector<BrowsingAction>& candidates,
        const std::vector<float>& action_values);
    
    // Action execution
    void execute_action();
    void execute_click_action();
    void execute_scroll_action();
    void execute_type_action();
    void execute_navigate_action();
    void execute_wait_action();
    
    // Motor command generation
    std::vector<float> convert_action_to_motor_command(const BrowsingAction& action);
    void sendMotorCommand(const std::vector<float>& motor_command);
    
    // ========================================================================
    // ATTENTION AND CONTROL MECHANISMS
    // ========================================================================
    
    // Attention allocation
    void updateAttentionMechanism(const std::vector<float>& sensory_input);
    void allocateAttention(const std::map<std::string, float>& attention_demands);
    std::vector<float> computeAttentionWeights(const std::vector<float>& context);
    void applyAttentionalModulation();
    
    // Executive control
    void processExecutiveControl(const std::vector<float>& goal_state);
    void updateCognitiveLoad();
    void manageResourceAllocation();
    void coordinateModuleActivation();
    
    // ========================================================================
    // LEARNING AND MEMORY SYSTEMS
    // ========================================================================
    
    // Memory management
    void consolidateMemories();
    void updateWorkingMemory(const MemoryTrace& trace);
    std::vector<MemoryTrace> retrieveSimilarEpisodes(const std::vector<float>& current_state);
    void strengthenMemoryTrace(const std::string& episode_key, float reinforcement);
    
    // Learning and adaptation
    void updateActionValues(const BrowsingAction& action, float reward);
    void adaptModuleSpecializations();
    void updateInterModuleWeights();
    void performMetaLearning();
    
    // ========================================================================
    // STATE MANAGEMENT AND PERSISTENCE
    // ========================================================================
    
    // Module state persistence
    bool saveModuleState(const std::string& module_name, const std::string& filename) const;
    bool loadModuleState(const std::string& module_name, const std::string& filename);
    bool saveControllerState(const std::string& filename) const override;
    bool loadControllerState(const std::string& filename) override;
    
    // System state queries
    ModuleState getModuleState(const std::string& module_name) const;
    std::map<std::string, ModuleState> getAllModuleStates() const;
    PerformanceMetrics getPerformanceMetrics() const;
    
    // ========================================================================
    // DEBUGGING AND MONITORING
    // ========================================================================
    
    // Performance monitoring
    void updatePerformanceMetrics();
    void logModuleActivity(const std::string& activity) const;
    std::string generateStatusReport() const;
    
    // Diagnostic interface
    void enableDiagnosticMode(bool enable);
    std::vector<float> getModuleActivationPattern() const;
    std::map<std::string, float> getInterModuleConnectivity() const;
    
protected:
    // ========================================================================
    // INTERNAL HELPER METHODS
    // ========================================================================
    
    // Initialization helpers
    void initializeNeuromodulatorController();
    void initializeModularNetwork();
    void initializeMemorySystem();
    void setupDefaultModuleConnections();
    
    // Processing helpers
    void processModuleUpdates(double dt);
    void processAttentionAllocation();
    void processNeuromodulation();
    void processMemoryConsolidation();
    
    // Communication helpers
    void routeInterModuleSignals();
    void updateSignalPriorities();
    void manageSignalQueue();
    
    // Validation and error handling
    bool validateModuleState(const std::string& module_name) const;
    void handleModuleError(const std::string& module_name, const std::string& error);
    void performSystemHealthCheck();
};

/**
 * @brief Autonomous learning agent class that integrates the ControllerModule
 * 
 * This class provides the high-level interface for the autonomous browsing agent,
 * integrating decision-making, action execution, and learning capabilities.
 */
class AutonomousLearningAgent {
public:
    // Expose necessary types for DecisionAndActionSystems.cpp
    using BrowsingAction = ControllerModule::BrowsingAction;
    using MemoryTrace = ControllerModule::MemoryTrace;

private:
    std::unique_ptr<ControllerModule> controller_module_;
    std::unique_ptr<MemorySystem> memory_system_;
    ControllerModule::BrowsingAction selected_action_;
    
public:
    explicit AutonomousLearningAgent(const NetworkConfig& config);
    ~AutonomousLearningAgent() = default;
    
    // Core agent interface
    bool initialize();
    void update(double dt);
    void shutdown();
    
    // Decision and action interface (for DecisionAndActionSystems.cpp)
    std::vector<float> collect_inter_module_signals(const std::string& target_module);
    void distribute_module_output(const std::string& source_module, 
                                 const std::vector<float>& output_data);
    std::vector<BrowsingAction> generate_action_candidates();
    std::vector<float> evaluate_action_candidates(
        const std::vector<BrowsingAction>& candidates,
        const std::vector<MemoryTrace>& similar_episodes);
    BrowsingAction select_action_with_exploration(
        const std::vector<BrowsingAction>& candidates,
        const std::vector<float>& action_values);
    void execute_action();
    void execute_click_action();
    void execute_scroll_action();
    void execute_type_action();
    void execute_navigate_action();
    void execute_wait_action();
    
    // State access
    const BrowsingAction& getSelectedAction() const { return selected_action_; }
    MemorySystem* getMemorySystem() const { return memory_system_.get(); }
    ControllerModule* getControllerModule() const { return controller_module_.get(); }
};

#endif // CONTROLLER_MODULE_H