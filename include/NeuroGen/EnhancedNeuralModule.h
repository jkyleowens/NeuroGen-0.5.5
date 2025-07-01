#ifndef ENHANCED_NEURAL_MODULE_H
#define ENHANCED_NEURAL_MODULE_H

#include <NeuroGen/NeuralModule.h>
#include <NeuroGen/TopologyGenerator.h>
#include <memory>
#include <functional>
#include <chrono>
#include <queue>  // FIX: Added missing queue include

/**
 * Enhanced neural module with biological features:
 * - Central control/attention mechanism
 * - Internal feedback loops
 * - State saving/loading per module
 * - Inter-module communication pathways
 */
class EnhancedNeuralModule : public NeuralModule {
public:
    // Module state for serialization
    struct ModuleState {
        std::string module_name;
        std::vector<float> neuron_states;
        std::vector<float> synapse_weights;
        std::vector<float> neuromodulator_levels;
        float module_attention_weight;
        int developmental_stage;
        std::map<std::string, float> performance_metrics;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    // Inter-module connection specification
    struct InterModuleConnection {
        std::string source_port;
        std::string target_module;
        std::string target_port;
        float connection_strength;
        bool is_feedback;  // True for feedback connections
        float delay_ms;    // Transmission delay
    };

    // Constructor
    EnhancedNeuralModule(const std::string& name, const NetworkConfig& config)
        : NeuralModule(name, config), 
          attention_weight_(1.0f),
          is_active_(true),
          developmental_stage_(0),
          last_feedback_update_(std::chrono::steady_clock::now()) {}

    // Virtual destructor
    virtual ~EnhancedNeuralModule() = default;

    // ========================================================================
    // OVERRIDDEN VIRTUAL FUNCTIONS
    // ========================================================================
    
    /**
     * @brief Initialize the enhanced neural module
     * @return Success status of initialization
     */
    bool initialize() override;
    
    /**
     * @brief Update module with enhanced biological features
     * @param dt Time step in seconds
     * @param inputs Input vector to process (optional)
     * @param reward Reward signal for learning (optional)
     */
    void update(float dt, const std::vector<float>& inputs = {}, float reward = 0.0f) override;
    
    /**
     * @brief Get enhanced performance metrics including biological features
     * @return Map of performance metric names to values
     */
    std::map<std::string, float> getPerformanceMetrics() const override;
    
    // ========================================================================
    // STATE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Save complete module state including biological parameters
     * @return ModuleState structure with all state information
     */
    virtual ModuleState saveState() const;
    
    /**
     * @brief Load complete module state
     * @param state ModuleState structure to load
     */
    virtual void loadState(const ModuleState& state);
    
    /**
     * @brief Save state to file with enhanced serialization
     * @param filename Output filename
     * @return Success status
     */
    bool save_state(const std::string& filename) const override;
    
    /**
     * @brief Load state from file with enhanced deserialization
     * @param filename Input filename  
     * @return Success status
     */
    bool load_state(const std::string& filename) override;
    
    // ========================================================================
    // ATTENTION MECHANISM
    // ========================================================================
    
    /**
     * @brief Set attention weight for this module
     * @param weight Attention weight (0.0 to 1.0)
     */
    void setAttentionWeight(float weight) { 
        std::lock_guard<std::mutex> lock(module_mutex_);
        attention_weight_ = std::max(0.0f, std::min(1.0f, weight)); 
    }
    
    /**
     * @brief Get current attention weight
     * @return Current attention weight
     */
    float getAttentionWeight() const { 
        std::lock_guard<std::mutex> lock(module_mutex_);
        return attention_weight_; 
    }
    
    /**
     * @brief Apply attention-based modulation to module activity
     * @param global_attention Global attention signal
     */
    virtual void applyAttentionModulation(float global_attention);
    
    // ========================================================================
    // MODULE ACTIVITY CONTROL
    // ========================================================================
    
    /**
     * @brief Set module activity state with enhanced control
     * @param active Activity state to set
     */
    void setActive(bool active) { 
        std::lock_guard<std::mutex> lock(module_mutex_);
        is_active_ = active; 
        NeuralModule::set_active(active);
    }
    
    /**
     * @brief Check if module is active with enhanced state tracking
     * @return Current activity state
     */
    bool isActive() const { 
        std::lock_guard<std::mutex> lock(module_mutex_);
        return is_active_ && NeuralModule::is_active(); 
    }
    
    // ========================================================================
    // FEEDBACK LOOP MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Add internal feedback loop
     * @param from_port Source port name
     * @param to_port Target port name
     * @param gain Feedback gain factor
     */
    void addFeedbackLoop(const std::string& from_port, const std::string& to_port, 
                        float gain = 1.0f);
    
    /**
     * @brief Process all feedback loops
     * @param dt Time step for feedback processing
     */
    void processFeedback(float dt);
    
    /**
     * @brief Remove feedback loop
     * @param from_port Source port name
     * @param to_port Target port name
     */
    void removeFeedbackLoop(const std::string& from_port, const std::string& to_port);
    
    // ========================================================================
    // INTER-MODULE COMMUNICATION
    // ========================================================================
    
    /**
     * @brief Register inter-module connection
     * @param connection Connection specification
     */
    void registerInterModuleConnection(const InterModuleConnection& connection);
    
    /**
     * @brief Get all outgoing connections
     * @return Vector of outgoing connections
     */
    std::vector<InterModuleConnection> getOutgoingConnections() const { 
        std::lock_guard<std::mutex> lock(module_mutex_);
        return outgoing_connections_; 
    }
    
    /**
     * @brief Process inter-module communications with delays
     * @param dt Time step for processing
     */
    virtual void processInterModuleCommunication(float dt);
    
    // ========================================================================
    // NEUROMODULATION INTERFACE
    // ========================================================================
    
    /**
     * @brief Apply neuromodulation to the module
     * @param modulator_type Type of neuromodulator (dopamine, serotonin, etc.)
     * @param level Modulation level
     */
    virtual void applyNeuromodulation(const std::string& modulator_type, float level);
    
    /**
     * @brief Get current neuromodulator levels
     * @return Map of neuromodulator types to levels
     */
    std::map<std::string, float> getNeuromodulatorLevels() const {
        std::lock_guard<std::mutex> lock(module_mutex_);
        return neuromodulator_levels_;
    }
    
    // ========================================================================
    // DEVELOPMENT AND PLASTICITY
    // ========================================================================
    
    /**
     * @brief Set developmental stage
     * @param stage Developmental stage (0=embryonic, 1=infant, 2=child, 3=adult)
     */
    void setDevelopmentalStage(int stage) { 
        std::lock_guard<std::mutex> lock(module_mutex_);
        developmental_stage_ = stage; 
    }
    
    /**
     * @brief Get current developmental stage
     * @return Current developmental stage
     */
    int getDevelopmentalStage() const { 
        std::lock_guard<std::mutex> lock(module_mutex_);
        return developmental_stage_; 
    }
    
    /**
     * @brief Update developmental plasticity
     * @param dt Time step for development
     */
    virtual void updateDevelopmentalPlasticity(float dt);

protected:
    // Feedback loop structure
    struct FeedbackLoop {
        std::string from_port;
        std::string to_port;
        float gain;
        std::vector<float> buffer;  // Activity buffer for delay
        size_t buffer_index;
        size_t delay_samples;
        
        FeedbackLoop() : gain(1.0f), buffer_index(0), delay_samples(1) {}
    };
    
    // Communication buffer for delayed signals
    struct DelayedSignal {
        std::vector<float> signal;
        std::string target_module;
        std::string target_port;
        std::chrono::steady_clock::time_point send_time;
        float delay_ms;
    };
    
    // Enhanced state variables
    float attention_weight_;
    bool is_active_;
    int developmental_stage_;
    
    // Feedback and communication
    std::vector<FeedbackLoop> feedback_loops_;
    std::vector<InterModuleConnection> outgoing_connections_;
    std::queue<DelayedSignal> delayed_signals_;
    std::chrono::steady_clock::time_point last_feedback_update_;
    
    // Neuromodulation
    std::map<std::string, float> neuromodulator_levels_;
    
    // Thread safety for enhanced features
    mutable std::mutex module_mutex_;
    
    // ========================================================================
    // HELPER METHODS FOR BIOLOGICALLY-INSPIRED FEATURES
    // ========================================================================
    
    /**
     * @brief Update synaptic homeostasis
     * @param dt Time step for homeostasis
     */
    void updateSynapticHomeostasis(float dt);
    
    /**
     * @brief Update structural plasticity
     * @param dt Time step for structural changes
     */
    void updateStructuralPlasticity(float dt);
    
    /**
     * @brief Process local inhibition mechanisms
     */
    void processLocalInhibition();
    
    /**
     * @brief Apply developmental constraints
     * @param dt Time step for development
     */
    void applyDevelopmentalConstraints(float dt);
    
    /**
     * @brief Process delayed signals in communication buffer
     */
    void processDelayedSignals();
};

/**
 * Central Executive Module - Orchestrates attention and module activation
 */
class CentralExecutiveModule : public EnhancedNeuralModule {
public:
    CentralExecutiveModule(const std::string& name, const NetworkConfig& config);
    
    // Override virtual functions with correct signatures
    bool initialize() override;
    void update(float dt, const std::vector<float>& inputs = {}, float reward = 0.0f) override;
    
    // Executive control functions
    void allocateAttention(const std::map<std::string, float>& module_priorities);
    void orchestrateModules(const std::vector<std::string>& active_modules);
    std::vector<float> computeGlobalControlSignals();
    
    // Decision making
    std::string selectPrimaryModule(const std::map<std::string, float>& activations);
    void inhibitCompetingModules(const std::string& primary_module);
    
private:
    std::map<std::string, float> module_priorities_;
    std::string current_primary_module_;
    float global_inhibition_strength_;
};

/**
 * Specialized Memory Module - Long-term and working memory
 */
class MemoryModule : public EnhancedNeuralModule {
public:
    MemoryModule(const std::string& name, const NetworkConfig& config);
    
    // Override virtual functions
    bool initialize() override;
    void update(float dt, const std::vector<float>& inputs = {}, float reward = 0.0f) override;
    
    // Memory operations
    void storeEpisode(const std::vector<float>& episode_data, const std::string& context);
    std::vector<float> retrieveMemory(const std::vector<float>& cue, float threshold = 0.7f);
    void consolidateMemories(float dt);
    
    // Working memory
    void updateWorkingMemory(const std::vector<float>& current_input);
    std::vector<float> getWorkingMemoryContent() const;
    
private:
    struct MemoryTrace {
        std::vector<float> data;
        std::string context;
        float strength;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::vector<MemoryTrace> long_term_memory_;
    std::vector<float> working_memory_;
    float consolidation_threshold_;
};

#endif // ENHANCED_NEURAL_MODULE_H