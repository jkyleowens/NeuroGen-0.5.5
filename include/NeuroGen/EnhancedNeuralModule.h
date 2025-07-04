#ifndef ENHANCED_NEURAL_MODULE_H
#define ENHANCED_NEURAL_MODULE_H

#include <NeuroGen/NeuralModule.h>
#include <NeuroGen/TopologyGenerator.h>
#include <memory>
#include <functional>
#include <chrono>
#include <queue>
#include <map>

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
        bool is_active;
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
    
    /**
     * @brief Get current neural outputs from the module
     * @return Vector of current neuron outputs
     */
    std::vector<float> getOutputs() const;
    
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
     * @param weight Attention weight [0-1]
     */
    void setAttentionWeight(float weight) { attention_weight_ = weight; }
    
    /**
     * @brief Get current attention weight
     * @return Current attention weight
     */
    float getAttentionWeight() const { return attention_weight_; }
    
    /**
     * @brief Update attention based on context
     * @param context_vector Context information
     */
    void updateAttention(const std::vector<float>& context_vector);
    
    // ========================================================================
    // BIOLOGICAL FEATURES
    // ========================================================================
    
    /**
     * @brief Set developmental stage of the module
     * @param stage Developmental stage (0-5)
     */
    void setDevelopmentalStage(int stage) { developmental_stage_ = stage; }
    
    /**
     * @brief Get current developmental stage
     * @return Developmental stage
     */
    int getDevelopmentalStage() const { return developmental_stage_; }
    
    /**
     * @brief Apply neuromodulation to the module
     * @param modulator_type Type of neuromodulator
     * @param level Modulation level
     */
    virtual void applyNeuromodulation(const std::string& modulator_type, float level) override;
    
    /**
     * @brief Process internal feedback loops
     * @param dt Time step
     */
    void processFeedbackLoops(float dt);
    
    /**
     * @brief Add inter-module connection
     * @param connection Connection specification
     */
    void addInterModuleConnection(const InterModuleConnection& connection);
    
    /**
     * @brief Register inter-module connection (alias for addInterModuleConnection)
     * @param connection Connection specification
     */
    void registerInterModuleConnection(const InterModuleConnection& connection) {
        addInterModuleConnection(connection);
    }
    
    /**
     * @brief Send signal to connected modules
     * @param signal_data Data to send
     * @param target_port Target port name
     */
    void sendInterModuleSignal(const std::vector<float>& signal_data, 
                               const std::string& target_port);
    
    /**
     * @brief Receive signal from other modules
     * @param signal_data Received data
     * @param source_port Source port name
     */
    void receiveInterModuleSignal(const std::vector<float>& signal_data,
                                  const std::string& source_port);
    
    // ========================================================================
    // ACTIVITY AND STATUS
    // ========================================================================
    
    /**
     * @brief Check if module is currently active
     * @return Activity status
     */
    bool isActive() const { return is_active_; }
    
    /**
     * @brief Set module activity status
     * @param active New activity status
     */
    void setActive(bool active) { is_active_ = active; }
    
    /**
     * @brief Get module specialization type
     * @return Specialization description
     */
    virtual std::string getSpecializationType() const { return "general"; }

protected:
    // ========================================================================
    // PROTECTED MEMBER VARIABLES
    // ========================================================================
    
    float attention_weight_;
    bool is_active_;
    int developmental_stage_;
    std::chrono::steady_clock::time_point last_feedback_update_;
    
    // Inter-module communication
    std::vector<InterModuleConnection> connections_;
    std::map<std::string, std::queue<std::vector<float>>> input_buffers_;
    std::map<std::string, std::vector<float>> output_buffers_;
    
    // Feedback loop state
    std::vector<float> feedback_state_;
    float feedback_strength_;
    
    // Neuromodulator levels
    std::map<std::string, float> neuromodulator_levels_;
    
    // ========================================================================
    // PROTECTED HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Update internal biological processes
     * @param dt Time step
     */
    void updateBiologicalProcesses(float dt);
    
    /**
     * @brief Compute attention-weighted output
     * @param raw_output Raw module output
     * @return Attention-weighted output
     */
    std::vector<float> applyAttentionWeighting(const std::vector<float>& raw_output) const;
    
    /**
     * @brief Process developmental changes
     * @param dt Time step
     */
    void updateDevelopmentalState(float dt);
    
    /**
     * @brief Process inter-module communication signals
     */
    void processInterModuleCommunication();
};

#endif // ENHANCED_NEURAL_MODULE_H