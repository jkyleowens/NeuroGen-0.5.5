#ifndef ENHANCED_NEURAL_MODULE_H
#define ENHANCED_NEURAL_MODULE_H

#include <NeuroGen/NeuralModule.h>
#include <NeuroGen/TopologyGenerator.h>
#include <memory>
#include <functional>
#include <map>      // <<< FIX: Added missing include for std::map
#include <string>   // <<< FIX: Added missing include for std::string
#include <vector>   // <<< FIX: Added missing include for std::vector

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
    };
    
    // Inter-module connection specification
    struct InterModuleConnection {
        std::string source_port;
        std::string target_module;
        std::string target_port;
        float connection_strength;
        bool is_feedback;  // True for feedback connections
    };

    EnhancedNeuralModule(const std::string& name, const NetworkConfig& config)
        : NeuralModule(name, config), 
          attention_weight_(1.0f),
          is_active_(true),
          developmental_stage_(0) {}

    // State management
    virtual ModuleState saveState() const;
    virtual void loadState(const ModuleState& state);
    
    // Attention mechanism
    void setAttentionWeight(float weight) { attention_weight_ = weight; }
    float getAttentionWeight() const { return attention_weight_; }
    
    // Module activity control
    void setActive(bool active) { is_active_ = active; }
    bool isActive() const { return is_active_; }
    
    // Feedback loop management
    void addFeedbackLoop(const std::string& from_port, const std::string& to_port, 
                        float gain = 1.0f);
    void processFeedback(double dt);
    
    // Inter-module communication
    void registerInterModuleConnection(const InterModuleConnection& connection);
    std::vector<InterModuleConnection> getOutgoingConnections() const { 
        return outgoing_connections_; 
    }
    
    // Neuromodulation interface
    virtual void applyNeuromodulation(const std::string& modulator_type, float level);
    
    // Development and plasticity
    void setDevelopmentalStage(int stage) { developmental_stage_ = stage; }
    int getDevelopmentalStage() const { return developmental_stage_; }
    
    // Performance monitoring
    virtual std::map<std::string, float> getPerformanceMetrics() const;
    
protected:
    // Feedback loop structure
    struct FeedbackLoop {
        std::string from_port;
        std::string to_port;
        float gain;
        std::vector<float> buffer;  // Activity buffer for delay
        size_t buffer_index;
    };
    
    float attention_weight_;
    bool is_active_;
    int developmental_stage_;
    
    std::vector<FeedbackLoop> feedback_loops_;
    std::vector<InterModuleConnection> outgoing_connections_;
    std::map<std::string, float> neuromodulator_levels_;
    
    // Helper methods for biologically-inspired features
    void updateSynapticHomeostasis(double dt);
    void updateStructuralPlasticity(double dt);
    void processLocalInhibition();
};

/**
 * Central Executive Module - Orchestrates attention and module activation
 */
class CentralExecutiveModule : public EnhancedNeuralModule {
public:
    CentralExecutiveModule(const std::string& name, const NetworkConfig& config);
    
    void initialize() override;
    void update(double dt) override;
    
    // Attention control
    void updateAttentionWeights(const std::map<std::string, float>& module_activities);
    std::map<std::string, float> getAttentionDistribution() const { 
        return attention_distribution_; 
    }
    
    // Module activation control
    void setModuleActivation(const std::string& module_name, bool active);
    bool shouldModuleBeActive(const std::string& module_name) const;
    
private:
    std::map<std::string, float> attention_distribution_;
    std::map<std::string, bool> module_activation_states_;
    
    // Winner-take-all mechanism for attention
    void applyWinnerTakeAll();
    
    // Predictive activation based on context
    void predictiveActivation(const std::vector<float>& context_vector);
};

/**
 * Memory Consolidation Module - Handles long-term memory formation
 */
class MemoryConsolidationModule : public EnhancedNeuralModule {
public:
    MemoryConsolidationModule(const std::string& name, const NetworkConfig& config);
    
    void initialize() override;
    void update(double dt) override;
    
    // Memory operations
    void consolidateMemory(const std::vector<float>& pattern, const std::string& context);
    std::vector<float> recallMemory(const std::string& context);
    
    // Sleep-like consolidation phases
    void enterConsolidationPhase();
    void exitConsolidationPhase();
    
private:
    bool in_consolidation_phase_;
    std::map<std::string, std::vector<std::vector<float>>> context_memories_;
    
    // Replay buffer for offline learning
    struct ReplayItem {
        std::vector<float> pattern;
        std::string context;
        float importance;
    };
    std::vector<ReplayItem> replay_buffer_;
    
    void performReplay(size_t num_items);
};

#endif // ENHANCED_NEURAL_MODULE_H