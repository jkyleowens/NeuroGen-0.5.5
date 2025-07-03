#ifndef ENHANCED_LEARNING_SYSTEM_H
#define ENHANCED_LEARNING_SYSTEM_H

// Core NeuroGen includes with proper paths
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/cuda/GridBlockUtils.cuh>

// Enhanced learning rule components
#include <NeuroGen/LearningRuleConstants.h>
#include <NeuroGen/cuda/EnhancedSTDPKernel.cuh>
#include <NeuroGen/cuda/EligibilityAndRewardKernels.cuh>
#include <NeuroGen/cuda/RewardModulationKernel.cuh>
#include <NeuroGen/cuda/HebbianLearningKernel.cuh>
#include <NeuroGen/cuda/HomeostaticMechanismsKernel.cuh>

#include <cuda_runtime.h>
#include <vector>
#include <memory>

// Enhanced learning system constants
#define BASELINE_DOPAMINE 0.4f
#define UPDATE_FREQUENCY 50.0f    // ms
#define TARGET_ACTIVITY_LEVEL 0.1f
#define PROTEIN_SYNTHESIS_THRESHOLD 0.8f

/**
 * Enhanced Learning System Manager
 * Coordinates multiple biologically-inspired learning mechanisms
 * in a neurobiologically realistic manner
 */
class EnhancedLearningSystem {
private:
    // GPU memory pointers
    GPUSynapse* d_synapses_;
    GPUNeuronState* d_neurons_;
    float* d_network_stats_;
    float* d_trace_stats_;
    float* d_correlation_matrix_;
    
    // Network parameters
    int num_synapses_;
    int num_neurons_;
    int correlation_matrix_size_;
    
    // Learning state variables
    float current_dopamine_level_;
    float current_reward_signal_;
    float predicted_reward_;
    float prediction_error_;
    float network_reward_trace_;
    float protein_synthesis_signal_;
    
    // Timing and execution control
    float last_update_time_;
    float plasticity_update_interval_;
    float homeostatic_update_interval_;
    float trace_update_interval_;
    
    // Performance monitoring
    float total_weight_change_;
    float average_trace_activity_;
    int plasticity_updates_count_;
    
    // CUDA execution parameters
    dim3 synapse_grid_, synapse_block_;
    dim3 neuron_grid_, neuron_block_;
    
public:
    /**
     * Constructor initializes the enhanced learning system
     */
    EnhancedLearningSystem(int num_synapses, int num_neurons);
    
    /**
     * Destructor cleans up GPU memory
     */
    ~EnhancedLearningSystem();
    
    /**
     * Main update function coordinating all learning mechanisms
     * This is called from the main network simulation loop
     */
    void updateLearning(GPUSynapse* synapses, 
                       GPUNeuronState* neurons,
                       float current_time, 
                       float dt,
                       float external_reward = 0.0f);
    
    /**
     * Set external reward signal for the network
     */
    void setRewardSignal(float reward) {
        current_reward_signal_ = reward;
    }
    
    /**
     * Trigger protein synthesis for late-phase plasticity
     */
    void triggerProteinSynthesis(float strength = 1.0f);
    /**
     * Get learning system statistics
     */
    struct LearningStats {
        float total_weight_change;
        float average_trace_activity;
        float current_dopamine_level;
        float prediction_error;
        float network_activity;
        int plasticity_updates;
    };
    
    LearningStats getStatistics() const;
    
    /**
     * Reset learning system state (for episodic learning)
     */
    void resetEpisode(bool reset_traces = true, bool reset_rewards = true);

private:
    /**
     * Initialize GPU memory for learning system
     */
    void initializeGPUMemory();
    
    /**
     * Configure CUDA execution parameters
     */
    void configureExecutionParameters();
    
    /**
     * Update eligibility traces across all timescales
     */
    void updateEligibilityTraces(float current_time, float dt);
    
    /**
     * Update enhanced STDP with biological realism
     */
    void updateEnhancedSTDP(float current_time, float dt);
    
    /**
     * Update Hebbian learning mechanisms
     */
    void updateHebbianLearning(float current_time, float dt);
    
    /**
     * Update reward prediction and dopaminergic modulation
     */
    void updateRewardModulation(float current_time, float dt, float external_reward);
    
    /**
     * Update metaplasticity mechanisms
     */
    void updateMetaplasticity(float current_time, float dt);
    
    /**
     * Update synaptic scaling for homeostasis
     */
    void updateSynapticScaling(float current_time, float dt);
    
    /**
     * Update weight normalization
     */
    void updateWeightNormalization();
    
    /**
     * Update activity regulation
     */
    void updateActivityRegulation(float current_time, float dt);
    
    /**
     * Update network monitoring
     */
    void updateNetworkMonitoring();
    
    /**
     * Update late-phase plasticity
     */
    void updateLatePhrasePlasticity(float current_time, float dt);
    
    /**
     * Check network stability and apply emergency measures if needed
     */
    void checkNetworkStability(float current_time);
    
    /**
     * Clean up GPU memory
     */
    void cleanupGPUMemory();
    
    /**
     * Check for CUDA errors
     */
    void checkCudaErrors();
};

#endif // ENHANCED_LEARNING_SYSTEM_H