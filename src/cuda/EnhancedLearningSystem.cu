// ============================================================================
// ENHANCED LEARNING SYSTEM IMPLEMENTATION
// File: src/cuda/EnhancedLearningSystem.cu
// ============================================================================

#include "NeuroGen/EnhancedLearningSystem.h"
#include <cuda_runtime.h>
#include <cstdio>

// ============================================================================
// IMPLEMENTATION OF EnhancedLearningSystem METHODS
// ============================================================================

void EnhancedLearningSystem::updateLearning(GPUSynapse* synapses, 
                                           GPUNeuronState* neurons,
                                           float current_time, 
                                           float dt,
                                           float external_reward) {
    
    // Store device pointers
    d_synapses_ = synapses;
    d_neurons_ = neurons;
    
    // Update timing
    float time_since_last_update = current_time - last_update_time_;
    
    // ========================================
    // PHASE 1: ELIGIBILITY TRACE UPDATES (High Frequency)
    // ========================================
    if (time_since_last_update >= trace_update_interval_) {
        updateEligibilityTraces(current_time, dt);
    }
    
    // ========================================
    // PHASE 2: CORE PLASTICITY MECHANISMS (Medium Frequency)
    // ========================================
    if (time_since_last_update >= plasticity_update_interval_) {
        
        // Update STDP with enhanced mechanisms
        updateEnhancedSTDP(current_time, dt);
        
        // Update Hebbian learning
        updateHebbianLearning(current_time, dt);
        
        // Update reward prediction and modulation
        updateRewardModulation(current_time, dt, external_reward);
        
        // Update metaplasticity
        updateMetaplasticity(current_time, dt);
    }
    
    // ========================================
    // PHASE 3: HOMEOSTATIC MECHANISMS (Low Frequency)
    // ========================================
    if (time_since_last_update >= homeostatic_update_interval_) {
        
        // Synaptic scaling
        updateSynapticScaling(current_time, dt);
        
        // Weight normalization
        updateWeightNormalization();
        
        // Activity regulation
        updateActivityRegulation(current_time, dt);
        
        // Network monitoring
        updateNetworkMonitoring();
    }
    
    // ========================================
    // PHASE 4: LATE-PHASE PLASTICITY (Conditional)
    // ========================================
    if (protein_synthesis_signal_ > PROTEIN_SYNTHESIS_THRESHOLD) {
        updateLatePhrasePlasticity(current_time, dt);
    }
    
    // ========================================
    // PHASE 5: EMERGENCY STABILIZATION (As Needed)
    // ========================================
    checkNetworkStability(current_time);
    
    // Update timing and statistics
    last_update_time_ = current_time;
    plasticity_updates_count_++;
    
    // Synchronize GPU execution
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    checkCudaErrors();
}

void EnhancedLearningSystem::resetEpisode(bool reset_traces, bool reset_rewards) {
    if (reset_traces) {
        eligibilityTraceResetKernel<<<synapse_grid_, synapse_block_>>>(
            d_synapses_, num_synapses_, true, true, false);
    }
    
    if (reset_rewards) {
        current_reward_signal_ = 0.0f;
        predicted_reward_ = 0.0f;
        prediction_error_ = 0.0f;
        network_reward_trace_ = 0.0f;
    }
    
    cudaDeviceSynchronize();
}

// ============================================================================
// PRIVATE METHOD IMPLEMENTATIONS
// ============================================================================

void EnhancedLearningSystem::initializeGPUMemory() {
    // Allocate network statistics arrays
    cudaMalloc(&d_network_stats_, 4 * sizeof(float));
    cudaMalloc(&d_trace_stats_, 4 * sizeof(float));
    
    // Allocate correlation matrix
    int matrix_elements = correlation_matrix_size_ * correlation_matrix_size_;
    cudaMalloc(&d_correlation_matrix_, matrix_elements * sizeof(float));
    
    // Initialize arrays to zero
    cudaMemset(d_network_stats_, 0, 4 * sizeof(float));
    cudaMemset(d_trace_stats_, 0, 4 * sizeof(float));
    cudaMemset(d_correlation_matrix_, 0, matrix_elements * sizeof(float));
}

void EnhancedLearningSystem::configureExecutionParameters() {
    // Configure for synapses
    synapse_block_ = dim3(256);
    synapse_grid_ = dim3((num_synapses_ + synapse_block_.x - 1) / synapse_block_.x);
    
    // Configure for neurons
    neuron_block_ = dim3(256);
    neuron_grid_ = dim3((num_neurons_ + neuron_block_.x - 1) / neuron_block_.x);
}

void EnhancedLearningSystem::updateEligibilityTraces(float current_time, float dt) {
    eligibilityTraceUpdateKernel<<<synapse_grid_, synapse_block_>>>(
        d_synapses_, d_neurons_, current_time, dt, num_synapses_);
    
    // Monitor trace statistics
    traceMonitoringKernel<<<synapse_grid_, synapse_block_>>>(
        d_synapses_, num_synapses_, d_trace_stats_);
}

void EnhancedLearningSystem::updateEnhancedSTDP(float current_time, float dt) {
    enhancedSTDPKernel<<<synapse_grid_, synapse_block_>>>(
        d_synapses_, d_neurons_, current_time, dt, num_synapses_);
}

void EnhancedLearningSystem::updateHebbianLearning(float current_time, float dt) {
    // Core Hebbian learning
    hebbianLearningKernel<<<synapse_grid_, synapse_block_>>>(
        d_synapses_, d_neurons_, current_time, dt, num_synapses_);
    
    // BCM rule for sliding threshold plasticity
    bcmLearningKernel<<<synapse_grid_, synapse_block_>>>(
        d_synapses_, d_neurons_, HEBBIAN_LEARNING_RATE * 0.1f, dt, num_synapses_);
    
    // Correlation-based learning
    correlationLearningKernel<<<synapse_grid_, synapse_block_>>>(
        d_synapses_, d_neurons_, d_correlation_matrix_, 
        HEBBIAN_LEARNING_RATE * 0.05f, dt, num_synapses_, correlation_matrix_size_);
}

void EnhancedLearningSystem::updateRewardModulation(float current_time, float dt, float external_reward) {
    // Compute reward prediction error
    float* d_predicted_reward;
    float* d_prediction_error;
    float* d_dopamine_level;
    
    cudaMalloc(&d_predicted_reward, sizeof(float));
    cudaMalloc(&d_prediction_error, sizeof(float));
    cudaMalloc(&d_dopamine_level, sizeof(float));
    
    cudaMemcpy(d_dopamine_level, &current_dopamine_level_, sizeof(float), cudaMemcpyHostToDevice);
    
    rewardPredictionErrorKernel<<<1, 1>>>(
        d_neurons_, external_reward, d_predicted_reward, 
        d_prediction_error, d_dopamine_level, current_time, dt, num_neurons_);
    
    // Copy results back
    cudaMemcpy(&predicted_reward_, d_predicted_reward, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&prediction_error_, d_prediction_error, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&current_dopamine_level_, d_dopamine_level, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Apply reward modulation to synaptic plasticity
    rewardModulationKernel<<<synapse_grid_, synapse_block_>>>(
        d_synapses_, d_neurons_, external_reward, current_dopamine_level_,
        prediction_error_, current_time, dt, num_synapses_);
    
    // Update dopamine sensitivity adaptation
    float average_reward = network_reward_trace_ * 0.1f; // Simplified
    dopamineSensitivityAdaptationKernel<<<synapse_grid_, synapse_block_>>>(
        d_synapses_, d_neurons_, average_reward, current_time, dt, num_synapses_);
    
    // Update network reward trace
    rewardTraceUpdateKernel<<<1, 1>>>(
        &network_reward_trace_, external_reward, dt);
    
    // Cleanup temporary GPU memory
    cudaFree(d_predicted_reward);
    cudaFree(d_prediction_error);
    cudaFree(d_dopamine_level);
}

void EnhancedLearningSystem::updateMetaplasticity(float current_time, float dt) {
    updateMetaplasticityKernel<<<synapse_grid_, synapse_block_>>>(
        d_synapses_, d_neurons_, current_time, dt, num_synapses_);
}

void EnhancedLearningSystem::updateSynapticScaling(float current_time, float dt) {
    // Compute scaling factors
    synapticScalingKernel<<<neuron_grid_, neuron_block_>>>(
        d_neurons_, d_synapses_, current_time, dt, num_synapses_, num_neurons_);
    
    // Apply scaling factors
    applySynapticScalingKernel<<<synapse_grid_, synapse_block_>>>(
        d_synapses_, d_neurons_, num_synapses_);
}

void EnhancedLearningSystem::updateWeightNormalization() {
    int* d_synapse_counts;
    cudaMalloc(&d_synapse_counts, num_neurons_ * sizeof(int));
    cudaMemset(d_synapse_counts, 0, num_neurons_ * sizeof(int));
    
    weightNormalizationKernel<<<neuron_grid_, neuron_block_>>>(
        d_synapses_, d_synapse_counts, num_synapses_, num_neurons_);
    
    cudaFree(d_synapse_counts);
}

void EnhancedLearningSystem::updateActivityRegulation(float current_time, float dt) {
    activityRegulationKernel<<<neuron_grid_, neuron_block_>>>(
        d_neurons_, current_time, dt, num_neurons_);
}

void EnhancedLearningSystem::updateNetworkMonitoring() {
    // Reset statistics
    cudaMemset(d_network_stats_, 0, 4 * sizeof(float));
    
    networkHomeostaticMonitoringKernel<<<neuron_grid_, neuron_block_>>>(
        d_neurons_, d_synapses_, d_network_stats_, num_neurons_, num_synapses_);
}

void EnhancedLearningSystem::updateLatePhrasePlasticity(float current_time, float dt) {
    latePhaseePlasticityKernel<<<synapse_grid_, synapse_block_>>>(
        d_synapses_, d_neurons_, protein_synthesis_signal_, 
        current_time, dt, num_synapses_);
    
    // Decay protein synthesis signal
    protein_synthesis_signal_ *= expf(-dt / 30000.0f); // 30-second decay
}

void EnhancedLearningSystem::checkNetworkStability(float current_time) {
    // Get network activity level
    float network_stats[4];
    cudaMemcpy(network_stats, d_network_stats_, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    float network_activity = network_stats[0];
    float emergency_threshold = TARGET_ACTIVITY_LEVEL * num_neurons_;
    
    // Apply emergency stabilization if needed
    emergencyStabilizationKernel<<<max(synapse_grid_.x, neuron_grid_.x), 256>>>(
        d_synapses_, d_neurons_, network_activity, emergency_threshold,
        num_synapses_, num_neurons_);
}

void EnhancedLearningSystem::cleanupGPUMemory() {
    if (d_network_stats_) cudaFree(d_network_stats_);
    if (d_trace_stats_) cudaFree(d_trace_stats_);
    if (d_correlation_matrix_) cudaFree(d_correlation_matrix_);
}

void EnhancedLearningSystem::checkCudaErrors() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in Enhanced Learning System: %s\n", 
               cudaGetErrorString(error));
    }
}

EnhancedLearningSystem::LearningStats EnhancedLearningSystem::getStatistics() const {
    LearningStats stats;
    stats.total_weight_change = total_weight_change_;
    stats.average_trace_activity = average_trace_activity_;
    stats.current_dopamine_level = current_dopamine_level_;
    stats.prediction_error = prediction_error_;
    stats.plasticity_updates = plasticity_updates_count_;
    
    // Get network activity from GPU
    float network_stats[4] = {0};
    cudaMemcpy(network_stats, d_network_stats_, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    stats.network_activity = network_stats[0];
    
    return stats;
}
