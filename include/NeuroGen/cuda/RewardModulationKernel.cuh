// ============================================================================
// CUDA COMPILATION FIXES
// File: include/NeuroGen/cuda/RewardModulationKernel.cuh
// ============================================================================

#ifndef REWARD_MODULATION_KERNEL_CUH
#define REWARD_MODULATION_KERNEL_CUH

// CRITICAL FIX: Add proper CUDA math includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <math.h>        // For host functions
#include <cmath>         // For C++ math functions
#include "NeuroGen/cuda/GPUNeuralStructures.h"

// CUDA math functions are in global namespace when compiling .cu files
// Make sure this file is compiled as .cu, not .cpp

/**
 * @brief Enhanced reward modulation kernel with proper CUDA math functions
 */
__global__ void rewardModulationKernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float reward_signal,
    float current_time,
    float dt,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // FIX: Use CUDA device functions instead of host functions
    float decay_factor = __expf(-dt / 1000.0f); // Use __expf for device code
    
    // Enhanced reward modulation with biological realism
    float dopamine_concentration = reward_signal * (1.0f + sinf(current_time * 0.001f));
    
    // Apply reward-dependent plasticity
    float eligibility_weighted_change = synapse.eligibility_trace * dopamine_concentration * dt;
    synapse.weight += eligibility_weighted_change * 0.001f; // Learning rate
    
    // Bound synaptic weights
    synapse.weight = fmaxf(0.0f, fminf(synapse.weight, 10.0f));
    
    // Update neuromodulator dynamics
    synapse.dopamine_level = synapse.dopamine_level * decay_factor + dopamine_concentration * dt;
    synapse.eligibility_trace *= decay_factor;
}

/**
 * @brief Reward prediction error computation kernel
 */
__global__ void rewardPredictionErrorKernel(
    float* predicted_rewards,
    float* actual_rewards,
    float* rpe_output,
    int num_timesteps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_timesteps) return;
    
    rpe_output[idx] = actual_rewards[idx] - predicted_rewards[idx];
}

#endif // REWARD_MODULATION_KERNEL_CUH



