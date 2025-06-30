// ============================================================================
// CUDA WRAPPER FUNCTIONS IMPLEMENTATION
// File: src/cuda/LearningSystemWrappers.cu
// ============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

// Include GPU structures (must be available in .cu compilation)
#include <NeuroGen/cuda/GPUNeuralStructures.h>

// ============================================================================
// CUDA KERNEL IMPLEMENTATIONS
// ============================================================================

/**
 * @brief Eligibility trace reset kernel with biological decay dynamics
 */
__global__ void eligibility_trace_reset_kernel(GPUSynapse* synapses, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Biological eligibility trace reset with protein degradation
    synapse.eligibility_trace *= 0.95f; // Rapid degradation
    synapse.protein_synthesis_rate *= 0.99f; // Slower protein degradation
    
    // Reset dopamine sensitivity if below threshold
    if (synapse.dopamine_sensitivity < 0.1f) {
        synapse.dopamine_sensitivity = 0.5f; // Baseline sensitivity
    }
}

/**
 * @brief Enhanced STDP kernel with multi-factor plasticity
 */
__global__ void enhanced_stdp_kernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Validate neuron indices
    if (synapse.pre_neuron_id >= 0 && synapse.post_neuron_id >= 0) {
        const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_id];
        const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_id];
        
        // Calculate spike timing difference
        float delta_t = post_neuron.last_spike_time - pre_neuron.last_spike_time;
        
        // Biological STDP with calcium-dependent modulation
        float stdp_window = 20.0f; // 20ms STDP window
        float calcium_factor = post_neuron.calcium_concentration / 1.0f; // Normalize
        
        if (fabsf(delta_t) < stdp_window) {
            float stdp_magnitude;
            
            if (delta_t > 0) {
                // LTP: Post before pre (causal)
                stdp_magnitude = __expf(-delta_t / 10.0f) * calcium_factor;
                synapse.weight += stdp_magnitude * 0.01f * dt;
            } else {
                // LTD: Pre before post (anti-causal)
                stdp_magnitude = __expf(delta_t / 10.0f) * calcium_factor;
                synapse.weight -= stdp_magnitude * 0.005f * dt;
            }
            
            // Update eligibility trace
            synapse.eligibility_trace += stdp_magnitude * 0.1f;
        }
        
        // Bound synaptic weight
        synapse.weight = fmaxf(0.0f, fminf(synapse.weight, 5.0f));
    }
}

/**
 * @brief Eligibility trace update kernel with biological dynamics
 */
__global__ void eligibility_trace_update_kernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Exponential decay of eligibility traces
    float decay_rate = 1.0f / 1000.0f; // 1 second time constant
    float decay_factor = __expf(-dt * decay_rate);
    
    synapse.eligibility_trace *= decay_factor;
    
    // Update based on recent synaptic activity
    if (synapse.pre_neuron_id >= 0 && synapse.post_neuron_id >= 0) {
        const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_id];
        const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_id];
        
        // Add contribution from recent activity
        float activity_contribution = pre_neuron.membrane_potential * post_neuron.membrane_potential;
        synapse.eligibility_trace += activity_contribution * 0.001f * dt;
    }
    
    // Bound eligibility trace
    synapse.eligibility_trace = fmaxf(0.0f, fminf(synapse.eligibility_trace, 2.0f));
}

/**
 * @brief Trace monitoring kernel for statistics collection
 */
__global__ void trace_monitoring_kernel(
    const GPUSynapse* synapses,
    int num_synapses,
    float* trace_stats
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // Use shared memory for efficient reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_max[256];
    
    int tid = threadIdx.x;
    shared_sum[tid] = 0.0f;
    shared_max[tid] = 0.0f;
    
    // Process multiple synapses per thread if necessary
    if (idx < num_synapses) {
        float trace_value = synapses[idx].eligibility_trace;
        shared_sum[tid] = trace_value;
        shared_max[tid] = trace_value;
    }
    
    __syncthreads();
    
    // Reduction for statistics
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write results to global memory
    if (tid == 0) {
        atomicAdd(&trace_stats[0], shared_sum[0]); // Sum
        atomicMax((int*)&trace_stats[1], __float_as_int(shared_max[0])); // Max
    }
}

/**
 * @brief Reward modulation kernel with dopaminergic dynamics
 */
__global__ void reward_modulation_kernel(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float reward,
    float current_time,
    float dt,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    // Dopaminergic modulation with realistic dynamics
    float dopamine_release = reward * (1.0f + 0.1f * sinf(current_time * 0.01f));
    float dopamine_decay = __expf(-dt / 100.0f); // 100ms dopamine half-life
    
    // Update dopamine concentration
    synapse.dopamine_level = synapse.dopamine_level * dopamine_decay + dopamine_release * dt;
    
    // Apply reward-dependent plasticity
    float eligibility_weighted_change = synapse.eligibility_trace * synapse.dopamine_level;
    float learning_rate = 0.001f * synapse.dopamine_sensitivity;
    
    synapse.weight += eligibility_weighted_change * learning_rate * dt;
    
    // Update dopamine sensitivity (metaplasticity)
    synapse.dopamine_sensitivity += (dopamine_release - 0.5f) * 0.0001f * dt;
    synapse.dopamine_sensitivity = fmaxf(0.1f, fminf(synapse.dopamine_sensitivity, 2.0f));
    
    // Bound synaptic weight
    synapse.weight = fmaxf(0.0f, fminf(synapse.weight, 10.0f));
}

/**
 * @brief Hebbian learning kernel with BCM-like threshold adaptation
 */
__global__ void hebbian_learning_kernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    if (synapse.pre_neuron_id >= 0 && synapse.post_neuron_id >= 0) {
        const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_id];
        const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_id];
        
        // BCM-like plasticity with sliding threshold
        float pre_activity = pre_neuron.membrane_potential / 70.0f; // Normalize
        float post_activity = post_neuron.membrane_potential / 70.0f;
        
        // Sliding threshold based on recent activity
        float activity_threshold = post_neuron.average_activity;
        
        // BCM rule: LTP if above threshold, LTD if below
        float plasticity_signal = post_activity * (post_activity - activity_threshold) * pre_activity;
        
        // Apply Hebbian change
        float hebbian_rate = 0.0001f;
        synapse.weight += plasticity_signal * hebbian_rate * dt;
        
        // Bound weight
        synapse.weight = fmaxf(0.0f, fminf(synapse.weight, 5.0f));
    }
}

/**
 * @brief BCM learning kernel with explicit threshold dynamics
 */
__global__ void bcm_learning_kernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float learning_rate,
    float dt,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    if (synapse.pre_neuron_id >= 0 && synapse.post_neuron_id >= 0) {
        const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_id];
        const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_id];
        
        // BCM plasticity with homeostatic threshold
        float pre_rate = pre_neuron.firing_rate;
        float post_rate = post_neuron.firing_rate;
        float threshold = post_neuron.bcm_threshold;
        
        // BCM learning rule
        float weight_change = learning_rate * pre_rate * post_rate * (post_rate - threshold) * dt;
        
        synapse.weight += weight_change;
        synapse.weight = fmaxf(0.0f, fminf(synapse.weight, 5.0f));
    }
}

/**
 * @brief Correlation-based learning kernel with matrix operations
 */
__global__ void correlation_learning_kernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float* correlation_matrix,
    float learning_rate,
    float dt,
    int num_synapses,
    int matrix_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    GPUSynapse& synapse = synapses[idx];
    
    if (synapse.pre_neuron_id >= 0 && synapse.post_neuron_id >= 0 && 
        synapse.pre_neuron_id < matrix_size && synapse.post_neuron_id < matrix_size) {
        
        const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_id];
        const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_id];
        
        // Update correlation matrix
        int matrix_idx = synapse.post_neuron_id * matrix_size + synapse.pre_neuron_id;
        float correlation = pre_neuron.membrane_potential * post_neuron.membrane_potential;
        
        // Exponential moving average of correlations
        correlation_matrix[matrix_idx] = correlation_matrix[matrix_idx] * 0.99f + correlation * 0.01f;
        
        // Apply correlation-based learning
        float correlation_strength = correlation_matrix[matrix_idx];
        synapse.weight += correlation_strength * learning_rate * dt;
        
        // Bound weight
        synapse.weight = fmaxf(0.0f, fminf(synapse.weight, 5.0f));
    }
}

/**
 * @brief Reward prediction error computation kernel
 */
__global__ void reward_prediction_error_kernel(
    const float* actual_reward,
    float* predicted_rewards,
    int num_timesteps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_timesteps) return;
    
    // Simple TD error computation
    float prediction_error = *actual_reward - predicted_rewards[idx];
    
    // Update prediction using simple learning rule
    predicted_rewards[idx] += 0.1f * prediction_error;
    
    // Store error for further processing (could be expanded)
    // This is a simplified version - full implementation would include
    // temporal difference learning with value function approximation
}

// ============================================================================
// C++ WRAPPER FUNCTIONS (called from CPU code)
// ============================================================================

extern "C" {

void launch_eligibility_reset_wrapper(void* d_synapses, int num_synapses) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    eligibility_trace_reset_kernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses), num_synapses);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in eligibility_reset: %s\n", cudaGetErrorString(error));
    }
}

void launch_enhanced_stdp_wrapper(
    void* d_synapses, 
    const void* d_neurons,
    float current_time, 
    float dt, 
    int num_synapses
) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    enhanced_stdp_kernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        current_time, dt, num_synapses);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in enhanced_stdp: %s\n", cudaGetErrorString(error));
    }
}

void launch_eligibility_update_wrapper(
    void* d_synapses,
    const void* d_neurons,
    float current_time,
    float dt,
    int num_synapses
) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    eligibility_trace_update_kernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        current_time, dt, num_synapses);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in eligibility_update: %s\n", cudaGetErrorString(error));
    }
}

void launch_trace_monitoring_wrapper(
    const void* d_synapses,
    int num_synapses,
    void* d_trace_stats
) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    trace_monitoring_kernel<<<grid, block>>>(
        static_cast<const GPUSynapse*>(d_synapses),
        num_synapses,
        static_cast<float*>(d_trace_stats));
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in trace_monitoring: %s\n", cudaGetErrorString(error));
    }
}

void launch_reward_modulation_wrapper(
    void* d_synapses,
    void* d_neurons,
    float reward,
    float current_time,
    float dt,
    int num_synapses
) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    reward_modulation_kernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<GPUNeuronState*>(d_neurons),
        reward, current_time, dt, num_synapses);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in reward_modulation: %s\n", cudaGetErrorString(error));
    }
}

void launch_hebbian_learning_wrapper(
    void* d_synapses,
    const void* d_neurons,
    float current_time,
    float dt,
    int num_synapses
) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    hebbian_learning_kernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        current_time, dt, num_synapses);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in hebbian_learning: %s\n", cudaGetErrorString(error));
    }
}

void launch_bcm_learning_wrapper(
    void* d_synapses,
    const void* d_neurons,
    float learning_rate,
    float dt,
    int num_synapses
) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    bcm_learning_kernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        learning_rate, dt, num_synapses);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in bcm_learning: %s\n", cudaGetErrorString(error));
    }
}

void launch_correlation_learning_wrapper(
    void* d_synapses,
    const void* d_neurons,
    void* d_correlation_matrix,
    float learning_rate,
    float dt,
    int num_synapses,
    int matrix_size
) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    correlation_learning_kernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        static_cast<float*>(d_correlation_matrix),
        learning_rate, dt, num_synapses, matrix_size);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in correlation_learning: %s\n", cudaGetErrorString(error));
    }
}

void launch_reward_prediction_error_wrapper(
    const void* d_actual_reward,
    void* d_predicted_rewards,
    int num_timesteps
) {
    dim3 block(256);
    dim3 grid((num_timesteps + block.x - 1) / block.x);
    
    reward_prediction_error_kernel<<<grid, block>>>(
        static_cast<const float*>(d_actual_reward),
        static_cast<float*>(d_predicted_rewards),
        num_timesteps);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in reward_prediction_error: %s\n", cudaGetErrorString(error));
    }
}

} // extern "C"