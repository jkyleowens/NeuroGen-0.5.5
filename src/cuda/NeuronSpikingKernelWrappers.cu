// ============================================================================
// CUDA KERNEL WRAPPER FUNCTIONS FOR NEUROSPIKINGKERNELS
// File: src/cuda/NeuronSpikingKernelWrappers.cu
// ============================================================================

#include "NeuroGen/cuda/NeuronSpikingKernels.cuh"
#include "NeuroGen/cuda/GPUNeuralStructures.h"
#include "NeuroGen/cuda/NeuronModelConstants.h"
#include <cuda_runtime.h>

// ============================================================================
// EXTERNAL LINKAGE WRAPPER FUNCTIONS
// ============================================================================

/**
 * @brief Host wrapper for updateNeuronSpikes kernel with correct signature
 * 
 * This wrapper ensures proper linkage for the NetworkCUDA class while
 * maintaining the biologically accurate spike detection implementation.
 */
extern "C" void updateNeuronSpikes(GPUNeuronState* neurons, float threshold, int num_neurons) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    // Launch the actual CUDA kernel with proper parameters
    updateNeuronSpikes<<<grid, block>>>(neurons, threshold, num_neurons);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in updateNeuronSpikes: %s\n", cudaGetErrorString(error));
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

/**
 * @brief Host wrapper for countSpikesKernel with correct signature
 * 
 * This wrapper provides the exact function signature expected by NetworkCUDA
 * while leveraging our advanced spike counting implementation.
 */
extern "C" void countSpikesKernel(const GPUNeuronState* neurons, int* spike_count, int num_neurons) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    // Launch the actual CUDA kernel
    countSpikesKernel<<<grid, block>>>(neurons, spike_count, num_neurons);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in countSpikesKernel: %s\n", cudaGetErrorString(error));
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

/**
 * @brief Advanced wrapper for comprehensive spike processing
 * 
 * This wrapper provides a unified interface for all spike-related processing,
 * enabling the breakthrough neural architecture to handle complex spike dynamics
 * with optimal performance.
 */
extern "C" void processNeuralSpikes(GPUNeuronState* neurons, int* spike_count, 
                                   float threshold, float current_time, 
                                   int num_neurons, float dt) {
    if (!neurons || !spike_count || num_neurons <= 0) {
        printf("Error: Invalid parameters for processNeuralSpikes\n");
        return;
    }
    
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    // Reset spike counter
    cudaMemset(spike_count, 0, sizeof(int));
    
    // Step 1: Update neuron spike states with biological realism
    updateNeuronSpikes<<<grid, block>>>(neurons, num_neurons, current_time, dt);
    cudaDeviceSynchronize();
    
    // Step 2: Count spikes for network statistics
    countSpikesKernel<<<grid, block>>>(neurons, spike_count, num_neurons, current_time);
    cudaDeviceSynchronize();
    
    // Check for any errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in processNeuralSpikes: %s\n", cudaGetErrorString(error));
    }
}

/**
 * @brief Wrapper for modular spike processing with attention mechanisms
 * 
 * This advanced wrapper supports the modular neural architecture by providing
 * module-aware spike processing with attention-based modulation.
 */
extern "C" void processModularSpikes(GPUNeuronState* neurons, int* spike_count,
                                    int* module_assignments, float* attention_weights,
                                    float threshold, float current_time,
                                    int num_neurons, int num_modules, float dt) {
    if (!neurons || !spike_count || num_neurons <= 0) {
        return;
    }
    
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    // Reset spike counter
    cudaMemset(spike_count, 0, sizeof(int));
    
    // Process spikes with modular awareness
    if (module_assignments && attention_weights) {
        // Use advanced modular spike processing
        launchProcessModularInteractions(neurons, num_neurons, module_assignments,
                                       attention_weights, nullptr, current_time);
    }
    
    // Standard spike detection and counting
    updateNeuronSpikes<<<grid, block>>>(neurons, threshold, num_neurons);
    countSplikesKernel<<<grid, block>>>(neurons, spike_count, num_neurons, current_time);
    
    cudaDeviceSynchronize();
}

// ============================================================================
// COMPATIBILITY LAYER FOR LEGACY INTERFACES
// ============================================================================

/**
 * @brief Legacy compatibility wrapper for older NetworkCUDA interfaces
 */
extern "C" void launchSpikeDetection(GPUNeuronState* d_neurons, int* d_spike_count,
                                    int num_neurons, float threshold, float current_time) {
    processNeuralSpikes(d_neurons, d_spike_count, threshold, current_time, num_neurons, 0.1f);
}

/**
 * @brief Simplified interface for basic spike counting
 */
extern "C" int countActiveNeurons(const GPUNeuronState* neurons, int num_neurons, float threshold) {
    if (!neurons || num_neurons <= 0) return 0;
    
    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    
    countSplikesKernel(neurons, d_count, num_neurons);
    
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
    
    return h_count;
}

// ============================================================================
// KERNEL FUNCTION DECLARATIONS FOR PROPER LINKAGE
// ============================================================================

// Ensure the actual kernel functions are properly declared and linked
__global__ void updateNeuronSpikes(GPUNeuronState* neurons, float threshold, int num_neurons);
__global__ void countSplikesKernel(const GPUNeuronState* neurons, int* spike_count, 
                                   int num_neurons, float current_time);
__global__ void updateNeuronSpikes(GPUNeuronState* neurons, int num_neurons, 
                                  float current_time, float dt);

// Advanced modular processing functions
extern "C" void launchProcessModularInteractions(GPUNeuronState* neurons, int num_neurons,
                                                int* module_assignments, float* attention_weights,
                                                float* global_inhibition, float current_time);

// ============================================================================
// PERFORMANCE MONITORING FUNCTIONS
// ============================================================================

/**
 * @brief Monitor spike processing performance for optimization
 */
extern "C" float benchmarkSpikeProcessing(GPUNeuronState* neurons, int num_neurons, 
                                         int iterations = 100) {
    if (!neurons || num_neurons <= 0) return 0.0f;
    
    int* d_spike_count;
    cudaMalloc(&d_spike_count, sizeof(int));
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        updateNeuronSpikes(neurons, 30.0f, num_neurons);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        updateNeuronSpikes(neurons, 30.0f, num_neurons);
        countSplikesKernel(neurons, d_spike_count, num_neurons);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    cudaFree(d_spike_count);
    
    return static_cast<float>(duration.count()) / iterations; // microseconds per iteration
}

/**
 * @brief Validate spike processing accuracy for debugging
 */
extern "C" bool validateSpikeProcessing(GPUNeuronState* neurons, int num_neurons) {
    if (!neurons || num_neurons <= 0) return false;
    
    // Copy neurons to host for validation
    std::vector<GPUNeuronState> h_neurons(num_neurons);
    cudaMemcpy(h_neurons.data(), neurons, num_neurons * sizeof(GPUNeuronState), 
               cudaMemcpyDeviceToHost);
    
    // Basic validation checks
    bool valid = true;
    for (int i = 0; i < num_neurons; i++) {
        const auto& neuron = h_neurons[i];
        
        // Check for reasonable voltage values
        if (neuron.V < -100.0f || neuron.V > 100.0f) {
            printf("Warning: Neuron %d has unreasonable voltage: %f\n", i, neuron.V);
            valid = false;
        }
        
        // Check for reasonable calcium values
        for (int c = 0; c < 4; c++) {
            if (neuron.ca_conc[c] < 0.0f || neuron.ca_conc[c] > 50.0f) {
                printf("Warning: Neuron %d compartment %d has unreasonable calcium: %f\n", 
                       i, c, neuron.ca_conc[c]);
                valid = false;
            }
        }
    }
    
    return valid;
}