#ifndef KERNEL_LAUNCH_WRAPPERS_CUH
#define KERNEL_LAUNCH_WRAPPERS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/cuda/CorticalColumn.h>

/**
 * @brief Comprehensive kernel wrapper declarations for breakthrough neural network
 * 
 * This header provides the complete interface for GPU-accelerated biological neural
 * simulation, including advanced plasticity, neuromodulation, and cortical column
 * processing that enables your brain-mimicking technology.
 */

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CORE NEURAL SIMULATION KERNELS
// ============================================================================

/**
 * @brief Advanced Runge-Kutta 4th order neuron state integration
 * Provides biologically accurate temporal dynamics for breakthrough realism
 */
void launchRK4NeuronUpdateKernel(GPUNeuronState* neurons, int N, float dt, float current_time);

/**
 * @brief Process synaptic input propagation across cortical modules
 */
void launchSynapseInputKernelInternal(GPUSynapse* synapses, GPUNeuronState* neurons, int num_synapses);

/**
 * @brief Update synaptic state variables and neurotransmitter dynamics
 */
void launchUpdateSynapseStatesKernel(dim3 blocks, dim3 threads, GPUSynapse* synapses, int num_synapses, float dt);

/**
 * @brief Process spike generation and propagation events
 */
void launchProcessSpikesKernel(dim3 blocks, dim3 threads, GPUNeuronState* neurons, 
                              int* spike_counts, float current_time, int num_neurons);

/**
 * @brief Apply external input currents to specified neurons
 */
void launchApplyInputCurrentsKernel(dim3 blocks, dim3 threads, GPUNeuronState* neurons, 
                                   const float* input_data, int input_size, int num_neurons);

// ============================================================================
// NEURAL STATE INITIALIZATION AND RESET KERNELS
// ============================================================================

/**
 * @brief Initialize neuron states with biologically realistic parameters
 */
void launchInitializeNeuronStatesKernel(dim3 blocks, dim3 threads, GPUNeuronState* neurons, int num_neurons);

/**
 * @brief Initialize synapse states with distance-based connectivity
 */
void launchInitializeSynapseStatesKernel(dim3 blocks, dim3 threads, GPUSynapse* synapses, int num_synapses);

/**
 * @brief Initialize random number generator states for stochastic processes
 */
void launchInitializeRandomStatesKernel(dim3 blocks, dim3 threads, curandState* states, 
                                       int num_states, unsigned long seed);

/**
 * @brief Reset neuron states to resting conditions
 */
void launchResetNeuronStatesKernel(dim3 blocks, dim3 threads, GPUNeuronState* neurons, int num_neurons);

/**
 * @brief Reset synapse states and eligibility traces
 */
void launchResetSynapseStatesKernel(dim3 blocks, dim3 threads, GPUSynapse* synapses, int num_synapses);

/**
 * @brief Reset spike detection flags for next timestep
 */
void resetNeuronStatesWrapper(dim3 blocks, dim3 threads, GPUNeuronState* neurons, int num_neurons);

// ============================================================================
// ADVANCED PLASTICITY AND LEARNING KERNELS
// ============================================================================

/**
 * @brief Apply reward-modulated plasticity for breakthrough learning
 */
void launchApplyRewardModulationKernel(dim3 blocks, dim3 threads, GPUSynapse* synapses, 
                                      float reward, int num_synapses);

/**
 * @brief Update Hebbian learning and STDP mechanisms
 */
void launchApplyHebbianLearningKernel(dim3 blocks, dim3 threads, GPUSynapse* synapses, 
                                     GPUNeuronState* neurons, int num_synapses);

/**
 * @brief Update eligibility traces for temporal credit assignment
 */
void launchUpdateEligibilityTracesKernel(dim3 blocks, dim3 threads, GPUSynapse* synapses, 
                                        GPUNeuronState* neurons, float dt, int num_synapses);

/**
 * @brief Apply homeostatic scaling for long-term stability
 */
void launchApplyHomeostaticScalingKernel(dim3 blocks, dim3 threads, GPUSynapse* synapses, int num_synapses);

/**
 * @brief Process structural plasticity (synapse creation/deletion)
 */
void launchStructuralPlasticityKernel(dim3 blocks, dim3 threads, GPUSynapse* synapses, 
                                     GPUNeuronState* neurons, int num_synapses, float dt);

// ============================================================================
// CORTICAL COLUMN AND MODULAR ARCHITECTURE KERNELS
// ============================================================================

/**
 * @brief Initialize cortical columns with specialized functions
 */
void launchInitializeCorticalColumnsKernel(dim3 blocks, dim3 threads, CorticalColumn* columns, int num_columns);

/**
 * @brief Update cortical column dynamics and inter-column communication
 */
void launchUpdateCorticalColumnsKernel(dim3 blocks, dim3 threads, CorticalColumn* columns, 
                                      GPUNeuronState* neurons, int num_columns, float dt);

/**
 * @brief Process inter-column connectivity and specialization
 */
void launchColumnConnectivityKernel(dim3 blocks, dim3 threads, CorticalColumn* columns, 
                                   int num_columns, float connection_probability);

/**
 * @brief Update column specialization based on input patterns
 */
void launchColumnSpecializationKernel(dim3 blocks, dim3 threads, CorticalColumn* columns, 
                                     const float* input_patterns, int num_columns, int pattern_size);

// ============================================================================
// NEUROMODULATION AND BRAIN-LIKE DYNAMICS KERNELS
// ============================================================================

/**
 * @brief Update neuromodulator concentrations and effects
 */
void launchUpdateNeuromodulationKernel(dim3 blocks, dim3 threads, GPUNeuronState* neurons, 
                                      CorticalColumn* columns, float dopamine, float acetylcholine,
                                      float serotonin, float norepinephrine, int num_neurons);

/**
 * @brief Apply attention-based modulation to neural activity
 */
void launchApplyAttentionModulationKernel(dim3 blocks, dim3 threads, GPUNeuronState* neurons, 
                                         CorticalColumn* columns, const float* attention_weights, 
                                         int num_neurons);

/**
 * @brief Process oscillatory dynamics and phase synchronization
 */
void launchOscillationSynchronizationKernel(dim3 blocks, dim3 threads, GPUNeuronState* neurons, 
                                           CorticalColumn* columns, float target_frequency, 
                                           int num_neurons);

// ============================================================================
// ANALYSIS AND MONITORING KERNELS
// ============================================================================

/**
 * @brief Calculate network statistics and health metrics
 */
void launchCalculateNetworkStatsKernel(dim3 blocks, dim3 threads, const GPUNeuronState* neurons, 
                                      const GPUSynapse* synapses, float* output_stats, 
                                      int num_neurons, int num_synapses);

/**
 * @brief Extract network output with biological encoding
 */
void launchExtractOutputKernel(dim3 blocks, dim3 threads, const GPUNeuronState* neurons, 
                              float* output_buffer, int output_size, float current_time);

/**
 * @brief Monitor network criticality and stability
 */
void launchCriticalityMonitoringKernel(dim3 blocks, dim3 threads, const GPUNeuronState* neurons, 
                                      const CorticalColumn* columns, float* criticality_metrics, 
                                      int num_neurons);

// ============================================================================
// UTILITY AND DEBUGGING KERNELS
// ============================================================================

/**
 * @brief Test GPU memory accessibility and performance
 */
void testMemoryAccessWrapper(GPUNeuronState* neurons, int num_neurons);

/**
 * @brief Initialize neuron compartment structures
 */
void initializeNeuronCompartments(GPUNeuronState* neurons, int num_neurons);

/**
 * @brief Validate neural network state integrity
 */
void launchValidateNetworkStateKernel(dim3 blocks, dim3 threads, const GPUNeuronState* neurons, 
                                     const GPUSynapse* synapses, int* error_flags, 
                                     int num_neurons, int num_synapses);

// ============================================================================
// WRAPPER FUNCTION DECLARATIONS
// ============================================================================

/**
 * @brief High-level wrapper functions for complex kernel sequences
 */

// Comprehensive network update sequence
void updateFullNetworkWrapper(GPUNeuronState* neurons, GPUSynapse* synapses, 
                             CorticalColumn* columns, const float* inputs, 
                             float reward, float dt, int num_neurons, 
                             int num_synapses, int num_columns);

// Plasticity update sequence
void updatePlasticityWrapper(GPUSynapse* synapses, GPUNeuronState* neurons, 
                           float reward, float learning_rate, float dt, 
                           int num_synapses);

// Cortical column update sequence  
void updateColumnsWrapper(CorticalColumn* columns, GPUNeuronState* neurons, 
                         const float* attention_weights, float dt, 
                         int num_columns, int num_neurons);

// Neuromodulation update sequence
void updateNeuromodulationWrapper(GPUNeuronState* neurons, CorticalColumn* columns,
                                 float dopamine, float acetylcholine, 
                                 float serotonin, float norepinephrine,
                                 int num_neurons, int num_columns);

// Homeostatic scaling sequence
void applyHomeostaticScalingWrapper(dim3 blocks, dim3 threads, GPUSynapse* synapses, int num_synapses);

// ============================================================================
// ADVANCED BIOLOGICAL MECHANISM KERNELS
// ============================================================================

/**
 * @brief Simulate vesicle depletion and recovery
 */
void launchVesicleDepletionKernel(dim3 blocks, dim3 threads, GPUSynapse* synapses, 
                                 float depletion_rate, float recovery_rate, float dt, 
                                 int num_synapses);

/**
 * @brief Update calcium dynamics and buffering
 */
void launchCalciumDynamicsKernel(dim3 blocks, dim3 threads, GPUNeuronState* neurons, 
                                float* calcium_levels, float dt, int num_neurons);

/**
 * @brief Process metabolic constraints and energy usage
 */
void launchMetabolicConstraintsKernel(dim3 blocks, dim3 threads, GPUNeuronState* neurons, 
                                     CorticalColumn* columns, float energy_budget, 
                                     int num_neurons);

/**
 * @brief Simulate glial cell interactions and support
 */
void launchGlialInteractionKernel(dim3 blocks, dim3 threads, GPUNeuronState* neurons, 
                                 float* glial_factors, float dt, int num_neurons);

// ============================================================================
// PERFORMANCE OPTIMIZATION KERNELS
// ============================================================================

/**
 * @brief Optimized sparse matrix operations for connectivity
 */
void launchSparseConnectivityKernel(dim3 blocks, dim3 threads, const int* connectivity_matrix, 
                                   const float* weights, GPUNeuronState* neurons, 
                                   int num_neurons, int max_connections);

/**
 * @brief Vectorized neuron update for maximum throughput
 */
void launchVectorizedNeuronUpdateKernel(dim3 blocks, dim3 threads, GPUNeuronState* neurons, 
                                       const float* input_currents, float dt, 
                                       int num_neurons);

/**
 * @brief Memory-coalesced synapse processing
 */
void launchCoalescedSynapseKernel(dim3 blocks, dim3 threads, GPUSynapse* synapses, 
                                 const GPUNeuronState* pre_neurons, 
                                 GPUNeuronState* post_neurons, int num_synapses);

#ifdef __cplusplus
}
#endif

// ============================================================================
// INLINE UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Calculate optimal kernel launch parameters for given workload
 */
inline void calculateOptimalLaunchParams(int workload_size, dim3& blocks, dim3& threads) {
    threads.x = (workload_size < 256) ? workload_size : 256;
    threads.y = 1;
    threads.z = 1;
    
    blocks.x = (workload_size + threads.x - 1) / threads.x;
    blocks.y = 1;
    blocks.z = 1;
    
    // Clamp to maximum grid size
    if (blocks.x > 65535) {
        blocks.y = (blocks.x + 65535 - 1) / 65535;
        blocks.x = 65535;
    }
}

/**
 * @brief Validate kernel launch parameters for safety
 */
inline bool validateLaunchParams(dim3 blocks, dim3 threads, int expected_workload) {
    int total_threads = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
    return (total_threads >= expected_workload) && 
           (threads.x <= 1024) && (threads.y <= 1024) && (threads.z <= 1024) &&
           (blocks.x <= 65535) && (blocks.y <= 65535) && (blocks.z <= 65535);
}

/**
 * @brief Get recommended shared memory size for kernel
 */
inline size_t getRecommendedSharedMemory(int threads_per_block, size_t per_thread_memory) {
    size_t total = threads_per_block * per_thread_memory;
    // Clamp to maximum shared memory (48KB typical)
    return (total > 49152) ? 49152 : total;
}

#endif // KERNEL_LAUNCH_WRAPPERS_CUH