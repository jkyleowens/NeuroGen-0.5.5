#ifndef ELIGIBILITY_AND_REWARD_KERNELS_CUH
#define ELIGIBILITY_AND_REWARD_KERNELS_CUH

// >>> FIX: Added the missing include for the GPU data structures.
// This defines GPUSynapse and GPUNeuronState, resolving the "undefined identifier" errors.
#include <NeuroGen/cuda/GPUNeuralStructures.h>
// <<< END FIX

/**
 * @brief Applies reward (dopamine) signal to consolidate synaptic changes.
 */
__global__ void applyRewardKernel(
    GPUSynapse* synapses,
    float reward,
    float dt,
    int num_synapses
);

/**
 * @brief Adapts the sensitivity of synapses to neuromodulators like dopamine.
 */
__global__ void adaptNeuromodulationKernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_dopamine,
    int num_synapses,
    float current_time // Pass current_time as a parameter
);

#endif // ELIGIBILITY_AND_REWARD_KERNELS_CUH