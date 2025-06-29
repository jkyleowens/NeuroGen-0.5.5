#ifndef HOMEOSTATIC_MECHANISMS_KERNEL_CUH
#define HOMEOSTATIC_MECHANISMS_KERNEL_CUH

#include <NeuroGen/cuda/GPUNeuralStructures.h>

/**
 * @brief Adjusts incoming synaptic weights to maintain a target firing rate.
 */
__global__ void synapticScalingKernel(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int num_neurons,
    int total_synapses,
    float current_time
);

/**
 * @brief Adjusts a neuron's intrinsic excitability to maintain a target activity level.
 */
__global__ void intrinsicPlasticityKernel(GPUNeuronState* neurons, int num_neurons);

#endif // HOMEOSTATIC_MECHANISMS_KERNEL_CUH