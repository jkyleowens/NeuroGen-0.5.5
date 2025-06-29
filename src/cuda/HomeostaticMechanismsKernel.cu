#include "NeuroGen/cuda/HomeostaticMechanismsKernel.cuh"
#include "NeuroGen/cuda/NeuronModelConstants.h"

// --- Kernel for Synaptic Scaling ---
__global__ void synapticScalingKernel(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    int num_neurons,
    int total_synapses,
    float current_time)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[neuron_idx];

    // --- 1. Update long-term average firing rate ---
    float decay = expf(-0.1f / NeuronModelConstants::HOMEOSTATIC_TIMESCALE);
    bool just_spiked = (neuron.last_spike_time >= current_time - 0.1f);
    float instantaneous_rate = just_spiked ? 10.0f : 0.0f; // Simplified rate in Hz
    neuron.average_firing_rate = neuron.average_firing_rate * decay + instantaneous_rate * (1.0f - decay);

    // --- 2. Calculate Scaling Adjustment ---
    float rate_error = neuron.average_firing_rate - NeuronModelConstants::TARGET_FIRING_RATE;
    float scaling_adjustment = -rate_error * 0.0001f; // Slow adjustment

    neuron.synaptic_scaling_factor = fmaxf(0.5f, fminf(1.5f, neuron.synaptic_scaling_factor + scaling_adjustment));
}

// --- Kernel for Intrinsic Plasticity ---
__global__ void intrinsicPlasticityKernel(GPUNeuronState* neurons, int num_neurons) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[neuron_idx];

    // --- 1. Update long-term average activity (membrane potential) ---
    float decay = expf(-0.1f / NeuronModelConstants::HOMEOSTATIC_TIMESCALE);
    neuron.average_activity = neuron.average_activity * decay + neuron.V * (1.0f - decay);

    // --- 2. Adjust Intrinsic Excitability ---
    float activity_error = neuron.average_activity - NeuronModelConstants::RESTING_POTENTIAL;
    float excitability_change = -activity_error * 0.00005f; // Very slow adjustment

    neuron.excitability = fmaxf(0.8f, fminf(1.2f, neuron.excitability + excitability_change));
}