#include "NeuroGen/cuda/EnhancedSTDPKernel.cuh"
#include "NeuroGen/cuda/NeuronModelConstants.h" // Correct header for constants
#include "NeuroGen/LearningRuleConstants.h"
#include <math_constants.h> // For CUDART_PI_F

// >>> FIX: The function signature now exactly matches the declaration in the .cuh file.
__global__ void enhancedSTDPKernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    // Use the correct data structures as defined in the header.
    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;

    const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_idx];
    const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];

    // --- 1. Get Spike Timing Information ---
    float t_pre = pre_neuron.last_spike_time;
    float t_post = post_neuron.last_spike_time;
    float dt_spike = t_post - t_pre;

    // --- 2. Calculate Calcium-Dependent Plasticity ---
    // Get local calcium from the correct compartment of the postsynaptic neuron.
    int compartment_idx = synapse.post_compartment;
    float local_calcium = post_neuron.ca_conc[compartment_idx];
    float calcium_factor = 0.0f;

    // Use the correct, namespaced constants.
    if (local_calcium > NeuronModelConstants::CA_THRESHOLD_LTP) {
        calcium_factor = (local_calcium - NeuronModelConstants::CA_THRESHOLD_LTP) / (1.5f - NeuronModelConstants::CA_THRESHOLD_LTP);
        calcium_factor = fminf(1.0f, calcium_factor);
    } else if (local_calcium > NeuronModelConstants::CA_THRESHOLD_LTD) {
        calcium_factor = (local_calcium - NeuronModelConstants::CA_THRESHOLD_LTD) / (NeuronModelConstants::CA_THRESHOLD_LTP - NeuronModelConstants::CA_THRESHOLD_LTD);
        calcium_factor = -fminf(1.0f, 1.0f - calcium_factor);
    }

    // --- 3. Calculate Spike-Timing-Dependent Component ---
    float timing_factor = 0.0f;
    if (fabsf(dt_spike) < 50.0f) { // 50ms STDP window
        if (dt_spike > 0) { // Pre-before-post (LTP)
            timing_factor = NeuronModelConstants::STDP_A_PLUS * expf(-dt_spike / NeuronModelConstants::STDP_TAU_PLUS);
        } else { // Post-before-pre (LTD)
            timing_factor = -NeuronModelConstants::STDP_A_MINUS * expf(dt_spike / NeuronModelConstants::STDP_TAU_MINUS);
        }
    }

    // --- 4. Combine Factors to Update Eligibility Trace ---
    // The fast trace represents the immediate potential for change.
    if (fabsf(calcium_factor) > 0.01f && fabsf(timing_factor) > 0.0001f) {
        float dw = timing_factor * (1.0f + fabsf(calcium_factor)) * synapse.plasticity_modulation;
        // The eligibility_trace will be used by the reward modulation kernel later.
        atomicAdd(&synapse.eligibility_trace, dw);
    }
}