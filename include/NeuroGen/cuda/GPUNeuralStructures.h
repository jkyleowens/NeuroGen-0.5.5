#ifndef GPU_NEURAL_STRUCTURES_H
#define GPU_NEURAL_STRUCTURES_H

// This header defines the core data structures used on the GPU.

struct GPUSynapse {
    // --- Core Connectivity & State ---
    int pre_neuron_idx;
    int post_neuron_idx;
    int post_compartment;
    int active;

    // --- Weight & Delay ---
    float weight;
    float delay;

    // --- Plasticity & State Tracking ---
    float eligibility_trace;
    float plasticity_modulation;
    float effective_weight;
    float last_pre_spike_time;
    float last_post_spike_time;
    float last_active_time;
    float activity_metric;
    float max_weight;
    float min_weight;
    float dopamine_sensitivity;

    // >>> FIX: Added missing member for acetylcholine neuromodulation.
    float acetylcholine_sensitivity;
    // <<< END FIX
};

struct GPUNeuronState {
    float V;
    float u;
    float I_syn[4];
    float ca_conc[4];
    float last_spike_time;
    float average_firing_rate;
    float average_activity;
    float excitability;
    float synaptic_scaling_factor;
};

#endif // GPU_NEURAL_STRUCTURES_H