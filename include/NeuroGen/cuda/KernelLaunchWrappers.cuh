#ifndef KERNEL_LAUNCH_WRAPPERS_CUH
#define KERNEL_LAUNCH_WRAPPERS_CUH

#include <NeuroGen/cuda/GPUNeuralStructures.h>

namespace KernelLaunchWrappers {

// (Other function declarations remain the same)
void initialize_ion_channels(GPUNeuronState* neurons, int num_neurons);
void update_neuron_states(GPUNeuronState* neurons, float current_time, float dt, int num_neurons);
void update_calcium_dynamics(GPUNeuronState* neurons, float current_time, float dt, int num_neurons);
void run_stdp_and_eligibility(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses
);
void apply_reward_and_adaptation(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float reward,
    float current_time,
    float dt,
    int num_synapses
);

// --- FIX: The function signature is updated to accept current_time. ---
void run_homeostatic_mechanisms(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    float current_time, // Added missing parameter
    int num_neurons,
    int num_synapses
);

} // namespace KernelLaunchWrappers

#endif // KERNEL_LAUNCH_WRAPPERS_CUH