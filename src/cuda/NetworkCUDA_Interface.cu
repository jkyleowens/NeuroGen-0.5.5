#include <NeuroGen/cuda/NetworkCUDA_Interface.h>

// --- Constructor ---
// Initializes the underlying NetworkCUDA object and copies initial data to the GPU.
NetworkCUDA_Interface::NetworkCUDA_Interface(
    const NetworkConfig& config,
    const std::vector<GPUNeuronState>& neurons,
    const std::vector<GPUSynapse>& synapses)
{
    // Create the CUDA network manager.
    cuda_network_ = std::make_unique<NetworkCUDA>(config);

    // Perform the initial copy of the network topology to the GPU.
    cuda_network_->copy_to_gpu(neurons, synapses);
}

// --- Destructor ---
// The default destructor is sufficient as the unique_ptr will handle cleanup.
NetworkCUDA_Interface::~NetworkCUDA_Interface() = default;

// --- Simulation Step ---
// Delegates the step call to the underlying CUDA network object.
void NetworkCUDA_Interface::step(float current_time, float dt, float reward, const std::vector<float>& inputs) {
    if (cuda_network_) {
        cuda_network_->simulate_step(current_time, dt, reward, inputs);
    }
}

// --- Get Statistics ---
// Fetches the latest stats from the CUDA network object.
NetworkStats NetworkCUDA_Interface::get_stats() const {
    NetworkStats stats = {};
    if (cuda_network_) {
        cuda_network_->get_stats(stats);
    }
    return stats;
}

// --- Get Full Network State ---
// Copies the entire state of neurons and synapses from the GPU to host memory.
void NetworkCUDA_Interface::get_network_state(std::vector<GPUNeuronState>& neurons, std::vector<GPUSynapse>& synapses) {
    if (cuda_network_) {
        cuda_network_->copy_from_gpu(neurons, synapses);
    }
}