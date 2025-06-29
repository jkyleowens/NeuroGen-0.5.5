#include "NeuroGen/cuda/NetworkCUDA.cuh"
#include "NeuroGen/cuda/KernelLaunchWrappers.cuh" // Includes all kernel launchers
#include <iostream>
#include <stdexcept>
#include <string>

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::string error_msg = std::string("CUDA Error: ") + cudaGetErrorString(err) + \
                                " at " + __FILE__ + ":" + std::to_string(__LINE__) + \
                                " in function '" + __func__ + "'"; \
        std::cerr << error_msg << std::endl; \
        throw std::runtime_error(error_msg); \
    } \
}

NetworkCUDA::NetworkCUDA(const NetworkConfig& config)
    // --- FIX: Use correct members from NetworkConfig. Total neurons = hidden_size.
    : config_(config),
      num_neurons_(config.hidden_size),
      num_synapses_(config.totalSynapses) {
    // --- END FIX ---
    cudaStreamCreate(&stream_);
    allocate_gpu_memory();
    initialize_gpu_state();
    std::cout << "NetworkCUDA initialized for " << num_neurons_ << " neurons and " << num_synapses_ << " synapses." << std::endl;
}

NetworkCUDA::~NetworkCUDA() {
    free_gpu_memory();
    cudaStreamDestroy(stream_);
}

void NetworkCUDA::simulate_step(float current_time, float dt, float reward, const std::vector<float>& inputs) {
    if (inputs.size() > 0) {
        // Copy new input currents to the device
        CUDA_CHECK(cudaMemcpyAsync(d_input_currents_, inputs.data(), inputs.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
    }

    // --- FIX: Call the correct, existing kernel wrappers ---
    // Note: The logic for synaptic input processing and application is typically part of the main neuron update kernel.
    // The wrappers are now called in a logical sequence for a simulation step.

    // 1. Update neuron states (potential, recovery, etc.)
    KernelLaunchWrappers::update_neuron_states(d_neurons_, current_time, dt, num_neurons_);

    // 2. Update calcium dynamics based on new neuron states
    KernelLaunchWrappers::update_calcium_dynamics(d_neurons_, current_time, dt, num_neurons_);

    // 3. Run STDP to calculate immediate eligibility for plasticity
    KernelLaunchWrappers::run_stdp_and_eligibility(d_synapses_, d_neurons_, current_time, dt, num_synapses_);

    // 4. Apply reward signals and adapt neuromodulation
    KernelLaunchWrappers::apply_reward_and_adaptation(d_synapses_, d_neurons_, reward, current_time, dt, num_synapses_);

    // 5. Run long-term homeostatic mechanisms
    KernelLaunchWrappers::run_homeostatic_mechanisms(d_neurons_, d_synapses_, current_time, num_neurons_, num_synapses_);
    // --- END FIX ---
}

void NetworkCUDA::get_stats(NetworkStats& stats) const {
    // For now, we just copy basic counts. A more detailed implementation
    // would run kernels to gather dynamic stats like firing rates.
    stats.active_neuron_count = num_neurons_;
    stats.active_synapses = num_synapses_;
}

void NetworkCUDA::copy_to_gpu(const std::vector<GPUNeuronState>& neurons, const std::vector<GPUSynapse>& synapses) {
    if (neurons.size() != num_neurons_ || synapses.size() != num_synapses_) {
        throw std::runtime_error("Mismatched sizes in copy_to_gpu.");
    }
    CUDA_CHECK(cudaMemcpyAsync(d_neurons_, neurons.data(), num_neurons_ * sizeof(GPUNeuronState), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_synapses_, synapses.data(), num_synapses_ * sizeof(GPUSynapse), cudaMemcpyHostToDevice, stream_));
    cudaStreamSynchronize(stream_);
}

void NetworkCUDA::copy_from_gpu(std::vector<GPUNeuronState>& neurons, std::vector<GPUSynapse>& synapses) {
    if (neurons.size() != num_neurons_ || synapses.size() != num_synapses_) {
        throw std::runtime_error("Mismatched sizes in copy_from_gpu.");
    }
    CUDA_CHECK(cudaMemcpyAsync(neurons.data(), d_neurons_, num_neurons_ * sizeof(GPUNeuronState), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(synapses.data(), d_synapses_, num_synapses_ * sizeof(GPUSynapse), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);
}

void NetworkCUDA::allocate_gpu_memory() {
    // --- FIX: Use correct data types and config members ---
    CUDA_CHECK(cudaMalloc(&d_neurons_, num_neurons_ * sizeof(GPUNeuronState)));
    CUDA_CHECK(cudaMalloc(&d_synapses_, num_synapses_ * sizeof(GPUSynapse)));
    // Use input_size for the input current buffer
    CUDA_CHECK(cudaMalloc(&d_input_currents_, config_.input_size * sizeof(float)));
    // --- END FIX ---
}

void NetworkCUDA::free_gpu_memory() {
    cudaFree(d_neurons_);
    cudaFree(d_synapses_);
    cudaFree(d_input_currents_);
    d_neurons_ = nullptr;
    d_synapses_ = nullptr;
    d_input_currents_ = nullptr;
}

void NetworkCUDA::initialize_gpu_state() {
    KernelLaunchWrappers::initialize_ion_channels(d_neurons_, num_neurons_);
}