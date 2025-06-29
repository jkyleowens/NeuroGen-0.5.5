#include <NeuroGen/cuda/NetworkCUDA.cuh>
#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>
#include <NeuroGen/cuda/CudaUtils.h>
#include <stdexcept>
#include <iostream>

// Global network statistics definition (as it's declared extern in the header)
__managed__ NetworkStats g_stats;

NetworkCUDA::NetworkCUDA(const NetworkConfig& config)
    : config(config),
      d_neurons(nullptr),
      d_synapses(nullptr),
      d_calcium_levels(nullptr),
      d_neuron_spike_counts(nullptr),
      d_random_states(nullptr),
      d_cortical_columns(nullptr),
      d_input_currents(nullptr),
      current_time_ms(0.0f),
      network_initialized(false),
      plasticity_enabled(true),
      current_learning_rate(0.01f) {
    try {
        initializeNetwork();
        network_initialized = true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize NetworkCUDA: " << e.what() << std::endl;
        cleanup();
        throw;
    }
}

NetworkCUDA::~NetworkCUDA() {
    cleanup();
}

void NetworkCUDA::update(float dt_ms, const std::vector<float>& input_currents, float reward) {
    if (!network_initialized) return;

    current_time_ms += dt_ms;
    g_stats.current_reward = reward;
    g_stats.total_simulation_time += dt_ms;

    int num_neurons = getNumNeurons();
    int num_synapses = getNumSynapses();

    // Copy input currents to device
    if (!input_currents.empty()) {
        cudaMemcpy(d_input_currents, input_currents.data(), input_currents.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Launch kernels
    launchSynapseInputKernelInternal(d_synapses, d_neurons, num_synapses);
    launchRK4NeuronUpdateKernel(d_neurons, num_neurons, dt_ms, current_time_ms);
    
    // Placeholder for other updates like plasticity, etc.
    // updatePlasticity(reward);

    updateNetworkStatistics();
}

std::vector<float> NetworkCUDA::getOutput() const {
    int num_neurons = getNumNeurons();
    std::vector<float> output(num_neurons);
    std::vector<GPUNeuronState> host_neurons(num_neurons);
    cudaMemcpy(host_neurons.data(), d_neurons, num_neurons * sizeof(GPUNeuronState), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_neurons; ++i) {
        output[i] = host_neurons[i].voltage;
    }
    return output;
}

void NetworkCUDA::reset() {
    current_time_ms = 0.0f;
    g_stats.reset();
    
    int num_neurons = getNumNeurons();
    dim3 blocks, threads;
    calculateGridBlockSize(num_neurons, blocks, threads);
    resetNeuronStatesWrapper(blocks, threads, d_neurons, num_neurons);
}

NetworkStats NetworkCUDA::getStats() const {
    return g_stats;
}

void NetworkCUDA::setRewardSignal(float reward) {
    g_stats.current_reward = reward;
}

void NetworkCUDA::printNetworkState() const {
    // Implementation can be filled in to print relevant stats
    std::cout << "Network State - Time: " << current_time_ms << "ms, Spikes: " << g_stats.total_spikes << std::endl;
}

std::vector<float> NetworkCUDA::getNeuronVoltages() const {
    return getOutput();
}

std::vector<float> NetworkCUDA::getSynapticWeights() const {
    int num_synapses = getNumSynapses();
    std::vector<float> weights(num_synapses);
    std::vector<GPUSynapse> host_synapses(num_synapses);
    cudaMemcpy(host_synapses.data(), d_synapses, num_synapses * sizeof(GPUSynapse), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_synapses; ++i) {
        weights[i] = host_synapses[i].weight;
    }
    return weights;
}

void NetworkCUDA::initializeNetwork() {
    allocateDeviceMemory();
    initializeDeviceArrays();
}

void NetworkCUDA::cleanup() {
    cudaFree(d_neurons);
    cudaFree(d_synapses);
    cudaFree(d_calcium_levels);
    cudaFree(d_neuron_spike_counts);
    cudaFree(d_random_states);
    cudaFree(d_cortical_columns);
    cudaFree(d_input_currents);
}

void NetworkCUDA::allocateDeviceMemory() {
    int num_neurons = getNumNeurons();
    int num_synapses = getNumSynapses();
    
    cudaMalloc(&d_neurons, num_neurons * sizeof(GPUNeuronState));
    cudaMalloc(&d_synapses, num_synapses * sizeof(GPUSynapse));
    cudaMalloc(&d_input_currents, num_neurons * sizeof(float));
    cudaMalloc(&d_random_states, num_neurons * sizeof(curandState));
    // Allocate other device memory...
}

void NetworkCUDA::initializeDeviceArrays() {
    int num_neurons = getNumNeurons();
    dim3 blocks, threads;
    calculateGridBlockSize(num_neurons, blocks, threads);
    initializeNeuronStatesWrapper(blocks, threads, d_neurons, num_neurons);
    initializeRandomStatesWrapper(blocks, threads, d_random_states, num_neurons, time(0));
}

void NetworkCUDA::calculateGridBlockSize(int n_elements, dim3& grid, dim3& block) const {
    block.x = 256;
    grid.x = (n_elements + block.x - 1) / block.x;
}

void NetworkCUDA::updateNetworkStatistics() {
    // This would involve a kernel to gather statistics from the GPU
}

void NetworkCUDA::validateConfig() const {
    // Add validation logic for the config
}