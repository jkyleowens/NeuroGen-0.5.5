#include "NeuroGen/cuda/NetworkCUDA.cuh"
#include "NeuroGen/NetworkException.h"
#include "NeuroGen/cuda/CudaUtils.h"
#include "NeuroGen/cuda/KernelLaunchWrappers.h" // <-- ADDED: For all kernel wrappers
#include "NeuroGen/cuda/GPUNeuralStructures.h"  // <-- ADDED: For GPUCorticalColumn etc.
#include "NeuroGen/global.h"                    // <-- ADDED: For g_stats
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

// Constructor, Destructor, and Core Interface
NetworkCUDA::NetworkCUDA(const NetworkConfig& config) :
    config(config),
    initialized(false),
    plasticity_enabled(true),
    learning_rate(0.01f),
    d_neurons(nullptr),
    d_synapses(nullptr),
    d_cortical_columns(nullptr),
    d_random_states(nullptr),
    d_input_currents(nullptr),
    d_reward_signal(nullptr),
    d_neuron_spike_counts(nullptr)
{
    try {
        validateConfig();
        initializeNetwork();
    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        cleanup();
        throw;
    }
}

NetworkCUDA::~NetworkCUDA() {
    cleanup();
}

void NetworkCUDA::update(float dt, const std::vector<float>& input, float reward) {
    if (!initialized) {
        throw NetworkException(NetworkError::NETWORK_NOT_INITIALIZED, "Cannot update uninitialized network.");
    }

    // Copy input data to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_input_currents, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_reward_signal, &reward, sizeof(float), cudaMemcpyHostToDevice));

    // Launch CUDA kernels for one time step
    applyInputCurrentsWrapper(d_neurons, d_input_currents, config.numInputs, config.numNeurons);
    processSynapticInputsWrapper(d_neurons, d_synapses, config.numSynapses, config.numNeurons);
    updateNeuronsWrapper(dt);
    processSpikingWrapper();
    if (plasticity_enabled) {
        applyPlasticityWrapper(dt);
    }

    // Update stats (can be done less frequently for performance)
    // For now, we do it every step
    g_stats.total_spikes += 0; // Placeholder, update properly
}

std::vector<float> NetworkCUDA::getOutput() const {
    if (!initialized) {
        throw NetworkException(NetworkError::NETWORK_NOT_INITIALIZED, "Cannot get output from uninitialized network.");
    }
    std::vector<float> output(config.numOutputs);
    // Assuming output neurons are the last 'numOutputs' neurons
    CUDA_CHECK_ERROR(cudaMemcpy(output.data(), d_neurons + (config.numNeurons - config.numOutputs), config.numOutputs * sizeof(float), cudaMemcpyDeviceToHost));
    return output;
}

void NetworkCUDA::reset() {
    if (!initialized) {
        throw NetworkException(NetworkError::NETWORK_NOT_INITIALIZED, "Cannot reset uninitialized network.");
    }
    initializeDeviceArrays();
}

NetworkStats NetworkCUDA::getStats() const {
    return g_stats;
}

// Control Methods
void NetworkCUDA::setLearningRate(float rate) {
    if (rate < 0) {
        throw NetworkException(NetworkError::INVALID_INPUT, "Learning rate must be non-negative.");
    }
    learning_rate = rate;
}

void NetworkCUDA::setRewardSignal(float reward) {
     if (!initialized) return;
     CUDA_CHECK_ERROR(cudaMemcpy(d_reward_signal, &reward, sizeof(float), cudaMemcpyHostToDevice));
}

void NetworkCUDA::enablePlasticity(bool enable) {
    plasticity_enabled = enable;
}

// Monitoring Methods
void NetworkCUDA::printNetworkState() const {
    if (!initialized) {
        std::cout << "Network is not initialized." << std::endl;
        return;
    }
    std::cout << "--- Network State ---" << std::endl;
    std::cout << "Neurons: " << config.numNeurons << ", Synapses: " << config.numSynapses << std::endl;
    std::cout << "Plasticity enabled: " << (plasticity_enabled ? "Yes" : "No") << std::endl;
    std::cout << "Average firing rate: " << g_stats.getAverageFiringRate() << " Hz" << std::endl;
}

std::vector<float> NetworkCUDA::getNeuronVoltages() const {
    if (!initialized) return {};
    std::vector<float> voltages(config.numNeurons);
    // You would need a kernel to extract just voltages, or copy the whole struct array
    // This is simplified and inefficient for demonstration
    std::vector<GPUNeuron> host_neurons(config.numNeurons);
    CUDA_CHECK_ERROR(cudaMemcpy(host_neurons.data(), d_neurons, config.numNeurons * sizeof(GPUNeuron), cudaMemcpyDeviceToHost));
    for(size_t i = 0; i < config.numNeurons; ++i) voltages[i] = host_neurons[i].V_m;
    return voltages;
}

std::vector<float> NetworkCUDA::getSynapticWeights() const {
     if (!initialized) {
        throw NetworkException(NetworkError::NETWORK_NOT_INITIALIZED, "Cannot get weights from uninitialized network.");
    }
    std::vector<float> weights(config.numSynapses);
    std::vector<GPUSynapse> host_synapses(config.numSynapses);
    CUDA_CHECK_ERROR(cudaMemcpy(host_synapses.data(), d_synapses, config.numSynapses * sizeof(GPUSynapse), cudaMemcpyDeviceToHost));
    for(size_t i = 0; i < config.numSynapses; ++i) weights[i] = host_synapses[i].weight;
    return weights;
}

// Initialization and Cleanup
void NetworkCUDA::initializeNetwork() {
    CudaUtils::selectOptimalDevice();
    allocateDeviceMemory();
    initializeDeviceArrays();
    initialized = true;
    std::cout << "CUDA Network Initialized." << std::endl;
}

void NetworkCUDA::allocateDeviceMemory() {
    CUDA_CHECK_ERROR(cudaMalloc(&d_neurons, config.numNeurons * sizeof(GPUNeuron)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_synapses, config.numSynapses * sizeof(GPUSynapse)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_random_states, config.numNeurons * sizeof(curandState)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_input_currents, config.numInputs * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_reward_signal, sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_neuron_spike_counts, config.numNeurons * sizeof(int)));
    if (config.numColumns > 0) {
       CUDA_CHECK_ERROR(cudaMalloc(&d_cortical_columns, config.numColumns * sizeof(CorticalColumn)));
    }
}

void NetworkCUDA::initializeDeviceArrays() {
    dim3 blocks, threads;
    calculateGridBlockSize(config.numNeurons, blocks, threads);
    
    initializeNeuronStatesWrapper(blocks, threads, d_neurons, config.numNeurons);
    initializeRandomStatesWrapper(blocks, threads, d_random_states, time(0));
    CUDA_CHECK_ERROR(cudaMemset(d_neuron_spike_counts, 0, config.numNeurons * sizeof(int)));
    
    if (config.numColumns > 0) {
        initializeColumns();
    }
    generateDistanceBasedSynapses();
}

void NetworkCUDA::initializeColumns() {
    if (config.numColumns > 0) {
        initializeCorticalColumnsWrapper(d_cortical_columns, config.numColumns);
    }
}

void NetworkCUDA::cleanup() {
    if (d_neurons) cudaFree(d_neurons);
    if (d_synapses) cudaFree(d_synapses);
    if (d_random_states) cudaFree(d_random_states);
    if (d_input_currents) cudaFree(d_input_currents);
    if (d_reward_signal) cudaFree(d_reward_signal);
    if (d_neuron_spike_counts) cudaFree(d_neuron_spike_counts);
    if (d_cortical_columns) cudaFree(d_cortical_columns);
    initialized = false;
}

// Kernel Launch Wrappers
void NetworkCUDA::updateNeuronsWrapper(float dt) {
    dim3 blocks, threads;
    calculateGridBlockSize(config.numNeurons, blocks, threads);
    updateNeuronStatesWrapper(blocks, threads, d_neurons, config.numNeurons, dt);
}

void NetworkCUDA::updateSynapsesWrapper(float dt) {
    dim3 blocks, threads;
    calculateGridBlockSize(config.numSynapses, blocks, threads);
    // Note: The original call was incorrect. There's no single 'updateSynapseStatesWrapper'
    // It's broken into parts like plasticity. This section may need refactoring based
    // on the intended simulation logic.
}

void NetworkCUDA::applyPlasticityWrapper(float dt) {
    dim3 blocks, threads;
    calculateGridBlockSize(config.numSynapses, blocks, threads);
    
    // This is a more complete plasticity sequence
    updateEligibilityTracesWrapper(blocks, threads, d_synapses, d_neurons, config.numSynapses);
    applyRewardModulationWrapper(blocks, threads, d_synapses, d_reward_signal, config.numSynapses);
    applyHebbianLearningWrapper(blocks, threads, d_synapses, d_neurons, config.numSynapses, learning_rate);
}

void NetworkCUDA::processSpikingWrapper() {
    dim3 blocks, threads;
    calculateGridBlockSize(config.numNeurons, blocks, threads);
    processSpikesWrapper(blocks, threads, d_neurons, d_neuron_spike_counts, d_synapses, config.numNeurons);
}

// Utility Methods
void NetworkCUDA::calculateGridBlockSize(int total_items, dim3& grid, dim3& block) const {
    const int block_size = 256;
    block = dim3(block_size);
    grid = dim3((total_items + block_size - 1) / block_size);
}

void NetworkCUDA::validateConfig() const {
    if (config.numNeurons <= 0 || config.numNeurons > 1000000) {
        throw NetworkException(NetworkError::CONFIGURATION_ERROR, "Invalid number of neurons.");
    }
    if (config.numSynapses < 0 || config.numSynapses > 100000000) {
         throw NetworkException(NetworkError::CONFIGURATION_ERROR, "Invalid number of synapses.");
    }
    //... add more checks
}

// --- ADDED MISSING IMPLEMENTATIONS ---
void NetworkCUDA::generateDistanceBasedSynapses() {
    // This needs a proper implementation based on cortical columns and distance
    std::cout << "Placeholder: Generating distance-based synapses." << std::endl;
}

size_t NetworkCUDA::estimateMemoryRequirements() const {
    size_t mem_size = 0;
    mem_size += config.numNeurons * sizeof(GPUNeuron);
    mem_size += config.numSynapses * sizeof(GPUSynapse);
    mem_size += config.numNeurons * sizeof(curandState);
    mem_size += config.numInputs * sizeof(float);
    mem_size += sizeof(float); // reward
    mem_size += config.numNeurons * sizeof(int); // spike counts
    if (config.numColumns > 0) {
        mem_size += config.numColumns * sizeof(CorticalColumn);
    }
    return mem_size;
}

void NetworkCUDA::checkCudaErrors() const {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw NetworkException(NetworkError::CUDA_ERROR, cudaGetErrorString(err));
    }
}