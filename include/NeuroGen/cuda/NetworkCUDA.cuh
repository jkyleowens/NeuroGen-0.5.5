#ifndef NETWORK_CUDA_CUH
#define NETWORK_CUDA_CUH

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/NetworkStats.h"
#include "NeuroGen/cuda/GPUNeuralStructures.h"

/**
 * @class NetworkCUDA
 * @brief Manages the CUDA implementation of the neural network.
 *
 * This class handles GPU memory allocation, data transfers between host and device,
 * and the orchestration of CUDA kernel launches for the simulation.
 */
class NetworkCUDA {
public:
    NetworkCUDA(const NetworkConfig& config);
    ~NetworkCUDA();

    // High-level simulation control
    void simulate_step(float current_time, float dt, float reward, const std::vector<float>& inputs);

    // Data management
    void copy_to_gpu(const std::vector<GPUNeuronState>& neurons, const std::vector<GPUSynapse>& synapses);
    void copy_from_gpu(std::vector<GPUNeuronState>& neurons, std::vector<GPUSynapse>& synapses);
    void get_stats(NetworkStats& stats) const;

private:
    void allocate_gpu_memory();
    void free_gpu_memory();
    void initialize_gpu_state();

    const NetworkConfig& config_;
    int num_neurons_;
    int num_synapses_;

    // GPU device pointers
    GPUNeuronState* d_neurons_ = nullptr;
    GPUSynapse* d_synapses_ = nullptr;
    float* d_input_currents_ = nullptr; // For applying external stimuli

    // CUDA stream for asynchronous operations
    cudaStream_t stream_;
};

#endif // NETWORK_CUDA_CUH