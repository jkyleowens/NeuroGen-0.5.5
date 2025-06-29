#ifndef NETWORK_CUDA_CUH
#define NETWORK_CUDA_CUH

#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <iostream>

#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/NetworkStats.h>
#include <NeuroGen/cuda/GPUNeuralStructures.h> // Include the full definitions

// Forward declarations are no longer necessary as we include the header above
// struct GPUNeuronState;
// struct GPUSynapse;
struct CorticalColumn;

class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& message, int code)
        : std::runtime_error(message), message_(message), error_code_(code) {}

    int getErrorCode() const { return error_code_; }

private:
    std::string message_;
    int error_code_;
};

class NetworkCUDA {
public:
    NetworkCUDA(const NetworkConfig& config);
    ~NetworkCUDA();

    void update(float dt_ms, const std::vector<float>& input_currents, float reward);
    std::vector<float> getOutput() const;
    void reset();
    NetworkStats getStats() const;

    void setRewardSignal(float reward);
    void printNetworkState() const;
    
    std::vector<float> getNeuronVoltages() const;
    std::vector<float> getSynapticWeights() const;

private:
    void initializeNetwork();
    void cleanup();
    void allocateDeviceMemory();
    void initializeDeviceArrays();
    void updateNetworkStatistics();
    void validateConfig() const;
    void calculateGridBlockSize(int n_elements, dim3& grid, dim3& block) const;

    int getNumNeurons() const { return config.getInt("network.num_neurons"); }
    int getNumSynapses() const { return config.getInt("network.num_synapses"); }

    const NetworkConfig& config;
    GPUNeuronState* d_neurons;
    GPUSynapse* d_synapses;
    float* d_calcium_levels;
    int* d_neuron_spike_counts;
    curandState* d_random_states;
    CorticalColumn* d_cortical_columns;
    float* d_input_currents;

    float current_time_ms;
    bool network_initialized;
    bool plasticity_enabled;
    float current_learning_rate;
};

#endif // NETWORK_CUDA_CUH