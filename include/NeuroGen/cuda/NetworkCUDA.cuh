#ifndef NETWORK_CUDA_CUH
#define NETWORK_CUDA_CUH

#include "NeuroGen/NetworkInterface.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/NetworkStats.h"
#include "NeuroGen/cuda/GPUNeuralStructures.h"
#include <vector>
#include <string>
#include <curand_kernel.h>
#include <dim3.h> // For dim3 type

class NetworkCUDA : public NetworkInterface {
public:
    explicit NetworkCUDA(const NetworkConfig& config);
    ~NetworkCUDA() override;

    void update(float dt, const std::vector<float>& input, float reward) override;
    std::vector<float> getOutput() const override;
    void reset() override;
    NetworkStats getStats() const override;

    // Control methods
    void setLearningRate(float rate);
    void setRewardSignal(float reward) override;
    void enablePlasticity(bool enable); // <-- ADDED

    // Monitoring
    void printNetworkState() const override;
    std::vector<float> getNeuronVoltages() const;
    std::vector<float> getSynapticWeights() const;
    size_t estimateMemoryRequirements() const; // <-- ADDED

private:
    // Initialization
    void initializeNetwork();
    void validateConfig() const;
    void allocateDeviceMemory();
    void initializeDeviceArrays();
    void initializeColumns();
    void initializeColumnSpecializations();
    void generateDistanceBasedSynapses();

    // Core update wrappers
    void updateNeuronsWrapper(float dt);
    void updateSynapsesWrapper(float dt);
    void applyPlasticityWrapper(float dt);
    void processSpikingWrapper();

    // Memory management
    void cleanup();

    // Utility
    void calculateGridBlockSize(int total_items, dim3& grid, dim3& block) const;
    void checkCudaErrors() const; // <-- ADDED

    // Member variables
    const NetworkConfig& config;
    bool initialized;
    bool plasticity_enabled; // <-- ADDED
    float learning_rate;     // <-- ADDED

    // Device pointers
    GPUNeuron* d_neurons;
    GPUSynapse* d_synapses;
    CorticalColumn* d_cortical_columns;
    curandState* d_random_states;
    float* d_input_currents;
    float* d_reward_signal;
    int* d_neuron_spike_counts;

    // Host-side stats
    NetworkStats stats;
};

#endif // NETWORK_CUDA_CUH