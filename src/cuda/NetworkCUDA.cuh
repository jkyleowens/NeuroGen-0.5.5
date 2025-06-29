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
#include <NeuroGen/cuda/GPUNeuralStructures.h>

// Forward declarations
struct CorticalColumn;

/**
 * @brief Exception class for CUDA-related errors in neural network operations
 */
class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& message, int code)
        : std::runtime_error(message), message_(message), error_code_(code) {}

    int getErrorCode() const { return error_code_; }

private:
    std::string message_;
    int error_code_;
};

/**
 * @brief GPU-accelerated biologically-inspired neural network implementation
 * 
 * This class implements a modular neural network architecture with cortical columns,
 * neuromodulation, and advanced plasticity mechanisms. Designed for real-time brain
 * simulation with biological realism.
 */
class NetworkCUDA {
public:
    /**
     * @brief Constructor - Initialize CUDA neural network
     * @param config Network configuration parameters
     */
    explicit NetworkCUDA(const NetworkConfig& config);
    
    /**
     * @brief Destructor - Clean up GPU resources
     */
    ~NetworkCUDA();

    // ========================================================================
    // CORE SIMULATION INTERFACE
    // ========================================================================
    
    /**
     * @brief Update network state for one timestep
     * @param dt_ms Timestep duration in milliseconds
     * @param input_currents External input currents to neurons
     * @param reward Reward signal for learning algorithms
     */
    void update(float dt_ms, const std::vector<float>& input_currents, float reward);
    
    /**
     * @brief Get current network output (neuron voltages/activities)
     * @return Vector of output values
     */
    std::vector<float> getOutput() const;
    
    /**
     * @brief Reset network to initial state
     */
    void reset();
    
    /**
     * @brief Get comprehensive network statistics
     * @return Current NetworkStats object
     */
    NetworkStats getStats() const;

    // ========================================================================
    // LEARNING AND MODULATION INTERFACE
    // ========================================================================
    
    /**
     * @brief Set reward signal for reinforcement learning
     * @param reward Reward value (can be positive or negative)
     */
    void setRewardSignal(float reward);
    
    /**
     * @brief Enable/disable synaptic plasticity
     * @param enable True to enable plasticity, false to disable
     */
    void setPlasticityEnabled(bool enable) { plasticity_enabled = enable; }
    
    /**
     * @brief Set learning rate for plastic changes
     * @param rate Learning rate (0.0 to 1.0)
     */
    void setLearningRate(float rate) { current_learning_rate = rate; }
    
    /**
     * @brief Print current network state for debugging
     */
    void printNetworkState() const;
    
    // ========================================================================
    // DATA ACCESS INTERFACE
    // ========================================================================
    
    /**
     * @brief Get all neuron membrane voltages
     * @return Vector of neuron voltages in mV
     */
    std::vector<float> getNeuronVoltages() const;
    
    /**
     * @brief Get all synaptic weights
     * @return Vector of synaptic weight values
     */
    std::vector<float> getSynapticWeights() const;
    
    /**
     * @brief Get number of neurons in the network
     * @return Total neuron count
     */
    int getNumNeurons() const { 
        return static_cast<int>(config.numColumns * config.neuronsPerColumn); 
    }
    
    /**
     * @brief Get number of synapses in the network
     * @return Total synapse count
     */
    int getNumSynapses() const { 
        return static_cast<int>(config.totalSynapses); 
    }
    
    /**
     * @brief Get number of cortical columns
     * @return Column count
     */
    int getNumColumns() const {
        return static_cast<int>(config.numColumns);
    }
    
    /**
     * @brief Get neurons per cortical column
     * @return Neurons per column count
     */
    int getNeuronsPerColumn() const {
        return static_cast<int>(config.neuronsPerColumn);
    }

    // ========================================================================
    // STATE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Check if network is properly initialized
     * @return True if network is ready for simulation
     */
    bool isInitialized() const { return network_initialized; }
    
    /**
     * @brief Get current simulation time
     * @return Time in milliseconds
     */
    float getCurrentTime() const { return current_time_ms; }
    
    /**
     * @brief Get network configuration
     * @return Reference to configuration object
     */
    const NetworkConfig& getConfig() const { return config; }

private:
    // ========================================================================
    // INITIALIZATION AND CLEANUP
    // ========================================================================
    
    /**
     * @brief Initialize the complete neural network on GPU
     */
    void initializeNetwork();
    
    /**
     * @brief Initialize cortical columns
     */
    void initializeColumns();
    
    /**
     * @brief Clean up all GPU resources
     */
    void cleanup();
    
    /**
     * @brief Allocate GPU memory for all network components
     */
    void allocateDeviceMemory();
    
    /**
     * @brief Initialize device arrays with default values
     */
    void initializeDeviceArrays();
    
    /**
     * @brief Validate network configuration parameters
     */
    void validateConfig() const;

    // ========================================================================
    // COMPUTATIONAL UTILITIES
    // ========================================================================
    
    /**
     * @brief Calculate optimal CUDA grid and block dimensions
     * @param n_elements Number of elements to process
     * @param grid Output grid dimensions
     * @param block Output block dimensions
     */
    void calculateGridBlockSize(int n_elements, dim3& grid, dim3& block) const;
    
    /**
     * @brief Update network statistics from GPU state
     */
    void updateNetworkStatistics();

    // ========================================================================
    // BIOLOGICAL MECHANISM IMPLEMENTATIONS
    // ========================================================================
    
    /**
     * @brief Update neuromodulator concentrations based on network activity
     * @param reward Current reward signal for dopamine modulation
     */
    void updateNeuromodulation(float reward);
    
    /**
     * @brief Apply homeostatic scaling mechanisms for network stability
     */
    void applyHomeostaticScaling();
    
    /**
     * @brief Monitor network health and criticality for optimal brain-like operation
     */
    void monitorNetworkHealth();
    
    /**
     * @brief Estimate total GPU memory requirements for network
     * @return Required memory in bytes
     */
    size_t estimateMemoryRequirements() const;
    
    /**
     * @brief Initialize cortical column specializations for modular processing
     */
    void initializeColumnSpecializations();

    // ========================================================================
    // NETWORK GENERATION AND TOPOLOGY
    // ========================================================================
    
    /**
     * @brief Generate synaptic connections based on distance and biological principles
     */
    void generateDistanceBasedSynapses();
    
    /**
     * @brief Initialize neuron positions within cortical columns for spatial organization
     */
    void initializeNeuronPositions();

    // ========================================================================
    // KERNEL WRAPPER METHODS
    // ========================================================================
    
    /**
     * @brief Update neuron states using CUDA kernels
     * @param dt_ms Timestep in milliseconds
     */
    void updateNeuronsWrapper(float dt_ms);
    
    /**
     * @brief Update synapse states using CUDA kernels
     * @param dt_ms Timestep in milliseconds
     */
    void updateSynapsesWrapper(float dt_ms);
    
    /**
     * @brief Apply plasticity rules using CUDA kernels
     * @param reward Current reward signal
     */
    void applyPlasticityWrapper(float reward);
    
    /**
     * @brief Process spiking events using CUDA kernels
     */
    void processSpikingWrapper();

    // ========================================================================
    // MEMBER VARIABLES
    // ========================================================================
    
    /** Network configuration parameters */
    const NetworkConfig& config;
    
    /** GPU device arrays */
    GPUNeuronState* d_neurons;           ///< Neuron state array on GPU
    GPUSynapse* d_synapses;              ///< Synapse array on GPU  
    float* d_calcium_levels;             ///< Calcium concentration array
    int* d_neuron_spike_counts;          ///< Spike count array
    curandState* d_random_states;        ///< Random number generator states
    CorticalColumn* d_cortical_columns;  ///< Cortical column structures
    float* d_input_currents;             ///< External input current array

    /** Simulation state */
    float current_time_ms;               ///< Current simulation time (ms)
    bool network_initialized;            ///< Initialization status flag
    bool plasticity_enabled;             ///< Plasticity on/off flag
    float current_learning_rate;         ///< Current learning rate

    /** Performance monitoring */
    mutable float last_kernel_time;      ///< Last kernel execution time
    mutable size_t memory_usage;         ///< Current GPU memory usage
};

// ============================================================================
// INLINE IMPLEMENTATIONS FOR PERFORMANCE
// ============================================================================

inline std::vector<float> NetworkCUDA::getNeuronVoltages() const {
    if (!network_initialized) {
        return std::vector<float>();
    }
    
    int num_neurons = getNumNeurons();
    std::vector<GPUNeuronState> host_neurons(num_neurons);
    
    cudaMemcpy(host_neurons.data(), d_neurons, 
               num_neurons * sizeof(GPUNeuronState), 
               cudaMemcpyDeviceToHost);
    
    std::vector<float> voltages(num_neurons);
    for (int i = 0; i < num_neurons; ++i) {
        voltages[i] = host_neurons[i].voltage;
    }
    
    return voltages;
}

inline std::vector<float> NetworkCUDA::getSynapticWeights() const {
    if (!network_initialized) {
        return std::vector<float>();
    }
    
    int num_synapses = getNumSynapses();
    std::vector<GPUSynapse> host_synapses(num_synapses);
    
    cudaMemcpy(host_synapses.data(), d_synapses,
               num_synapses * sizeof(GPUSynapse),
               cudaMemcpyDeviceToHost);
    
    std::vector<float> weights(num_synapses);
    for (int i = 0; i < num_synapses; ++i) {
        weights[i] = host_synapses[i].weight;
    }
    
    return weights;
}

#endif // NETWORK_CUDA_CUH