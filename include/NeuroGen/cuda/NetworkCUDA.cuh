#ifndef NETWORK_CUDA_CUH
#define NETWORK_CUDA_CUH

// ============================================================================
// CUDA HEADERS AND GPU MEMORY MANAGEMENT
// ============================================================================
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cusparse.h>

// ============================================================================
// NEUREGEN FRAMEWORK HEADERS
// ============================================================================
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/NetworkStats.h>

// ============================================================================
// STANDARD LIBRARY HEADERS (CUDA-COMPATIBLE)
// ============================================================================
#include <vector>
#include <string>
#include <utility>  // For std::pair

// ============================================================================
// FORWARD DECLARATIONS - CONSISTENT WITH CPU CODE
// ============================================================================
class Neuron;       // FIX: Use class to match Neuron.h
struct Synapse;     // Use struct to match Synapse.h
class Network;
class NeuralModule;

// ============================================================================
// PERFORMANCE METRICS STRUCTURE
// ============================================================================
struct PerformanceMetrics {
    float simulation_fps;              // Simulation frames per second
    float biological_time_factor;      // Biological time scaling factor
    float memory_bandwidth_gbps;       // Memory bandwidth in GB/s
    float gpu_utilization_percent;     // GPU utilization percentage
    float neuron_update_time_ms;       // Time to update neurons (ms)
    float synapse_update_time_ms;      // Time to update synapses (ms)
    float plasticity_update_time_ms;   // Time for plasticity updates (ms)
    float kernel_execution_time_ms;    // Kernel execution time (ms)
    size_t total_memory_used_mb;       // Total GPU memory used (MB)
    size_t peak_memory_used_mb;        // Peak GPU memory used (MB)
};

/**
 * @brief GPU-accelerated Neural Network Manager
 * 
 * This class provides CUDA-accelerated neural network simulation with
 * seamless integration to the CPU-based modular neural network architecture.
 * Optimized for biological realism and high-performance simulation.
 */
class NetworkCUDA {
public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Construct CUDA neural network manager
     * @param config Network configuration parameters
     */
    explicit NetworkCUDA(const NetworkConfig& config);
    
    /**
     * @brief Destructor with automatic resource cleanup
     */
    ~NetworkCUDA();
    
    /**
     * @brief Initialize CUDA resources and allocate GPU memory
     * @return Success status of initialization
     */
    bool initialize();
    
    /**
     * @brief Clean up all CUDA resources
     */
    void cleanup();
    
    // ========================================================================
    // NETWORK CONSTRUCTION AND MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Transfer network from CPU to GPU
     * @param cpu_network Source CPU network
     * @return Transfer success status
     */
    bool transferFromCPU(const Network& cpu_network);
    
    /**
     * @brief Transfer network from GPU to CPU
     * @param cpu_network Target CPU network
     * @return Transfer success status
     */
    bool transferToCPU(Network& cpu_network);
    
    /**
     * @brief Add neuron to GPU network
     * @param neuron_data GPU neuron structure
     * @return Neuron ID on GPU
     */
    size_t addNeuron(const GPUNeuronState& neuron_data);
    
    /**
     * @brief Add synapse to GPU network
     * @param synapse_data GPU synapse structure
     * @return Synapse ID on GPU
     */
    size_t addSynapse(const GPUSynapse& synapse_data);
    
    // ========================================================================
    // SIMULATION INTERFACE
    // ========================================================================
    
    /**
     * @brief Update neural network simulation on GPU
     * @param dt Time step in milliseconds
     * @param input_currents External input currents
     * @param reward_signal Global reward signal for learning
     */
    void update(float dt, const std::vector<float>& input_currents = {}, 
               float reward_signal = 0.0f);
    
    /**
     * @brief Get current neuron outputs from GPU
     * @return Vector of neuron activation values
     */
    std::vector<float> getOutputs() const;
    
    /**
     * @brief Get specific neuron states from GPU
     * @param neuron_ids Vector of neuron IDs to query
     * @return Vector of neuron states
     */
    std::vector<GPUNeuronState> getNeuronStates(const std::vector<size_t>& neuron_ids) const;
    
    // ========================================================================
    // MODULAR NEURAL NETWORK SUPPORT
    // ========================================================================
    
    /**
     * @brief Register neural module for GPU processing
     * @param module_id Unique module identifier
     * @param neuron_range Range of neurons belonging to this module
     * @param config Module-specific configuration
     */
    void registerModule(size_t module_id, const std::pair<size_t, size_t>& neuron_range,
                       const NetworkConfig& config);
    
    /**
     * @brief Update specific neural module on GPU
     * @param module_id Module identifier
     * @param dt Time step
     * @param module_inputs Input signals for this module
     * @param attention_weight Attention modulation factor
     */
    void updateModule(size_t module_id, float dt, 
                     const std::vector<float>& module_inputs,
                     float attention_weight = 1.0f);
    
    /**
     * @brief Get module-specific outputs
     * @param module_id Module identifier
     * @return Vector of module output activations
     */
    std::vector<float> getModuleOutputs(size_t module_id) const;
    
    // ========================================================================
    // INTER-MODULE COMMUNICATION
    // ========================================================================
    
    /**
     * @brief Transfer signals between modules on GPU
     * @param source_module Source module ID
     * @param target_module Target module ID
     * @param signal_data Signal values to transfer
     * @param connection_strength Strength of inter-module connection
     */
    void transferInterModuleSignals(size_t source_module, size_t target_module,
                                  const std::vector<float>& signal_data,
                                  float connection_strength = 1.0f);
    
    /**
     * @brief Process all pending inter-module communications
     */
    void processInterModuleCommunication();
    
    // ========================================================================
    // LEARNING AND PLASTICITY
    // ========================================================================
    
    /**
     * @brief Apply reward-modulated learning on GPU
     * @param reward_signal Global reward signal
     * @param learning_rate Learning rate parameter
     */
    void applyRewardModulatedLearning(float reward_signal, float learning_rate);
    
    /**
     * @brief Update synaptic plasticity using STDP
     * @param plasticity_window STDP time window in milliseconds
     * @param learning_rate Plasticity learning rate
     */
    void updateSynapticPlasticity(float plasticity_window = 20.0f, 
                                float learning_rate = 0.01f);
    
    /**
     * @brief Apply homeostatic scaling to maintain network stability
     * @param target_activity Target average activity level
     * @param scaling_rate Rate of homeostatic adjustment
     */
    void applyHomeostaticScaling(float target_activity = 0.1f, 
                               float scaling_rate = 0.001f);
    
    // ========================================================================
    // PERFORMANCE MONITORING
    // ========================================================================
    
    /**
     * @brief Get comprehensive network statistics from GPU
     * @return Current network statistics
     */
    NetworkStats getNetworkStats() const;
    
    /**
     * @brief Get GPU performance metrics  
     * @return Performance metrics structure
     */
    PerformanceMetrics getPerformanceMetrics() const;
    
    /**
     * @brief Reset performance counters
     */
    void resetPerformanceMetrics();
    
    // ========================================================================
    // MEMORY MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Get current GPU memory usage
     * @return Memory usage in bytes
     */
    size_t getMemoryUsage() const;
    
    /**
     * @brief Optimize GPU memory layout for better performance
     */
    void optimizeMemoryLayout();
    
    /**
     * @brief Check if CUDA is available and initialized
     * @return CUDA availability status
     */
    bool isCudaAvailable() const;
    
    // ========================================================================
    // STATE SERIALIZATION
    // ========================================================================
    
    /**
     * @brief Save GPU network state to file
     * @param filename Output filename
     * @return Save success status
     */
    bool saveGPUState(const std::string& filename) const;
    
    /**
     * @brief Load GPU network state from file
     * @param filename Input filename
     * @return Load success status
     */
    bool loadGPUState(const std::string& filename);
    
    // ========================================================================
    // CUDA INTERFACE METHODS (PUBLIC ACCESS REQUIRED)
    // ========================================================================
    
    /**
     * @brief Copy data to GPU
     */
    void copy_to_gpu(const std::vector<GPUNeuronState>& neurons, 
                     const std::vector<GPUSynapse>& synapses);
    
    /**
     * @brief Copy data from GPU
     */
    void copy_from_gpu(std::vector<GPUNeuronState>& neurons, 
                       std::vector<GPUSynapse>& synapses);
    
    /**
     * @brief Simulate one step
     */
    void simulate_step(float current_time, float dt, float reward, 
                       const std::vector<float>& inputs);
    
    /**
     * @brief Get network statistics (snake_case version)
     */
    void get_stats(NetworkStats& stats) const;

private:
    // ========================================================================
    // INTERNAL GPU DATA STRUCTURES
    // ========================================================================
    
    // Network configuration and state
    NetworkConfig config_;
    bool system_initialized_;
    
    // Network dimensions
    int num_neurons_;
    int num_synapses_;
    
    // Simulation state
    float current_simulation_time_;
    size_t simulation_step_count_;
    
    // GPU memory pointers - Core neural structures
    GPUNeuronState* d_neurons_;
    GPUSynapse* d_synapses_;
    float* d_input_currents_;
    float* d_output_activations_;
    curandState* d_random_states_;
    
    // GPU memory pointers - Statistics and monitoring
    int* d_spike_count_;
    NetworkStats* d_network_stats_;
    
    // Memory tracking
    float allocated_memory_mb_;
    size_t total_gpu_memory_;
    size_t allocated_memory_;
    
    // Module management (using vectors instead of STL maps for CUDA compatibility)
    struct ModuleInfo {
        size_t module_id;
        size_t start_neuron;
        size_t end_neuron;
        NetworkConfig config;
        float* d_module_inputs;
        float* d_module_outputs;
    };
    std::vector<ModuleInfo> registered_modules_;
    
    // Performance tracking
    mutable PerformanceMetrics performance_metrics_;
    
    // Performance tracking
    mutable NetworkStats gpu_stats_;
    
    // CUDA handles
    cublasHandle_t cublas_handle_;
    cusparseHandle_t cusparse_handle_;
    cudaStream_t computation_stream_;
    cudaStream_t memory_stream_;
    
    // ========================================================================
    // INTERNAL HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Initialize CUDA handles and streams
     */
    bool initializeCudaHandles();
    
    /**
     * @brief Allocate GPU memory for network structures
     */
    bool allocateGPUMemory();
    
    /**
     * @brief Initialize random number generators on GPU
     */
    bool initializeRandomStates();
    
    /**
     * @brief Launch neural update kernels
     */
    void launchNeuronUpdateKernels(float dt, float reward_signal);
    
    /**
     * @brief Launch synaptic processing kernels
     */
    void launchSynapseProcessingKernels(float dt);
    
    /**
     * @brief Launch plasticity update kernels
     */
    void launchPlasticityKernels(float learning_rate);
    
    /**
     * @brief Update performance statistics
     */
    void updatePerformanceStats();
    
    /**
     * @brief Check for CUDA errors and handle them
     */
    bool checkCudaErrors(const std::string& operation) const;
    
    // ========================================================================
    // ADDITIONAL MISSING METHODS FROM IMPLEMENTATION
    // ========================================================================
    
    /**
     * @brief Calculate neuron firing rate from GPU neuron state
     */
    float calculateNeuronFiringRate(const GPUNeuronState& gpu_neuron) const;
    
    /**
     * @brief Update synaptic plasticity for a specific synapse
     */
    void updateSynapticPlasticity(GPUSynapse& gpu_synapse, float dt, float reward);
    
    /**
     * @brief Get incoming synapses for a neuron
     */
    std::vector<size_t> getIncomingSynapses(size_t neuron_id) const;
    
    /**
     * @brief Synchronize with CPU network
     */
    void synchronize_with_cpu_network(void* cpu_network_ptr, const std::string& sync_direction);
    
    /**
     * @brief Get network output (alternative method name)
     */
    std::vector<float> get_output() const;
    
    /**
     * @brief Allocate GPU memory (snake_case version)
     */
    void allocate_gpu_memory();
    
    /**
     * @brief Initialize GPU state (snake_case version)
     */
    void initialize_gpu_state();
    
    /**
     * @brief Cleanup resources (snake_case version)
     */
    void cleanup_resources();
    
    /**
     * @brief Check if synapse exists
     */
    bool synapseExists(size_t pre_id, size_t post_id) const;
    
    /**
     * @brief Check if neuron is active
     */
    bool isNeuronActive(size_t neuron_id) const;
    
    /**
     * @brief Get performance metrics (snake_case version)
     */
    PerformanceMetrics get_performance_metrics() const;
    
    /**
     * @brief Synchronize configurations
     */
    void synchronize_configurations(const NetworkConfig& cpu_config, NetworkConfig& gpu_config);
    
    /**
     * @brief Save GPU state (snake_case version)
     */
    bool save_gpu_state(const std::string& filename) const;
    
    /**
     * @brief Load GPU state (snake_case version)
     */
    bool load_gpu_state(const std::string& filename);
};

// ============================================================================
// GLOBAL CUDA UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Initialize CUDA device and check compatibility
 * @return Device initialization success status
 */
bool initializeCudaDevice();

/**
 * @brief Get optimal CUDA block and grid dimensions
 * @param num_elements Number of elements to process
 * @return Optimal launch configuration
 */
dim3 getOptimalLaunchConfig(size_t num_elements);

/**
 * @brief Synchronize all CUDA operations and check for errors
 * @return Synchronization success status
 */
bool synchronizeCuda();

/**
 * @brief Clean up global CUDA resources
 */
void cleanupCuda();

#endif // NETWORK_CUDA_CUH