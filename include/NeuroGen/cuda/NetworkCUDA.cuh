#ifndef NETWORK_CUDA_CUH
#define NETWORK_CUDA_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <memory>
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/NetworkStats.h"
#include "NeuroGen/cuda/GPUNeuralStructures.h"

/**
 * @class NetworkCUDA
 * @brief Clean, production-ready CUDA neural network implementation
 *
 * This implementation provides a stable foundation for biologically-inspired
 * neural computation while maintaining compatibility with existing systems.
 * 
 * Key features:
 * - Multi-compartment neuron processing with calcium dynamics
 * - STDP-based synaptic plasticity with reward modulation
 * - Homeostatic regulation for network stability
 * - Efficient GPU memory management and kernel execution
 * - Real-time spike detection and processing
 * - Seamless integration with existing kernel infrastructure
 *
 * This clean implementation serves as the foundation for advanced modular
 * architectures while ensuring system stability and performance.
 */
class NetworkCUDA {
public:
    // ========================================
    // CORE INTERFACE
    // ========================================
    
    /**
     * @brief Initialize CUDA neural network with given configuration
     * @param config Network configuration parameters
     */
    explicit NetworkCUDA(const NetworkConfig& config);
    
    /**
     * @brief Clean destructor with proper resource management
     */
    ~NetworkCUDA();
    
    /**
     * @brief Execute one simulation timestep
     * @param current_time Current simulation time (ms)
     * @param dt Integration timestep (ms)
     * @param reward Global reward signal for learning
     * @param inputs External input currents
     */
    void simulate_step(float current_time, float dt, float reward, const std::vector<float>& inputs);
    
    /**
     * @brief Get current network statistics
     * @param stats Reference to NetworkStats structure to populate
     */
    void get_stats(NetworkStats& stats) const;
    
    // ========================================
    // DATA MANAGEMENT
    // ========================================
    
    /**
     * @brief Copy network state from host to GPU
     * @param neurons Vector of neuron states
     * @param synapses Vector of synapse states
     */
    void copy_to_gpu(const std::vector<GPUNeuronState>& neurons, 
                     const std::vector<GPUSynapse>& synapses);
    
    /**
     * @brief Copy network state from GPU to host
     * @param neurons Vector to receive neuron states
     * @param synapses Vector to receive synapse states
     */
    void copy_from_gpu(std::vector<GPUNeuronState>& neurons, 
                       std::vector<GPUSynapse>& synapses);
    
    // ========================================
    // SYSTEM MONITORING
    // ========================================
    
    /**
     * @brief Get allocated GPU memory in MB
     * @return Memory usage in megabytes
     */
    float get_allocated_memory_mb() const;
    
    /**
     * @brief Get current simulation step count
     * @return Number of simulation steps executed
     */
    int get_simulation_step_count() const;
    
    /**
     * @brief Get current simulation time
     * @return Current simulation time in milliseconds
     */
    float get_current_simulation_time() const;
    
    /**
     * @brief Check if system is properly initialized
     * @return True if system is ready for simulation
     */
    bool is_initialized() const { return system_initialized_; }
    
    // ========================================
    // ADVANCED CONTROL (FOR FUTURE EXTENSION)
    // ========================================
    
    /**
     * @brief Access device neuron array (for advanced integrations)
     * @return Pointer to GPU neuron array
     */
    GPUNeuronState* get_device_neurons() const { return d_neurons_; }
    
    /**
     * @brief Access device synapse array (for advanced integrations)
     * @return Pointer to GPU synapse array
     */
    GPUSynapse* get_device_synapses() const { return d_synapses_; }
    
    /**
     * @brief Get number of neurons in network
     * @return Total neuron count
     */
    int get_num_neurons() const { return num_neurons_; }
    
    /**
     * @brief Get number of synapses in network
     * @return Total synapse count
     */
    int get_num_synapses() const { return num_synapses_; }
    
    /**
     * @brief Get CUDA stream for advanced kernel launches
     * @return CUDA stream handle
     */
    cudaStream_t get_cuda_stream() const { return stream_; }

private:
    // ========================================
    // SYSTEM STATE
    // ========================================
    
    const NetworkConfig& config_;              // Network configuration
    int num_neurons_;                          // Total number of neurons
    int num_synapses_;                         // Total number of synapses
    float current_simulation_time_;            // Current simulation time
    int simulation_step_count_;                // Number of steps executed
    bool system_initialized_;                  // System initialization status
    float allocated_memory_mb_;                // Allocated GPU memory
    
    // ========================================
    // GPU MEMORY POINTERS
    // ========================================
    
    GPUNeuronState* d_neurons_;                // GPU neuron state array
    GPUSynapse* d_synapses_;                   // GPU synapse array
    float* d_input_currents_;                  // External input buffer
    int* d_spike_count_;                       // Spike count buffer
    NetworkStats* d_network_stats_;            // Statistics buffer
    
    // ========================================
    // CUDA EXECUTION CONTEXT
    // ========================================
    
    cudaStream_t stream_;                      // CUDA stream for async operations
    
    // ========================================
    // CORE SIMULATION COMPONENTS
    // ========================================
    
    /**
     * @brief Update neural dynamics (voltage, calcium, etc.)
     */
    void update_neural_dynamics(float current_time, float dt);
    
    /**
     * @brief Update synaptic plasticity and learning
     */
    void update_synaptic_plasticity(float current_time, float dt, float reward);
    
    /**
     * @brief Apply homeostatic regulation mechanisms
     */
    void apply_homeostatic_regulation(float current_time, float dt);
    
    /**
     * @brief Process spike detection and recording
     */
    void process_spike_detection(float current_time);
    
    /**
     * @brief Process external input currents
     */
    void process_external_inputs(const std::vector<float>& inputs);
    
    // ========================================
    // SYSTEM MANAGEMENT
    // ========================================
    
    /**
     * @brief Allocate all GPU memory buffers
     */
    void allocate_gpu_memory();
    
    /**
     * @brief Clean up all allocated resources
     */
    void cleanup_resources();
    
    /**
     * @brief Initialize neural states and parameters
     */
    void initialize_gpu_state();
    
    /**
     * @brief Validate system initialization
     */
    void validate_initialization();
    
    // ========================================
    // UTILITY FUNCTIONS
    // ========================================
    
    /**
     * @brief Calculate CUDA grid size for given number of elements
     */
    dim3 get_grid_size(int num_elements, int block_size = 256) const;
    
    // Disable copy constructor and assignment operator
    NetworkCUDA(const NetworkCUDA&) = delete;
    NetworkCUDA& operator=(const NetworkCUDA&) = delete;
};

// ============================================================================
// CUDA KERNEL DECLARATIONS
// ============================================================================

/**
 * @brief Apply external input currents to neurons
 */
__global__ void apply_input_currents_kernel(GPUNeuronState* neurons, 
                                           const float* input_currents,
                                           int num_neurons, float dt);

/**
 * @brief Initialize synapse parameters with proper biological variability
 */
__global__ void initialize_synapses_kernel(GPUSynapse* synapses, int num_synapses,
                                          float min_weight, float max_weight,
                                          float weight_std, unsigned long seed);

/**
 * @brief Compute comprehensive network statistics on GPU
 */
__global__ void compute_network_statistics(const GPUNeuronState* neurons,
                                          const GPUSynapse* synapses,
                                          NetworkStats* stats,
                                          int num_neurons, int num_synapses,
                                          float current_time);

// ============================================================================
// EXTENDED NETWORK STATISTICS (FOR MODULAR INTEGRATION)
// ============================================================================

/**
 * @brief Extended network statistics structure for advanced monitoring
 * 
 * This structure can be extended to support modular architecture statistics
 * without breaking the current clean implementation.
 */
struct ExtendedNetworkStats : public NetworkStats {
    // Performance metrics
    float step_execution_time_us;            // Last step execution time
    float gpu_utilization_percent;           // GPU computational utilization
    float memory_bandwidth_utilization;      // Memory bandwidth usage
    
    // Neural activity metrics
    float network_synchronization_index;     // Measure of network synchrony
    float excitation_inhibition_balance;     // E/I balance measure
    float firing_rate_variance;              // Variance in firing rates
    
    // Plasticity metrics
    float average_synaptic_change_rate;      // Rate of weight changes
    float plasticity_stability_index;        // Stability of plastic changes
    float homeostatic_pressure_index;        // Homeostatic regulation pressure
    
    // System health metrics
    float numerical_stability_score;         // Numerical stability measure
    float convergence_rate;                  // Learning convergence rate
    bool critical_warnings;                  // Any critical system warnings
    
    // Future modular extension points
    void* module_specific_data;              // Pointer for modular extensions
    int num_active_modules;                  // Number of active modules
    float inter_module_communication_rate;   // Rate of inter-module communication
};

// ============================================================================
// CONFIGURATION EXTENSIONS (FOR MODULAR ARCHITECTURE)
// ============================================================================

/**
 * @brief Extended configuration structure for advanced features
 * 
 * This extends NetworkConfig for future modular architecture support
 * while maintaining backward compatibility.
 */
struct ExtendedNetworkConfig : public NetworkConfig {
    // Modular architecture parameters (for future use)
    bool enable_modular_architecture = false;
    int num_modules = 1;
    float inter_module_connection_density = 0.1f;
    
    // Advanced biological features
    bool enable_dendritic_computation = true;
    bool enable_calcium_dependent_plasticity = true;
    bool enable_homeostatic_intrinsic_plasticity = true;
    
    // Performance optimization
    bool enable_adaptive_timestep = false;
    float target_step_time_ms = 1.0f;
    bool enable_gpu_optimizations = true;
    
    // Advanced plasticity features
    bool enable_metaplasticity = false;
    bool enable_structural_plasticity = false;
    float structural_change_rate = 0.001f;
    
    // Neuromodulation
    bool enable_dopamine_modulation = true;
    bool enable_acetylcholine_modulation = false;
    bool enable_serotonin_modulation = false;
    
    ExtendedNetworkConfig() : NetworkConfig() {
        // Initialize extended parameters with safe defaults
    }
    
    ExtendedNetworkConfig(const NetworkConfig& base) : NetworkConfig(base) {
        // Convert from base config
    }
};

#endif // NETWORK_CUDA_CUH