#ifndef NETWORK_CUDA_CUH
#define NETWORK_CUDA_CUH

// ============================================================================
// CUDA SYSTEM INCLUDES - Optimized for Neural Computation
// ============================================================================
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <memory>
#include <string>

// ============================================================================
// NEUREGEN FRAMEWORK INCLUDES - Complete Integration
// ============================================================================
#include <NeuroGen/Neuron.h>              // CRITICAL: Add CPU Neuron class
#include <NeuroGen/Synapse.h>             // CRITICAL: Add CPU Synapse class  
#include <NeuroGen/Network.h>             // CRITICAL: Add CPU Network class
#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>
#include <NeuroGen/cuda/NeuronSpikingKernels.cuh>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/NetworkStats.h>

// ============================================================================
// FORWARD DECLARATIONS - CPU-GPU Bridge Interface
// ============================================================================
class Neuron;
class Synapse;
class Network;
class NeuralModule;

// Advanced forward declarations for modular integration
struct GPUModuleState;
struct GPUAttentionWeights;
struct GPUHomeostasis;

/**
 * @class NetworkCUDA
 * @brief Advanced CUDA neural network with seamless CPU integration
 *
 * This revolutionary implementation bridges sophisticated CPU-side neural
 * computation with massively parallel GPU acceleration, maintaining the
 * biological realism and modular architecture of the breakthrough Network class
 * while achieving cortical-scale simulation performance.
 * 
 * Key breakthrough features:
 * - Seamless CPU-GPU neural state synchronization
 * - GPU-accelerated STDP with biological time constants
 * - Massively parallel homeostatic regulation
 * - Real-time structural plasticity with GPU efficiency
 * - Modular attention mechanisms for hierarchical processing
 * - Biologically-realistic membrane dynamics at scale
 * - Independent module state management with GPU persistence
 * 
 * The system maintains perfect synchronization with the CPU Network class
 * while enabling cortical-column scale simulations (>1M neurons) with
 * biological temporal precision and sophisticated learning dynamics.
 */
class NetworkCUDA {
public:
    // ========================================================================
    // CONSTRUCTION AND LIFECYCLE - Advanced Resource Management
    // ========================================================================
    
    /**
     * @brief Constructs GPU neural network with CPU-compatible interface
     * @param config Comprehensive network configuration with biological parameters
     */
    explicit NetworkCUDA(const NetworkConfig& config);
    
    /**
     * @brief Destructor with intelligent resource cleanup and state preservation
     */
    ~NetworkCUDA();
    
    // Prevent copying for GPU resource integrity
    NetworkCUDA(const NetworkCUDA&) = delete;
    NetworkCUDA& operator=(const NetworkCUDA&) = delete;
    
    // Enable efficient moving for modular transfers
    NetworkCUDA(NetworkCUDA&&) = default;
    NetworkCUDA& operator=(NetworkCUDA&&) = default;

    // ========================================================================
    // CORE SIMULATION INTERFACE - Brain-Scale Performance
    // ========================================================================
    
    /**
     * @brief Execute high-performance simulation timestep with biological fidelity
     * @param current_time Current simulation time (ms) for temporal coding
     * @param dt Integration timestep (ms) - optimized for stability and speed
     * @param reward Global reward signal for dopaminergic-like modulation
     * @param inputs External input currents with spatial encoding
     * 
     * Performs cortical-scale neural simulation including:
     * - Parallel membrane dynamics across all neurons (>1M simultaneous)
     * - GPU-accelerated synaptic transmission with realistic delays
     * - Massively parallel STDP with biological time constants
     * - Real-time homeostatic regulation and stability control
     * - Dynamic attention modulation for modular coordination
     */
    void simulate_step(float current_time, float dt, float reward, const std::vector<float>& inputs);
    
    /**
     * @brief Get comprehensive network statistics with GPU-accelerated analysis
     * @param stats Reference to NetworkStats structure for real-time monitoring
     */
    void get_stats(NetworkStats& stats) const;
    
    /**
     * @brief Retrieves biologically-encoded network output at cortical scale
     * @return Vector of output patterns (firing rates, spike timing, population vectors)
     */
    std::vector<float> get_output() const;
    
    /**
     * @brief Reset network to biological resting state while preserving learned structure
     * 
     * Efficiently resets >1M neural states while maintaining:
     * - Learned synaptic weight distributions
     * - Structural connectivity patterns  
     * - Modular specialization parameters
     * - Homeostatic set points and scaling factors
     */
    void reset();

    // ========================================================================
    // CPU-GPU BRIDGE INTERFACE - Seamless Integration
    // ========================================================================
    
    /**
     * @brief Synchronizes GPU network with CPU Network class state
     * @param cpu_network Reference to CPU Network for state synchronization
     * @param sync_direction Direction of synchronization ("cpu_to_gpu", "gpu_to_cpu", "bidirectional")
     * 
     * Enables seamless switching between CPU detailed analysis and GPU 
     * high-performance simulation while maintaining biological accuracy.
     */
    void synchronize_with_cpu_network(void* cpu_network_ptr, const std::string& sync_direction = "bidirectional");
    
    /**
     * @brief Copies optimized network state from host to GPU with validation
     * @param neurons Vector of GPU-optimized neuron states
     * @param synapses Vector of GPU-optimized synapse states
     */
    void copy_to_gpu(const std::vector<GPUNeuronState>& neurons, 
                     const std::vector<GPUSynapse>& synapses);
    
    /**
     * @brief Retrieves complete network state from GPU with biological encoding
     * @param neurons Vector to receive current neuron states
     * @param synapses Vector to receive current synapse states  
     */
    void copy_from_gpu(std::vector<GPUNeuronState>& neurons, 
                       std::vector<GPUSynapse>& synapses);

    // ========================================================================
    // MODULAR ARCHITECTURE SUPPORT - Advanced Neural Coordination
    // ========================================================================
    
    /**
     * @brief Associates GPU network with neural module for hierarchical processing
     * @param module_id Unique identifier for parent neural module
     * @param attention_weights GPU-optimized attention weight array
     */
    void set_parent_module(size_t module_id, const std::vector<float>& attention_weights);
    
    /**
     * @brief Configures inter-module communication pathways
     * @param source_populations Source neural populations for communication
     * @param target_populations Target neural populations for integration
     * @param communication_weights Synaptic weights for inter-module connections
     */
    void configure_module_communication(const std::vector<size_t>& source_populations,
                                       const std::vector<size_t>& target_populations,
                                       const std::vector<float>& communication_weights);
    
    /**
     * @brief Applies module-specific attention modulation with GPU efficiency
     * @param attention_pattern Spatial attention pattern for neural populations
     * @param modulation_strength Strength of attentional modulation (0.0-2.0)
     */
    void apply_attention_modulation(const std::vector<float>& attention_pattern, float modulation_strength);

    // ========================================================================
    // ADVANCED BIOLOGICAL FEATURES - GPU-Accelerated Neuroscience
    // ========================================================================
    
    /**
     * @brief Enables GPU-accelerated structural plasticity with biological constraints
     * @param enabled Whether to allow massively parallel synaptic pruning/growth
     * @param update_frequency Frequency of structural updates (steps between updates)
     */
    void set_structural_plasticity(bool enabled, int update_frequency = 1000);
    
    /**
     * @brief Configures GPU-accelerated homeostatic regulation
     * @param target_activity Desired network activity level (Hz)
     * @param regulation_strength Homeostatic feedback strength (0.0-1.0)
     * @param time_constant Regulation time constant (ms)
     */
    void set_homeostatic_regulation(float target_activity, float regulation_strength, float time_constant);
    
    /**
     * @brief Sets reward learning parameters for GPU-accelerated dopaminergic modulation
     * @param learning_rate Base learning rate for reward-modulated plasticity
     * @param eligibility_decay Time constant for eligibility trace decay (ms)
     * @param reward_prediction_error_threshold Threshold for significant reward prediction errors
     */
    void configure_reward_learning(float learning_rate, float eligibility_decay, float reward_prediction_error_threshold);

    // ========================================================================
    // STATE PERSISTENCE - Independent GPU Module Management
    // ========================================================================
    
    /**
     * @brief Saves complete GPU network state with biological encoding
     * @param filename Base filename for GPU-optimized state files
     * @return Success status of GPU save operation
     * 
     * Saves comprehensive GPU state including:
     * - All neural membrane states and ion channel configurations
     * - Complete synaptic weight matrices and plasticity states
     * - Homeostatic scaling factors and set points
     * - Structural connectivity patterns and formation timestamps
     * - Module-specific attention weights and coordination parameters
     */
    bool save_gpu_state(const std::string& filename) const;
    
    /**
     * @brief Loads complete GPU network state for rapid deployment
     * @param filename Base filename for GPU state files
     * @return Success status of GPU load operation
     */
    bool load_gpu_state(const std::string& filename);

    // ========================================================================
    // SYSTEM MONITORING - Real-Time Performance Analysis
    // ========================================================================
    
    /**
     * @brief Get GPU memory utilization with detailed breakdown
     * @return Memory usage in megabytes with allocation details
     */
    float get_allocated_memory_mb() const;
    
    /**
     * @brief Get simulation performance metrics
     * @return Performance data including kernel execution times and throughput
     */
    struct PerformanceMetrics {
        float simulation_fps;           // Simulation frames per second
        float biological_time_factor;   // Real-time to biological time ratio
        float kernel_execution_time_ms; // Average kernel execution time
        float memory_bandwidth_gbps;    // Memory bandwidth utilization
        float gpu_utilization_percent;  // GPU compute utilization
    };
    PerformanceMetrics get_performance_metrics() const;
    
    /**
     * @brief Check if GPU system is ready for cortical-scale simulation
     * @return True if system is initialized and validated
     */
    bool is_initialized() const { return system_initialized_; }

    // ========================================================================
    // ADVANCED GPU ACCESS - For Expert Integration
    // ========================================================================
    
    /**
     * @brief Access GPU neuron array for advanced kernel integration
     * @return Device pointer to GPU neuron state array
     */
    GPUNeuronState* get_device_neurons() const { return d_neurons_; }
    
    /**
     * @brief Access GPU synapse array for custom connectivity analysis
     * @return Device pointer to GPU synapse state array
     */
    GPUSynapse* get_device_synapses() const { return d_synapses_; }
    
    /**
     * @brief Get CUDA stream for asynchronous operation integration
     * @return CUDA stream handle for advanced GPU coordination
     */
    cudaStream_t get_cuda_stream() const { return stream_; }

private:
    // ========================================================================
    // CPU-GPU BRIDGE METHODS - Seamless Neural Integration
    // ========================================================================
    
    /**
     * @brief Calculates GPU-accelerated firing rate compatible with CPU Network
     * @param gpu_neuron Reference to GPU neuron state for analysis
     * @return Biologically-realistic firing rate (Hz) matching CPU calculations
     */
    float calculateNeuronFiringRate(const GPUNeuronState& gpu_neuron) const;
    
    /**
     * @brief GPU-accelerated synaptic plasticity matching CPU Network dynamics
     * @param gpu_synapse Reference to GPU synapse for plasticity update
     * @param dt Time step for integration
     * @param reward Global reward signal for modulation
     */
    void updateSynapticPlasticity(GPUSynapse& gpu_synapse, float dt, float reward);
    
    /**
     * @brief Efficient GPU-based connectivity queries for modular analysis
     * @param neuron_id Target neuron identifier
     * @return GPU-optimized vector of incoming synapse indices
     */
    std::vector<size_t> getIncomingSynapses(size_t neuron_id) const;
    
    /**
     * @brief GPU-based synapse existence check for structural plasticity
     * @param pre_id Presynaptic neuron identifier
     * @param post_id Postsynaptic neuron identifier
     * @return True if direct synaptic connection exists
     */
    bool synapseExists(size_t pre_id, size_t post_id) const;
    
    /**
     * @brief GPU-accelerated neuron activity detection
     * @param neuron_id Target neuron identifier
     * @return True if neuron is in active state based on GPU computation
     */
    bool isNeuronActive(size_t neuron_id) const;

    // ========================================================================
    // GPU MEMORY MANAGEMENT - Optimized Resource Allocation
    // ========================================================================
    
    /**
     * @brief Allocates optimized GPU memory for cortical-scale simulation
     */
    void allocate_gpu_memory();
    
    /**
     * @brief Initializes GPU neural states with biological parameter distributions
     */
    void initialize_gpu_state();
    
    /**
     * @brief Validates GPU system integrity and biological parameter bounds
     */
    void validate_initialization();
    
    /**
     * @brief Intelligent cleanup of GPU resources with state preservation options
     */
    void cleanup_resources();
    
    /**
     * @brief Optimizes GPU memory layout for cache-efficient neural computation
     */
    void optimize_memory_layout();

    // ========================================================================
    // UTILITY METHODS - Performance Optimization
    // ========================================================================
    
    /**
     * @brief Calculates optimal CUDA grid configuration for neural kernels
     * @param num_elements Number of neural elements to process
     * @param block_size Preferred CUDA block size (default optimized for neural computation)
     * @return Optimal grid dimension configuration
     */
    dim3 get_grid_size(int num_elements, int block_size = 256) const;
    
    /**
     * @brief Synchronizes CPU-GPU neural parameter types for seamless integration
     * @param cpu_config CPU Network configuration  
     * @param gpu_config GPU-optimized configuration
     */
    void synchronize_configurations(const NetworkConfig& cpu_config, NetworkConfig& gpu_config);

    // ========================================================================
    // MEMBER VARIABLES - High-Performance Neural State
    // ========================================================================
    
    // Core configuration and system state
    NetworkConfig config_;
    bool system_initialized_;
    int num_neurons_;
    int num_synapses_;
    float current_simulation_time_;
    int simulation_step_count_;
    
    // GPU memory arrays for massive parallel processing
    GPUNeuronState* d_neurons_;          // Device neuron state array
    GPUSynapse* d_synapses_;             // Device synapse state array  
    float* d_input_currents_;            // Device input current buffer
    int* d_spike_count_;                 // Device spike counting buffer
    NetworkStats* d_network_stats_;      // Device statistics buffer
    
    // Advanced GPU state for biological realism
    float* d_attention_weights_;         // Device attention modulation weights
    float* d_homeostatic_scaling_;       // Device homeostatic scaling factors
    float* d_reward_traces_;             // Device reward eligibility traces
    
    // CUDA execution and synchronization
    cudaStream_t stream_;                // CUDA stream for asynchronous execution
    float allocated_memory_mb_;          // Total allocated GPU memory
    
    // Performance monitoring
    mutable PerformanceMetrics performance_metrics_;
    
    // Modular architecture state
    size_t parent_module_id_;
    std::vector<size_t> connected_modules_;
    
    // Biological realism parameters
    float homeostatic_target_activity_;
    float homeostatic_regulation_strength_;
    float reward_learning_rate_;
    float eligibility_decay_rate_;
    bool structural_plasticity_enabled_;
    int structural_update_frequency_;
    
    // Friend classes for advanced neural integration
    friend class Network;
    friend class NeuralModule;
    friend class NetworkCUDA_Interface;
};

// ============================================================================
// CUDA KERNEL DECLARATIONS - High-Performance Neural Computation
// ============================================================================

/**
 * @brief GPU kernel for massively parallel neuron membrane dynamics
 * @param neurons Device array of neuron states (>1M parallel processing)
 * @param input_currents Device array of input currents with spatial encoding
 * @param num_neurons Number of neurons for parallel processing
 * @param dt Integration time step optimized for stability and speed
 * @param current_time Current simulation time for temporal coding
 */
__global__ void update_neurons_kernel(GPUNeuronState* neurons, 
                                     const float* input_currents,
                                     int num_neurons, float dt, float current_time);

/**
 * @brief GPU kernel for massively parallel synaptic transmission and plasticity
 * @param synapses Device array of synapse states
 * @param neurons Device array of neuron states for pre/post-synaptic access
 * @param num_synapses Number of synapses for parallel processing
 * @param dt Integration time step for plasticity dynamics
 * @param reward Global reward signal for dopaminergic-like modulation
 */
__global__ void update_synapses_kernel(GPUSynapse* synapses,
                                      GPUNeuronState* neurons,
                                      int num_synapses, float dt, float reward);

/**
 * @brief GPU kernel for real-time homeostatic regulation at cortical scale
 * @param neurons Device array of neuron states
 * @param synapses Device array of synapse states  
 * @param num_neurons Number of neurons for homeostatic regulation
 * @param target_activity Target activity level for stability
 * @param regulation_strength Homeostatic feedback strength
 */
__global__ void homeostatic_regulation_kernel(GPUNeuronState* neurons,
                                              GPUSynapse* synapses,
                                              int num_neurons, float target_activity,
                                              float regulation_strength);

/**
 * @brief GPU kernel for structural plasticity with biological constraints
 * @param synapses Device array of synapse states
 * @param neurons Device array of neuron states
 * @param num_synapses Number of synapses for structural evaluation
 * @param pruning_threshold Activity threshold for synaptic pruning
 * @param growth_probability Probability of new synapse formation
 */
__global__ void structural_plasticity_kernel(GPUSynapse* synapses,
                                            GPUNeuronState* neurons,
                                            int num_synapses, float pruning_threshold,
                                            float growth_probability);

/**
 * @brief GPU kernel for comprehensive network statistics computation
 * @param neurons Device array of neuron states
 * @param synapses Device array of synapse states
 * @param stats Device pointer to network statistics structure
 * @param num_neurons Number of neurons for statistical analysis
 * @param num_synapses Number of synapses for connectivity analysis
 * @param current_time Current simulation time for temporal metrics
 */
__global__ void compute_network_statistics_kernel(const GPUNeuronState* neurons,
                                                  const GPUSynapse* synapses,
                                                  NetworkStats* stats,
                                                  int num_neurons, int num_synapses,
                                                  float current_time);

/**
 * @brief GPU kernel for attention-modulated neural activity
 * @param neurons Device array of neuron states
 * @param attention_weights Device array of attention modulation weights
 * @param num_neurons Number of neurons for attention processing
 * @param modulation_strength Overall strength of attentional modulation
 */
__global__ void attention_modulation_kernel(GPUNeuronState* neurons,
                                           const float* attention_weights,
                                           int num_neurons, float modulation_strength);

#endif // NETWORK_CUDA_CUH