// ============================================================================
// ENHANCED LEARNING SYSTEM - PROPER CPU/GPU INTERFACE
// File: include/NeuroGen/EnhancedLearningSystem.h
// ============================================================================

#ifndef ENHANCED_LEARNING_SYSTEM_H
#define ENHANCED_LEARNING_SYSTEM_H

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>

// Forward declarations to avoid CUDA header conflicts in C++ compilation
struct GPUSynapse;
struct GPUNeuronState;
struct ModuleState;

/**
 * @brief Enhanced Learning System with Clean CPU/GPU Interface
 * 
 * This breakthrough implementation provides biologically-inspired learning
 * mechanisms while maintaining clean separation between CPU and GPU code.
 * All CUDA kernel launches are wrapped in C++ functions to ensure proper
 * compilation with standard C++ compilers.
 */
class EnhancedLearningSystem {
private:
    // GPU memory pointers (managed internally)
    void* d_synapses_ptr_;
    void* d_neurons_ptr_;
    void* d_reward_signals_ptr_;
    void* d_attention_weights_ptr_;
    void* d_trace_stats_ptr_;
    void* d_correlation_matrix_ptr_;
    
    // Network parameters
    int num_neurons_;
    int num_synapses_;
    int num_modules_;
    size_t correlation_matrix_size_;
    
    // Learning parameters
    float learning_rate_;
    float eligibility_decay_;
    float reward_scaling_;
    float baseline_dopamine_;
    
    // Module-specific state (CPU side)
    std::vector<ModuleState> module_states_;
    std::vector<float> module_attention_;
    std::vector<float> module_learning_rates_;
    
    // Performance tracking
    mutable std::atomic<float> average_eligibility_trace_;
    mutable std::atomic<float> learning_progress_;
    mutable std::atomic<float> total_weight_change_;
    
    // CUDA context management
    bool cuda_initialized_;
    cudaStream_t learning_stream_;
    cudaStream_t attention_stream_;
    
    // Thread safety
    mutable std::mutex learning_mutex_;

public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    EnhancedLearningSystem();
    ~EnhancedLearningSystem();
    
    /**
     * @brief Initialize the learning system with specified network dimensions
     * @param num_neurons Total number of neurons in the network
     * @param num_synapses Total number of synapses in the network
     * @param num_modules Number of independent neural modules
     * @return Success status of initialization
     */
    bool initialize(int num_neurons, int num_synapses, int num_modules = 1);
    
    /**
     * @brief Configure learning parameters for biological realism
     * @param lr Base learning rate for plasticity mechanisms
     * @param decay Eligibility trace decay time constant
     * @param scaling Reward signal scaling factor
     */
    void configure_learning_parameters(float lr, float decay, float scaling);
    
    /**
     * @brief Setup modular architecture with independent learning
     * @param module_sizes Vector containing neuron count for each module
     */
    void setup_modular_architecture(const std::vector<int>& module_sizes);
    
    // ========================================================================
    // MAIN LEARNING INTERFACE - NO CUDA SYNTAX IN HEADERS
    // ========================================================================
    
    /**
     * @brief Execute comprehensive learning update across all mechanisms
     * @param current_time Current simulation time (ms)
     * @param dt Time step for integration (ms)
     * @param reward_signal Global reward signal for dopaminergic modulation
     */
    void update_learning(float current_time, float dt, float reward_signal);
    
    /**
     * @brief Update attention-modulated learning for module coordination
     * @param attention_weights Per-module attention weights
     * @param dt Time step for integration
     */
    void update_attention_learning(const std::vector<float>& attention_weights, float dt);
    
    /**
     * @brief Apply module-specific learning with independent state management
     * @param module_id Target module identifier
     * @param module_reward Module-specific reward signal
     * @param dt Time step for integration
     */
    void update_modular_learning(int module_id, float module_reward, float dt);
    
    /**
     * @brief Execute STDP and eligibility trace updates
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_stdp_and_eligibility(float current_time, float dt);
    
    /**
     * @brief Apply reward modulation to eligible synapses
     * @param reward_signal Dopaminergic reward signal
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void apply_reward_modulation(float reward_signal, float current_time, float dt);
    
    /**
     * @brief Execute correlation-based learning mechanisms
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void update_correlation_learning(float current_time, float dt);
    
    /**
     * @brief Compute and apply homeostatic regulation
     * @param target_activity Desired network activity level
     * @param dt Time step
     */
    void apply_homeostatic_regulation(float target_activity, float dt);
    
    // ========================================================================
    // STATE MANAGEMENT AND PERSISTENCE
    // ========================================================================
    
    /**
     * @brief Save complete learning state to file
     * @param filename Base filename for state persistence
     * @return Success status of save operation
     */
    bool save_learning_state(const std::string& filename) const;
    
    /**
     * @brief Load complete learning state from file
     * @param filename Base filename for state loading
     * @return Success status of load operation
     */
    bool load_learning_state(const std::string& filename);
    
    /**
     * @brief Reset all learning state to baseline
     */
    void reset_learning_state();
    
    /**
     * @brief Save module-specific learning state
     * @param module_id Target module identifier
     * @param filename Filename for module state
     * @return Success status
     */
    bool save_module_learning_state(int module_id, const std::string& filename) const;
    
    /**
     * @brief Load module-specific learning state
     * @param module_id Target module identifier
     * @param filename Filename for module state
     * @return Success status
     */
    bool load_module_learning_state(int module_id, const std::string& filename);
    
    // ========================================================================
    // MONITORING AND ANALYSIS
    // ========================================================================
    
    /**
     * @brief Get average eligibility trace magnitude across network
     * @return Average eligibility trace value
     */
    float get_average_eligibility_trace() const;
    
    /**
     * @brief Get overall learning progress metric
     * @return Learning progress value [0.0, 1.0]
     */
    float get_learning_progress() const;
    
    /**
     * @brief Get learning rates for all modules
     * @return Vector of per-module learning rates
     */
    std::vector<float> get_module_learning_rates() const;
    
    /**
     * @brief Get correlation statistics for network analysis
     * @param stats Output vector for correlation statistics
     */
    void get_correlation_statistics(std::vector<float>& stats) const;
    
    /**
     * @brief Get total weight change magnitude since last reset
     * @return Total synaptic weight change
     */
    float get_total_weight_change() const;
    
    /**
     * @brief Get detailed learning statistics for performance analysis
     * @param detailed_stats Output vector for comprehensive statistics
     */
    void get_detailed_learning_statistics(std::vector<float>& detailed_stats) const;
    
    // ========================================================================
    // ADVANCED CONFIGURATION
    // ========================================================================
    
    /**
     * @brief Enable or disable specific learning mechanisms
     * @param enable_stdp Enable STDP plasticity
     * @param enable_homeostatic Enable homeostatic regulation
     * @param enable_correlation Enable correlation-based learning
     */
    void configure_learning_mechanisms(bool enable_stdp, bool enable_homeostatic, bool enable_correlation);
    
    /**
     * @brief Set module-specific learning parameters
     * @param module_id Target module
     * @param learning_rate Module learning rate
     * @param plasticity_threshold Activation threshold for plasticity
     */
    void set_module_learning_parameters(int module_id, float learning_rate, float plasticity_threshold);
    
    /**
     * @brief Configure reward prediction error computation
     * @param prediction_window Time window for prediction (ms)
     * @param error_sensitivity Sensitivity to prediction errors
     */
    void configure_reward_prediction(float prediction_window, float error_sensitivity);

private:
    // ========================================================================
    // INTERNAL IMPLEMENTATION - CUDA WRAPPER FUNCTIONS
    // ========================================================================
    
    /**
     * @brief Initialize CUDA resources and memory
     * @return Success status of CUDA initialization
     */
    bool initialize_cuda_resources();
    
    /**
     * @brief Cleanup CUDA resources
     */
    void cleanup_cuda_resources();
    
    /**
     * @brief Configure optimal CUDA execution parameters
     */
    void configure_cuda_execution_parameters();
    
    /**
     * @brief Launch eligibility trace reset operation
     */
    void launch_eligibility_reset();
    
    /**
     * @brief Launch STDP update kernels
     * @param current_time Current simulation time
     * @param dt Time step
     */
    void launch_stdp_updates(float current_time, float dt);
    
    /**
     * @brief Launch reward modulation kernels
     * @param reward Reward signal
     * @param current_time Current time
     * @param dt Time step
     */
    void launch_reward_modulation(float reward, float current_time, float dt);
    
    /**
     * @brief Launch correlation-based learning kernels
     * @param current_time Current time
     * @param dt Time step
     */
    void launch_correlation_learning(float current_time, float dt);
    
    /**
     * @brief Launch homeostatic regulation kernels
     * @param target_activity Target activity level
     * @param dt Time step
     */
    void launch_homeostatic_regulation(float target_activity, float dt);
    
    /**
     * @brief Update performance metrics from GPU
     */
    void update_performance_metrics();
    
    /**
     * @brief Validate CUDA operation success
     * @param operation_name Name of operation for error reporting
     * @return Success status
     */
    bool validate_cuda_operation(const std::string& operation_name) const;
};

// ========================================================================
// MODULE STATE STRUCTURE FOR CPU-SIDE MANAGEMENT
// ========================================================================

struct ModuleState {
    int module_id;
    int num_neurons;
    int num_synapses;
    float learning_rate;
    float attention_weight;
    float activity_level;
    float plasticity_threshold;
    bool is_active;
    
    // Learning progress tracking
    float total_weight_change;
    float average_eligibility;
    float reward_prediction_error;
    
    // Timing information
    float last_update_time;
    float total_update_time;
    
    ModuleState() : module_id(-1), num_neurons(0), num_synapses(0),
                   learning_rate(0.001f), attention_weight(1.0f),
                   activity_level(0.0f), plasticity_threshold(0.5f),
                   is_active(false), total_weight_change(0.0f),
                   average_eligibility(0.0f), reward_prediction_error(0.0f),
                   last_update_time(0.0f), total_update_time(0.0f) {}
};

#endif // ENHANCED_LEARNING_SYSTEM_H