#ifndef NETWORK_STATS_H
#define NETWORK_STATS_H

#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <atomic>
#include <memory>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define STATS_HOST_DEVICE __host__ __device__
    #define STATS_DEVICE __device__
    #define STATS_MANAGED __managed__
#else
    #define STATS_HOST_DEVICE
    #define STATS_DEVICE  
    #define STATS_MANAGED
#endif


enum class SystemState {
    STABLE,
    LEARNING,
    PRUNING,
    GROWING,
    DEGRADED,
    INACTIVE
};

/**
 * @brief Comprehensive network statistics for biologically-inspired neural networks
 * 
 * This class tracks both computational performance and biological realism metrics
 * for modular neural networks with cortical columns, neuromodulation, and plasticity.
 * Optimized for real-time brain simulation and supports both CPU and CUDA backends.
 */
class NetworkStats {
public:
    // ========================================================================
    // CORE SIMULATION METRICS
    // ========================================================================
    
    /** Current simulation time in milliseconds */
    float current_time_ms = 0.0f;
    
    /** Total elapsed simulation time */
    float total_simulation_time = 0.0f;
    
    /** Current reward signal for learning algorithms */
    float current_reward = 0.0f;
    
    /** Cumulative reward over simulation */
    float cumulative_reward = 0.0f;
    
    /** Average reward over recent time window */
    float average_reward = 0.0f;
    
    /** Number of simulation steps completed */
    uint64_t simulation_steps = 0;
    
    // ========================================================================
    // NETWORK ACTIVITY METRICS
    // ========================================================================
    
    /** Total number of spikes generated this timestep */
    uint32_t current_spike_count = 0;
    
    /** Total spikes across entire simulation */
    uint64_t total_spike_count = 0;
    
    /** Alias for backward compatibility */
    uint64_t& total_spikes = total_spike_count;
    
    /** Mean firing rate across all neurons (Hz) */
    float mean_firing_rate = 0.0f;
    
    /** Standard deviation of firing rates */
    float firing_rate_std = 0.0f;
    
    /** Network-wide synchrony measure (0-1) */
    float network_synchrony = 0.0f;
    
    /** Percentage of neurons active this timestep */
    float neuron_activity_ratio = 0.0f;
    
    /** Number of neurons that spiked */
    uint32_t active_neuron_count = 0;
    
    /** Total number of neurons in network */
    uint32_t total_neurons = 0;
    
    /** Population vector magnitude (activity coherence) */
    float population_vector_strength = 0.0f;
    
    // ========================================================================
    // SYNAPTIC AND CONNECTIVITY METRICS  
    // ========================================================================
    
    /** Total number of synapses in network */
    uint32_t total_synapses = 0;
    
    /** Number of active synapses (weight > threshold) */
    uint32_t active_synapses = 0;
    
    /** Mean synaptic weight */
    float mean_synaptic_weight = 0.0f;
    
    /** Standard deviation of synaptic weights */
    float synaptic_weight_std = 0.0f;
    
    /** Rate of synaptic weight changes */
    float plasticity_rate = 0.0f;
    
    /** Excitatory/Inhibitory balance ratio */
    float excitation_inhibition_ratio = 0.0f;
    
    /** Average connectivity per neuron */
    float mean_connectivity = 0.0f;
    
    /** Network clustering coefficient */
    float clustering_coefficient = 0.0f;
    
    /** Average path length in connectivity graph */
    float average_path_length = 0.0f;
    
    // ========================================================================
    // NEUROMODULATION METRICS
    // ========================================================================
    
    /** Current dopamine concentration */
    float dopamine_level = 0.5f;
    
    /** Current acetylcholine concentration */
    float acetylcholine_level = 0.5f;
    
    /** Current serotonin concentration */
    float serotonin_level = 0.5f;
    
    /** Current norepinephrine concentration */
    float norepinephrine_level = 0.5f;
    
    /** Rate of neuromodulator release */
    float neuromodulator_release_rate = 0.0f;
    
    /** Overall modulatory influence on plasticity */
    float modulation_strength = 1.0f;
    
    // ========================================================================
    // CORTICAL COLUMN METRICS (for modular architecture)
    // ========================================================================
    
    /** Number of cortical columns */
    uint32_t num_columns = 0;
    
    /** Activity per column */
    std::vector<float> column_activity;
    
    /** Inter-column synchronization */
    float inter_column_sync = 0.0f;
    
    /** Dominant frequency per column */
    std::vector<float> column_frequencies;
    
    /** Column specialization index */
    std::vector<float> column_specialization;
    
    // ========================================================================
    // LEARNING AND ADAPTATION METRICS
    // ========================================================================
    
    /** Current learning rate */
    float learning_rate = 0.001f;
    
    /** Hebbian learning activity */
    float hebbian_activity = 0.0f;
    
    /** Homeostatic scaling factor */
    float homeostatic_scaling = 1.0f;
    
    /** Structural plasticity events (synapse creation/deletion) */
    uint32_t structural_changes = 0;
    
    /** Metaplasticity state (learning-to-learn) */
    float metaplasticity_factor = 1.0f;
    
    /** Network entropy (measure of information processing) */
    float network_entropy = 0.0f;
    
    // ========================================================================
    // COMPUTATIONAL PERFORMANCE METRICS
    // ========================================================================
    
    /** CUDA kernel execution time (microseconds) */
    float cuda_kernel_time_us = 0.0f;
    
    /** Memory transfer time */
    float memory_transfer_time_us = 0.0f;
    
    /** Total computation time per timestep */
    float computation_time_us = 0.0f;
    
    /** Memory usage in bytes */
    size_t memory_usage_bytes = 0;
    
    /** GPU memory utilization percentage */
    float gpu_memory_utilization = 0.0f;
    
    /** Simulation speed (real-time factor) */
    float simulation_speed_factor = 1.0f;
    
    /** Number of CUDA blocks launched */
    uint32_t cuda_blocks_launched = 0;
    
    /** Average threads per block */
    float avg_threads_per_block = 0.0f;
    
    // ========================================================================
    // BIOLOGICAL REALISM METRICS
    // ========================================================================
    
    /** Calcium concentration dynamics */
    float mean_calcium_level = 0.0f;
    
    /** Vesicle pool depletion rate */
    float vesicle_depletion_rate = 0.0f;
    
    /** Network criticality measure (edge of chaos) */
    float criticality_index = 0.0f;
    
    /** Avalanche size distribution exponent */
    float avalanche_exponent = 0.0f;
    
    /** Oscillation power in different frequency bands */
    float theta_power = 0.0f;      // 4-8 Hz
    float alpha_power = 0.0f;      // 8-13 Hz  
    float beta_power = 0.0f;       // 13-30 Hz
    float gamma_power = 0.0f;      // 30-100 Hz
    
    /** Phase-locking between regions */
    float phase_coherence = 0.0f;
    
    // ========================================================================
    // ERROR AND STABILITY METRICS
    // ========================================================================
    
    /** Numerical stability indicator */
    float numerical_stability = 1.0f;
    
    /** Maximum voltage deviation */
    float max_voltage_deviation = 0.0f;
    
    /** Weight saturation percentage */
    float weight_saturation = 0.0f;
    
    /** Simulation divergence indicator */
    bool simulation_stable = true;
    
    /** CUDA error count */
    uint32_t cuda_error_count = 0;
    
    // ========================================================================
    // TIMING AND PROFILING
    // ========================================================================
    
    /** Detailed timing breakdown */
    struct TimingProfile {
        float neuron_update_time = 0.0f;
        float synapse_update_time = 0.0f;
        float plasticity_update_time = 0.0f;
        float spike_processing_time = 0.0f;
        float memory_copy_time = 0.0f;
        float total_time = 0.0f;
    } timing_profile;
    
    /** Performance history for trend analysis */
    std::vector<float> performance_history;
    
    // ========================================================================
    // CONSTRUCTORS AND BASIC OPERATIONS
    // ========================================================================
    
    STATS_HOST_DEVICE NetworkStats() {
        reset();
    }
    
    /** Reset all statistics to initial state */
    STATS_HOST_DEVICE void reset() {
        // Core metrics
        current_time_ms = 0.0f;
        total_simulation_time = 0.0f;
        current_reward = 0.0f;
        cumulative_reward = 0.0f;
        average_reward = 0.0f;
        simulation_steps = 0;
        
        // Activity metrics
        current_spike_count = 0;
        total_spike_count = 0;
        mean_firing_rate = 0.0f;
        firing_rate_std = 0.0f;
        network_synchrony = 0.0f;
        neuron_activity_ratio = 0.0f;
        active_neuron_count = 0;
        population_vector_strength = 0.0f;
        
        // Synaptic metrics
        mean_synaptic_weight = 0.0f;
        synaptic_weight_std = 0.0f;
        plasticity_rate = 0.0f;
        excitation_inhibition_ratio = 1.0f;
        mean_connectivity = 0.0f;
        clustering_coefficient = 0.0f;
        average_path_length = 0.0f;
        
        // Neuromodulation
        dopamine_level = 0.5f;
        acetylcholine_level = 0.5f;
        serotonin_level = 0.5f;
        norepinephrine_level = 0.5f;
        neuromodulator_release_rate = 0.0f;
        modulation_strength = 1.0f;
        
        // Learning metrics
        learning_rate = 0.001f;
        hebbian_activity = 0.0f;
        homeostatic_scaling = 1.0f;
        structural_changes = 0;
        metaplasticity_factor = 1.0f;
        network_entropy = 0.0f;
        
        // Performance metrics
        cuda_kernel_time_us = 0.0f;
        memory_transfer_time_us = 0.0f;
        computation_time_us = 0.0f;
        memory_usage_bytes = 0;
        gpu_memory_utilization = 0.0f;
        simulation_speed_factor = 1.0f;
        
        // Biological metrics
        mean_calcium_level = 0.0f;
        vesicle_depletion_rate = 0.0f;
        criticality_index = 0.0f;
        avalanche_exponent = 0.0f;
        theta_power = 0.0f;
        alpha_power = 0.0f;
        beta_power = 0.0f;
        gamma_power = 0.0f;
        phase_coherence = 0.0f;
        
        // Stability metrics
        numerical_stability = 1.0f;
        max_voltage_deviation = 0.0f;
        weight_saturation = 0.0f;
        simulation_stable = true;
        cuda_error_count = 0;
        
        // Reset timing profile
        timing_profile = TimingProfile{};
        
        // Clear history vectors
        #ifndef __CUDA_ARCH__  // Host code only
        column_activity.clear();
        column_frequencies.clear();
        column_specialization.clear();
        performance_history.clear();
        #endif
    }
    
    // ========================================================================
    // UPDATE METHODS
    // ========================================================================
    
    /** Update core simulation metrics */
    STATS_HOST_DEVICE void updateSimulationMetrics(float dt_ms, uint64_t step_count) {
        current_time_ms += dt_ms;
        total_simulation_time += dt_ms;
        simulation_steps = step_count;
        
        // Update average reward with exponential decay
        const float alpha = 0.01f; // Decay factor
        average_reward = alpha * current_reward + (1.0f - alpha) * average_reward;
        cumulative_reward += current_reward;
    }
    
    /** Update activity metrics */
    STATS_HOST_DEVICE void updateActivityMetrics(uint32_t spikes_this_step, 
                                                uint32_t active_neurons,
                                                uint32_t total_neurons) {
        current_spike_count = spikes_this_step;
        total_spike_count += spikes_this_step;
        active_neuron_count = active_neurons;
        
        if (total_neurons > 0) {
            neuron_activity_ratio = static_cast<float>(active_neurons) / total_neurons;
        }
        
        // Update mean firing rate (exponential moving average)
        const float dt = 0.001f; // Assume 1ms timestep for rate calculation
        const float alpha = 0.05f;
        float instantaneous_rate = static_cast<float>(spikes_this_step) / (total_neurons * dt);
        mean_firing_rate = alpha * instantaneous_rate + (1.0f - alpha) * mean_firing_rate;
    }
    
    /** Update synaptic metrics */
    STATS_HOST_DEVICE void updateSynapticMetrics(float mean_weight, float weight_std,
                                                uint32_t active_syns, uint32_t total_syns) {
        mean_synaptic_weight = mean_weight;
        synaptic_weight_std = weight_std;
        active_synapses = active_syns;
        total_synapses = total_syns;
        
        if (total_syns > 0) {
            mean_connectivity = static_cast<float>(active_syns) / total_syns;
        }
    }
    
    /** Update neuromodulator levels */
    STATS_HOST_DEVICE void updateNeuromodulation(float da, float ach, float ser, float nor) {
        dopamine_level = da;
        acetylcholine_level = ach;
        serotonin_level = ser;
        norepinephrine_level = nor;
        
        // Calculate overall modulation strength
        modulation_strength = (da + ach + ser + nor) / 4.0f;
    }
    
    /** Update performance metrics */
    STATS_HOST_DEVICE void updatePerformanceMetrics(float kernel_time, float transfer_time,
                                                   size_t memory_bytes) {
        cuda_kernel_time_us = kernel_time;
        memory_transfer_time_us = transfer_time;
        computation_time_us = kernel_time + transfer_time;
        memory_usage_bytes = memory_bytes;
        
        // Calculate simulation speed factor
        const float real_time_us = 1000.0f; // 1ms in microseconds
        if (computation_time_us > 0) {
            simulation_speed_factor = real_time_us / computation_time_us;
        }
    }
    
    // ========================================================================
    // ANALYSIS METHODS
    // ========================================================================
    
    /** Check if network is in healthy operating range */
    STATS_HOST_DEVICE bool isNetworkHealthy() const {
        return simulation_stable &&
               mean_firing_rate > 0.1f && mean_firing_rate < 100.0f &&
               numerical_stability > 0.9f &&
               cuda_error_count == 0;
    }
    
    /** Get overall network efficiency score (0-1) */
    STATS_HOST_DEVICE float getNetworkEfficiency() const {
        float activity_score = (neuron_activity_ratio > 0.05f && neuron_activity_ratio < 0.8f) ? 1.0f : 0.5f;
        float plasticity_score = (plasticity_rate > 0.0f && plasticity_rate < 0.1f) ? 1.0f : 0.5f;
        float stability_score = numerical_stability;
        
        return (activity_score + plasticity_score + stability_score) / 3.0f;
    }
    
    /** Get computational performance score */
    STATS_HOST_DEVICE float getPerformanceScore() const {
        return (simulation_speed_factor > 1.0f) ? 1.0f : simulation_speed_factor;
    }
    
    // ========================================================================
    // UTILITY METHODS
    // ========================================================================
    
    /** Convert statistics to human-readable string */
    std::string toString() const;
    
    /** Export detailed statistics to JSON format */
    std::string toJSON() const;
    
    /** Save statistics to binary file */
    bool saveToFile(const std::string& filename) const;
    
    /** Load statistics from binary file */
    bool loadFromFile(const std::string& filename);
};

// ============================================================================
// GLOBAL STATISTICS INSTANCES
// ============================================================================

#ifdef __CUDACC__
/** Global managed statistics for CUDA kernels */
extern STATS_MANAGED NetworkStats g_stats;
#else
/** Global statistics instance for CPU code */
extern NetworkStats g_stats;
#endif

// ============================================================================
// CONVENIENCE MACROS FOR STATISTICS UPDATES
// ============================================================================

#define UPDATE_SIMULATION_STATS(dt, steps) \
    g_stats.updateSimulationMetrics(dt, steps)

#define UPDATE_ACTIVITY_STATS(spikes, active, total) \
    g_stats.updateActivityMetrics(spikes, active, total)

#define UPDATE_SYNAPTIC_STATS(mean_w, std_w, active_s, total_s) \
    g_stats.updateSynapticMetrics(mean_w, std_w, active_s, total_s)

#define UPDATE_NEUROMOD_STATS(da, ach, ser, nor) \
    g_stats.updateNeuromodulation(da, ach, ser, nor)

#define UPDATE_PERFORMANCE_STATS(kernel_t, transfer_t, mem_bytes) \
    g_stats.updatePerformanceMetrics(kernel_t, transfer_t, mem_bytes)

#define STATS_RECORD_ERROR() \
    do { g_stats.cuda_error_count++; g_stats.simulation_stable = false; } while(0)

#endif // NETWORK_STATS_H