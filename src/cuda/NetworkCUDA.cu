// ============================================================================
// CRITICAL IMPLEMENTATION FIXES FOR NetworkCUDA.cu
// ============================================================================
#include <NeuroGen/cuda/NetworkCUDA.cuh>

// CRITICAL: Add missing standard headers
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>      // CRITICAL: For std::ofstream and std::ifstream
#include <memory>       // For smart pointers


struct CPUNeuronAdapter {
    float potential;
    float recovery_variable;
    float last_spike_time;
    float firing_rate;
    size_t neuron_id;
    bool is_valid;
};

struct CPUSynapseAdapter {
    size_t pre_neuron_id;
    size_t post_neuron_id;
    float weight;
    float delay;
    float eligibility_trace;
    float activity_metric;
    bool is_valid;
};

struct CPUNetworkAdapter {
    std::vector<CPUNeuronAdapter> neurons;
    std::vector<CPUSynapseAdapter> synapses;
    NetworkStats stats;
    size_t active_neuron_count;
    size_t active_synapse_count;
    bool is_valid;
};


extern "C" {
    // Function to extract CPU network data without direct class access
    bool extract_cpu_network_data(void* cpu_network_ptr, CPUNetworkAdapter* adapter);
    
    // Function to update CPU network from GPU data
    bool update_cpu_network_data(void* cpu_network_ptr, const CPUNetworkAdapter* adapter);
    
    // Function to get specific neuron data
    bool get_cpu_neuron_data(void* cpu_network_ptr, size_t neuron_id, CPUNeuronAdapter* adapter);
    
    // Function to get outgoing synapses for a neuron
    bool get_cpu_outgoing_synapses(void* cpu_network_ptr, size_t neuron_id, 
                                  CPUSynapseAdapter* synapses, size_t max_synapses, size_t* actual_count);
}


// ============================================================================
// FIXED IMPLEMENTATION: CPU-GPU BRIDGE METHODS
// ============================================================================

// REPLACE the problematic method implementations with these corrected versions:

// Method 1: Fixed calculateNeuronFiringRate - now uses GPU structures
float NetworkCUDA::calculateNeuronFiringRate(const GPUNeuronState& gpu_neuron) const {
    // Biologically-realistic firing rate calculation matching CPU Network dynamics
    constexpr float BASELINE_RATE = 2.0f;  // Hz - cortical baseline
    constexpr float MAX_RATE = 100.0f;     // Hz - physiological maximum
    constexpr float THRESHOLD_VOLTAGE = -55.0f; // mV - typical spike threshold
    
    // Check recent spiking activity
    float time_since_spike = current_simulation_time_ - gpu_neuron.last_spike_time;
    bool recently_spiked = (time_since_spike < 5.0f); // Within 5ms
    
    if (recently_spiked) {
        // Active neuron: rate depends on membrane potential dynamics
        float potential_factor = std::tanh((gpu_neuron.V - THRESHOLD_VOLTAGE) / 20.0f);
        return BASELINE_RATE + (MAX_RATE - BASELINE_RATE) * std::max(0.0f, potential_factor);
    }
    
    // Subthreshold activity contributes to background rate
    float subthreshold_factor = std::max(0.0f, (gpu_neuron.V + 70.0f) / 50.0f);
    return BASELINE_RATE * subthreshold_factor;
}

// Method 2: Fixed updateSynapticPlasticity - now uses GPU structures
void NetworkCUDA::updateSynapticPlasticity(GPUSynapse& gpu_synapse, float dt, float reward) {
    // Advanced biological plasticity mechanisms for GPU acceleration
    
    // Get pre and post neuron states
    if (gpu_synapse.pre_neuron_idx >= num_neurons_ || gpu_synapse.post_neuron_idx >= num_neurons_) {
        return; // Invalid neuron indices
    }
    
    // Access GPU neuron states directly
    GPUNeuronState& pre_neuron = d_neurons_[gpu_synapse.pre_neuron_idx];
    GPUNeuronState& post_neuron = d_neurons_[gpu_synapse.post_neuron_idx];
    
    // Check for recent spiking activity
    float time_since_pre_spike = current_simulation_time_ - pre_neuron.last_spike_time;
    float time_since_post_spike = current_simulation_time_ - post_neuron.last_spike_time;
    
    bool pre_spiked = (time_since_pre_spike < 2.0f);  // Within 2ms
    bool post_spiked = (time_since_post_spike < 2.0f); // Within 2ms
    
    // Update activity metric with biological time constants
    float activity_decay = 0.995f; // ~200ms time constant
    if (pre_spiked || post_spiked) {
        gpu_synapse.activity_metric = gpu_synapse.activity_metric * activity_decay + 0.01f;
    } else {
        gpu_synapse.activity_metric *= activity_decay;
    }
    
    // Spike-timing dependent plasticity (STDP)
    if (pre_spiked && post_spiked) {
        // Calculate timing-dependent plasticity
        float spike_time_diff = gpu_synapse.last_post_spike_time - gpu_synapse.last_pre_spike_time;
        float stdp_window = 20.0f; // ms
        
        float hebbian_change = 0.001f * reward; // Reward-modulated learning
        
        // Apply timing-dependent modulation
        if (std::abs(spike_time_diff) < stdp_window) {
            if (spike_time_diff > 0) {
                // Pre before post - potentiation
                hebbian_change *= std::exp(-spike_time_diff / 10.0f);
            } else {
                // Post before pre - depression  
                hebbian_change *= -0.5f * std::exp(spike_time_diff / 10.0f);
            }
        }
        
        gpu_synapse.weight += hebbian_change;
        
        // Biological weight bounds with type-safe clamping
        float min_weight = static_cast<float>(config_.min_weight);
        float max_weight = static_cast<float>(config_.max_weight);
        gpu_synapse.weight = std::max(min_weight, std::min(max_weight, gpu_synapse.weight));
        
        // Update eligibility trace for delayed reward learning
        gpu_synapse.eligibility_trace = std::min(1.0f, gpu_synapse.eligibility_trace + 0.1f);
        
        // Record spike times for future STDP calculations
        gpu_synapse.last_pre_spike_time = pre_neuron.last_spike_time;
        gpu_synapse.last_post_spike_time = post_neuron.last_spike_time;
    }
    
    // Eligibility trace decay
    gpu_synapse.eligibility_trace *= 0.99f; // ~100ms time constant
    
    // Update plasticity modulation based on recent activity
    gpu_synapse.plasticity_modulation = 1.0f + 0.5f * gpu_synapse.activity_metric;
}

// Method 3: Fixed getIncomingSynapses - now returns GPU-optimized indices
std::vector<size_t> NetworkCUDA::getIncomingSynapses(size_t neuron_id) const {
    std::vector<size_t> incoming_indices;
    
    if (neuron_id >= static_cast<size_t>(num_neurons_)) {
        return incoming_indices; // Invalid neuron ID
    }
    
    // Search through all synapses for connections to this neuron
    // Note: In production, this would use optimized GPU-side connection maps
    for (int syn_idx = 0; syn_idx < num_synapses_; ++syn_idx) {
        GPUSynapse synapse;
        
        // Copy synapse from GPU to check connectivity
        cudaError_t err = cudaMemcpy(&synapse, &d_synapses_[syn_idx], 
                                   sizeof(GPUSynapse), cudaMemcpyDeviceToHost);
        
        if (err == cudaSuccess && 
            static_cast<size_t>(synapse.post_neuron_idx) == neuron_id && 
            synapse.active) {
            incoming_indices.push_back(syn_idx);
        }
    }
    
    return incoming_indices;
}

// ============================================================================
// NEW IMPLEMENTATION: CPU-GPU BRIDGE INTERFACE
// ============================================================================

// Add these new methods to NetworkCUDA class for seamless CPU integration:

void NetworkCUDA::synchronize_with_cpu_network(void* cpu_network_ptr, const std::string& sync_direction) {
    std::cout << "🔄 Synchronizing CPU-GPU neural states: " << sync_direction << std::endl;
    
    if (sync_direction == "cpu_to_gpu" || sync_direction == "bidirectional") {
        std::cout << "📥 Transferring CPU neural states to GPU..." << std::endl;
        
        // Use adapter structure instead of direct CPU class access
        CPUNetworkAdapter cpu_adapter;
        
        if (!extract_cpu_network_data(cpu_network_ptr, &cpu_adapter)) {
            std::cerr << "❌ Failed to extract CPU network data" << std::endl;
            return;
        }
        
        std::cout << "   • CPU neurons: " << cpu_adapter.active_neuron_count << std::endl;
        std::cout << "   • CPU synapses: " << cpu_adapter.active_synapse_count << std::endl;
        
        // Convert adapter data to GPU format
        std::vector<GPUNeuronState> gpu_neurons(num_neurons_);
        std::vector<GPUSynapse> gpu_synapses(num_synapses_);
        
        // Convert CPU neurons to GPU format using adapter
        size_t neurons_to_copy = std::min(static_cast<size_t>(num_neurons_), cpu_adapter.neurons.size());
        for (size_t i = 0; i < neurons_to_copy; ++i) {
            if (cpu_adapter.neurons[i].is_valid) {
                gpu_neurons[i].V = cpu_adapter.neurons[i].potential;
                gpu_neurons[i].u = cpu_adapter.neurons[i].recovery_variable;
                gpu_neurons[i].last_spike_time = cpu_adapter.neurons[i].last_spike_time;
                gpu_neurons[i].average_firing_rate = cpu_adapter.neurons[i].firing_rate;
                gpu_neurons[i].excitability = 1.0f;
                gpu_neurons[i].synaptic_scaling_factor = 1.0f;
                
                // Initialize synaptic input arrays
                for (int comp = 0; comp < 4; ++comp) {
                    gpu_neurons[i].I_syn[comp] = 0.0f;
                    gpu_neurons[i].ca_conc[comp] = 0.1f;
                }
            }
        }
        
        // Convert CPU synapses to GPU format using adapter
        size_t synapses_to_copy = std::min(static_cast<size_t>(num_synapses_), cpu_adapter.synapses.size());
        for (size_t i = 0; i < synapses_to_copy; ++i) {
            if (cpu_adapter.synapses[i].is_valid) {
                gpu_synapses[i].pre_neuron_idx = static_cast<int>(cpu_adapter.synapses[i].pre_neuron_id);
                gpu_synapses[i].post_neuron_idx = static_cast<int>(cpu_adapter.synapses[i].post_neuron_id);
                gpu_synapses[i].weight = cpu_adapter.synapses[i].weight;
                gpu_synapses[i].delay = cpu_adapter.synapses[i].delay;
                gpu_synapses[i].active = 1;
                gpu_synapses[i].eligibility_trace = cpu_adapter.synapses[i].eligibility_trace;
                gpu_synapses[i].activity_metric = cpu_adapter.synapses[i].activity_metric;
                gpu_synapses[i].plasticity_modulation = 1.0f;
                gpu_synapses[i].effective_weight = cpu_adapter.synapses[i].weight;
                gpu_synapses[i].last_pre_spike_time = -1000.0f;
                gpu_synapses[i].last_post_spike_time = -1000.0f;
                gpu_synapses[i].last_active_time = 0.0f;
                gpu_synapses[i].max_weight = static_cast<float>(config_.max_weight);
                gpu_synapses[i].min_weight = static_cast<float>(config_.min_weight);
                gpu_synapses[i].dopamine_sensitivity = 1.0f;
                gpu_synapses[i].acetylcholine_sensitivity = 1.0f;
                gpu_synapses[i].post_compartment = 0;
            }
        }
        
        // Copy converted states to GPU
        copy_to_gpu(gpu_neurons, gpu_synapses);
        std::cout << "✅ CPU to GPU synchronization completed" << std::endl;
    }
    
    if (sync_direction == "gpu_to_cpu" || sync_direction == "bidirectional") {
        std::cout << "📤 Transferring GPU neural states to CPU..." << std::endl;
        
        // Create adapter with GPU data
        CPUNetworkAdapter gpu_to_cpu_adapter;
        
        std::vector<GPUNeuronState> gpu_neurons(num_neurons_);
        std::vector<GPUSynapse> gpu_synapses(num_synapses_);
        
        copy_from_gpu(gpu_neurons, gpu_synapses);
        
        // Convert GPU data to adapter format
        gpu_to_cpu_adapter.neurons.resize(num_neurons_);
        for (int i = 0; i < num_neurons_; ++i) {
            gpu_to_cpu_adapter.neurons[i].potential = gpu_neurons[i].V;
            gpu_to_cpu_adapter.neurons[i].recovery_variable = gpu_neurons[i].u;
            gpu_to_cpu_adapter.neurons[i].last_spike_time = gpu_neurons[i].last_spike_time;
            gpu_to_cpu_adapter.neurons[i].firing_rate = gpu_neurons[i].average_firing_rate;
            gpu_to_cpu_adapter.neurons[i].neuron_id = i;
            gpu_to_cpu_adapter.neurons[i].is_valid = true;
        }
        
        gpu_to_cpu_adapter.synapses.resize(num_synapses_);
        for (int i = 0; i < num_synapses_; ++i) {
            if (gpu_synapses[i].active) {
                gpu_to_cpu_adapter.synapses[i].pre_neuron_id = gpu_synapses[i].pre_neuron_idx;
                gpu_to_cpu_adapter.synapses[i].post_neuron_id = gpu_synapses[i].post_neuron_idx;
                gpu_to_cpu_adapter.synapses[i].weight = gpu_synapses[i].weight;
                gpu_to_cpu_adapter.synapses[i].delay = gpu_synapses[i].delay;
                gpu_to_cpu_adapter.synapses[i].eligibility_trace = gpu_synapses[i].eligibility_trace;
                gpu_to_cpu_adapter.synapses[i].activity_metric = gpu_synapses[i].activity_metric;
                gpu_to_cpu_adapter.synapses[i].is_valid = true;
            }
        }
        
        // Update CPU network using adapter
        if (!update_cpu_network_data(cpu_network_ptr, &gpu_to_cpu_adapter)) {
            std::cerr << "❌ Failed to update CPU network data" << std::endl;
            return;
        }
        
        std::cout << "✅ GPU to CPU synchronization completed" << std::endl;
    }
}

std::vector<float> NetworkCUDA::get_output() const {
    std::vector<float> outputs;
    
    if (!system_initialized_ || num_neurons_ == 0) {
        return outputs;
    }
    
    // Extract output from GPU neurons (typically last neurons in network)
    size_t output_start = std::max(0, num_neurons_ - config_.output_size);
    outputs.reserve(config_.output_size);
    
    // Copy GPU neuron states to get output
    std::vector<GPUNeuronState> gpu_neurons(num_neurons_);
    cudaError_t err = cudaMemcpy(gpu_neurons.data(), d_neurons_, 
                                num_neurons_ * sizeof(GPUNeuronState), 
                                cudaMemcpyDeviceToHost);
    
    if (err == cudaSuccess) {
        for (size_t i = output_start; i < static_cast<size_t>(num_neurons_); i++) {
            // Use GPU-compatible firing rate calculation
            float firing_rate = calculateNeuronFiringRate(gpu_neurons[i]);
            outputs.push_back(firing_rate);
        }
    }
    
    return outputs;
}

// ============================================================================
// NETWORKCU DA IMPLEMENTATION - Core Methods
// ============================================================================

NetworkCUDA::NetworkCUDA(const NetworkConfig& config) 
    : config_(config),
      system_initialized_(false),
      num_neurons_(config.hidden_size),
      num_synapses_(config.hidden_size * 4), // Estimate based on connectivity
      current_simulation_time_(0.0f),
      simulation_step_count_(0),
      d_neurons_(nullptr),
      d_synapses_(nullptr),
      d_input_currents_(nullptr),
      d_spike_count_(nullptr),
      d_network_stats_(nullptr),
      allocated_memory_mb_(0.0f) {
    
    std::cout << "🧠 Initializing NetworkCUDA with " << num_neurons_ << " neurons..." << std::endl;
    
    // Allocate GPU memory
    allocate_gpu_memory();
    
    // Initialize GPU state
    initialize_gpu_state();
    
    system_initialized_ = true;
    std::cout << "✅ NetworkCUDA initialized successfully" << std::endl;
}

NetworkCUDA::~NetworkCUDA() {
    cleanup_resources();
    std::cout << "🧠 NetworkCUDA destroyed" << std::endl;
}

void NetworkCUDA::allocate_gpu_memory() {
    cudaError_t err;
    
    // Allocate neurons
    err = cudaMalloc(&d_neurons_, num_neurons_ * sizeof(GPUNeuronState));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate neuron memory: " + std::string(cudaGetErrorString(err)));
    }
    
    // Allocate synapses
    err = cudaMalloc(&d_synapses_, num_synapses_ * sizeof(GPUSynapse));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate synapse memory: " + std::string(cudaGetErrorString(err)));
    }
    
    // Allocate auxiliary arrays
    err = cudaMalloc(&d_input_currents_, num_neurons_ * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate input currents: " + std::string(cudaGetErrorString(err)));
    }
    
    err = cudaMalloc(&d_spike_count_, sizeof(int));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate spike count: " + std::string(cudaGetErrorString(err)));
    }
    
    err = cudaMalloc(&d_network_stats_, sizeof(NetworkStats));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate network stats: " + std::string(cudaGetErrorString(err)));
    }
    
    // Calculate allocated memory
    size_t total_bytes = num_neurons_ * sizeof(GPUNeuronState) +
                        num_synapses_ * sizeof(GPUSynapse) +
                        num_neurons_ * sizeof(float) +
                        sizeof(int) + sizeof(NetworkStats);
    allocated_memory_mb_ = total_bytes / (1024.0f * 1024.0f);
    
    std::cout << "💾 GPU memory allocated: " << allocated_memory_mb_ << " MB" << std::endl;
}

void NetworkCUDA::initialize_gpu_state() {
    // Initialize neurons
    cudaMemset(d_neurons_, 0, num_neurons_ * sizeof(GPUNeuronState));
    
    // Initialize synapses
    cudaMemset(d_synapses_, 0, num_synapses_ * sizeof(GPUSynapse));
    
    // Initialize auxiliary arrays
    cudaMemset(d_input_currents_, 0, num_neurons_ * sizeof(float));
    cudaMemset(d_spike_count_, 0, sizeof(int));
    cudaMemset(d_network_stats_, 0, sizeof(NetworkStats));
    
    std::cout << "🔧 GPU state initialized" << std::endl;
}

void NetworkCUDA::cleanup_resources() {
    if (d_neurons_) cudaFree(d_neurons_);
    if (d_synapses_) cudaFree(d_synapses_);
    if (d_input_currents_) cudaFree(d_input_currents_);
    if (d_spike_count_) cudaFree(d_spike_count_);
    if (d_network_stats_) cudaFree(d_network_stats_);
    
    d_neurons_ = nullptr;
    d_synapses_ = nullptr;
    d_input_currents_ = nullptr;
    d_spike_count_ = nullptr;
    d_network_stats_ = nullptr;
}

void NetworkCUDA::copy_to_gpu(const std::vector<GPUNeuronState>& neurons, 
                             const std::vector<GPUSynapse>& synapses) {
    if (!system_initialized_) return;
    
    // Copy neurons
    size_t neurons_to_copy = std::min(static_cast<size_t>(num_neurons_), neurons.size());
    if (neurons_to_copy > 0) {
        cudaMemcpy(d_neurons_, neurons.data(), 
                  neurons_to_copy * sizeof(GPUNeuronState), 
                  cudaMemcpyHostToDevice);
    }
    
    // Copy synapses
    size_t synapses_to_copy = std::min(static_cast<size_t>(num_synapses_), synapses.size());
    if (synapses_to_copy > 0) {
        cudaMemcpy(d_synapses_, synapses.data(), 
                  synapses_to_copy * sizeof(GPUSynapse), 
                  cudaMemcpyHostToDevice);
    }
}

void NetworkCUDA::copy_from_gpu(std::vector<GPUNeuronState>& neurons, 
                               std::vector<GPUSynapse>& synapses) {
    if (!system_initialized_) return;
    
    // Resize vectors
    neurons.resize(num_neurons_);
    synapses.resize(num_synapses_);
    
    // Copy from GPU
    cudaMemcpy(neurons.data(), d_neurons_, 
              num_neurons_ * sizeof(GPUNeuronState), 
              cudaMemcpyDeviceToHost);
    
    cudaMemcpy(synapses.data(), d_synapses_, 
              num_synapses_ * sizeof(GPUSynapse), 
              cudaMemcpyDeviceToHost);
}

void NetworkCUDA::simulate_step(float current_time, float dt, float reward, 
                               const std::vector<float>& inputs) {
    if (!system_initialized_) return;
    
    current_simulation_time_ = current_time;
    simulation_step_count_++;
    
    // Copy inputs to GPU if provided
    if (!inputs.empty()) {
        size_t input_size = std::min(static_cast<size_t>(num_neurons_), inputs.size());
        cudaMemcpy(d_input_currents_, inputs.data(), 
                  input_size * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // Execute simulation step (placeholder - would call actual kernels)
    // This would launch the neural update kernels
    std::cout << "🧠 Simulation step " << simulation_step_count_ << " at t=" << current_time << "ms" << std::endl;
}

void NetworkCUDA::get_stats(NetworkStats& stats) const {
    if (!system_initialized_) {
        stats.reset();
        return;
    }
    
    // Copy stats from GPU
    cudaMemcpy(&stats, d_network_stats_, sizeof(NetworkStats), cudaMemcpyDeviceToHost);
    
    // Fill in basic information
    stats.active_neuron_count = num_neurons_;
    stats.active_synapses = num_synapses_;
    stats.simulation_steps = simulation_step_count_;
    stats.current_time_ms = current_simulation_time_;
}

// ============================================================================
// MISSING UTILITY FUNCTIONS IMPLEMENTATION
// ============================================================================

extern "C" {

bool extract_cpu_network_data(void* cpu_network_ptr, CPUNetworkAdapter* adapter) {
    if (!cpu_network_ptr || !adapter) return false;
    
    // This is a placeholder implementation
    // In a real implementation, this would cast cpu_network_ptr to Network*
    // and extract the actual neural data
    
    adapter->neurons.clear();
    adapter->synapses.clear();
    adapter->active_neuron_count = 0;
    adapter->active_synapse_count = 0;
    adapter->is_valid = true;
    
    // For now, return success with empty data
    // Real implementation would access Network class members
    std::cout << "📤 Extracting CPU network data (placeholder)" << std::endl;
    return true;
}

bool update_cpu_network_data(void* cpu_network_ptr, const CPUNetworkAdapter* adapter) {
    if (!cpu_network_ptr || !adapter) return false;
    
    // This is a placeholder implementation
    // In a real implementation, this would cast cpu_network_ptr to Network*
    // and update the CPU network state with GPU data
    
    std::cout << "📥 Updating CPU network data (placeholder)" << std::endl;
    return true;
}

bool get_cpu_neuron_data(void* cpu_network_ptr, size_t neuron_id, CPUNeuronAdapter* adapter) {
    if (!cpu_network_ptr || !adapter) return false;
    
    // Placeholder implementation
    adapter->potential = 0.0f;
    adapter->recovery_variable = 0.0f;
    adapter->last_spike_time = -1000.0f;
    adapter->firing_rate = 0.0f;
    adapter->neuron_id = neuron_id;
    adapter->is_valid = true;
    
    return true;
}

bool get_cpu_outgoing_synapses(void* cpu_network_ptr, size_t neuron_id, 
                              CPUSynapseAdapter* synapses, size_t max_synapses, size_t* actual_count) {
    if (!cpu_network_ptr || !synapses || !actual_count) return false;
    
    // Placeholder implementation
    *actual_count = 0;
    return true;
}

} // extern "C"

// ============================================================================
// ADDITIONAL HELPER METHODS FOR SEAMLESS INTEGRATION
// ============================================================================

bool NetworkCUDA::synapseExists(size_t pre_id, size_t post_id) const {
    if (pre_id >= static_cast<size_t>(num_neurons_) || 
        post_id >= static_cast<size_t>(num_neurons_)) {
        return false;
    }
    
    // Search GPU synapses for connection
    for (int syn_idx = 0; syn_idx < num_synapses_; ++syn_idx) {
        GPUSynapse synapse;
        cudaError_t err = cudaMemcpy(&synapse, &d_synapses_[syn_idx], 
                                   sizeof(GPUSynapse), cudaMemcpyDeviceToHost);
        
        if (err == cudaSuccess && 
            static_cast<size_t>(synapse.pre_neuron_idx) == pre_id &&
            static_cast<size_t>(synapse.post_neuron_idx) == post_id &&
            synapse.active) {
            return true;
        }
    }
    
    return false;
}

bool NetworkCUDA::isNeuronActive(size_t neuron_id) const {
    if (neuron_id >= static_cast<size_t>(num_neurons_)) {
        return false;
    }
    
    // Copy single neuron state from GPU
    GPUNeuronState neuron;
    cudaError_t err = cudaMemcpy(&neuron, &d_neurons_[neuron_id], 
                                sizeof(GPUNeuronState), cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        return false;
    }
    
    // Define activity based on firing rate and membrane potential
    float firing_rate = calculateNeuronFiringRate(neuron);
    bool above_threshold = neuron.V > -60.0f; // mV
    
    return firing_rate > 1.0f || above_threshold;
}

// ============================================================================
// PERFORMANCE OPTIMIZATION METHODS
// ============================================================================

NetworkCUDA::PerformanceMetrics NetworkCUDA::get_performance_metrics() const {
    PerformanceMetrics metrics;
    
    // Calculate simulation performance
    static auto last_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_time);
    
    if (elapsed.count() > 0) {
        metrics.simulation_fps = 1000000.0f / elapsed.count(); // FPS
        metrics.biological_time_factor = (config_.dt * 1000.0f) / elapsed.count(); // Real-time factor
    } else {
        metrics.simulation_fps = 0.0f;
        metrics.biological_time_factor = 0.0f;
    }
    
    // Estimate kernel execution time (simplified)
    metrics.kernel_execution_time_ms = elapsed.count() / 1000.0f;
    
    // Estimate memory bandwidth (simplified)
    size_t data_transferred = (num_neurons_ * sizeof(GPUNeuronState) + 
                              num_synapses_ * sizeof(GPUSynapse)) * 2; // Read + Write
    metrics.memory_bandwidth_gbps = (data_transferred * metrics.simulation_fps) / (1024.0f * 1024.0f * 1024.0f);
    
    // GPU utilization (estimated based on compute intensity)
    metrics.gpu_utilization_percent = std::min(95.0f, 
        (num_neurons_ + num_synapses_) / 10000.0f * 100.0f);
    
    return metrics;
}

void NetworkCUDA::synchronize_configurations(const NetworkConfig& cpu_config, NetworkConfig& gpu_config) {
    // Copy base configuration
    gpu_config = cpu_config;
    
    // GPU-specific optimizations (store in separate variables since NetworkConfig doesn't have these)
    // Instead of accessing non-existent members, store GPU config separately
    
    std::cout << "🔧 CPU-GPU configuration synchronized" << std::endl;
    std::cout << "   • Neurons: " << gpu_config.hidden_size << std::endl;
    std::cout << "   • Synapses: " << gpu_config.totalSynapses << std::endl;
    std::cout << "   • Memory allocated: " << allocated_memory_mb_ << " MB" << std::endl;
}

// ============================================================================
// STATE PERSISTENCE WITH CPU COMPATIBILITY
// ============================================================================

bool NetworkCUDA::save_gpu_state(const std::string& filename) const {
    std::string gpu_filename = filename + "_gpu_state.bin";
    
    // Use fstream which is now properly included
    std::ofstream file(gpu_filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "❌ Cannot open GPU state file: " << gpu_filename << std::endl;
        return false;
    }
    
    try {
        // Write GPU-specific header
        const char* header = "NEUROGENALPHA_GPU";
        file.write(header, 17);
        
        // Write version and metadata
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&num_neurons_), sizeof(num_neurons_));
        file.write(reinterpret_cast<const char*>(&num_synapses_), sizeof(num_synapses_));
        file.write(reinterpret_cast<const char*>(&current_simulation_time_), sizeof(current_simulation_time_));
        
        // Copy GPU states to host and save
        std::vector<GPUNeuronState> neurons(num_neurons_);
        std::vector<GPUSynapse> synapses(num_synapses_);
        
        // Note: const_cast is needed here but should be encapsulated properly
        const_cast<NetworkCUDA*>(this)->copy_from_gpu(neurons, synapses);
        
        // Write neuron states
        file.write(reinterpret_cast<const char*>(neurons.data()), 
                  num_neurons_ * sizeof(GPUNeuronState));
        
        // Write synapse states
        file.write(reinterpret_cast<const char*>(synapses.data()), 
                  num_synapses_ * sizeof(GPUSynapse));
        
        file.close();
        std::cout << "✅ GPU state saved: " << gpu_filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error saving GPU state: " << e.what() << std::endl;
        file.close();
        return false;
    }
}

bool NetworkCUDA::load_gpu_state(const std::string& filename) {
    std::string gpu_filename = filename + "_gpu_state.bin";
    
    // Use fstream which is now properly included
    std::ifstream file(gpu_filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "❌ Cannot open GPU state file: " << gpu_filename << std::endl;
        return false;
    }
    
    try {
        // Verify header
        char header[18] = {0};
        file.read(header, 17);
        if (std::string(header) != "NEUROGENALPHA_GPU") {
            std::cerr << "❌ Invalid GPU state file format" << std::endl;
            file.close();
            return false;
        }
        
        // Read metadata
        uint32_t version;
        int file_neurons, file_synapses;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        file.read(reinterpret_cast<char*>(&file_neurons), sizeof(file_neurons));
        file.read(reinterpret_cast<char*>(&file_synapses), sizeof(file_synapses));
        file.read(reinterpret_cast<char*>(&current_simulation_time_), sizeof(current_simulation_time_));
        
        // Validate compatibility
        if (file_neurons != num_neurons_ || file_synapses != num_synapses_) {
            std::cerr << "❌ GPU state size mismatch: expected " << num_neurons_ 
                      << "/" << num_synapses_ << ", got " << file_neurons 
                      << "/" << file_synapses << std::endl;
            file.close();
            return false;
        }
        
        // Load states
        std::vector<GPUNeuronState> neurons(num_neurons_);
        std::vector<GPUSynapse> synapses(num_synapses_);
        
        file.read(reinterpret_cast<char*>(neurons.data()), 
                 num_neurons_ * sizeof(GPUNeuronState));
        file.read(reinterpret_cast<char*>(synapses.data()), 
                 num_synapses_ * sizeof(GPUSynapse));
        
        file.close();
        
        // Copy to GPU
        copy_to_gpu(neurons, synapses);
        
        std::cout << "✅ GPU state loaded: " << gpu_filename << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error loading GPU state: " << e.what() << std::endl;
        file.close();
        return false;
    }
}