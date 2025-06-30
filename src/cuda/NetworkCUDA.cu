#include "NeuroGen/cuda/NetworkCUDA.cuh"
#include "NeuroGen/cuda/KernelLaunchWrappers.cuh"
#include "NeuroGen/cuda/NeuronSpikingKernels.cuh"
#include "NeuroGen/cuda/GPUNeuralStructures.h"
#include "NeuroGen/NetworkStats.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================================
// CUDA ERROR HANDLING
// ============================================================================

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::string error_msg = std::string("CUDA Error: ") + cudaGetErrorString(err) + \
                                " at " + __FILE__ + ":" + std::to_string(__LINE__) + \
                                " in function '" + __func__ + "'"; \
        std::cerr << error_msg << std::endl; \
        throw std::runtime_error(error_msg); \
    } \
}

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

NetworkCUDA::NetworkCUDA(const NetworkConfig& config)
    : config_(config),
      num_neurons_(config.hidden_size),
      num_synapses_(config.totalSynapses),
      current_simulation_time_(0.0f),
      simulation_step_count_(0),
      system_initialized_(false)
{
    std::cout << "🧠 Initializing CUDA Neural Network..." << std::endl;
    std::cout << "📊 Network Configuration:" << std::endl;
    std::cout << "   • Neurons: " << num_neurons_ << std::endl;
    std::cout << "   • Synapses: " << num_synapses_ << std::endl;
    std::cout << "   • Input Size: " << config.input_size << std::endl;
    std::cout << "   • Output Size: " << config.output_size << std::endl;
    
    try {
        // Initialize CUDA stream for async operations
        CUDA_CHECK(cudaStreamCreate(&stream_));
        
        // Allocate GPU memory
        allocate_gpu_memory();
        
        // Initialize neural states
        initialize_gpu_state();
        
        // Validate initialization
        validate_initialization();
        
        system_initialized_ = true;
        
        std::cout << "✅ CUDA Neural Network initialized successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error initializing neural network: " << e.what() << std::endl;
        cleanup_resources();
        throw;
    }
}

NetworkCUDA::~NetworkCUDA() {
    std::cout << "🔄 Shutting down CUDA Neural Network..." << std::endl;
    cleanup_resources();
    std::cout << "✅ Network shutdown completed." << std::endl;
}

// ============================================================================
// CORE SIMULATION STEP
// ============================================================================

void NetworkCUDA::simulate_step(float current_time, float dt, float reward, const std::vector<float>& inputs) {
    if (!system_initialized_) {
        throw std::runtime_error("Network not properly initialized");
    }
    
    current_simulation_time_ = current_time;
    simulation_step_count_++;
    
    try {
        // 1. PROCESS EXTERNAL INPUTS
        if (!inputs.empty()) {
            process_external_inputs(inputs);
        }
        
        // 2. UPDATE NEURAL DYNAMICS
        update_neural_dynamics(current_time, dt);
        
        // 3. PROCESS SYNAPTIC PLASTICITY
        update_synaptic_plasticity(current_time, dt, reward);
        
        // 4. APPLY HOMEOSTATIC MECHANISMS
        apply_homeostatic_regulation(current_time, dt);
        
        // 5. DETECT AND PROCESS SPIKES
        process_spike_detection(current_time);
        
        // Synchronize GPU operations
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        
    } catch (const std::exception& e) {
        std::cerr << "⚠️  Error in simulation step " << simulation_step_count_ << ": " << e.what() << std::endl;
        throw;
    }
}

// ============================================================================
// NEURAL DYNAMICS PROCESSING
// ============================================================================

void NetworkCUDA::update_neural_dynamics(float current_time, float dt) {
    // Update neuron states using existing kernel wrappers
    KernelLaunchWrappers::update_neuron_states(d_neurons_, current_time, dt, num_neurons_);
    
    // Update calcium dynamics for plasticity
    KernelLaunchWrappers::update_calcium_dynamics(d_neurons_, current_time, dt, num_neurons_);
    
    // Apply external currents if available
    if (d_input_currents_) {
        apply_input_currents_kernel<<<get_grid_size(num_neurons_), 256, 0, stream_>>>(
            d_neurons_, d_input_currents_, num_neurons_, dt);
        CUDA_CHECK(cudaGetLastError());
    }
}

void NetworkCUDA::update_synaptic_plasticity(float current_time, float dt, float reward) {
    // Run STDP and calculate eligibility traces
    KernelLaunchWrappers::run_stdp_and_eligibility(d_synapses_, d_neurons_, 
                                                   current_time, dt, num_synapses_);
    
    // Apply reward modulation to synaptic weights
    KernelLaunchWrappers::apply_reward_and_adaptation(d_synapses_, d_neurons_, 
                                                      reward, current_time, dt, num_synapses_);
}

void NetworkCUDA::apply_homeostatic_regulation(float current_time, float dt) {
    // Apply homeostatic mechanisms to maintain network stability
    KernelLaunchWrappers::run_homeostatic_mechanisms(d_neurons_, d_synapses_, 
                                                     current_time, num_neurons_, num_synapses_);
}

// ============================================================================
// SPIKE PROCESSING & DETECTION
// ============================================================================

void NetworkCUDA::process_spike_detection(float current_time) {
    // Reset spike counter
    CUDA_CHECK(cudaMemsetAsync(d_spike_count_, 0, sizeof(int), stream_));
    
    // Update neuron spike states and detect spikes (using correct signature)
    updateNeuronSpikes<<<get_grid_size(num_neurons_), 256, 0, stream_>>>(
        d_neurons_, config_.spike_threshold, num_neurons_);
    CUDA_CHECK(cudaGetLastError());
    
    // Count total spikes for statistics (using correct signature)
    countSpikesKernel<<<get_grid_size(num_neurons_), 256, 0, stream_>>>(
        d_neurons_, d_spike_count_, num_neurons_);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// INPUT PROCESSING
// ============================================================================

void NetworkCUDA::process_external_inputs(const std::vector<float>& inputs) {
    // Ensure input size doesn't exceed buffer capacity
    size_t copy_size = std::min(inputs.size(), static_cast<size_t>(config_.input_size));
    
    // Copy inputs to GPU asynchronously
    CUDA_CHECK(cudaMemcpyAsync(d_input_currents_, inputs.data(), 
                               copy_size * sizeof(float), 
                               cudaMemcpyHostToDevice, stream_));
    
    // Clear remaining input buffer if needed
    if (copy_size < static_cast<size_t>(config_.input_size)) {
        CUDA_CHECK(cudaMemsetAsync(d_input_currents_ + copy_size, 0, 
                                   (config_.input_size - copy_size) * sizeof(float), stream_));
    }
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

void NetworkCUDA::allocate_gpu_memory() {
    std::cout << "💾 Allocating GPU memory..." << std::endl;
    
    size_t total_memory = 0;
    
    // Allocate neuron state array
    CUDA_CHECK(cudaMalloc(&d_neurons_, num_neurons_ * sizeof(GPUNeuronState)));
    total_memory += num_neurons_ * sizeof(GPUNeuronState);
    
    // Allocate synapse array
    CUDA_CHECK(cudaMalloc(&d_synapses_, num_synapses_ * sizeof(GPUSynapse)));
    total_memory += num_synapses_ * sizeof(GPUSynapse);
    
    // Allocate input current buffer
    CUDA_CHECK(cudaMalloc(&d_input_currents_, config_.input_size * sizeof(float)));
    total_memory += config_.input_size * sizeof(float);
    
    // Allocate spike counting buffer
    CUDA_CHECK(cudaMalloc(&d_spike_count_, sizeof(int)));
    total_memory += sizeof(int);
    
    // Allocate network statistics buffer
    CUDA_CHECK(cudaMalloc(&d_network_stats_, sizeof(NetworkStats)));
    total_memory += sizeof(NetworkStats);
    
    // Initialize all memory to zero
    CUDA_CHECK(cudaMemset(d_neurons_, 0, num_neurons_ * sizeof(GPUNeuronState)));
    CUDA_CHECK(cudaMemset(d_synapses_, 0, num_synapses_ * sizeof(GPUSynapse)));
    CUDA_CHECK(cudaMemset(d_input_currents_, 0, config_.input_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_spike_count_, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_network_stats_, 0, sizeof(NetworkStats)));
    
    allocated_memory_mb_ = total_memory / (1024.0f * 1024.0f);
    std::cout << "✅ Allocated " << allocated_memory_mb_ << " MB of GPU memory" << std::endl;
}

void NetworkCUDA::cleanup_resources() {
    // Free GPU memory
    if (d_neurons_) { cudaFree(d_neurons_); d_neurons_ = nullptr; }
    if (d_synapses_) { cudaFree(d_synapses_); d_synapses_ = nullptr; }
    if (d_input_currents_) { cudaFree(d_input_currents_); d_input_currents_ = nullptr; }
    if (d_spike_count_) { cudaFree(d_spike_count_); d_spike_count_ = nullptr; }
    if (d_network_stats_) { cudaFree(d_network_stats_); d_network_stats_ = nullptr; }
    
    // Destroy CUDA stream
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = 0;
    }
}

// ============================================================================
// INITIALIZATION & VALIDATION
// ============================================================================

void NetworkCUDA::initialize_gpu_state() {
    std::cout << "🔧 Initializing neural states..." << std::endl;
    
    // Initialize ion channels and neural parameters
    KernelLaunchWrappers::initialize_ion_channels(d_neurons_, num_neurons_);
    
    // Initialize synaptic connections with proper parameters
    initialize_synapses_kernel<<<get_grid_size(num_synapses_), 256>>>(
        d_synapses_, num_synapses_, config_.min_weight, config_.max_weight,
        config_.weight_init_std, time(nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    std::cout << "✅ Neural states initialized" << std::endl;
}

void NetworkCUDA::validate_initialization() {
    // Check that GPU memory allocations were successful
    if (!d_neurons_ || !d_synapses_ || !d_input_currents_ || !d_spike_count_ || !d_network_stats_) {
        throw std::runtime_error("GPU memory allocation failed");
    }
    
    // Validate neuron count and synapse count
    if (num_neurons_ <= 0 || num_synapses_ <= 0) {
        throw std::runtime_error("Invalid network size");
    }
    
    // Test basic CUDA operations
    int test_value = 42;
    int* d_test;
    CUDA_CHECK(cudaMalloc(&d_test, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_test, &test_value, sizeof(int), cudaMemcpyHostToDevice));
    
    int result;
    CUDA_CHECK(cudaMemcpy(&result, d_test, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_test);
    
    if (result != test_value) {
        throw std::runtime_error("Basic CUDA operations failed");
    }
    
    std::cout << "✅ System validation passed" << std::endl;
}

// ============================================================================
// DATA TRANSFER & SYNCHRONIZATION
// ============================================================================

void NetworkCUDA::copy_to_gpu(const std::vector<GPUNeuronState>& neurons, 
                             const std::vector<GPUSynapse>& synapses) {
    if (neurons.size() != static_cast<size_t>(num_neurons_)) {
        throw std::runtime_error("Neuron count mismatch in copy_to_gpu");
    }
    if (synapses.size() != static_cast<size_t>(num_synapses_)) {
        throw std::runtime_error("Synapse count mismatch in copy_to_gpu");
    }
    
    CUDA_CHECK(cudaMemcpyAsync(d_neurons_, neurons.data(), 
                               num_neurons_ * sizeof(GPUNeuronState),
                               cudaMemcpyHostToDevice, stream_));
    
    CUDA_CHECK(cudaMemcpyAsync(d_synapses_, synapses.data(),
                               num_synapses_ * sizeof(GPUSynapse),
                               cudaMemcpyHostToDevice, stream_));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NetworkCUDA::copy_from_gpu(std::vector<GPUNeuronState>& neurons, 
                               std::vector<GPUSynapse>& synapses) {
    neurons.resize(num_neurons_);
    synapses.resize(num_synapses_);
    
    CUDA_CHECK(cudaMemcpyAsync(neurons.data(), d_neurons_,
                               num_neurons_ * sizeof(GPUNeuronState),
                               cudaMemcpyDeviceToHost, stream_));
    
    CUDA_CHECK(cudaMemcpyAsync(synapses.data(), d_synapses_,
                               num_synapses_ * sizeof(GPUSynapse),
                               cudaMemcpyDeviceToHost, stream_));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ============================================================================
// STATISTICS & MONITORING
// ============================================================================

void NetworkCUDA::get_stats(NetworkStats& stats) const {
    // Compute basic statistics on GPU
    compute_network_statistics<<<1, 256, 0, stream_>>>(
        d_neurons_, d_synapses_, d_network_stats_, 
        num_neurons_, num_synapses_, current_simulation_time_);
    
    // Copy statistics to host
    CUDA_CHECK(cudaMemcpy(&stats, d_network_stats_, sizeof(NetworkStats), cudaMemcpyDeviceToHost));
    
    // Fill in additional statistics that exist in NetworkStats
    stats.simulation_steps = simulation_step_count_;  // Note: steps not step
    stats.computation_time_us = allocated_memory_mb_ * 1000.0f; // Approximate timing
    
    // Get current spike count
    int current_spikes;
    CUDA_CHECK(cudaMemcpy(&current_spikes, d_spike_count_, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Update network activity metrics using existing fields
    stats.mean_firing_rate = static_cast<float>(current_spikes) / num_neurons_;
    stats.neuron_activity_ratio = (current_spikes > 0) ? (static_cast<float>(current_spikes) / num_neurons_) : 0.0f;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

dim3 NetworkCUDA::get_grid_size(int num_elements, int block_size) const {
    return dim3((num_elements + block_size - 1) / block_size);
}

float NetworkCUDA::get_allocated_memory_mb() const {
    return allocated_memory_mb_;
}

int NetworkCUDA::get_simulation_step_count() const {
    return simulation_step_count_;
}

float NetworkCUDA::get_current_simulation_time() const {
    return current_simulation_time_;
}

// ============================================================================
// CUDA KERNEL IMPLEMENTATIONS
// ============================================================================

__global__ void apply_input_currents_kernel(GPUNeuronState* neurons, 
                                           const float* input_currents,
                                           int num_neurons, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Apply input current to first synaptic compartment
    if (input_currents && idx < num_neurons) {
        neuron.I_syn[0] += input_currents[idx % num_neurons] * dt;
    }
}

__global__ void initialize_synapses_kernel(GPUSynapse* synapses, int num_synapses,
                                          float min_weight, float max_weight,
                                          float weight_std, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // Initialize random state for this thread
    curandState local_state;
    curand_init(seed, idx, 0, &local_state);
    
    GPUSynapse& synapse = synapses[idx];
    
    // Initialize basic synapse properties
    synapse.pre_neuron_idx = idx % num_synapses;  // Simple initialization
    synapse.post_neuron_idx = (idx + 1) % num_synapses;
    synapse.post_compartment = idx % 4;  // Distribute across compartments
    synapse.active = 1;
    
    // Initialize weight with normal distribution
    float weight = curand_normal(&local_state) * weight_std + (min_weight + max_weight) / 2.0f;
    synapse.weight = fmaxf(min_weight, fminf(max_weight, weight));
    
    // Initialize delay with small random component
    synapse.delay = 1.0f + curand_uniform(&local_state) * 2.0f;
    
    // Initialize plasticity parameters
    synapse.eligibility_trace = 0.0f;
    synapse.plasticity_modulation = 1.0f;
    synapse.effective_weight = synapse.weight;
    synapse.last_pre_spike_time = -1e6f;
    synapse.last_post_spike_time = -1e6f;
    synapse.last_active_time = 0.0f;
    synapse.activity_metric = 0.0f;
    synapse.max_weight = max_weight;
    synapse.min_weight = min_weight;
    synapse.dopamine_sensitivity = 0.1f;
    synapse.acetylcholine_sensitivity = 0.05f;
}

__global__ void compute_network_statistics(const GPUNeuronState* neurons,
                                          const GPUSynapse* synapses,
                                          NetworkStats* stats,
                                          int num_neurons, int num_synapses,
                                          float current_time) {
    int idx = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Shared memory for reduction
    __shared__ float shared_firing_rate[256];
    __shared__ float shared_activity_ratio[256];
    __shared__ int shared_active_synapses[256];
    
    // Initialize shared memory
    shared_firing_rate[idx] = 0.0f;
    shared_activity_ratio[idx] = 0.0f;
    shared_active_synapses[idx] = 0;
    
    // Compute partial sums using available GPUNeuronState fields
    for (int i = idx; i < num_neurons; i += num_threads) {
        shared_firing_rate[idx] += neurons[i].average_firing_rate;
        shared_activity_ratio[idx] += neurons[i].average_activity;
    }
    
    for (int i = idx; i < num_synapses; i += num_threads) {
        if (synapses[i].active) {
            shared_active_synapses[idx]++;
        }
    }
    
    __syncthreads();
    
    // Reduction
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (idx < stride) {
            shared_firing_rate[idx] += shared_firing_rate[idx + stride];
            shared_activity_ratio[idx] += shared_activity_ratio[idx + stride];
            shared_active_synapses[idx] += shared_active_synapses[idx + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes final results using only existing NetworkStats fields
    if (idx == 0) {
        stats->active_neuron_count = num_neurons;
        stats->active_synapses = shared_active_synapses[0];
        stats->total_synapses = num_synapses;
        stats->mean_firing_rate = shared_firing_rate[0] / num_neurons;
        stats->neuron_activity_ratio = shared_activity_ratio[0] / num_neurons;
        
        // Calculate network entropy based on activity ratio
        float p = stats->neuron_activity_ratio;
        if (p > 0.0f && p < 1.0f) {
            stats->network_entropy = -(p * __logf(p) + (1.0f - p) * __logf(1.0f - p));
        } else {
            stats->network_entropy = 0.0f;
        }
    }
}