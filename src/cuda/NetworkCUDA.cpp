#include <NeuroGen/cuda/NetworkCUDA.cuh>
#include <NeuroGen/cuda/KernelLaunchWrappers.cuh>
#include <NeuroGen/cuda/CudaUtils.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/NetworkStats.h>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <memory>
#include <random>
#include <cmath>
#include <chrono>

// Global network statistics definition (CUDA managed memory for seamless GPU/CPU access)
__managed__ NetworkStats g_stats;

// ============================================================================
// CONSTRUCTOR AND DESTRUCTOR
// ============================================================================

NetworkCUDA::NetworkCUDA(const NetworkConfig& config)
    : config(config),
      d_neurons(nullptr),
      d_synapses(nullptr),
      d_calcium_levels(nullptr),
      d_neuron_spike_counts(nullptr),
      d_random_states(nullptr),
      d_cortical_columns(nullptr),
      d_input_currents(nullptr),
      current_time_ms(0.0f),
      network_initialized(false),
      plasticity_enabled(true),
      current_learning_rate(0.01f),
      last_kernel_time(0.0f),
      memory_usage(0)
{
    std::cout << "[NetworkCUDA] Initializing breakthrough modular neural network..." << std::endl;
    
    try {
        // Validate and finalize configuration for biological realism
        validateConfig();
        
        // Initialize CUDA device and select optimal GPU
        int device_id = CudaUtils::selectOptimalDevice();
        CudaUtils::printDeviceInfo(device_id);
        
        std::cout << "[NetworkCUDA] Network architecture:" << std::endl;
        std::cout << "  Cortical Columns: " << getNumColumns() << std::endl;
        std::cout << "  Neurons per Column: " << getNeuronsPerColumn() << std::endl;
        std::cout << "  Total Neurons: " << getNumNeurons() << std::endl;
        std::cout << "  Total Synapses: " << getNumSynapses() << std::endl;
        
        // Initialize GPU-accelerated modular architecture
        allocateDeviceMemory();
        initializeDeviceArrays();
        initializeNetwork();
        initializeColumns();
        
        // Reset global statistics for this simulation
        g_stats.reset();
        g_stats.num_columns = static_cast<uint32_t>(getNumColumns());
        g_stats.column_activity.resize(getNumColumns(), 0.0f);
        g_stats.column_frequencies.resize(getNumColumns(), 0.0f);
        g_stats.column_specialization.resize(getNumColumns(), 0.5f);
        
        network_initialized = true;
        
        std::cout << "[NetworkCUDA] Breakthrough neural architecture initialized successfully!" << std::endl;
        std::cout << "[NetworkCUDA] Ready for biological-speed brain simulation." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to initialize breakthrough neural network: " << e.what() << std::endl;
        cleanup();
        throw;
    }
}

NetworkCUDA::~NetworkCUDA() {
    std::cout << "[NetworkCUDA] Shutting down modular neural network..." << std::endl;
    cleanup();
    std::cout << "[NetworkCUDA] Network cleanup completed." << std::endl;
}

// ============================================================================
// CORE SIMULATION METHODS
// ============================================================================

void NetworkCUDA::update(float dt_ms, const std::vector<float>& input_currents, float reward) {
    if (!network_initialized) {
        throw std::runtime_error("[NetworkCUDA] Cannot update: Network not initialized");
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Update simulation time and reward signals
        current_time_ms += dt_ms;
        g_stats.updateSimulationMetrics(dt_ms, static_cast<uint64_t>(current_time_ms / dt_ms));
        g_stats.current_reward = reward;
        g_stats.cumulative_reward += reward;

        // 1. Apply external input currents to network
        if (!input_currents.empty()) {
            size_t copy_size = std::min(input_currents.size(), static_cast<size_t>(getNumNeurons()));
            CUDA_CHECK_ERROR(cudaMemcpy(d_input_currents, input_currents.data(), 
                                      copy_size * sizeof(float), cudaMemcpyHostToDevice));
            
            // Apply input currents using optimized CUDA kernel
            dim3 input_blocks, input_threads;
            calculateGridBlockSize(static_cast<int>(copy_size), input_blocks, input_threads);
            launchApplyInputCurrentsKernel(input_blocks, input_threads, 
                                         d_neurons, d_input_currents, 
                                         static_cast<int>(copy_size), getNumNeurons());
        }

        // 2. Process synaptic inputs across cortical columns
        updateSynapsesWrapper(dt_ms);

        // 3. Update neuron states using biologically-accurate integration
        updateNeuronsWrapper(dt_ms);

        // 4. Process spike generation and propagation
        processSpikingWrapper();

        // 5. Apply reward-modulated plasticity (breakthrough learning mechanism)
        if (plasticity_enabled && std::abs(reward) > 1e-6f) {
            applyPlasticityWrapper(reward);
        }

        // 6. Update neuromodulator concentrations based on network activity
        updateNeuromodulation(reward);

        // 7. Apply homeostatic mechanisms for stability
        if (static_cast<int>(current_time_ms / dt_ms) % 1000 == 0) {
            applyHomeostaticScaling();
        }

        // 8. Update comprehensive network statistics
        updateNetworkStatistics();

        // 9. Monitor for criticality and biological realism
        monitorNetworkHealth();

        // Record performance metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        g_stats.updatePerformanceMetrics(static_cast<float>(duration.count()), 0.0f, memory_usage);

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Network update failed at time " << current_time_ms << "ms: " << e.what() << std::endl;
        g_stats.simulation_stable = false;
        g_stats.cuda_error_count++;
        throw;
    }
}

std::vector<float> NetworkCUDA::getOutput() const {
    if (!network_initialized) {
        return std::vector<float>();
    }
    
    int num_neurons = getNumNeurons();
    std::vector<float> output(num_neurons);
    
    // Extract network output using high-performance memory transfer
    std::vector<GPUNeuronState> host_neurons(num_neurons);
    CUDA_CHECK_ERROR(cudaMemcpy(host_neurons.data(), d_neurons, 
                               num_neurons * sizeof(GPUNeuronState), 
                               cudaMemcpyDeviceToHost));
    
    // Extract meaningful output signals (not just raw voltages)
    for (int i = 0; i < num_neurons; ++i) {
        // Use spike-rate encoding for more biological output
        float spike_rate = host_neurons[i].spike_count * 1000.0f / (current_time_ms + 1e-6f);
        output[i] = std::tanh(spike_rate / 50.0f); // Normalize to [-1, 1] range
    }
    
    return output;
}

void NetworkCUDA::reset() {
    if (!network_initialized) {
        return;
    }
    
    std::cout << "[NetworkCUDA] Resetting breakthrough neural network to initial state..." << std::endl;
    
    current_time_ms = 0.0f;
    g_stats.reset();
    
    // Reset all neuron states to resting conditions
    int num_neurons = getNumNeurons();
    dim3 blocks, threads;
    calculateGridBlockSize(num_neurons, blocks, threads);
    
    launchResetNeuronStatesKernel(blocks, threads, d_neurons, num_neurons);
    CUDA_KERNEL_CHECK();
    
    // Reset synaptic states and eligibility traces
    int num_synapses = getNumSynapses();
    calculateGridBlockSize(num_synapses, blocks, threads);
    launchResetSynapseStatesKernel(blocks, threads, d_synapses, num_synapses);
    CUDA_KERNEL_CHECK();
    
    std::cout << "[NetworkCUDA] Network reset completed." << std::endl;
}

NetworkStats NetworkCUDA::getStats() const {
    return g_stats;
}

// ============================================================================
// LEARNING AND MODULATION METHODS  
// ============================================================================

void NetworkCUDA::setRewardSignal(float reward) {
    g_stats.current_reward = reward;
    
    // Update neuromodulator concentrations based on reward
    if (reward > 0.0f) {
        g_stats.dopamine_level = std::min(1.0f, g_stats.dopamine_level + reward * 0.1f);
    } else {
        g_stats.dopamine_level = std::max(0.0f, g_stats.dopamine_level + reward * 0.05f);
    }
}

void NetworkCUDA::printNetworkState() const {
    std::cout << "=== NeuroGen Breakthrough Neural Network State ===" << std::endl;
    std::cout << "Cortical Columns: " << getNumColumns() << std::endl;
    std::cout << "Neurons per Column: " << getNeuronsPerColumn() << std::endl;
    std::cout << "Total Neurons: " << getNumNeurons() << std::endl;
    std::cout << "Total Synapses: " << getNumSynapses() << std::endl;
    std::cout << "Simulation Time: " << current_time_ms << " ms" << std::endl;
    std::cout << "Plasticity: " << (plasticity_enabled ? "ENABLED" : "disabled") << std::endl;
    std::cout << "Learning Rate: " << current_learning_rate << std::endl;
    std::cout << "Network Health: " << (g_stats.isNetworkHealthy() ? "HEALTHY" : "WARNING") << std::endl;
    std::cout << "Current Spikes: " << g_stats.current_spike_count << " (Total: " << g_stats.total_spike_count << ")" << std::endl;
    std::cout << "Mean Firing Rate: " << g_stats.mean_firing_rate << " Hz" << std::endl;
    std::cout << "Network Efficiency: " << (g_stats.getNetworkEfficiency() * 100.0f) << "%" << std::endl;
    std::cout << "=============================================" << std::endl;
}

// ============================================================================
// PRIVATE INITIALIZATION METHODS
// ============================================================================

void NetworkCUDA::validateConfig() const {
    if (!config.validate()) {
        throw std::invalid_argument("[NetworkCUDA] Invalid network configuration");
    }
    
    if (config.numColumns <= 0 || config.neuronsPerColumn <= 0) {
        throw std::invalid_argument("[NetworkCUDA] Invalid cortical column configuration");
    }
    
    if (config.totalSynapses <= 0) {
        throw std::invalid_argument("[NetworkCUDA] Invalid synapse count");
    }
    
    // Check GPU memory requirements
    auto memory_info = CudaUtils::getMemoryInfo();
    size_t required_memory = estimateMemoryRequirements();
    
    if (required_memory > memory_info.first) {
        std::cerr << "[WARNING] Required GPU memory (" << (required_memory / 1024 / 1024) 
                  << " MB) may exceed available memory (" << (memory_info.first / 1024 / 1024) << " MB)" << std::endl;
    }
}

void NetworkCUDA::allocateDeviceMemory() {
    std::cout << "[NetworkCUDA] Allocating GPU memory for modular architecture..." << std::endl;
    
    int num_neurons = getNumNeurons();
    int num_synapses = getNumSynapses();
    int num_columns = getNumColumns();
    
    // Allocate neuron state arrays
    CUDA_CHECK_ERROR(cudaMalloc(&d_neurons, num_neurons * sizeof(GPUNeuronState)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_calcium_levels, num_neurons * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_neuron_spike_counts, num_neurons * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_input_currents, num_neurons * sizeof(float)));
    
    // Allocate synapse arrays  
    CUDA_CHECK_ERROR(cudaMalloc(&d_synapses, num_synapses * sizeof(GPUSynapse)));
    
    // Allocate cortical column structures
    CUDA_CHECK_ERROR(cudaMalloc(&d_cortical_columns, num_columns * sizeof(CorticalColumn)));
    
    // Allocate random number generator states
    CUDA_CHECK_ERROR(cudaMalloc(&d_random_states, num_neurons * sizeof(curandState)));
    
    // Calculate total memory usage
    memory_usage = (num_neurons * (sizeof(GPUNeuronState) + sizeof(float) + sizeof(int) + sizeof(float) + sizeof(curandState))) +
                   (num_synapses * sizeof(GPUSynapse)) +
                   (num_columns * sizeof(CorticalColumn));
    
    std::cout << "[NetworkCUDA] Allocated " << (memory_usage / 1024 / 1024) << " MB of GPU memory" << std::endl;
}

void NetworkCUDA::initializeDeviceArrays() {
    std::cout << "[NetworkCUDA] Initializing GPU arrays with biological parameters..." << std::endl;
    
    int num_neurons = getNumNeurons();
    int num_synapses = getNumSynapses();
    
    // Zero-initialize arrays
    CUDA_CHECK_ERROR(cudaMemset(d_calcium_levels, 0, num_neurons * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(d_neuron_spike_counts, 0, num_neurons * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemset(d_input_currents, 0, num_neurons * sizeof(float)));
    
    // Initialize neurons with biologically realistic parameters
    dim3 blocks, threads;
    calculateGridBlockSize(num_neurons, blocks, threads);
    launchInitializeNeuronStatesKernel(blocks, threads, d_neurons, num_neurons);
    CUDA_KERNEL_CHECK();
    
    // Initialize synapses with distance-based connectivity
    calculateGridBlockSize(num_synapses, blocks, threads);
    launchInitializeSynapseStatesKernel(blocks, threads, d_synapses, num_synapses);
    CUDA_KERNEL_CHECK();
    
    // Initialize random number generators
    launchInitializeRandomStatesKernel(blocks, threads, d_random_states, num_neurons, 
                                     static_cast<unsigned long>(std::time(nullptr)));
    CUDA_KERNEL_CHECK();
}

void NetworkCUDA::initializeNetwork() {
    std::cout << "[NetworkCUDA] Initializing breakthrough modular network topology..." << std::endl;
    
    // Generate biologically-inspired cortical column connectivity
    generateDistanceBasedSynapses();
    
    // Initialize cortical column specializations
    initializeColumnSpecializations();
    
    std::cout << "[NetworkCUDA] Network topology initialization completed." << std::endl;
}

void NetworkCUDA::initializeColumns() {
    if (d_cortical_columns && getNumColumns() > 0) {
        std::cout << "[NetworkCUDA] Initializing " << getNumColumns() << " cortical columns..." << std::endl;
        
        dim3 blocks, threads;
        calculateGridBlockSize(getNumColumns(), blocks, threads);
        launchInitializeCorticalColumnsKernel(blocks, threads, d_cortical_columns, getNumColumns());
        CUDA_KERNEL_CHECK();
        
        std::cout << "[NetworkCUDA] Cortical column initialization completed." << std::endl;
    }
}

void NetworkCUDA::cleanup() {
    if (d_neurons) { cudaFree(d_neurons); d_neurons = nullptr; }
    if (d_synapses) { cudaFree(d_synapses); d_synapses = nullptr; }
    if (d_calcium_levels) { cudaFree(d_calcium_levels); d_calcium_levels = nullptr; }
    if (d_neuron_spike_counts) { cudaFree(d_neuron_spike_counts); d_neuron_spike_counts = nullptr; }
    if (d_random_states) { cudaFree(d_random_states); d_random_states = nullptr; }
    if (d_cortical_columns) { cudaFree(d_cortical_columns); d_cortical_columns = nullptr; }
    if (d_input_currents) { cudaFree(d_input_currents); d_input_currents = nullptr; }
    
    network_initialized = false;
}

// ============================================================================
// COMPUTATIONAL UTILITIES
// ============================================================================

void NetworkCUDA::calculateGridBlockSize(int n_elements, dim3& grid, dim3& block) const {
    block = CudaUtils::makeSafeBlock(CudaUtils::getOptimalBlockSize(n_elements));
    grid = CudaUtils::makeSafeGrid(n_elements, block.x);
}

void NetworkCUDA::updateNetworkStatistics() {
    // Update activity metrics from GPU state
    uint32_t total_spikes = 0;
    uint32_t active_neurons = 0;
    
    // Copy spike counts from GPU
    std::vector<int> host_spike_counts(getNumNeurons());
    CUDA_CHECK_ERROR(cudaMemcpy(host_spike_counts.data(), d_neuron_spike_counts,
                               getNumNeurons() * sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < getNumNeurons(); ++i) {
        if (host_spike_counts[i] > 0) {
            active_neurons++;
            total_spikes += host_spike_counts[i];
        }
    }
    
    g_stats.updateActivityMetrics(total_spikes, active_neurons, static_cast<uint32_t>(getNumNeurons()));
    g_stats.updateSynapticMetrics(0.1f, 0.05f, static_cast<uint32_t>(getNumSynapses() * 0.8f), 
                                 static_cast<uint32_t>(getNumSynapses()));
}

// ============================================================================
// ADVANCED BIOLOGICAL MECHANISMS
// ============================================================================

// ============================================================================
// ADVANCED BIOLOGICAL MECHANISMS
// ============================================================================

void NetworkCUDA::updateNeuromodulation(float reward) {
    // Update dopamine based on reward prediction error - critical for breakthrough learning
    float dopamine_change = reward * 0.1f;
    g_stats.dopamine_level = std::clamp(g_stats.dopamine_level + dopamine_change, 0.0f, 1.0f);
    
    // Update other neuromodulators based on network state for biological realism
    g_stats.acetylcholine_level = 0.5f + 0.3f * g_stats.neuron_activity_ratio;
    g_stats.serotonin_level = 0.5f + 0.2f * (g_stats.network_synchrony - 0.5f);
    g_stats.norepinephrine_level = 0.5f + 0.2f * std::abs(reward);
    
    // Apply neuromodulation to cortical columns for breakthrough modular processing
    if (d_cortical_columns && getNumColumns() > 0) {
        dim3 blocks, threads;
        calculateGridBlockSize(getNumColumns(), blocks, threads);
        launchUpdateNeuromodulationKernel(blocks, threads, d_neurons, d_cortical_columns,
                                         g_stats.dopamine_level, g_stats.acetylcholine_level,
                                         g_stats.serotonin_level, g_stats.norepinephrine_level,
                                         getNumNeurons());
        NEURAL_KERNEL_CHECK();
    }
    
    g_stats.updateNeuromodulation(g_stats.dopamine_level, g_stats.acetylcholine_level,
                                 g_stats.serotonin_level, g_stats.norepinephrine_level);
}

void NetworkCUDA::applyHomeostaticScaling() {
    // Apply homeostatic scaling to maintain network stability - crucial for brain-like operation
    dim3 blocks, threads;
    calculateGridBlockSize(getNumSynapses(), blocks, threads);
    launchApplyHomeostaticScalingKernel(blocks, threads, d_synapses, getNumSynapses());
    NEURAL_KERNEL_CHECK();
    
    // Update homeostatic metrics
    g_stats.homeostatic_scaling = 1.0f - 0.1f * std::abs(g_stats.mean_firing_rate / 50.0f - 1.0f);
}

void NetworkCUDA::monitorNetworkHealth() {
    // Monitor for pathological activity patterns that would compromise brain-like behavior
    if (g_stats.mean_firing_rate > 200.0f) {
        std::cerr << "[WARNING] Excessive firing rate detected: " << g_stats.mean_firing_rate << " Hz" << std::endl;
        g_stats.simulation_stable = false;
    }
    
    if (g_stats.network_synchrony > 0.95f) {
        std::cerr << "[WARNING] Pathological synchronization detected: " << g_stats.network_synchrony << std::endl;
        g_stats.simulation_stable = false;
    }
    
    // Update criticality index for optimal brain-like operation at edge of chaos
    g_stats.criticality_index = 1.0f - std::abs(g_stats.mean_firing_rate / 50.0f - 1.0f);
    
    // Monitor cortical column health for modular architecture integrity
    if (d_cortical_columns && getNumColumns() > 0) {
        dim3 blocks, threads;
        calculateGridBlockSize(getNumColumns(), blocks, threads);
        
        // Launch criticality monitoring kernel for breakthrough brain-like dynamics
        float criticality_buffer[64]; // Assuming max 64 columns for now
        float* d_criticality_buffer;
        NEURAL_CUDA_CHECK(cudaMalloc(&d_criticality_buffer, getNumColumns() * sizeof(float)));
        
        launchCriticalityMonitoringKernel(blocks, threads, d_neurons, d_cortical_columns, 
                                        d_criticality_buffer, getNumNeurons());
        NEURAL_KERNEL_CHECK();
        
        // Copy results back and analyze
        NEURAL_CUDA_CHECK(cudaMemcpy(criticality_buffer, d_criticality_buffer, 
                                   getNumColumns() * sizeof(float), cudaMemcpyDeviceToHost));
        
        float mean_criticality = 0.0f;
        for (int i = 0; i < getNumColumns(); ++i) {
            mean_criticality += criticality_buffer[i];
        }
        mean_criticality /= getNumColumns();
        g_stats.criticality_index = mean_criticality;
        
        NEURAL_CUDA_CHECK(cudaFree(d_criticality_buffer));
    }
}

size_t NetworkCUDA::estimateMemoryRequirements() const {
    // Calculate comprehensive memory requirements for breakthrough neural architecture
    size_t neuron_memory = static_cast<size_t>(getNumNeurons()) * 
                          (sizeof(GPUNeuronState) + sizeof(float) * 2 + sizeof(int) + sizeof(curandState));
    
    size_t synapse_memory = static_cast<size_t>(getNumSynapses()) * sizeof(GPUSynapse);
    
    size_t column_memory = static_cast<size_t>(getNumColumns()) * sizeof(CorticalColumn);
    
    // Add overhead for temporary buffers and kernel workspace
    size_t overhead = (neuron_memory + synapse_memory) * 0.2f; // 20% overhead
    
    return neuron_memory + synapse_memory + column_memory + overhead;
}

void NetworkCUDA::initializeColumnSpecializations() {
    // Initialize different cortical columns with specialized functions for breakthrough modularity
    if (getNumColumns() <= 0) return;
    
    for (int col = 0; col < getNumColumns(); ++col) {
        // Create functional specialization patterns that enhance brain-like processing
        float specialization_value = 0.5f + 0.3f * std::sin(col * 0.1f);
        g_stats.column_specialization[col] = specialization_value;
        
        // Set up different column types for cognitive functions
        int specialization_type = col % 4; // 4 types: sensory, motor, association, executive
        g_stats.column_frequencies[col] = 8.0f + 4.0f * specialization_type; // Different frequency bands
    }
    
    std::cout << "[NetworkCUDA] Initialized " << getNumColumns() 
              << " cortical columns with breakthrough functional specializations." << std::endl;
}

// ============================================================================
// KERNEL WRAPPER IMPLEMENTATIONS
// ============================================================================

void NetworkCUDA::updateNeuronsWrapper(float dt_ms) {
    dim3 blocks, threads;
    calculateGridBlockSize(getNumNeurons(), blocks, threads);
    
    auto start = std::chrono::high_resolution_clock::now();
    launchRK4NeuronUpdateKernel(d_neurons, getNumNeurons(), dt_ms, current_time_ms);
    NEURAL_KERNEL_CHECK();
    auto end = std::chrono::high_resolution_clock::now();
    
    last_kernel_time = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    g_stats.timing_profile.neuron_update_time = last_kernel_time;
}

void NetworkCUDA::updateSynapsesWrapper(float dt_ms) {
    dim3 blocks, threads;
    calculateGridBlockSize(getNumSynapses(), blocks, threads);
    
    auto start = std::chrono::high_resolution_clock::now();
    launchUpdateSynapseStatesKernel(blocks, threads, d_synapses, getNumSynapses(), dt_ms);
    NEURAL_KERNEL_CHECK();
    auto end = std::chrono::high_resolution_clock::now();
    
    g_stats.timing_profile.synapse_update_time = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

void NetworkCUDA::applyPlasticityWrapper(float reward) {
    dim3 blocks, threads;
    calculateGridBlockSize(getNumSynapses(), blocks, threads);
    
    auto start = std::chrono::high_resolution_clock::now();
    launchApplyRewardModulationKernel(blocks, threads, d_synapses, reward, getNumSynapses());
    NEURAL_KERNEL_CHECK();
    auto end = std::chrono::high_resolution_clock::now();
    
    g_stats.timing_profile.plasticity_update_time = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

void NetworkCUDA::processSpikingWrapper() {
    dim3 blocks, threads;
    calculateGridBlockSize(getNumNeurons(), blocks, threads);
    
    auto start = std::chrono::high_resolution_clock::now();
    launchProcessSpikesKernel(blocks, threads, d_neurons, d_neuron_spike_counts, current_time_ms, getNumNeurons());
    NEURAL_KERNEL_CHECK();
    auto end = std::chrono::high_resolution_clock::now();
    
    g_stats.timing_profile.spike_processing_time = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}