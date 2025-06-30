extern "C" {

// Enhanced STDP kernel wrapper
void launch_enhanced_stdp_wrapper(void* d_synapses, const void* d_neurons,
                                 void* d_plasticity_states, void* d_stdp_config,
                                 void* d_global_neuromodulators, float current_time,
                                 float dt, int num_synapses) {
    
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    enhancedSTDPKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        static_cast<GPUPlasticityState*>(d_plasticity_states),
        static_cast<GPUNetworkConfig*>(d_stdp_config),
        static_cast<GPUNeuromodulatorState*>(d_global_neuromodulators),
        current_time, dt, num_synapses
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in enhanced STDP kernel: %s\n", cudaGetErrorString(error));
    }
}

// BCM learning kernel wrapper
void launch_bcm_learning_wrapper(void* d_synapses, const void* d_neurons,
                                void* d_plasticity_states, float current_time,
                                float dt, int num_synapses) {
    
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    bcmLearningKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        static_cast<GPUPlasticityState*>(d_plasticity_states),
        current_time, dt, num_synapses
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in BCM learning kernel: %s\n", cudaGetErrorString(error));
    }
}

// Homeostatic regulation kernel wrapper
void launch_homeostatic_regulation_wrapper(void* d_synapses, void* d_neurons,
                                          float target_activity, float regulation_strength,
                                          float dt, int num_neurons, int num_synapses) {
    
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    homeostaticRegulationKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<GPUNeuronState*>(d_neurons),
        target_activity, regulation_strength, dt, num_neurons, num_synapses
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in homeostatic regulation kernel: %s\n", cudaGetErrorString(error));
    }
}

// Dopamine update kernel wrapper
void launch_dopamine_update_wrapper(void* d_da_neurons, void* d_network_neurons,
                                   float reward_signal, float predicted_reward,
                                   float current_time, float dt, int num_da_neurons,
                                   int num_network_neurons) {
    
    dim3 block(256);
    dim3 grid((num_da_neurons + block.x - 1) / block.x);
    
    dopamineUpdateKernel<<<grid, block>>>(
        static_cast<GPUNeuronState*>(d_da_neurons),
        static_cast<GPUNeuronState*>(d_network_neurons),
        reward_signal, predicted_reward, current_time, dt,
        num_da_neurons, num_network_neurons
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in dopamine update kernel: %s\n", cudaGetErrorString(error));
    }
}

// Utility functions
void cuda_memory_copy_wrapper(void* dst, const void* src, size_t size, int direction) {
    cudaMemcpyKind kind = (direction == 0) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    cudaMemcpy(dst, src, size, kind);
}

int cuda_check_last_error_wrapper(void) {
    cudaError_t error = cudaGetLastError();
    return (error == cudaSuccess) ? 0 : 1;
}

void cuda_device_synchronize_wrapper(void) {
    cudaDeviceSynchronize();
}

} // extern "C"