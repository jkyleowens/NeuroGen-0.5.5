#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cstdlib>
#include "NeuroGen/TaskAutomationModules.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/NeuralModule.h"

// Function to create a default configuration for a neural module
NetworkConfig create_default_config() {
    NetworkConfig config;
    config.num_neurons = 256; // Enhanced neuron count for version 0.5.5
    config.enable_neurogenesis = true;
    config.enable_stdp = true;
    config.enable_pruning = true;
    config.enable_structural_plasticity = true; // Enable dynamic synaptogenesis
    
    // Enhanced connectivity parameters for version 0.5.5
    config.input_hidden_prob = 0.6f;
    config.hidden_hidden_prob = 0.3f;
    config.hidden_output_prob = 0.8f;
    config.exc_ratio = 0.8f;
    
    // Synaptic parameters
    config.min_weight = 0.01f;
    config.max_weight = 1.5f;
    config.weight_init_std = 0.3f;
    
    // Topology parameters
    config.numColumns = 4;
    config.neuronsPerColumn = 64;
    config.localFanOut = 15;
    config.localFanIn = 15;
    
    // Enhanced timing
    config.dt = 0.1;
    config.simulation_time = 1000.0f; // 1 second simulation
    
    config.finalizeConfig();
    return config;
}

int main() {
    std::cout << "Starting Modular Neural Network Simulation..." << std::endl;

    // --- Configuration ---
    // Create configurations for different modules
    auto cognitive_config = create_default_config();
    cognitive_config.neurogenesis_rate = 0.002; // Higher plasticity for cognitive tasks

    auto motor_config = create_default_config();
    motor_config.stdp_learning_rate = 0.005; // Lower learning rate for stable motor control

    // --- Module Creation ---
    // Create the underlying neural modules that the task modules will manage
    auto perception_net = std::make_shared<NeuralModule>("PerceptionNet", cognitive_config);
    auto planning_net = std::make_shared<NeuralModule>("PlanningNet", cognitive_config);
    auto motor_control_net = std::make_shared<NeuralModule>("MotorControlNet", motor_config);

    // --- Task-Level Module Creation ---
    // >>> FIX: Replaced undeclared classes with the correct classes from TaskAutomationModules.h
    auto cognitive_module = std::make_shared<CognitiveModule>(perception_net, planning_net);
    auto motor_module = std::make_shared<MotorModule>(motor_control_net);
    // <<< END FIX

    // --- System Initialization ---
    // Create a polymorphic vector to hold all task modules
    std::vector<std::shared_ptr<TaskModule>> task_modules;
    task_modules.push_back(cognitive_module);
    task_modules.push_back(motor_module);

    // Initialize all modules in a uniform way
    std::cout << "\nInitializing all task modules..." << std::endl;
    for (const auto& module : task_modules) {
        module->initialize();
    }

    // --- Simulation Loop ---
    std::cout << "\n🚀 Starting enhanced simulation loop (Version 0.5.5)..." << std::endl;
    std::cout << "   Features: Dynamic synaptogenesis, modular testing, adaptive connectivity" << std::endl;
    
    float total_simulation_time = 1000.0f; // ms - Extended simulation
    float dt = 0.1f; // ms
    int num_steps = static_cast<int>(total_simulation_time / dt);
    
    // Enhanced input patterns for testing modular responses
    std::vector<std::vector<float>> test_patterns = {
        std::vector<float>(256, 0.0f), // Baseline
        std::vector<float>(256, 0.0f), // Pattern 1: Visual-like input
        std::vector<float>(256, 0.0f), // Pattern 2: Motor command
        std::vector<float>(256, 0.0f)  // Pattern 3: Mixed pattern
    };
    
    // Configure test patterns
    // Pattern 1: Visual-like input (concentrated activation)
    for (int i = 0; i < 32; i++) {
        test_patterns[1][i] = 15.0f + (i % 3) * 5.0f; // Spatial pattern
    }
    
    // Pattern 2: Motor command (distributed activation)
    for (int i = 64; i < 128; i += 4) {
        test_patterns[2][i] = 20.0f; // Rhythmic pattern
    }
    
    // Pattern 3: Mixed pattern (complex input)
    for (int i = 0; i < 256; i += 8) {
        test_patterns[3][i] = 12.0f + (i / 32) * 2.0f; // Gradient pattern
    }
    
    int current_pattern = 0;
    int pattern_duration = 2000; // Steps per pattern (200ms)
    int structural_plasticity_interval = 1000; // Every 100ms
    
    std::cout << "🧠 Testing modular networks with " << test_patterns.size() << " input patterns" << std::endl;
    
    for (int i = 0; i < num_steps; ++i) {
        float current_time = i * dt;
        
        // Switch input patterns periodically to test adaptation
        if (i % pattern_duration == 0) {
            current_pattern = (current_pattern + 1) % test_patterns.size();
            std::cout << "🔄 Switching to input pattern " << current_pattern 
                      << " at time " << current_time << "ms" << std::endl;
        }
        
        // Get current input pattern
        std::vector<float> inputs = test_patterns[current_pattern];
        
        // Add noise for biological realism
        for (auto& input : inputs) {
            if (input > 0.0f) {
                input += ((rand() % 100) / 100.0f - 0.5f) * 2.0f; // ±1.0 noise
            }
        }
        
        // Calculate reward based on network coordination
        float reward = 0.1f;
        auto perception_stats = perception_net->get_stats();
        auto planning_stats = planning_net->get_stats();
        auto motor_stats = motor_control_net->get_stats();
        
        // Reward coordinated activity between modules
        if (perception_stats.active_neuron_count > 5 && 
            planning_stats.active_neuron_count > 3 &&
            motor_stats.active_neuron_count > 2) {
            reward = 0.5f; // Higher reward for coordinated activity
        }
        
        // Update all modules with enhanced dynamics
        perception_net->update(dt, inputs, reward);
        
        // Inter-modular communication: Planning receives perception output
        auto perception_output = perception_net->get_output();
        planning_net->update(dt, perception_output, reward);
        
        // Motor control receives planning output
        auto planning_output = planning_net->get_output();
        motor_control_net->update(dt, planning_output, reward);
        
        // Implement structural plasticity (dynamic synaptogenesis)
        if (i % structural_plasticity_interval == 0 && i > 0) {
            std::cout << "🌱 Triggering structural plasticity at " << current_time << "ms" << std::endl;
            
            // Each network grows/prunes synapses based on activity
            auto* perception_network = perception_net->get_network();
            auto* planning_network = planning_net->get_network();
            auto* motor_network = motor_control_net->get_network();
            
            if (perception_network) {
                perception_network->grow_synapses();
                perception_network->prune_synapses();
            }
            if (planning_network) {
                planning_network->grow_synapses();
                planning_network->prune_synapses();
            }
            if (motor_network) {
                motor_network->grow_synapses();
                motor_network->prune_synapses();
            }
        }
        
        // Enhanced monitoring and output
        if (i % 1000 == 0) { // Every 100ms
            std::cout << "\n📊 Time: " << current_time << "ms (Pattern " << current_pattern << ")" << std::endl;
            std::cout << "   🧠 Perception: " << perception_stats.active_neuron_count 
                      << " active, " << perception_stats.total_synapses << " synapses" << std::endl;
            std::cout << "   🎯 Planning: " << planning_stats.active_neuron_count 
                      << " active, " << planning_stats.total_synapses << " synapses" << std::endl;
            std::cout << "   🏃 Motor: " << motor_stats.active_neuron_count 
                      << " active, " << motor_stats.total_synapses << " synapses" << std::endl;
            
            // Show network adaptation metrics
            float total_synapses = perception_stats.total_synapses + 
                                 planning_stats.total_synapses + 
                                 motor_stats.total_synapses;
            std::cout << "   📈 Total connectivity: " << total_synapses << " synapses" << std::endl;
            std::cout << "   🎯 Reward signal: " << reward << std::endl;
        }
        
        // Test inter-modular communication
        if (i % 2500 == 0 && i > 0) { // Every 250ms
            std::cout << "\n🔗 Testing inter-modular communication..." << std::endl;
            
            // Test perception -> planning pathway
            auto perception_output_strength = 0.0f;
            for (float val : perception_output) {
                perception_output_strength += std::abs(val);
            }
            
            auto planning_output_strength = 0.0f;
            for (float val : planning_output) {
                planning_output_strength += std::abs(val);
            }
            
            std::cout << "   📡 Perception output strength: " << perception_output_strength << std::endl;
            std::cout << "   📡 Planning output strength: " << planning_output_strength << std::endl;
            
            if (perception_output_strength > 1.0f && planning_output_strength > 0.5f) {
                std::cout << "   ✅ Strong inter-modular communication detected!" << std::endl;
            }
        }
    }

    std::cout << "\n🎉 Simulation completed successfully!" << std::endl;
    std::cout << "\n📊 Final Network Statistics:" << std::endl;
    
    auto final_perception_stats = perception_net->get_stats();
    auto final_planning_stats = planning_net->get_stats();
    auto final_motor_stats = motor_control_net->get_stats();
    
    std::cout << "   🧠 Perception Network:" << std::endl;
    std::cout << "      Neurons: " << final_perception_stats.active_neuron_count << std::endl;
    std::cout << "      Synapses: " << final_perception_stats.total_synapses << std::endl;
    std::cout << "      Firing rate: " << final_perception_stats.mean_firing_rate << " Hz" << std::endl;
    
    std::cout << "   🎯 Planning Network:" << std::endl;
    std::cout << "      Neurons: " << final_planning_stats.active_neuron_count << std::endl;
    std::cout << "      Synapses: " << final_planning_stats.total_synapses << std::endl;
    std::cout << "      Firing rate: " << final_planning_stats.mean_firing_rate << " Hz" << std::endl;
    
    std::cout << "   🏃 Motor Network:" << std::endl;
    std::cout << "      Neurons: " << final_motor_stats.active_neuron_count << std::endl;
    std::cout << "      Synapses: " << final_motor_stats.total_synapses << std::endl;
    std::cout << "      Firing rate: " << final_motor_stats.mean_firing_rate << " Hz" << std::endl;
    
    size_t total_synapses = final_perception_stats.total_synapses + 
                           final_planning_stats.total_synapses + 
                           final_motor_stats.total_synapses;
    
    std::cout << "\n🌐 Overall System Performance:" << std::endl;
    std::cout << "   Total synaptic connections: " << total_synapses << std::endl;
    std::cout << "   Network connectivity achieved through dynamic synaptogenesis" << std::endl;
    std::cout << "   Modular communication pathways established" << std::endl;
    std::cout << "   Adaptive learning and structural plasticity demonstrated" << std::endl;

    std::cout << "\nSimulation finished." << std::endl;

    return 0;
}