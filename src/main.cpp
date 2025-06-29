#include <iostream>
#include <vector>
#include <memory>
#include "NeuroGen/TaskAutomationModules.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/NeuralModule.h"

// Function to create a default configuration for a neural module
NetworkConfig create_default_config() {
    NetworkConfig config;
    config.num_neurons = 100; // Example: 100 neurons per module
    config.enable_neurogenesis = true;
    config.enable_stdp = true;
    config.enable_pruning = true;
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
    std::cout << "\nStarting simulation loop..." << std::endl;
    float total_simulation_time = 100.0f; // ms
    float dt = 0.1f; // ms
    int num_steps = static_cast<int>(total_simulation_time / dt);

    for (int i = 0; i < num_steps; ++i) {
        float current_time = i * dt;
        // In a real simulation, you would generate inputs here
        std::vector<float> inputs(100, 0.0f); // Example: 100 input channels
        if (i % 10 == 0) { // Stimulate every 10 steps
            inputs[0] = 10.0f; // Inject some current
        }

        // Update all modules
        perception_net->update(dt, inputs, 0.1f);
        planning_net->update(dt, perception_net->get_output(), 0.1f);
        motor_control_net->update(dt, planning_net->get_output(), 0.1f);
        
        if (i % 100 == 0) {
            std::cout << "Time: " << current_time << "ms, Perception Neurons: " 
                      << perception_net->get_stats().active_neuron_count << std::endl;
        }
    }

    std::cout << "\nSimulation finished." << std::endl;

    return 0;
}