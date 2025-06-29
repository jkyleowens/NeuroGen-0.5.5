#include <iostream>
#include <memory>
#include "NeuroGen/ModularNeuralNetwork.h"
#include "NeuroGen/TaskAutomationModules.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/Network.h"

int main() {
    std::cout << "--- Initializing Modular Neural Network ---" << std::endl;

    // 1. Create the main orchestrator for the network of networks.
    auto modular_net = std::make_unique<ModularNeuralNetwork>();

    // 2. Define configurations for our specialized sub-networks (modules).
    NetworkConfig sensory_config;
    // Configure sensory processing module
    sensory_config.input_size = 128;
    sensory_config.hidden_size = 256;
    sensory_config.output_size = 64;
    sensory_config.dt = 0.1;
    sensory_config.simulation_time = 1000.0;
    sensory_config.enable_stdp = true;
    sensory_config.enable_neurogenesis = true;
    sensory_config.enable_pruning = true;

    NetworkConfig action_config;
    // Configure action selection module
    action_config.input_size = 64;
    action_config.hidden_size = 128;
    action_config.output_size = 32;
    action_config.dt = 0.1;
    action_config.simulation_time = 1000.0;
    action_config.enable_stdp = true;
    action_config.enable_neurogenesis = true;
    action_config.enable_pruning = true;

    // 3. Create the specialized modules (agents).
    auto sensory_module = std::make_shared<SensoryProcessingModule>("SensoryModule", sensory_config);
    auto action_module = std::make_shared<ActionSelectionModule>("ActionModule", action_config);

    // 4. Add the modules to the orchestrator.
    modular_net->addModule(sensory_module);
    modular_net->addModule(action_module);

    // 5. Initialize all modules (this builds their internal neural circuits).
    std::cout << "\n--- Initializing all modules... ---" << std::endl;
    modular_net->initialize();
    std::cout << "--- Module initialization complete. ---" << std::endl;


    // 6. Connect the modules to form a cognitive pathway.
    // Connect the "OUTPUT" of the sensory module to the "SENSORY_INPUT" of the action module.
    std::cout << "\n--- Connecting modules... ---" << std::endl;
    try {
        modular_net->connect("SensoryModule", "OUTPUT", "ActionModule", "SENSORY_INPUT", 0.1, 0.5);
        std::cout << "--- Modules connected successfully. ---" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error during connection: " << e.what() << std::endl;
        return 1;
    }

    // 7. Run the simulation loop.
    std::cout << "\n--- Starting simulation... ---" << std::endl;
    const double time_step = 0.1; // ms
    const int simulation_steps = 1000;
    for (int i = 0; i < simulation_steps; ++i) {
        // In a real scenario, you would inject current into the sensory module here
        // Example: sensory_module->injectCurrentToPopulation("INPUT", sensory_data);
        
        modular_net->update(time_step);

        if (i % 100 == 0) {
            std::cout << "Simulation step: " << i << std::endl;
        }
    }

    std::cout << "\n--- Simulation finished. ---" << std::endl;

    return 0;
}
