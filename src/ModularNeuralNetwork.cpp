#include "NeuroGen/ModularNeuralNetwork.h"
#include <iostream>
#include <stdexcept>
#include <vector>

// --- Implementation matches the declaration in the header ---
void ModularNeuralNetwork::add_module(std::unique_ptr<NeuralModule> module) {
    if (module) {
        // Use get_name() which is the correct method on NeuralModule
        modules_[module->get_name()] = std::move(module);
    }
}

// --- Implementation matches the declaration in the header ---
NeuralModule* ModularNeuralNetwork::get_module(const std::string& name) const {
    auto it = modules_.find(name);
    if (it != modules_.end()) {
        return it->second.get();
    }
    return nullptr;
}

// --- Implementation matches the declaration in the header ---
void ModularNeuralNetwork::connect(const std::string& source_module_name, const std::string& source_port_name,
                                 const std::string& target_module_name, const std::string& target_port_name) {
    NeuralModule* source_module = get_module(source_module_name);
    NeuralModule* target_module = get_module(target_module_name);

    if (!source_module || !target_module) {
        throw std::runtime_error("One or both modules not found for connection.");
    }

    const auto& source_neurons = source_module->get_neuron_population(source_port_name);
    const auto& target_neurons = target_module->get_neuron_population(target_port_name);
    Network* target_network = target_module->get_network();

    if (!target_network) {
        throw std::runtime_error("Target module's internal network is null.");
    }

    std::cout << "Connecting '" << source_port_name << "' of module '" << source_module_name
              << "' to '" << target_port_name << "' of module '" << target_module_name << "'..." << std::endl;

    // Example connection logic: all-to-all
    for (size_t source_neuron_id : source_neurons) {
        for (size_t target_neuron_id : target_neurons) {
            target_network->createSynapse(source_neuron_id, target_neuron_id, "excitatory", 1, 0.1f);
        }
    }
}

// --- Implementation matches the declaration in the header ---
void ModularNeuralNetwork::run(float duration, float dt) {
    int num_steps = static_cast<int>(duration / dt);
    std::cout << "Running simulation for " << num_steps << " steps." << std::endl;

    for (int i = 0; i < num_steps; ++i) {
        // In a real scenario, you'd process inputs from external sources
        // and propagate outputs between connected modules.
        // This is a simplified loop for demonstration.
        for (auto const& [name, module] : modules_) {
            // Provide dummy inputs for this example simulation step
            std::vector<float> dummy_inputs(module->get_stats().active_neuron_count, 0.0f);
            float dummy_reward = 0.0f;
            module->update(dt, dummy_inputs, dummy_reward);
        }
    }
}

// --- Implementation matches the declaration in the header ---
void ModularNeuralNetwork::initialize() {
    std::cout << "Modular network initialized. " << modules_.size() << " modules loaded." << std::endl;
}