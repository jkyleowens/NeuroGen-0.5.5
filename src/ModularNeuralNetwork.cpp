#include "NeuroGen/ModularNeuralNetwork.h"
#include "NeuroGen/Network.h" // <<< The required include for the full Network definition
#include <stdexcept>
#include <utility>
#include <random>

void ModularNeuralNetwork::addModule(std::shared_ptr<NeuralModule> module) {
    if (module) {
        modules_[module->getName()] = std::move(module);
    }
}

void ModularNeuralNetwork::initialize() {
    for (auto const& [name, module] : modules_) {
        module->initialize();
    }
}

void ModularNeuralNetwork::connect(const std::string& source_module_name, const std::string& source_port_name,
                                   const std::string& target_module_name, const std::string& target_port_name,
                                   double probability, double weight) {
    try {
        auto& source_module = modules_.at(source_module_name);
        auto& target_module = modules_.at(target_module_name);

        const auto& source_neurons = source_module->getNeuronPopulation(source_port_name);
        const auto& target_neurons = target_module->getNeuronPopulation(target_port_name);

        if (source_neurons.empty() || target_neurons.empty()) {
            throw std::runtime_error("One or both neuron populations (ports) not found for connection.");
        }

        Network* target_network = target_module->getNetwork();
        if (!target_network) {
            throw std::runtime_error("Target module's network is null.");
        }

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<> dist(0.0, 1.0);

        for (const auto& source_id : source_neurons) {
            for (const auto& target_id : target_neurons) {
                if (dist(rng) < probability) {
                    // This call is now valid because Network.h has been included
                    target_network->createSynapse(source_id, target_id, "soma", 0, weight);
                }
            }
        }

    } catch (const std::out_of_range& e) {
        throw std::runtime_error("Module name not found in network: " + std::string(e.what()));
    }
}

void ModularNeuralNetwork::update(double dt) {
    for (auto const& [name, module] : modules_) {
        module->update(dt);
    }
}

std::shared_ptr<NeuralModule> ModularNeuralNetwork::getModule(const std::string& name) {
    auto it = modules_.find(name);
    if (it != modules_.end()) {
        return it->second;
    }
    return nullptr;
}