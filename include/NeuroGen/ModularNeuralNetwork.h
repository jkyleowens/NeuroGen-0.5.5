#ifndef MODULAR_NEURAL_NETWORK_H
#define MODULAR_NEURAL_NETWORK_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include "NeuroGen/NeuralModule.h"

/**
 * @brief Manages a collection of interconnected NeuralModules.
 */
class ModularNeuralNetwork {
public:
    ModularNeuralNetwork() = default;

    void addModule(std::shared_ptr<NeuralModule> module);
    void initialize();

    /**
     * @brief Establishes synaptic connections between two modules.
     * @param source_module_name The name of the module sending signals.
     * @param source_port_name The output port name of the source module.
     * @param target_module_name The name of the module receiving signals.
     * @param target_port_name The input port name of the target module.
     * @param probability The chance (0.0 to 1.0) of forming a synapse.
     * @param weight The initial weight for created synapses.
     */
    void connect(const std::string& source_module_name, const std::string& source_port_name,
                 const std::string& target_module_name, const std::string& target_port_name,
                 double probability, double weight);

    void update(double dt);
    std::shared_ptr<NeuralModule> getModule(const std::string& name);

private:
    // Corrected the typo from shared_dtr to shared_ptr
    std::unordered_map<std::string, std::shared_ptr<NeuralModule>> modules_;
};

#endif // MODULAR_NEURAL_NETWORK_H