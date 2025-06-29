#ifndef MODULAR_NEURAL_NETWORK_H
#define MODULAR_NEURAL_NETWORK_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "NeuroGen/NeuralModule.h"

/**
 * @class ModularNeuralNetwork
 * @brief Manages a collection of interconnected NeuralModules to form a large-scale network.
 *
 * This class serves as the main orchestrator, responsible for holding the modules,
 * establishing connections between them, and running the simulation for the entire system.
 */
class ModularNeuralNetwork {
public:
    /**
     * @brief Default constructor.
     */
    ModularNeuralNetwork() = default;

    /**
     * @brief Adds a neural module to the network. The ModularNeuralNetwork takes ownership.
     * @param module A unique_ptr to the NeuralModule to be added.
     */
    void add_module(std::unique_ptr<NeuralModule> module);

    /**
     * @brief Retrieves a pointer to a module by its name.
     * @param name The unique identifier of the module.
     * @return A raw pointer to the NeuralModule, or nullptr if not found.
     */
    NeuralModule* get_module(const std::string& name) const;

    /**
     * @brief Connects an output port of one module to an input port of another.
     * @param source_module_name The name of the module providing the output.
     * @param source_port_name The name of the output port on the source module.
     * @param target_module_name The name of the module receiving the input.
     * @param target_port_name The name of the input port on the target module.
     */
    void connect(const std::string& source_module_name, const std::string& source_port_name,
                 const std::string& target_module_name, const std::string& target_port_name);

    /**
     * @brief Runs the simulation for the entire modular network.
     * @param duration The total simulation time in milliseconds.
     * @param dt The time step for the simulation in milliseconds.
     */
    void run(float duration, float dt);

    /**
     * @brief Initializes the network. Placeholder for future initialization logic.
     */
    void initialize();

private:
    // A map to hold all the modules, with the module name as the key.
    std::unordered_map<std::string, std::unique_ptr<NeuralModule>> modules_;
};

#endif // MODULAR_NEURAL_NETWORK_H