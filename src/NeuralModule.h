#ifndef NEURAL_MODULE_H
#define NEURAL_MODULE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// Forward-declare dependencies to reduce header includes
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/Network.h> // <<< The critical fix: Include the full Network definition
#include <NeuroGen/Neuron.h>   // Also include Neuron for getPotential()
#include <utility>

/**
 * @brief Represents a self-contained, specialized neural circuit (an "agent").
 * Each NeuralModule encapsulates its own biological neural network and defines
 * named neuron populations that act as ports for inter-module communication.
 */
class NeuralModule {
public:
    /**
     * @brief Constructs a NeuralModule with its own internal network.
     * @param name A unique name for this module (e.g., "VisualCortex", "MotorControl").
     * @param config The configuration used to build the module's internal neural network.
     */
    NeuralModule(std::string name, const NetworkConfig& config);

    virtual ~NeuralModule() = default;

    /**
     * @brief Performs one-time setup of the module's internal network and neuron populations.
     * This is a pure virtual function and MUST be implemented by derived classes.
     */
    virtual void initialize() = 0;

    /**
     * @brief Advances the module's internal network simulation by one time step.
     * @param dt The simulation time step (e.g., 0.1 ms).
     */
    void update(double dt);

    /**
     * @brief Resets the module's internal network to its initial state.
     */
    void reset();

    // --- Accessors ---
    const std::string& getName() const;
    Network* getNetwork() const;

    /**
     * @brief Retrieves the IDs of neurons in a named population (port).
     * @param port_name The name of the neuron population (e.g., "OUTPUT").
     * @return A const reference to a vector of neuron IDs, or an empty vector if the port doesn't exist.
     */
    const std::vector<size_t>& getNeuronPopulation(const std::string& port_name) const;

    // --- External World I/O ---

    /**
     * @brief Injects external current into a specified neuron population.
     * @param port_name The name of the target population (e.g., "INPUT").
     * @param currents A vector of current values, one for each neuron in the population.
     */
    void injectCurrentToPopulation(const std::string& port_name, const std::vector<double>& currents);

    /**
     * @brief Retrieves the current membrane potentials of a neuron population.
     * @param port_name The name of the target population (e.g., "OUTPUT").
     * @return A vector of membrane potential values.
     */
    std::vector<double> getPotentialsFromPopulation(const std::string& port_name) const;

protected:
    /**
     * @brief Creates and registers a named population of neurons.
     * To be called by derived classes within their initialize() method.
     * @param port_name The name for this population (e.g., "INPUT", "OUTPUT").
     * @param neuron_ids A vector of neuron IDs belonging to this population.
     */
    void addNeuronPopulation(const std::string& port_name, std::vector<size_t> neuron_ids);

    std::string module_name_;
    std::unique_ptr<Network> internal_network_;
    std::unordered_map<std::string, std::vector<size_t>> neuron_populations_;
};

#endif // NEURAL_MODULE_H