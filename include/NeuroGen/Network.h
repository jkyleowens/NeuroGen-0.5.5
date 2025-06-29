#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <random>

#include <NeuroGen/Neuron.h>
#include <NeuroGen/Synapse.h>
#include <NeuroGen/NetworkStats.h>
#include <NeuroGen/NetworkConfig.h>

// Forward declaration to prevent circular dependency with NeuralModule
class NeuralModule;

/**
 * @class Network
 * @brief Manages the entire neural network, including its neurons, synapses, and overall dynamics.
 *
 * This class is the core of the CPU-based simulation, handling the lifecycle of all
 * neural components, their interactions, and the application of plasticity rules.
 */
class Network {
public:
    /**
     * @brief Constructs a Network based on a given configuration.
     * @param config A NetworkConfig struct detailing the network's parameters.
     */
    explicit Network(const NetworkConfig& config);

    /**
     * @brief Destructor for the Network class.
     */
    ~Network();

    /**
     * @brief Updates the entire network for a single time step.
     * @param dt The simulation time step in milliseconds.
     * @param input_currents A vector of input currents for the input neurons.
     * @param reward The global reward signal for this time step.
     */
    void update(float dt, const std::vector<float>& input_currents, float reward);

    /**
     * @brief Retrieves the current output of the network.
     * @return A vector of floats representing the output values (e.g., firing rates of output neurons).
     */
    std::vector<float> get_output() const;

    /**
     * @brief Resets the state of all neurons and synapses in the network.
     */
    void reset();

    /**
     * @brief Adds a neuron to the network.
     * @param neuron A unique_ptr to the Neuron to be added.
     */
    void add_neuron(std::unique_ptr<Neuron> neuron);

    /**
     * @brief Adds a synapse to the network.
     * @param synapse A unique_ptr to the Synapse to be added.
     */
    void add_synapse(std::unique_ptr<Synapse> synapse);

    /**
     * @brief Creates a new synapse and adds it to the network.
     * @param source_neuron_id The ID of the presynaptic neuron.
     * @param target_neuron_id The ID of the postsynaptic neuron.
     * @param type The type of the synapse (e.g., "excitatory").
     * @param delay The transmission delay in time steps.
     * @param weight The initial weight of the synapse.
     * @return A pointer to the newly created Synapse, or nullptr on failure.
     */
    Synapse* createSynapse(size_t source_neuron_id, size_t target_neuron_id, const std::string& type, int delay, float weight);

    /**
     * @brief Retrieves a pointer to a neuron by its ID.
     * @param neuron_id The unique ID of the neuron.
     * @return A pointer to the Neuron, or nullptr if not found.
     */
    Neuron* get_neuron(size_t neuron_id) const;

    /**
     * @brief Retrieves a pointer to a synapse by its ID.
     * @param synapse_id The unique ID of the synapse.
     * @return A pointer to the Synapse, or nullptr if not found.
     */
    Synapse* get_synapse(size_t synapse_id) const;

    /**
     * @brief Gets all synapses originating from a specific neuron.
     * @param neuron_id The ID of the source neuron.
     * @return A vector of pointers to the outgoing Synapses.
     */
    std::vector<Synapse*> getOutgoingSynapses(size_t neuron_id);

    /**
     * @brief Gets all synapses targeting a specific neuron.
     * @param neuron_id The ID of the target neuron.
     * @return A vector of pointers to the incoming Synapses.
     */
    std::vector<Synapse*> getIncomingSynapses(size_t neuron_id);

    /**
     * @brief Associates a neural module with this network.
     * @param module A pointer to the NeuralModule.
     */
    void set_module(NeuralModule* module);

    /**
     * @brief Retrieves the current network statistics.
     * @return A copy of the NetworkStats struct.
     */
    NetworkStats get_stats() const;

private:
    // Initialization methods
    void initialize_neurons();
    void initialize_synapses();

    // Update steps
    void gather_inputs();
    void update_neurons(float dt, const std::vector<float>& input_currents);
    void update_synapses(float dt, float reward);
    void apply_plasticity(float dt, float reward);
    void structural_plasticity();
    void update_stats(float dt);

    // Structural plasticity helpers
    void prune_synapses();
    void grow_synapses();
    bool shouldPruneSynapse(const Synapse& synapse) const;
    void createNewSynapseForNeuron(const Neuron& neuron);


    // Member variables
    const NetworkConfig& config_; // Reference to the network configuration
    std::vector<std::unique_ptr<Neuron>> neurons_; // Owns all neurons
    std::vector<std::unique_ptr<Synapse>> synapses_; // Owns all synapses

    // Maps for efficient lookups
    std::unordered_map<size_t, Neuron*> neuron_map_;
    std::unordered_map<size_t, Synapse*> synapse_map_;
    std::unordered_map<size_t, std::vector<Synapse*>> outgoing_synapse_map_;
    std::unordered_map<size_t, std::vector<Synapse*>> incoming_synapse_map_;

    NeuralModule* module_; // Associated neural module (not owned)
    NetworkStats stats_;   // Network statistics

    // Random number generation for plasticity
    std::mt19937 random_engine_;
};

#endif // NETWORK_H