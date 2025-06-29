#ifndef NETWORK_BUILDER_H
#define NETWORK_BUILDER_H

#include "NeuroGen/Network.h"
#include "NeuroGen/Neuron.h"
#include <vector>

/**
 * @brief A helper class to construct topologies within an existing Network instance.
 *
 * This builder is designed to be used within NeuralModule implementations to define
 * their internal neural circuits.
 */
class NetworkBuilder {
public:
    /**
     * @brief Constructs a builder to modify an existing network.
     * @param network A pointer to the Network instance to be modified.
     */
    explicit NetworkBuilder(Network* network);

    /**
     * @brief Adds a population of neurons to the network.
     * @param type The type of neurons (e.g., EXCITATORY, INHIBITORY).
     * @param count The number of neurons to create.
     * @param center The central 3D position for the new population.
     * @return A vector of IDs for the newly created neurons.
     */
    std::vector<size_t> addNeuronPopulation(NeuronType type, size_t count, const Position3D& center);

    /**
     * @brief Connects two populations of neurons.
     * @param source_population A vector of source neuron IDs.
     * @param target_population A vector of target neuron IDs.
     * @param probability The chance (0.0 to 1.0) of forming a synapse between any two neurons.
     * @param initial_weight The initial weight for the created synapses.
     */
    void connect(const std::vector<size_t>& source_population, const std::vector<size_t>& target_population, double probability, double initial_weight);

private:
    Network* network_; // Pointer to the network being modified
    std::mt19937 rng_; // Random number generator for probabilistic connections
};

#endif // NETWORK_BUILDER_H
