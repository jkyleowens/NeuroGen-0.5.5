#include "NeuroGen/NetworkBuilder.h"
#include "NeuroGen/Neuron.h" // Needed for Neuron class
#include <random>
#include <stdexcept>

NetworkBuilder::NetworkBuilder(Network* network) : network_(network) {
    if (!network_) {
        throw std::invalid_argument("NetworkBuilder must be initialized with a valid Network pointer.");
    }
    // Seed the random number generator
    std::random_device rd;
    rng_.seed(rd());
}

std::vector<size_t> NetworkBuilder::addNeuronPopulation(NeuronType type, size_t count, const Position3D& center) {
    std::vector<size_t> neuron_ids;
    neuron_ids.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        // For simplicity, placing all neurons at the center.
        // A more advanced implementation could distribute them in a radius.
        auto neuron = std::make_shared<Neuron>(0, type, network_); // ID will be assigned by network
        size_t new_id = network_->addNeuron(neuron, center);
        neuron_ids.push_back(new_id);
    }
    return neuron_ids;
}

void NetworkBuilder::connect(const std::vector<size_t>& source_population, const std::vector<size_t>& target_population, double probability, double initial_weight) {
    std::uniform_real_distribution<> dist(0.0, 1.0);

    for (const auto& source_id : source_population) {
        for (const auto& target_id : target_population) {
            if (source_id == target_id) continue; // No self-connections

            if (dist(rng_) < probability) {
                // Assuming a default compartment and receptor for simplicity
                network_->createSynapse(source_id, target_id, "soma", 0, initial_weight);
            }
        }
    }
}