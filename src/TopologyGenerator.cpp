#include <NeuroGen/TopologyGenerator.h>
#include <NeuroGen/NetworkLayer.h>
#include <NeuroGen/Neuron.h>
#include <stdexcept>
#include <cmath>
#include <random>
#include <iostream>

TopologyGenerator::TopologyGenerator(const NetworkConfig& config, unsigned int seed)
    : /*config_(config),*/ random_engine_(seed) {}

std::vector<NeuronModel> TopologyGenerator::createNeuronPopulation(
    int count, NeuronType type, const Position3D& position, float radius) {
    std::vector<NeuronModel> population;
    std::uniform_real_distribution<float> dist(-radius, radius);

    for (int i = 0; i < count; ++i) {
        NeuronModel neuron;
        neuron.id = next_neuron_id_++;
        neuron.type = type;
        neuron.position = {
            position.x + dist(random_engine_),
            position.y + dist(random_engine_),
            position.z + dist(random_engine_)
        };
        population.push_back(neuron);
    }
    return population;
}

std::vector<GPUSynapse> TopologyGenerator::connectPopulations(
    const std::vector<NeuronModel>& source_pop,
    const std::vector<NeuronModel>& target_pop,
    const std::vector<ConnectionRule>& rules) {
    std::vector<GPUSynapse> all_synapses;
    for (const auto& rule : rules) {
        applyDistanceDecayRule(rule, source_pop, target_pop, all_synapses);
    }
    return all_synapses;
}

void TopologyGenerator::applyDistanceDecayRule(const ConnectionRule& rule, const std::vector<NeuronModel>& source_pop, const std::vector<NeuronModel>& /*target_pop*/, std::vector<GPUSynapse>& synapses) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (const auto& source_neuron : source_pop) {
        for (const auto& target_neuron : source_pop) {
            if (source_neuron.id == target_neuron.id) continue;

            float distance = calculateDistance(source_neuron.position, target_neuron.position);
            float probability = rule.probability * std::exp(-distance / rule.decay_rate);

            if (dist(random_engine_) < probability) {
                GPUSynapse synapse;
                synapse.source_neuron_id = source_neuron.id;
                synapse.target_neuron_id = target_neuron.id;
                synapse.weight = 1.0f; 
                synapse.delay = 1.0f; 
                synapses.push_back(synapse);
            }
        }
    }
}

float TopologyGenerator::calculateDistance(const Position3D& p1, const Position3D& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) +
                     std::pow(p1.y - p2.y, 2) +
                     std::pow(p1.z - p2.z, 2));
}

void TopologyGenerator::setNextNeuronId(int id) {
    next_neuron_id_ = id;
}

void TopologyGenerator::setNextSynapseId(int id) {
    next_synapse_id_ = id;
}