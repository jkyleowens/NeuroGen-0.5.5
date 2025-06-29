#include "NeuroGen/Network.h"
#include "NeuroGen/NeuralModule.h"
#include <iostream>
#include <numeric>

// --- (Other functions like the constructor, destructor, update, etc. are here) ---
// (Scroll down to find the new function at the end)

Network::Network(const NetworkConfig& config)
    : config_(config), module_(nullptr) {
    initialize_neurons();
    initialize_synapses();
}

Network::~Network() = default;

void Network::update(float dt, const std::vector<float>& input_currents, float reward) {
    update_neurons(dt, input_currents);
    update_synapses(dt, reward);
    if (config_.structural_plasticity) {
        structural_plasticity();
    }
    update_stats(dt);
}

std::vector<float> Network::get_output() const {
    // Implementation for get_output
    return {};
}

void Network::reset() {
    // Implementation for reset
}

void Network::add_neuron(std::unique_ptr<Neuron> neuron) {
    if (neuron) {
        neuron_map_[neuron->get_id()] = neuron.get();
        neurons_.push_back(std::move(neuron));
    }
}

void Network::add_synapse(std::unique_ptr<Synapse> synapse) {
    if (synapse) {
        Synapse* syn_ptr = synapse.get();
        synapse_map_[syn_ptr->get_id()] = syn_ptr;
        outgoing_synapse_map_[syn_ptr->get_source_neuron_id()].push_back(syn_ptr);
        incoming_synapse_map_[syn_ptr->get_target_neuron_id()].push_back(syn_ptr);
        synapses_.push_back(std::move(synapse));
    }
}

Neuron* Network::get_neuron(size_t neuron_id) const {
    auto it = neuron_map_.find(neuron_id);
    return (it != neuron_map_.end()) ? it->second : nullptr;
}

Synapse* Network::get_synapse(size_t synapse_id) const {
    auto it = synapse_map_.find(synapse_id);
    return (it != synapse_map_.end()) ? it->second : nullptr;
}

std::vector<Synapse*> Network::getOutgoingSynapses(size_t neuron_id) {
    auto it = outgoing_synapse_map_.find(neuron_id);
    return (it != outgoing_synapse_map_.end()) ? it->second : std::vector<Synapse*>();
}

std::vector<Synapse*> Network::getIncomingSynapses(size_t neuron_id) {
    auto it = incoming_synapse_map_.find(neuron_id);
    return (it != incoming_synapse_map_.end()) ? it->second : std::vector<Synapse*>();
}

void Network::set_module(NeuralModule* module) {
    module_ = module;
}

NetworkStats Network::get_stats() const {
    return stats_;
}

void Network::initialize_neurons() {
    for (size_t i = 0; i < config_.num_neurons; ++i) {
        add_neuron(std::make_unique<Neuron>(i));
    }
}

void Network::initialize_synapses() {
    // Synapse initialization logic
}

void Network::update_neurons(float dt, const std::vector<float>& input_currents) {
    // Neuron update logic
}

void Network::update_synapses(float dt, float reward) {
    // Synapse update logic
}

void Network::apply_plasticity(float dt, float reward) {
    // Plasticity logic
}

void Network::structural_plasticity() {
    // Structural plasticity logic
}

void Network::update_stats(float dt) {
    // Stats update logic
}

void Network::prune_synapses() {
    // Pruning logic
}

void Network::grow_synapses() {
    // Growth logic
}

bool Network::shouldPruneSynapse(const Synapse& synapse) const {
    return false; // Pruning condition logic
}

void Network::createNewSynapseForNeuron(const Neuron& neuron) {
    // New synapse creation logic
}

/**
 * THIS IS THE NEWLY ADDED FUNCTION IMPLEMENTATION
 */
Synapse* Network::createSynapse(size_t source_neuron_id, size_t target_neuron_id, const std::string& type, int delay, float weight) {
    // Ensure both source and target neurons exist before creating a synapse
    Neuron* source_neuron = get_neuron(source_neuron_id);
    Neuron* target_neuron = get_neuron(target_neuron_id);

    if (!source_neuron || !target_neuron) {
        std::cerr << "Error: Cannot create synapse. Source or target neuron not found." << std::endl;
        return nullptr;
    }

    // Create the synapse
    size_t new_synapse_id = synapses_.size();
    auto new_synapse = std::make_unique<Synapse>(new_synapse_id, source_neuron_id, target_neuron_id, weight);
    
    // Set other properties if needed from the 'type' and 'delay' arguments...
    // For example:
    // new_synapse->set_delay(delay);

    // Add the new synapse to the network's data structures
    Synapse* syn_ptr = new_synapse.get();
    add_synapse(std::move(new_synapse));

    std::cout << "Created synapse from " << source_neuron_id << " to " << target_neuron_id << std::endl;

    return syn_ptr;
}