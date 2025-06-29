#include "NeuroGen/Network.h"
#include "NeuroGen/NeuralModule.h" //
#include <iostream>
#include <numeric>

Network::Network(const NetworkConfig& config)
    : config_(config), module_(nullptr) {
    initialize_neurons(); //
    initialize_synapses(); //
}

Network::~Network() = default;

void Network::update(float dt, const std::vector<float>& input_currents, float reward) {
    update_neurons(dt, input_currents);
    update_synapses(dt, reward);
    // This member name was fixed in a previous step.
    if (config_.enable_structural_plasticity) {
        structural_plasticity();
    }
    update_stats(dt);
}

void Network::add_neuron(std::unique_ptr<Neuron> neuron) {
    if (neuron) {
        neuron_map_[neuron->get_id()] = neuron.get();
        neurons_.push_back(std::move(neuron)); //
    }
}

// ... other existing Network.cpp functions (add_synapse, get_neuron, etc.)

// >>> THE FIX IS HERE <<<
void Network::initialize_neurons() {
    // 1. Create a default set of neuron parameters for the host-side Neuron object.
    // In a more advanced simulation, these parameters could be read from the NetworkConfig.
    NeuronParams host_neuron_params;

    // 2. Loop through the number of neurons specified in the high-level config.
    // We use hidden_size as the total number of neurons in this network instance.
    for (size_t i = 0; i < config_.hidden_size; ++i) {
        // 3. Call make_unique with the correct constructor arguments: the ID and the params struct.
        add_neuron(std::make_unique<Neuron>(i, host_neuron_params));
    }
}
// >>> END OF FIX <<<

void Network::initialize_synapses() {
    // Synapse initialization logic
}

void Network::update_neurons(float dt, const std::vector<float>& input_currents) {
    // Neuron update logic
}

// ... other functions