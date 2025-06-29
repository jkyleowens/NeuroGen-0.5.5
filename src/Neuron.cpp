#include <NeuroGen/Neuron.h>
#include <NeuroGen/Network.h> // Network is still needed for the pointer type in the constructor
#include <NeuroGen/cuda/NeuronModelConstants.h> // Include for SPIKE_THRESHOLD

#include <iostream>
#include <stdexcept>

// --- Constructor ---
// The 'parent_network' pointer is received but no longer stored as a member variable,
// resolving the "unused private field" warning.
Neuron::Neuron(int id, NeuronType type, Network* parent_network)
    : id(id), type(type), fired(false) {
    if (!parent_network) {
        throw std::invalid_argument("Neuron must be associated with a parent network.");
    }
    initializeParameters();
    reset();
}

// --- Core Simulation Methods ---

void Neuron::update(float dt, float input_current) {
    // Reset fire status at the beginning of the update cycle.
    fired = false;

    // Izhikevich model equations:
    // v' = 0.04v^2 + 5v + 140 - u + I
    // u' = a(bv - u)
    potential += dt * (0.04f * potential * potential + 5.0f * potential + 140.0f - recovery_variable + input_current);
    recovery_variable += dt * (izh_a * (izh_b * potential - recovery_variable));

    // Check for spike condition.
    // This block is simplified to resolve the error and correctly model the after-spike reset.
    if (potential >= SPIKE_THRESHOLD) {
        fired = true;
        potential = izh_c;        // After-spike reset of the membrane potential.
        recovery_variable += izh_d; // After-spike reset of the recovery variable.
    }
}

// Corrected implementation to provide all required arguments to the Synapse constructor
void Neuron::addSynapse(const std::shared_ptr<Neuron>& target_neuron, double weight, const std::string& compartment, size_t receptor_idx) {
    if (!target_neuron) {
        std::cerr << "Warning: Attempted to add synapse to a null target neuron." << std::endl;
        return;
    }
    // Construct a Synapse with all required arguments. 'this->id' is the pre-synaptic neuron's ID.
    synapses.emplace_back(this->id, target_neuron->getId(), compartment, receptor_idx, weight);
}


void Neuron::reset() {
    // Reset to a resting state.
    potential = izh_c;
    recovery_variable = izh_b * potential;
    fired = false;
    avg_firing_rate = 0.0f;
}

// --- Private Methods ---

void Neuron::initializeParameters() {
    // Corrected the switch to use the unscoped NeuronType enum
    switch (type) {
        case NeuronType::EXCITATORY:
            // Regular Spiking (RS) neuron parameters
            izh_a = 0.02f;
            izh_b = 0.2f;
            izh_c = -65.0f;
            izh_d = 8.0f;
            break;
        case NeuronType::INHIBITORY:
            // Fast Spiking (FS) neuron parameters
            izh_a = 0.1f;
            izh_b = 0.2f;
            izh_c = -65.0f;
            izh_d = 2.0f;
            break;
    }
    // Set a default homeostatic target firing rate
    homeostatic_target = 5.0f;
}