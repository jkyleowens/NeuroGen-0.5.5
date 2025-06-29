#include "NeuroGen/Neuron.h"

// Constructor now correctly initializes potential from the constants namespace
Neuron::Neuron(uint64_t id) :
    id(id),
    potential(NeuronModelConstants::RESTING_POTENTIAL),
    time_since_last_spike(100.0f), // Initialize to a large value
    spiked(false) {}

void Neuron::update(float dt, float total_input_current) {
    // Reset spike flag at the beginning of an update
    spiked = false;

    // If in refractory period, do nothing but advance time
    if (time_since_last_spike < NeuronModelConstants::ABSOLUTE_REFRACTORY_PERIOD) {
        time_since_last_spike += dt;
        return;
    }

    // Leaky integrate-and-fire model dynamics
    // dV/dt = (-(V - V_rest) + I*R) / tau
    float dV = dt * (-(potential - NeuronModelConstants::RESTING_POTENTIAL)
                     + total_input_current * NeuronModelConstants::MEMBRANE_RESISTANCE)
                     / NeuronModelConstants::MEMBRANE_TIME_CONSTANT;

    potential += dV;

    // Check for a spike
    if (potential >= NeuronModelConstants::SPIKE_THRESHOLD) {
        spiked = true;
        potential = NeuronModelConstants::RESET_POTENTIAL; // Reset potential
        time_since_last_spike = 0.0f; // Reset refractory timer
    }

    time_since_last_spike += dt;
}

bool Neuron::has_spiked() {
    return spiked;
}

uint64_t Neuron::get_id() const {
    return id;
}

float Neuron::get_potential() const {
    return potential;
}