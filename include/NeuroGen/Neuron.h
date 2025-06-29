#ifndef NEURON_H
#define NEURON_H

#include "NeuroGen/cuda/NeuronModelConstants.h"
#include <vector>
#include <cstdint>

class Neuron {
public:
    // Default constructor using constants from the namespace
    Neuron(uint64_t id);

    // Updates the neuron's state over a single timestep
    void update(float dt, float total_input_current);

    // Checks if the neuron has spiked and resets its state
    bool has_spiked();

    // Getters for state variables
    uint64_t get_id() const;
    float get_potential() const;

private:
    uint64_t id;                // Unique identifier for the neuron
    float potential;            // Membrane potential (in mV)
    float time_since_last_spike; // Time elapsed since the last spike (in ms)
    bool spiked;                // Flag indicating if a spike occurred in the current step
};

#endif // NEURON_H