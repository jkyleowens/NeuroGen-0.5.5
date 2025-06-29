#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <memory>
#include <string>
#include "NeuroGen/Synapse.h"

// Forward declaration to break circular dependency
class Network;

enum class NeuronType {
    EXCITATORY,
    INHIBITORY
};

class Neuron {
public:
    Neuron(int id, NeuronType type, Network* parent_network);

    void update(float dt, float input_current);
    void addSynapse(const std::shared_ptr<Neuron>& target_neuron, double weight, const std::string& compartment, size_t receptor_idx);
    void reset();

    bool hasFired() const { return fired; }
    int getId() const { return id; }
    NeuronType getType() const { return type; }
    float getPotential() const { return potential; }
    const std::vector<Synapse>& getSynapses() const { return synapses; }

private:
    void initializeParameters();

    int id;
    NeuronType type;
    // The unused 'parent_network' pointer has been removed.

    // Izhikevich model parameters
    float potential;
    float recovery_variable;
    float izh_a, izh_b, izh_c, izh_d;

    bool fired;
    std::vector<Synapse> synapses;

    // Homeostatic plasticity
    float avg_firing_rate;
    float homeostatic_target;
};

#endif // NEURON_H