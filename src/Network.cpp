#include <NeuroGen/Network.h>
#include <NeuroGen/Neuron.h>   // Required for std::shared_ptr<Neuron>
#include <NeuroGen/Synapse.h>  // Good practice to include for clarity, though not strictly needed for the fix
#include <stdexcept>
#include <iostream>
#include <algorithm>

// ====================================================================================
// CONSTRUCTOR / DESTRUCTOR
// ====================================================================================

/**
 * @brief Constructs a new Network instance.
 * @param config The configuration settings for this network module.
 */
Network::Network(const NetworkConfig& config) : config_(config), current_time_(0.0) {
    std::cout << "Network instance created." << std::endl;
}

/**
 * @brief Destructor for the Network class.
 */
Network::~Network() {
    std::cout << "Network instance destroyed." << std::endl;
}

// ====================================================================================
// NETWORK STRUCTURE MODIFICATION
// ====================================================================================

/**
 * @brief Adds a neuron to the network.
 * @param neuron A shared pointer to the neuron object to be added.
 * @param position The 3D spatial position of the neuron.
 * @return The ID of the newly added neuron.
 */
size_t Network::addNeuron(std::shared_ptr<Neuron> neuron, const Position3D& position) {
    if (!neuron) {
        throw std::invalid_argument("Cannot add a null neuron to the network.");
    }
    neurons_.push_back(neuron);
    neuron_positions_.push_back(position);
    return neurons_.size() - 1;
}

/**
 * @brief Creates a synaptic connection between two neurons.
 * @param pre_neuron_id The index of the source (presynaptic) neuron.
 * @param post_neuron_id The index of the target (postsynaptic) neuron.
 * @param post_compartment The dendritic compartment on the target neuron.
 * @param receptor_index The index of the receptor on the compartment.
 * @param weight The initial weight of the synapse.
 * @param delay The transmission delay of the synapse.
 * @return True if the synapse was created successfully, false otherwise.
 *
 * Fix: The function signature now exactly matches the declaration in Network.h,
 * removing the 'SynapseType' and 'innovation' parameters that caused the error.
 */
bool Network::createSynapse(size_t pre_neuron_id, size_t post_neuron_id, const std::string& post_compartment, size_t receptor_index, double weight, double delay) {
    if (pre_neuron_id >= neurons_.size() || post_neuron_id >= neurons_.size()) {
        return false; // Neuron index out of range
    }

    auto source_neuron = neurons_[pre_neuron_id];
    auto target_neuron = neurons_[post_neuron_id];

    // Based on the declaration in Neuron.h, the `addSynapse` method is called
    // on the source neuron.
    source_neuron->addSynapse(target_neuron, weight, post_compartment, receptor_index);

    // A more complete implementation would also add this synapse to the Network's
    // own tracking containers, e.g., synapses_, outgoing_synapses_, etc.

    return true;
}

// ====================================================================================
// SIMULATION AND STATE MANAGEMENT
// ====================================================================================

/**
 * @brief Advances the entire network simulation by one time step.
 * @param dt The duration of the simulation step.
 */
void Network::step(double dt) {
    for (auto& neuron : neurons_) {
        // The Neuron::update signature takes dt and an input current.
        // We pass 0.0 as a default external current for this step.
        neuron->update(dt, 0.0f);
    }
    current_time_ += dt;
}

/**
 * @brief Resets the internal state of all neurons in the network.
 */
void Network::reset() {
    for (auto& neuron : neurons_) {
        if (neuron) {
            neuron->reset();
        }
    }
    current_time_ = 0.0;
}

/**
 * @brief Injects an external current into a specific neuron for one time step.
 * @param neuron_id The index of the target neuron.
 * @param current The amount of current to inject.
 */
void Network::injectCurrent(size_t neuron_id, double current) {
    if (neuron_id >= neurons_.size()) {
        throw std::out_of_range("Neuron index out of range for current injection.");
    }
    // We can directly call the neuron's update method for a single step to apply the current.
    // A small, arbitrary delta-time is used here.
    neurons_[neuron_id]->update(0.01f, current);
}

// ====================================================================================
// DATA ACCESS
// ====================================================================================

/**
 * @brief Retrieves a specific neuron from the network.
 * @param neuron_id The index of the neuron to retrieve.
 * @return A shared pointer to the neuron, or nullptr if the ID is invalid.
 */
std::shared_ptr<Neuron> Network::getNeuron(size_t neuron_id) const {
    if (neuron_id >= neurons_.size()) {
        return nullptr;
    }
    return neurons_[neuron_id];
}