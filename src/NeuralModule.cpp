#include <NeuroGen/NeuralModule.h>

// --- Constructor ---
NeuralModule::NeuralModule(std::string name, const NetworkConfig& config)
    : module_name_(std::move(name)) {
    // Now that Network.h is included, std::make_unique has the full definition and works correctly.
    internal_network_ = std::make_unique<Network>(config);
}

// --- Core Simulation Methods ---
void NeuralModule::update(double dt) {
    if (internal_network_) {
        // This call is now valid because the compiler knows what 'step' is.
        internal_network_->step(dt);
    }
}

void NeuralModule::reset() {
    if (internal_network_) {
        // This call is now valid.
        internal_network_->reset();
    }
}

// --- Accessors ---
const std::string& NeuralModule::getName() const {
    return module_name_;
}

Network* NeuralModule::getNetwork() const {
    return internal_network_.get();
}

// --- Population and Port Management ---

const std::vector<size_t>& NeuralModule::getNeuronPopulation(const std::string& port_name) const {
    auto it = neuron_populations_.find(port_name);
    if (it != neuron_populations_.end()) {
        return it->second;
    }
    static const std::vector<size_t> empty_vector;
    return empty_vector;
}

void NeuralModule::addNeuronPopulation(const std::string& port_name, std::vector<size_t> neuron_ids) {
    neuron_populations_[port_name] = std::move(neuron_ids);
}

// --- External World Interaction ---

void NeuralModule::injectCurrentToPopulation(const std::string& port_name, const std::vector<double>& currents) {
    const auto& neuron_ids = getNeuronPopulation(port_name);
    if (neuron_ids.size() != currents.size()) {
        return;
    }
    for (size_t i = 0; i < neuron_ids.size(); ++i) {
        // This call is now valid.
        internal_network_->injectCurrent(neuron_ids[i], currents[i]);
    }
}

std::vector<double> NeuralModule::getPotentialsFromPopulation(const std::string& port_name) const {
    const auto& neuron_ids = getNeuronPopulation(port_name);
    std::vector<double> potentials;
    if (neuron_ids.empty()) {
        return potentials;
    }

    potentials.reserve(neuron_ids.size());
    for (const auto& neuron_id : neuron_ids) {
        // This call is now valid.
        auto neuron = internal_network_->getNeuron(neuron_id);
        if (neuron) {
            potentials.push_back(neuron->getPotential());
        }
    }
    return potentials;
}