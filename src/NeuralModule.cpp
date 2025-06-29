#include "NeuroGen/NeuralModule.h"
#include <stdexcept>
#include <iostream>
#include <vector>

// (Constructor and other methods remain the same)
NeuralModule::NeuralModule(std::string name, const NetworkConfig& config)
    : module_name_(std::move(name)),
      active_(true) {
    internal_network_ = std::make_unique<Network>(config);
    if (internal_network_) {
        internal_network_->set_module(this);
    }
}

NeuralModule::~NeuralModule() = default;

void NeuralModule::update(float dt, const std::vector<float>& inputs, float reward) {
    if (!active_ || !internal_network_) {
        return;
    }
    internal_network_->update(dt, inputs, reward);
}
// (set_active, is_active, get_name, etc. remain the same)
void NeuralModule::set_active(bool active) {
    active_ = active;
}

bool NeuralModule::is_active() const {
    return active_;
}

const std::string& NeuralModule::get_name() const {
    return module_name_;
}

std::vector<float> NeuralModule::get_output() const {
    if (!internal_network_) {
        return {};
    }
    return internal_network_->get_output();
}

std::vector<float> NeuralModule::get_neuron_potentials(const std::vector<size_t>& neuron_ids) const {
    std::vector<float> potentials;
    if (!internal_network_) {
        return potentials;
    }
    potentials.reserve(neuron_ids.size());

    for (const auto& neuron_id : neuron_ids) {
        auto neuron = internal_network_->get_neuron(neuron_id);
        if (neuron) {
            potentials.push_back(neuron->get_potential());
        } else {
            potentials.push_back(0.0f);
            std::cerr << "Warning: Neuron with ID " << neuron_id << " not found in module " << module_name_ << std::endl;
        }
    }
    return potentials;
}

NetworkStats NeuralModule::get_stats() const {
    if (!internal_network_) {
        return {};
    }
    return internal_network_->get_stats();
}

Network* NeuralModule::get_network() {
    return internal_network_.get();
}


// >>> FIX: Implementation for the new port management functions.
void NeuralModule::register_neuron_port(const std::string& port_name, const std::vector<size_t>& neuron_ids) {
    neuron_ports_[port_name] = neuron_ids;
}

const std::vector<size_t>& NeuralModule::get_neuron_population(const std::string& port_name) const {
    auto it = neuron_ports_.find(port_name);
    if (it == neuron_ports_.end()) {
        throw std::runtime_error("Neuron port not found: " + port_name);
    }
    return it->second;
}
// <<< END FIX