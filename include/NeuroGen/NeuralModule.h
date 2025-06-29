#ifndef NEURAL_MODULE_H
#define NEURAL_MODULE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map> // Include for std::unordered_map
#include "NeuroGen/Network.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/NetworkStats.h"

class NeuralModule {
public:
    NeuralModule(std::string name, const NetworkConfig& config);
    ~NeuralModule();

    void update(float dt, const std::vector<float>& inputs, float reward);
    void set_active(bool active);
    bool is_active() const;
    const std::string& get_name() const;
    std::vector<float> get_output() const;
    std::vector<float> get_neuron_potentials(const std::vector<size_t>& neuron_ids) const;
    NetworkStats get_stats() const;
    Network* get_network();

    // >>> FIX: Added functions to manage named neuron populations (ports).
    /**
     * @brief Registers a named group of neurons as an I/O port.
     * @param port_name The name of the port (e.g., "input", "output").
     * @param neuron_ids A vector of neuron IDs belonging to this port.
     */
    void register_neuron_port(const std::string& port_name, const std::vector<size_t>& neuron_ids);

    /**
     * @brief Retrieves the neuron IDs associated with a named port.
     * @param port_name The name of the port to look up.
     * @return A const reference to the vector of neuron IDs.
     */
    const std::vector<size_t>& get_neuron_population(const std::string& port_name) const;
    // <<< END FIX

private:
    std::string module_name_;
    bool active_;
    std::unique_ptr<Network> internal_network_;

    // >>> FIX: Added a map to store the named neuron ports.
    std::unordered_map<std::string, std::vector<size_t>> neuron_ports_;
    // <<< END FIX
};

#endif // NEURAL_MODULE_H