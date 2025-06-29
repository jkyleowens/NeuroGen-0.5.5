#ifndef SYNAPSE_H
#define SYNAPSE_H

#include <string>
#include <algorithm> // For std::fill

/**
 * @brief Synaptic connection between neurons with plasticity tracking
 */
struct Synapse {
    // >>> FIX: Added missing ID member
    size_t id;
    // <<< END FIX

    size_t pre_neuron_id;
    size_t post_neuron_id;
    std::string post_compartment;
    size_t receptor_index;

    double weight;
    double base_weight;
    double axonal_delay;        // ms

    // Plasticity tracking
    double last_pre_spike;
    double last_post_spike;
    double eligibility_trace;
    double activity_metric;     // Running average of usage

    // Structural plasticity
    double formation_time;
    double last_potentiation;
    double strength_history[10]; // Sliding window for pruning decisions
    size_t history_index;

    // >>> FIX: Updated constructor to accept the new ID and all necessary parameters
    Synapse(size_t id, size_t pre_id, size_t post_id, const std::string& compartment,
            size_t receptor_idx, double w = 0.1, double delay = 1.0)
        : id(id), // Initialize the new ID
          pre_neuron_id(pre_id),
          post_neuron_id(post_id),
          post_compartment(compartment),
          receptor_index(receptor_idx),
          weight(w),
          base_weight(w),
          axonal_delay(delay),
          last_pre_spike(-1000.0),
          last_post_spike(-1000.0),
          eligibility_trace(0.0),
          activity_metric(0.0),
          formation_time(0.0),
          last_potentiation(0.0),
          history_index(0) {
        std::fill(std::begin(strength_history), std::end(strength_history), w);
    }
    // <<< END FIX
};

#endif // SYNAPSE_H