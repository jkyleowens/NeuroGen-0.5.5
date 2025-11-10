// ============================================================================
// BIOLOGICAL NEURON MODULE IMPLEMENTATION
// File: src/BiologicalNeuronModule.cpp
// ============================================================================

#include "NeuroGen/BiologicalNeuronModule.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

BiologicalNeuronModule::BiologicalNeuronModule(
    const std::string& name,
    size_t num_neurons,
    const NetworkConfig& config)
    : name_(name),
      config_(config),
      current_time_(0.0f),
      rng_(std::random_device{}())
{
    neurons_.reserve(num_neurons);
    is_excitatory_.resize(num_neurons, true);
    firing_rate_history_.resize(num_neurons, 0.0f);
    spike_times_.resize(num_neurons, -1000.0f); // Long time ago
    input_currents_.resize(num_neurons, 0.0f);
}

bool BiologicalNeuronModule::initialize(float excitatory_ratio, float connection_probability) {
    std::cout << "🧠 Initializing biological module: " << name_ << std::endl;
    std::cout << "   Neurons: " << neurons_.capacity()
              << ", E/I ratio: " << excitatory_ratio << std::endl;

    // Create neurons with realistic parameter diversity
    std::uniform_real_distribution<float> param_noise(-0.1f, 0.1f);

    size_t num_excitatory = static_cast<size_t>(neurons_.capacity() * excitatory_ratio);

    for (size_t i = 0; i < neurons_.capacity(); ++i) {
        NeuronParams params;

        // Excitatory neurons (Regular Spiking)
        if (i < num_excitatory) {
            is_excitatory_[i] = true;
            params.a = 0.02f + param_noise(rng_) * 0.005f;
            params.b = 0.2f + param_noise(rng_) * 0.05f;
            params.c = -65.0f + param_noise(rng_) * 5.0f;
            params.d = 8.0f + param_noise(rng_) * 2.0f;
        }
        // Inhibitory neurons (Fast Spiking)
        else {
            is_excitatory_[i] = false;
            params.a = 0.1f + param_noise(rng_) * 0.02f;
            params.b = 0.2f + param_noise(rng_) * 0.05f;
            params.c = -65.0f + param_noise(rng_) * 5.0f;
            params.d = 2.0f + param_noise(rng_) * 0.5f;
        }

        neurons_.push_back(std::make_unique<Neuron>(i, params));
    }

    // Create synaptic connections
    createConnections(connection_probability);

    // Initialize synaptic weights
    initializeWeights();

    std::cout << "✅ Module initialized: " << synapses_.size() << " synapses created" << std::endl;

    return true;
}

// ============================================================================
// CORE PROCESSING
// ============================================================================

void BiologicalNeuronModule::update(
    float dt,
    const std::vector<float>& external_input,
    float reward_signal)
{
    current_time_ += dt;

    // 1. Apply external inputs
    if (!external_input.empty()) {
        for (size_t i = 0; i < std::min(external_input.size(), neurons_.size()); ++i) {
            input_currents_[i] += external_input[i];
        }
    }

    // 2. Compute synaptic currents from network activity
    computeSynapticCurrents();

    // 3. Update all neurons
    for (size_t i = 0; i < neurons_.size(); ++i) {
        neurons_[i]->update(dt, input_currents_[i]);

        // Record spike times for STDP
        if (neurons_[i]->has_spiked()) {
            spike_times_[i] = current_time_;
        }
    }

    // 4. Update spike traces for STDP
    updateSpikeTraces(dt);

    // 5. Apply STDP with dopamine modulation
    float dopamine = dopamine_level_ + reward_signal;
    dopamine = std::max(0.0f, std::min(1.0f, dopamine));
    applySTDP(dt, dopamine);

    // 6. Update short-term plasticity
    updateShortTermPlasticity(dt);

    // 7. Update firing rate estimates
    updateFiringRates(dt);

    // 8. Apply homeostatic plasticity (slower timescale)
    static int homeostasis_counter = 0;
    if (++homeostasis_counter >= 100) { // Apply every 100 steps
        applyHomeostasis(dt * 100, 5.0f);
        homeostasis_counter = 0;
    }

    // 9. Reset input currents for next step
    std::fill(input_currents_.begin(), input_currents_.end(), 0.0f);
}

std::vector<float> BiologicalNeuronModule::process(const std::vector<float>& input) {
    // Run one timestep with external input
    update(1.0f, input, 0.0f);

    // Return current spike outputs
    return getSpikeOutputs();
}

// ============================================================================
// LEARNING AND PLASTICITY
// ============================================================================

void BiologicalNeuronModule::applySTDP(float dt, float dopamine_level) {
    // STDP: Asymmetric Hebbian learning based on spike timing
    // ΔW = dopamine * (A_plus * exp(-Δt/τ_plus) - A_minus * exp(-Δt/τ_minus))

    for (auto& syn : synapses_) {
        if (!syn.is_plastic) continue;

        // Compute weight change based on spike traces
        float delta_w = 0.0f;

        // Pre-before-post: Potentiation
        delta_w += stdp_a_plus_ * syn.post_spike_trace * syn.pre_spike_trace;

        // Post-before-pre: Depression
        delta_w -= stdp_a_minus_ * syn.pre_spike_trace * syn.post_spike_trace;

        // Modulate by dopamine (reward signal)
        delta_w *= dopamine_level;

        // Update eligibility trace (for multi-step credit assignment)
        syn.eligibility_trace = 0.95f * syn.eligibility_trace + delta_w;

        // Apply weight change with bounds
        syn.weight += syn.eligibility_trace * dt * 0.01f;
        syn.weight = std::max(syn.weight_min, std::min(syn.weight_max, syn.weight));
    }
}

void BiologicalNeuronModule::applyHomeostasis(float dt, float target_rate) {
    // Homeostatic plasticity: Scale weights to maintain target activity

    float tau_homeo = homeo_tau_;
    float alpha = homeo_alpha_;

    for (size_t i = 0; i < neurons_.size(); ++i) {
        // Compute activity error
        float current_rate = firing_rate_history_[i];
        float error = target_rate - current_rate;

        // Scale all incoming synaptic weights
        for (auto& syn : synapses_) {
            if (syn.post_neuron_id == i) {
                // Slowly adjust weights to reach target activity
                float weight_adjustment = alpha * error * dt / tau_homeo;
                syn.weight += weight_adjustment;
                syn.weight = std::max(syn.weight_min, std::min(syn.weight_max, syn.weight));
            }
        }
    }
}

void BiologicalNeuronModule::updateShortTermPlasticity(float dt) {
    // Short-term plasticity: Synaptic facilitation and depression

    for (auto& syn : synapses_) {
        bool pre_spiked = neurons_[syn.pre_neuron_id]->has_spiked();

        if (pre_spiked) {
            // Facilitation: Increase release probability
            syn.facilitation += syn.utilization * (1.0f - syn.facilitation);

            // Depression: Decrease available resources
            syn.depression *= (1.0f - syn.utilization * syn.facilitation);

            // Compute released amount
            syn.released_amount = syn.weight * syn.facilitation * syn.depression;
        }

        // Recovery dynamics
        float tau_f = stp_tau_facil_;
        float tau_d = stp_tau_depress_;
        float tau_r = stp_tau_recover_;

        syn.facilitation += dt * (1.0f - syn.facilitation) / tau_f;
        syn.depression += dt * (1.0f - syn.depression) / tau_d;
        syn.available_resources += dt * (1.0f - syn.available_resources) / tau_r;

        // Clamp values
        syn.facilitation = std::max(1.0f, syn.facilitation);
        syn.depression = std::max(0.0f, std::min(1.0f, syn.depression));
        syn.available_resources = std::max(0.0f, std::min(1.0f, syn.available_resources));
    }
}

// ============================================================================
// STATE ACCESSORS
// ============================================================================

std::vector<float> BiologicalNeuronModule::getSpikeOutputs() const {
    std::vector<float> spikes(neurons_.size(), 0.0f);
    for (size_t i = 0; i < neurons_.size(); ++i) {
        spikes[i] = neurons_[i]->has_spiked() ? 1.0f : 0.0f;
    }
    return spikes;
}

std::vector<float> BiologicalNeuronModule::getMembranePotentials() const {
    std::vector<float> potentials(neurons_.size());
    for (size_t i = 0; i < neurons_.size(); ++i) {
        potentials[i] = neurons_[i]->get_potential();
    }
    return potentials;
}

std::vector<float> BiologicalNeuronModule::getFiringRates() const {
    return firing_rate_history_;
}

// ============================================================================
// NEUROMODULATION
// ============================================================================

void BiologicalNeuronModule::setNeuromodulators(
    float dopamine,
    float acetylcholine,
    float norepinephrine)
{
    dopamine_level_ = std::max(0.0f, std::min(1.0f, dopamine));
    acetylcholine_level_ = std::max(0.0f, std::min(1.0f, acetylcholine));
    norepinephrine_level_ = std::max(0.0f, std::min(1.0f, norepinephrine));

    // Acetylcholine enhances attention/learning
    stdp_a_plus_ = 0.01f * (1.0f + acetylcholine_level_);

    // Norepinephrine increases excitability
    // (Could modify neuron parameters dynamically here)
}

// ============================================================================
// INTERNAL METHODS
// ============================================================================

void BiologicalNeuronModule::computeSynapticCurrents() {
    // Compute synaptic input currents for all neurons

    for (const auto& syn : synapses_) {
        if (neurons_[syn.pre_neuron_id]->has_spiked()) {
            // Apply synaptic transmission with short-term plasticity
            float current = syn.released_amount;

            // Excitatory (positive) or inhibitory (negative)
            if (!syn.is_excitatory) {
                current = -current;
            }

            // Add to postsynaptic neuron's input
            input_currents_[syn.post_neuron_id] += current;
        }
    }
}

void BiologicalNeuronModule::updateSpikeTraces(float dt) {
    // Update STDP spike traces for all synapses

    for (auto& syn : synapses_) {
        // Decay existing traces
        syn.pre_spike_trace *= std::exp(-dt / stdp_tau_pre_);
        syn.post_spike_trace *= std::exp(-dt / stdp_tau_post_);

        // Add spike contributions
        if (neurons_[syn.pre_neuron_id]->has_spiked()) {
            syn.pre_spike_trace += 1.0f;
        }
        if (neurons_[syn.post_neuron_id]->has_spiked()) {
            syn.post_spike_trace += 1.0f;
        }
    }
}

void BiologicalNeuronModule::propagateSpikes() {
    // Handle synaptic delays (for future enhancement)
    // Currently immediate transmission
}

void BiologicalNeuronModule::updateFiringRates(float dt) {
    // Update running average of firing rates

    float tau_rate = 100.0f; // Time constant for rate estimation (ms)

    for (size_t i = 0; i < neurons_.size(); ++i) {
        float spike = neurons_[i]->has_spiked() ? 1.0f : 0.0f;

        // Exponential moving average
        firing_rate_history_[i] +=
            dt * (spike * (1000.0f / dt) - firing_rate_history_[i]) / tau_rate;
    }
}

void BiologicalNeuronModule::createConnections(float connection_probability) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    for (size_t pre = 0; pre < neurons_.size(); ++pre) {
        for (size_t post = 0; post < neurons_.size(); ++post) {
            // No self-connections
            if (pre == post) continue;

            // Probabilistic connectivity
            if (prob_dist(rng_) < connection_probability) {
                BiologicalSynapse syn;
                syn.pre_neuron_id = pre;
                syn.post_neuron_id = post;
                syn.is_excitatory = is_excitatory_[pre];

                // Set initial weight (will be initialized properly later)
                syn.weight = 1.0f;

                // Different STP parameters for E and I synapses
                if (syn.is_excitatory) {
                    syn.utilization = 0.5f;  // Moderate release probability
                } else {
                    syn.utilization = 0.7f;  // Higher for inhibitory (faster)
                }

                synapses_.push_back(syn);
            }
        }
    }
}

void BiologicalNeuronModule::initializeWeights() {
    std::normal_distribution<float> weight_dist(2.0f, 0.5f);

    for (auto& syn : synapses_) {
        // Excitatory synapses: positive weights
        if (syn.is_excitatory) {
            syn.weight = std::abs(weight_dist(rng_));
            syn.weight_min = 0.0f;
            syn.weight_max = 10.0f;
        }
        // Inhibitory synapses: stronger negative weights
        else {
            syn.weight = std::abs(weight_dist(rng_)) * 1.5f; // Stronger inhibition
            syn.weight_min = 0.0f;
            syn.weight_max = 15.0f;
        }
    }
}
