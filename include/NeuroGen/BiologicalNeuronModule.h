// ============================================================================
// BIOLOGICAL NEURON MODULE - REALISTIC SPIKING NEURAL NETWORK
// File: include/NeuroGen/BiologicalNeuronModule.h
// ============================================================================

#ifndef BIOLOGICAL_NEURON_MODULE_H
#define BIOLOGICAL_NEURON_MODULE_H

#include <vector>
#include <memory>
#include <string>
#include <random>
#include <NeuroGen/Neuron.h>
#include <NeuroGen/NetworkConfig.h>

/**
 * @brief Synaptic connection with biological properties
 *
 * Implements realistic synaptic dynamics including:
 * - Short-term plasticity (facilitation/depression)
 * - Neurotransmitter dynamics
 * - Spike-timing dependent plasticity (STDP)
 * - Homeostatic weight scaling
 */
struct BiologicalSynapse {
    size_t pre_neuron_id;           // Presynaptic neuron ID
    size_t post_neuron_id;          // Postsynaptic neuron ID

    // Synaptic weight and plasticity
    float weight;                    // Base synaptic weight
    float weight_min = 0.0f;        // Minimum weight (for stability)
    float weight_max = 10.0f;       // Maximum weight (for stability)

    // STDP variables
    float eligibility_trace = 0.0f;  // Eligibility trace for STDP
    float pre_spike_trace = 0.0f;    // Presynaptic spike trace
    float post_spike_trace = 0.0f;   // Postsynaptic spike trace

    // Short-term plasticity (STP)
    float facilitation = 1.0f;       // Facilitation variable (>= 1.0)
    float depression = 1.0f;         // Depression variable (0 to 1.0)
    float utilization = 0.5f;        // Baseline release probability

    // Neurotransmitter dynamics
    float available_resources = 1.0f; // Available neurotransmitter
    float released_amount = 0.0f;     // Currently released amount

    // Transmission delay (realistic axonal delay)
    float transmission_delay_ms = 1.0f;
    std::vector<std::pair<float, float>> spike_queue; // (time, amplitude) pairs

    // Connection type
    bool is_excitatory = true;
    bool is_plastic = true;          // Can this synapse learn?

    // Homeostatic scaling
    float target_activity = 0.1f;    // Target postsynaptic activity
    float activity_history = 0.0f;   // Sliding window of activity
};

/**
 * @brief Biologically Realistic Neural Module
 *
 * This module implements a truly biologically-inspired spiking neural network with:
 *
 * 1. **Izhikevich Spiking Neurons**
 *    - Realistic membrane dynamics
 *    - Various firing patterns (regular, bursting, fast-spiking, etc.)
 *    - Proper spike generation
 *
 * 2. **STDP Learning**
 *    - Spike-timing dependent plasticity
 *    - Asymmetric learning windows
 *    - Dopamine-modulated learning
 *
 * 3. **Synaptic Dynamics**
 *    - Short-term facilitation and depression
 *    - Neurotransmitter release and depletion
 *    - Realistic transmission delays
 *
 * 4. **Homeostatic Plasticity**
 *    - Activity-dependent weight scaling
 *    - Intrinsic excitability adaptation
 *    - Network stability mechanisms
 *
 * 5. **Neuromodulation**
 *    - Dopamine (reward/novelty)
 *    - Acetylcholine (attention)
 *    - Norepinephrine (arousal)
 */
class BiologicalNeuronModule {
public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================

    /**
     * @brief Construct biological neuron module
     * @param name Module identifier
     * @param num_neurons Number of neurons in module
     * @param config Network configuration
     */
    BiologicalNeuronModule(
        const std::string& name,
        size_t num_neurons,
        const NetworkConfig& config
    );

    /**
     * @brief Initialize neurons and synapses
     * @param excitatory_ratio Fraction of excitatory neurons (default: 0.8)
     * @param connection_probability Probability of connection between neurons
     * @return Success status
     */
    bool initialize(float excitatory_ratio = 0.8f, float connection_probability = 0.1f);

    // ========================================================================
    // CORE PROCESSING
    // ========================================================================

    /**
     * @brief Update all neurons and synapses for one time step
     * @param dt Time step in milliseconds
     * @param external_input External input currents (optional)
     * @param reward_signal Reward/dopamine signal for learning (optional)
     */
    void update(float dt, const std::vector<float>& external_input = {}, float reward_signal = 0.0f);

    /**
     * @brief Process input through network
     * @param input Input spike trains or currents
     * @return Output spike trains or firing rates
     */
    std::vector<float> process(const std::vector<float>& input);

    // ========================================================================
    // LEARNING AND PLASTICITY
    // ========================================================================

    /**
     * @brief Apply STDP learning rule to all plastic synapses
     * @param dt Time step
     * @param dopamine_level Dopamine modulation (0 to 1)
     */
    void applySTDP(float dt, float dopamine_level = 0.5f);

    /**
     * @brief Apply homeostatic plasticity to maintain network stability
     * @param dt Time step
     * @param target_rate Target average firing rate (Hz)
     */
    void applyHomeostasis(float dt, float target_rate = 5.0f);

    /**
     * @brief Update short-term plasticity for all synapses
     * @param dt Time step
     */
    void updateShortTermPlasticity(float dt);

    // ========================================================================
    // STATE ACCESSORS
    // ========================================================================

    /**
     * @brief Get current spike outputs (binary: 0 or 1)
     * @return Vector of spike states for all neurons
     */
    std::vector<float> getSpikeOutputs() const;

    /**
     * @brief Get current membrane potentials
     * @return Vector of potentials for all neurons
     */
    std::vector<float> getMembranePotentials() const;

    /**
     * @brief Get firing rates (averaged over time window)
     * @return Vector of firing rates (Hz) for all neurons
     */
    std::vector<float> getFiringRates() const;

    /**
     * @brief Get module name
     * @return Module identifier
     */
    std::string getName() const { return name_; }

    /**
     * @brief Get number of neurons
     * @return Neuron count
     */
    size_t getNumNeurons() const { return neurons_.size(); }

    // ========================================================================
    // NEUROMODULATION
    // ========================================================================

    /**
     * @brief Set neuromodulator levels
     * @param dopamine Dopamine level (0 to 1) - affects learning
     * @param acetylcholine Acetylcholine level (0 to 1) - affects attention
     * @param norepinephrine Norepinephrine level (0 to 1) - affects excitability
     */
    void setNeuromodulators(float dopamine, float acetylcholine, float norepinephrine);

private:
    // ========================================================================
    // MEMBER VARIABLES
    // ========================================================================

    std::string name_;
    NetworkConfig config_;

    // Neural components
    std::vector<std::unique_ptr<Neuron>> neurons_;
    std::vector<BiologicalSynapse> synapses_;
    std::vector<bool> is_excitatory_;  // Neuron types

    // State tracking
    std::vector<float> firing_rate_history_;  // For computing average rates
    std::vector<float> spike_times_;          // Last spike time for each neuron
    std::vector<float> input_currents_;       // Current synaptic inputs

    // STDP parameters
    float stdp_tau_pre_ = 20.0f;   // Presynaptic trace time constant (ms)
    float stdp_tau_post_ = 20.0f;  // Postsynaptic trace time constant (ms)
    float stdp_a_plus_ = 0.01f;    // Potentiation amplitude
    float stdp_a_minus_ = 0.01f;   // Depression amplitude

    // Short-term plasticity parameters
    float stp_tau_facil_ = 750.0f;    // Facilitation time constant (ms)
    float stp_tau_depress_ = 100.0f;  // Depression time constant (ms)
    float stp_tau_recover_ = 800.0f;  // Resource recovery time constant (ms)

    // Homeostatic parameters
    float homeo_tau_ = 10000.0f;      // Homeostatic time constant (ms)
    float homeo_alpha_ = 0.1f;        // Homeostatic learning rate

    // Neuromodulation levels
    float dopamine_level_ = 0.5f;
    float acetylcholine_level_ = 0.5f;
    float norepinephrine_level_ = 0.5f;

    // Simulation state
    float current_time_ = 0.0f;
    std::mt19937 rng_;

    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================

    /**
     * @brief Compute total synaptic current for each neuron
     */
    void computeSynapticCurrents();

    /**
     * @brief Update spike traces for STDP
     * @param dt Time step
     */
    void updateSpikeTraces(float dt);

    /**
     * @brief Propagate spikes through synapses with delays
     */
    void propagateSpikes();

    /**
     * @brief Update firing rate estimates
     * @param dt Time step
     */
    void updateFiringRates(float dt);

    /**
     * @brief Create random synaptic connections
     * @param connection_probability Probability of connection
     */
    void createConnections(float connection_probability);

    /**
     * @brief Initialize synaptic weights randomly
     */
    void initializeWeights();
};

#endif // BIOLOGICAL_NEURON_MODULE_H
