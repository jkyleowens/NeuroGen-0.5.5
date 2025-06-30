#include "NeuroGen/Network.h"
#include "NeuroGen/NeuralModule.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR  
// ============================================================================

Network::Network(const NetworkConfig& config)
    : config_(config), module_(nullptr), random_engine_(std::random_device{}()) {
    
    std::cout << "🧠 Initializing breakthrough neural network..." << std::endl;
    std::cout << "   • Target neurons: " << config.hidden_size << std::endl;
    std::cout << "   • Structural plasticity: " << (config.enable_structural_plasticity ? "ENABLED" : "disabled") << std::endl;
    
    initialize_neurons();
    initialize_synapses();
    
    // Initialize network statistics
    stats_.active_neuron_count = neurons_.size();
    stats_.active_synapses = synapses_.size();
    stats_.total_synapses = synapses_.size();
    stats_.simulation_steps = 0;
    stats_.mean_firing_rate = 0.0f;
    stats_.network_entropy = 0.0f;
    
    std::cout << "✅ Network initialized with " << neurons_.size() << " neurons and " 
              << synapses_.size() << " synapses" << std::endl;
}

Network::~Network() = default;

// ============================================================================
// CORE SIMULATION INTERFACE
// ============================================================================

void Network::update(float dt, const std::vector<float>& input_currents, float reward) {
    // Update internal simulation state
    stats_.simulation_steps++;
    
    // Core neural dynamics
    update_neurons(dt, input_currents);
    update_synapses(dt, reward);
    
    // Advanced biological mechanisms
    if (config_.enable_structural_plasticity) {
        structural_plasticity();
    }
    
    // Update network statistics
    update_stats(dt);
}

std::vector<float> Network::get_output() const {
    std::vector<float> outputs;
    
    if (neurons_.empty()) {
        return outputs;
    }
    
    // Extract output from the last neurons in the network (output layer)
    size_t output_start = std::max(0, static_cast<int>(neurons_.size()) - config_.output_size);
    outputs.reserve(config_.output_size);
    
    for (size_t i = output_start; i < neurons_.size(); i++) {
        if (neurons_[i]) {
            // Use firing rate as output
            float firing_rate = calculateNeuronFiringRate(*neurons_[i]);
            outputs.push_back(firing_rate);
        } else {
            outputs.push_back(0.0f);
        }
    }
    
    return outputs;
}

void Network::reset() {
    std::cout << "🔄 Resetting neural network state..." << std::endl;
    
    // Reset all neurons
    for (auto& neuron : neurons_) {
        if (neuron) {
            // Reset neuron to default state
            neuron.reset(); // Assuming Neuron has a reset method
        }
    }
    
    // Reset all synapses
    for (auto& synapse : synapses_) {
        if (synapse) {
            synapse->eligibility_trace = 0.0;
            synapse->activity_metric = 0.0;
            synapse->last_pre_spike = -1000.0;
            synapse->last_post_spike = -1000.0;
        }
    }
    
    // Reset statistics
    stats_.simulation_steps = 0;
    stats_.mean_firing_rate = 0.0f;
    stats_.network_entropy = 0.0f;
    
    std::cout << "✅ Network reset completed" << std::endl;
}

// ============================================================================
// NETWORK CONSTRUCTION INTERFACE
// ============================================================================

void Network::add_neuron(std::unique_ptr<Neuron> neuron) {
    if (neuron) {
        size_t neuron_id = neuron->get_id();
        neuron_map_[neuron_id] = neuron.get();
        neurons_.push_back(std::move(neuron));
    }
}

void Network::add_synapse(std::unique_ptr<Synapse> synapse) {
    if (synapse) {
        size_t synapse_id = synapse->id;
        synapse_map_[synapse_id] = synapse.get();
        
        // Update connection maps
        outgoing_synapse_map_[synapse->pre_neuron_id].push_back(synapse.get());
        incoming_synapse_map_[synapse->post_neuron_id].push_back(synapse.get());
        
        synapses_.push_back(std::move(synapse));
    }
}

Synapse* Network::createSynapse(size_t source_neuron_id, size_t target_neuron_id, 
                               const std::string& type, int delay, float weight) {
    // Validate neuron IDs
    if (neuron_map_.find(source_neuron_id) == neuron_map_.end() ||
        neuron_map_.find(target_neuron_id) == neuron_map_.end()) {
        std::cerr << "Error: Invalid neuron IDs for synapse creation: " 
                  << source_neuron_id << " -> " << target_neuron_id << std::endl;
        return nullptr;
    }
    
    // Generate unique synapse ID
    size_t synapse_id = synapses_.size();
    
    // Determine receptor type and compartment based on synapse type
    size_t receptor_index = 0; // Default to excitatory
    std::string compartment = "soma"; // Default compartment
    
    if (type == "inhibitory") {
        receptor_index = 1;
        weight = -std::abs(weight); // Ensure inhibitory weights are negative
    } else {
        weight = std::abs(weight); // Ensure excitatory weights are positive
    }
    
    // Create synapse with biological parameters
    auto synapse = std::make_unique<Synapse>(
        synapse_id, source_neuron_id, target_neuron_id, compartment, 
        receptor_index, weight, static_cast<double>(delay)
    );
    
    Synapse* synapse_ptr = synapse.get();
    add_synapse(std::move(synapse));
    
    return synapse_ptr;
}

// ============================================================================
// NETWORK ACCESS INTERFACE
// ============================================================================

Neuron* Network::get_neuron(size_t neuron_id) const {
    auto it = neuron_map_.find(neuron_id);
    return (it != neuron_map_.end()) ? it->second : nullptr;
}

Synapse* Network::get_synapse(size_t synapse_id) const {
    auto it = synapse_map_.find(synapse_id);
    return (it != synapse_map_.end()) ? it->second : nullptr;
}

std::vector<Synapse*> Network::getOutgoingSynapses(size_t neuron_id) {
    auto it = outgoing_synapse_map_.find(neuron_id);
    return (it != outgoing_synapse_map_.end()) ? it->second : std::vector<Synapse*>();
}

std::vector<Synapse*> Network::getIncomingSynapses(size_t neuron_id) {
    auto it = incoming_synapse_map_.find(neuron_id);
    return (it != incoming_synapse_map_.end()) ? it->second : std::vector<Synapse*>();
}

void Network::set_module(NeuralModule* module) {
    module_ = module;
    std::cout << "🔗 Neural module association established" << std::endl;
}

NetworkStats Network::get_stats() const {
    return stats_;
}

// ============================================================================
// PRIVATE IMPLEMENTATION: INITIALIZATION
// ============================================================================

void Network::initialize_neurons() {
    std::cout << "🧬 Initializing " << config_.hidden_size << " neurons with biological diversity..." << std::endl;
    
    // Create neurobiologically diverse neuron population
    NeuronParams params;
    std::uniform_real_distribution<float> variability(-0.05f, 0.05f);
    
    for (size_t i = 0; i < config_.hidden_size; ++i) {
        // Add biological variability to parameters
        NeuronParams varied_params = params;
        varied_params.a += variability(random_engine_);
        varied_params.b += variability(random_engine_);
        varied_params.c += variability(random_engine_);
        varied_params.d += variability(random_engine_);
        
        add_neuron(std::make_unique<Neuron>(i, varied_params));
    }
    
    std::cout << "✅ Neurons initialized with biological parameter diversity" << std::endl;
}

void Network::initialize_synapses() {
    std::cout << "🔗 Initializing synaptic connections..." << std::endl;
    
    if (neurons_.empty()) {
        std::cout << "⚠️  No neurons available for synapse creation" << std::endl;
        return;
    }
    
    // Create initial random connectivity with biological constraints
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> weight_dist(config_.min_weight, config_.max_weight);
    std::uniform_int_distribution<int> delay_dist(1, 5);
    
    size_t target_synapses = std::min(static_cast<size_t>(config_.totalSynapses), 
                                     neurons_.size() * neurons_.size() / 10);
    
    size_t created_synapses = 0;
    for (size_t pre = 0; pre < neurons_.size() && created_synapses < target_synapses; pre++) {
        for (size_t post = 0; post < neurons_.size() && created_synapses < target_synapses; post++) {
            if (pre == post) continue; // No self-connections
            
            // Biological connection probability (distance-dependent)
            float connection_prob = calculateConnectionProbability(pre, post);
            
            if (prob_dist(random_engine_) < connection_prob) {
                std::string synapse_type = (prob_dist(random_engine_) < config_.exc_ratio) ? 
                                          "excitatory" : "inhibitory";
                float weight = weight_dist(random_engine_);
                int delay = delay_dist(random_engine_);
                
                if (createSynapse(pre, post, synapse_type, delay, weight)) {
                    created_synapses++;
                }
            }
        }
    }
    
    std::cout << "✅ Created " << created_synapses << " synaptic connections" << std::endl;
}

// ============================================================================
// PRIVATE IMPLEMENTATION: NEURAL DYNAMICS
// ============================================================================

void Network::update_neurons(float dt, const std::vector<float>& input_currents) {
    if (neurons_.empty()) return;
    
    // Apply external inputs to input neurons
    size_t input_neurons = std::min(input_currents.size(), neurons_.size());
    
    for (size_t i = 0; i < input_neurons; i++) {
        if (neurons_[i] && i < input_currents.size()) {
            // Apply input current (implementation depends on Neuron interface)
            // neurons_[i]->add_input_current(input_currents[i]);
        }
    }
    
    // Update all neurons
    for (auto& neuron : neurons_) {
        if (neuron) {
            // Calculate total synaptic input
            float total_input = calculateTotalSynapticInput(neuron->get_id());
            neuron->update(dt, total_input);
        }
    }
}

void Network::update_synapses(float dt, float reward) {
    if (synapses_.empty()) return;
    
    // Update synaptic dynamics and plasticity
    for (auto& synapse : synapses_) {
        if (synapse) {
            updateSynapticPlasticity(*synapse, dt, reward);
        }
    }
}

void Network::structural_plasticity() {
    if (!config_.enable_structural_plasticity) return;
    
    // Implement synaptic pruning and growth
    prune_synapses();
    grow_synapses();
}

void Network::update_stats(float dt) {
    // Update network activity statistics
    stats_.simulation_steps++;
    
    // Calculate mean firing rate
    float total_firing_rate = 0.0f;
    int active_neurons = 0;
    
    for (const auto& neuron : neurons_) {
        if (neuron) {
            float firing_rate = calculateNeuronFiringRate(*neuron);
            total_firing_rate += firing_rate;
            if (firing_rate > 0.1f) active_neurons++;
        }
    }
    
    stats_.mean_firing_rate = (neurons_.empty()) ? 0.0f : total_firing_rate / neurons_.size();
    stats_.active_neuron_count = active_neurons;
    stats_.neuron_activity_ratio = (neurons_.empty()) ? 0.0f : 
                                  static_cast<float>(active_neurons) / neurons_.size();
    
    // Calculate network entropy
    if (stats_.neuron_activity_ratio > 0.0f && stats_.neuron_activity_ratio < 1.0f) {
        float p = stats_.neuron_activity_ratio;
        stats_.network_entropy = -(p * std::log2(p) + (1.0f - p) * std::log2(1.0f - p));
    }
    
    // Update synapse statistics
    int active_synapses = 0;
    for (const auto& synapse : synapses_) {
        if (synapse && synapse->activity_metric > 0.01) {
            active_synapses++;
        }
    }
    stats_.active_synapses = active_synapses;
}

// ============================================================================
// PRIVATE IMPLEMENTATION: STRUCTURAL PLASTICITY
// ============================================================================

void Network::prune_synapses() {
    // Remove weak or inactive synapses
    auto it = std::remove_if(synapses_.begin(), synapses_.end(),
        [this](const std::unique_ptr<Synapse>& synapse) {
            return synapse && shouldPruneSynapse(*synapse);
        });
    
    size_t pruned_count = std::distance(it, synapses_.end());
    if (pruned_count > 0) {
        synapses_.erase(it, synapses_.end());
        std::cout << "🌿 Pruned " << pruned_count << " weak synapses" << std::endl;
        
        // Rebuild connection maps
        rebuild_connection_maps();
    }
}

void Network::grow_synapses() {
    // Create new synaptic connections in active regions
    size_t max_new_synapses = synapses_.size() / 100; // Limit growth to 1% per step
    size_t created = 0;
    
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> neuron_dist(0, neurons_.size() - 1);
    
    for (size_t attempt = 0; attempt < max_new_synapses * 10 && created < max_new_synapses; attempt++) {
        size_t pre_id = neuron_dist(random_engine_);
        size_t post_id = neuron_dist(random_engine_);
        
        if (pre_id != post_id && !synapseExists(pre_id, post_id)) {
            // Check if both neurons are sufficiently active
            if (isNeuronActive(pre_id) && isNeuronActive(post_id)) {
                float weight = 0.1f; // Small initial weight
                std::string type = (prob_dist(random_engine_) < config_.exc_ratio) ? 
                                  "excitatory" : "inhibitory";
                
                if (createSynapse(pre_id, post_id, type, 1, weight)) {
                    created++;
                }
            }
        }
    }
    
    if (created > 0) {
        std::cout << "🌱 Grew " << created << " new synapses" << std::endl;
    }
}

bool Network::shouldPruneSynapse(const Synapse& synapse) const {
    // Multi-criteria pruning decision
    bool is_weak = std::abs(synapse.weight) < config_.min_weight * 0.1;
    bool is_inactive = synapse.activity_metric < 0.001;
    bool is_old_and_unused = (synapse.activity_metric < 0.01) && 
                            (stats_.simulation_steps - synapse.formation_time > 10000);
    
    return is_weak && (is_inactive || is_old_and_unused);
}

// ============================================================================
// PRIVATE IMPLEMENTATION: UTILITY FUNCTIONS
// ============================================================================

float Network::calculateConnectionProbability(size_t pre_id, size_t post_id) const {
    // Simple distance-based connection probability
    float base_prob = 0.1f;
    float distance_factor = 1.0f / (1.0f + std::abs(static_cast<int>(post_id - pre_id)) / 10.0f);
    return base_prob * distance_factor;
}

float Network::calculateTotalSynapticInput(size_t neuron_id) const {
    float total_input = 0.0f;
    
    auto incoming = getIncomingSynapses(neuron_id);
    for (const auto& synapse : incoming) {
        if (synapse) {
            // Check if presynaptic neuron recently spiked
            auto pre_neuron = get_neuron(synapse->pre_neuron_id);
            if (pre_neuron && pre_neuron->has_spiked()) {
                total_input += synapse->weight;
            }
        }
    }
    
    return total_input;
}

float Network::calculateNeuronFiringRate(const Neuron& neuron) const {
    // Estimate firing rate from recent activity
    return neuron.has_spiked() ? 50.0f : 0.0f; // Simplified estimation
}

void Network::updateSynapticPlasticity(Synapse& synapse, float dt, float reward) {
    // Update activity metric
    auto pre_neuron = get_neuron(synapse.pre_neuron_id);
    auto post_neuron = get_neuron(synapse.post_neuron_id);
    
    if (pre_neuron && post_neuron) {
        bool pre_spiked = pre_neuron->has_spiked();
        bool post_spiked = post_neuron->has_spiked();
        
        // Update activity
        if (pre_spiked || post_spiked) {
            synapse.activity_metric = synapse.activity_metric * 0.999f + 0.001f;
        } else {
            synapse.activity_metric *= 0.9999f;
        }
        
        // Simple STDP rule
        if (pre_spiked && post_spiked) {
            synapse.weight += 0.01f * reward; // Reward-modulated plasticity
            synapse.weight = std::max(config_.min_weight, 
                           std::min(config_.max_weight, synapse.weight));
        }
    }
}

bool Network::synapseExists(size_t pre_id, size_t post_id) const {
    auto outgoing = outgoing_synapse_map_.find(pre_id);
    if (outgoing != outgoing_synapse_map_.end()) {
        for (const auto& synapse : outgoing->second) {
            if (synapse && synapse->post_neuron_id == post_id) {
                return true;
            }
        }
    }
    return false;
}

bool Network::isNeuronActive(size_t neuron_id) const {
    auto neuron = get_neuron(neuron_id);
    return neuron && calculateNeuronFiringRate(*neuron) > 1.0f;
}

void Network::rebuild_connection_maps() {
    // Clear existing maps
    outgoing_synapse_map_.clear();
    incoming_synapse_map_.clear();
    synapse_map_.clear();
    
    // Rebuild from current synapse list
    for (auto& synapse : synapses_) {
        if (synapse) {
            synapse_map_[synapse->id] = synapse.get();
            outgoing_synapse_map_[synapse->pre_neuron_id].push_back(synapse.get());
            incoming_synapse_map_[synapse->post_neuron_id].push_back(synapse.get());
        }
    }
}