#include <NeuroGen/TopologyGenerator.h>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>

TopologyGenerator::TopologyGenerator(const NetworkConfig& config, unsigned int seed)
    : config_(config), random_engine_(seed), next_neuron_id_(0), next_synapse_id_(0) {}

std::vector<NeuronModel> TopologyGenerator::createNeuronPopulation(
    int count, NeuronType type, const Position3D& position, float radius) {
    
    std::vector<NeuronModel> population;
    std::uniform_real_distribution<float> dist(-radius, radius);
    
    // Determine if some neurons should be hub neurons for inter-module communication
    int hub_count = static_cast<int>(count * 0.05f); // 5% of neurons are hubs
    
    for (int i = 0; i < count; ++i) {
        NeuronModel neuron;
        neuron.id = next_neuron_id_++;
        neuron.type = type;
        neuron.position = Position3D(
            position.x + dist(random_engine_),
            position.y + dist(random_engine_),
            position.z + dist(random_engine_)
        );
        neuron.population_key = ""; // Will be set by caller
        neuron.module_id = -1; // Will be set by module
        neuron.intrinsic_excitability = 1.0f;
        neuron.is_hub_neuron = (i < hub_count);
        
        population.push_back(neuron);
    }
    return population;
}

std::vector<GPUSynapse> TopologyGenerator::generate(
    const std::map<std::string, std::vector<NeuronModel>>& populations,
    const std::vector<ConnectionRule>& rules) {
    
    std::vector<GPUSynapse> all_synapses;
    
    // Process each connection rule
    for (const auto& rule : rules) {
        auto source_it = populations.find(rule.source_pop_key);
        auto target_it = populations.find(rule.target_pop_key);
        
        if (source_it == populations.end() || target_it == populations.end()) {
            std::cerr << "Warning: Population not found for rule connecting " 
                      << rule.source_pop_key << " to " << rule.target_pop_key << std::endl;
            continue;
        }
        
        const auto& source_pop = source_it->second;
        const auto& target_pop = target_it->second;
        
        switch (rule.type) {
            case ConnectionRule::Type::DISTANCE_DECAY:
                applyDistanceDecayRule(rule, source_pop, target_pop, all_synapses);
                break;
            case ConnectionRule::Type::PROBABILISTIC:
                applyProbabilisticRule(rule, source_pop, target_pop, all_synapses);
                break;
            case ConnectionRule::Type::FIXED_IN_DEGREE:
                applyFixedDegreeRule(rule, source_pop, target_pop, all_synapses, true);
                break;
            case ConnectionRule::Type::FIXED_OUT_DEGREE:
                applyFixedDegreeRule(rule, source_pop, target_pop, all_synapses, false);
                break;
        }
    }
    
    return all_synapses;
}

std::vector<GPUSynapse> TopologyGenerator::connectPopulations(
    const std::vector<NeuronModel>& source_pop,
    const std::vector<NeuronModel>& target_pop,
    const std::vector<ConnectionRule>& rules) {
    
    std::vector<GPUSynapse> synapses;
    for (const auto& rule : rules) {
        switch (rule.type) {
            case ConnectionRule::Type::DISTANCE_DECAY:
                applyDistanceDecayRule(rule, source_pop, target_pop, synapses);
                break;
            case ConnectionRule::Type::PROBABILISTIC:
                applyProbabilisticRule(rule, source_pop, target_pop, synapses);
                break;
            case ConnectionRule::Type::FIXED_IN_DEGREE:
                applyFixedDegreeRule(rule, source_pop, target_pop, synapses, true);
                break;
            case ConnectionRule::Type::FIXED_OUT_DEGREE:
                applyFixedDegreeRule(rule, source_pop, target_pop, synapses, false);
                break;
        }
    }
    return synapses;
}

void TopologyGenerator::applyDistanceDecayRule(
    const ConnectionRule& rule, 
    const std::vector<NeuronModel>& source_pop, 
    const std::vector<NeuronModel>& target_pop, 
    std::vector<GPUSynapse>& synapses) {
    
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (const auto& source_neuron : source_pop) {
        for (const auto& target_neuron : target_pop) {
            // Skip self-connections unless explicitly allowed
            if (source_neuron.id == target_neuron.id) continue;
            
            float distance = calculateDistance(source_neuron.position, target_neuron.position);
            
            // Skip if beyond maximum distance
            if (distance > rule.max_distance) continue;
            
            // Calculate connection probability with distance decay
            float probability = rule.probability * std::exp(-distance / rule.decay_rate);
            
            // Boost probability for hub neurons (for inter-module connectivity)
            if (source_neuron.is_hub_neuron || target_neuron.is_hub_neuron) {
                probability *= hub_neuron_connection_boost_;
            }
            
            // Check if neurons are from different modules
            if (source_neuron.module_id != target_neuron.module_id && 
                source_neuron.module_id != -1 && target_neuron.module_id != -1) {
                probability *= inter_module_connection_prob_;
            }
            
            if (prob_dist(random_engine_) < probability) {
                synapses.push_back(createSynapse(source_neuron.id, target_neuron.id, rule));
            }
        }
    }
}

void TopologyGenerator::applyProbabilisticRule(
    const ConnectionRule& rule,
    const std::vector<NeuronModel>& source_pop,
    const std::vector<NeuronModel>& target_pop,
    std::vector<GPUSynapse>& synapses) {
    
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (const auto& source_neuron : source_pop) {
        for (const auto& target_neuron : target_pop) {
            if (source_neuron.id == target_neuron.id) continue;
            
            float probability = rule.probability;
            
            // Adjust for hub neurons
            if (source_neuron.is_hub_neuron || target_neuron.is_hub_neuron) {
                probability *= hub_neuron_connection_boost_;
            }
            
            // Adjust for inter-module connections
            if (source_neuron.module_id != target_neuron.module_id && 
                source_neuron.module_id != -1 && target_neuron.module_id != -1) {
                probability *= inter_module_connection_prob_;
            }
            
            if (prob_dist(random_engine_) < probability) {
                synapses.push_back(createSynapse(source_neuron.id, target_neuron.id, rule));
            }
        }
    }
}

void TopologyGenerator::applyFixedDegreeRule(
    const ConnectionRule& rule,
    const std::vector<NeuronModel>& source_pop,
    const std::vector<NeuronModel>& target_pop,
    std::vector<GPUSynapse>& synapses,
    bool is_in_degree) {
    
    if (is_in_degree) {
        // Fixed in-degree: each target receives fixed number of inputs
        for (const auto& target_neuron : target_pop) {
            std::vector<int> source_indices;
            for (size_t i = 0; i < source_pop.size(); ++i) {
                if (source_pop[i].id != target_neuron.id) {
                    source_indices.push_back(i);
                }
            }
            
            // Randomly select sources
            std::shuffle(source_indices.begin(), source_indices.end(), random_engine_);
            int connections = std::min(rule.degree, static_cast<int>(source_indices.size()));
            
            for (int i = 0; i < connections; ++i) {
                const auto& source_neuron = source_pop[source_indices[i]];
                synapses.push_back(createSynapse(source_neuron.id, target_neuron.id, rule));
            }
        }
    } else {
        // Fixed out-degree: each source sends fixed number of outputs
        for (const auto& source_neuron : source_pop) {
            std::vector<int> target_indices;
            for (size_t i = 0; i < target_pop.size(); ++i) {
                if (target_pop[i].id != source_neuron.id) {
                    target_indices.push_back(i);
                }
            }
            
            // Randomly select targets
            std::shuffle(target_indices.begin(), target_indices.end(), random_engine_);
            int connections = std::min(rule.degree, static_cast<int>(target_indices.size()));
            
            for (int i = 0; i < connections; ++i) {
                const auto& target_neuron = target_pop[target_indices[i]];
                synapses.push_back(createSynapse(source_neuron.id, target_neuron.id, rule));
            }
        }
    }
}

float TopologyGenerator::calculateDistance(const Position3D& p1, const Position3D& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) +
                     std::pow(p1.y - p2.y, 2) +
                     std::pow(p1.z - p2.z, 2));
}

GPUSynapse TopologyGenerator::createSynapse(int pre_id, int post_id, const ConnectionRule& rule) {
    GPUSynapse synapse;
    
    // Set neuron indices (these will need to be mapped to actual indices later)
    synapse.pre_neuron_idx = pre_id;
    synapse.post_neuron_idx = post_id;
    
    // Set weight and delay
    synapse.weight = generateWeight(rule.weight_mean, rule.weight_std_dev);
    synapse.delay = generateDelay(rule.delay_min, rule.delay_max);
    
    // Set receptor type
    synapse.receptor_index = rule.receptor_type;
    
    // Initialize other fields
    synapse.last_active = 0.0f;
    synapse.last_pre_spike_time = -1000.0f;
    synapse.activity_metric = 0.0f;
    synapse.last_potentiation = 0.0f;
    synapse.post_compartment = 0; // Default to soma
    
    // Additional fields from GPUSynapse structure
    synapse.effective_weight = synapse.weight;
    synapse.max_weight = synapse.weight * 3.0f; // Allow growth up to 3x initial
    synapse.min_weight = 0.0f;
    synapse.last_post_spike_time = -1000.0f;
    synapse.receptor_weight_fraction = 1.0f;
    
    return synapse;
}

float TopologyGenerator::generateWeight(float mean, float std_dev) {
    std::normal_distribution<float> weight_dist(mean, std_dev);
    float weight = weight_dist(random_engine_);
    return std::max(0.0f, weight); // Ensure non-negative
}

float TopologyGenerator::generateDelay(float min_delay, float max_delay) {
    std::uniform_real_distribution<float> delay_dist(min_delay, max_delay);
    return delay_dist(random_engine_);
}

void TopologyGenerator::setNextNeuronId(int id) {
    next_neuron_id_ = id;
}

void TopologyGenerator::setNextSynapseId(int id) {
    next_synapse_id_ = id;
}

void TopologyGenerator::initializeSpatialGrid(const std::vector<NeuronModel>& all_neurons) {
    // Find spatial bounds
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();
    float max_z = std::numeric_limits<float>::min();
    
    for (const auto& neuron : all_neurons) {
        min_x = std::min(min_x, neuron.position.x);
        min_y = std::min(min_y, neuron.position.y);
        min_z = std::min(min_z, neuron.position.z);
        max_x = std::max(max_x, neuron.position.x);
        max_y = std::max(max_y, neuron.position.y);
        max_z = std::max(max_z, neuron.position.z);
    }
    
    // Calculate grid dimensions
    int grid_x = static_cast<int>((max_x - min_x) / grid_bin_size_) + 1;
    int grid_y = static_cast<int>((max_y - min_y) / grid_bin_size_) + 1;
    int grid_z = static_cast<int>((max_z - min_z) / grid_bin_size_) + 1;
    
    // Initialize grid
    spatial_grid_.resize(grid_x);
    for (auto& plane : spatial_grid_) {
        plane.resize(grid_y);
        for (auto& row : plane) {
            row.resize(grid_z);
        }
    }
    
    // Populate grid with neuron indices
    grid_neurons_ = all_neurons;
    for (size_t i = 0; i < all_neurons.size(); ++i) {
        const auto& neuron = all_neurons[i];
        int x = static_cast<int>((neuron.position.x - min_x) / grid_bin_size_);
        int y = static_cast<int>((neuron.position.y - min_y) / grid_bin_size_);
        int z = static_cast<int>((neuron.position.z - min_z) / grid_bin_size_);
        
        if (x >= 0 && x < grid_x && y >= 0 && y < grid_y && z >= 0 && z < grid_z) {
            spatial_grid_[x][y][z].push_back(i);
        }
    }
}

std::vector<int> TopologyGenerator::getNearbyNeurons(const NeuronModel& neuron, float radius) {
    std::vector<int> nearby_indices;
    
    // Calculate grid bounds to search
    int search_bins = static_cast<int>(radius / grid_bin_size_) + 1;
    
    // This is a simplified version - in a full implementation,
    // you would calculate the actual grid coordinates and search neighboring bins
    
    return nearby_indices;
}