#ifndef TOPOLOGY_GENERATOR_H
#define TOPOLOGY_GENERATOR_H

#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <string>
#include <map>
#include <NeuroGen/cuda/GPUNeuralStructures.h>
#include <NeuroGen/NetworkConfig.h>

// Forward declaration
class Network;

// 3D position structure for spatial organization
struct Position3D {
    float x, y, z;
    
    Position3D(float x_ = 0.0f, float y_ = 0.0f, float z_ = 0.0f) 
        : x(x_), y(y_), z(z_) {}
};

// Neuron types enum
enum class NeuronType {
    EXCITATORY,
    INHIBITORY,
    MODULATORY
};

// Define receptor type constants if not already defined
#ifndef RECEPTOR_AMPA
#define RECEPTOR_AMPA 0
#define RECEPTOR_NMDA 1
#define RECEPTOR_GABA_A 2
#define RECEPTOR_GABA_B 3
#endif

// Connection rule for topology generation
struct ConnectionRule {
    enum class Type {
        PROBABILISTIC,      // Simple probability-based connection
        DISTANCE_DECAY,     // Connection probability decays with distance
        FIXED_IN_DEGREE,    // Each target neuron receives a fixed number of inputs
        FIXED_OUT_DEGREE    // Each source neuron sends a fixed number of outputs
    };

    std::string source_pop_key; // Key for the source population of neurons
    std::string target_pop_key; // Key for the target population

    Type type = Type::PROBABILISTIC;
    float probability = 0.1f;         // For PROBABILISTIC type
    float max_distance = 150.0f;      // For DISTANCE_DECAY type
    float decay_rate = 50.0f;         // For DISTANCE_DECAY type (renamed from decay_constant)
    int degree = 10;                  // For FIXED_IN_DEGREE / FIXED_OUT_DEGREE

    // Synaptic properties for this rule
    float weight_mean = 0.1f;
    float weight_std_dev = 0.05f;
    float delay_min = 1.0f;           // ms
    float delay_max = 5.0f;           // ms
    int receptor_type = RECEPTOR_AMPA;
};

// Abstract representation of a neuron for topology generation
struct NeuronModel {
    int id;
    NeuronType type;
    Position3D position;
    std::string population_key;
    
    // Additional properties for modular organization
    int module_id;                    // Which module this neuron belongs to
    float intrinsic_excitability;     // Base excitability level
    bool is_hub_neuron;              // Whether this is a hub neuron for inter-module communication
};

class TopologyGenerator {
public:
    // Constructor matching the implementation
    TopologyGenerator(const NetworkConfig& config, unsigned int seed = std::random_device{}());
    
    // Destructor
    ~TopologyGenerator() = default;

    // Main function to generate synapses based on neuron populations and rules
    std::vector<GPUSynapse> generate(
        const std::map<std::string, std::vector<NeuronModel>>& populations,
        const std::vector<ConnectionRule>& rules
    );
    
    // Create a population of neurons
    std::vector<NeuronModel> createNeuronPopulation(
        int count, 
        NeuronType type, 
        const Position3D& position, 
        float radius
    );
    
    // Connect two populations based on rules
    std::vector<GPUSynapse> connectPopulations(
        const std::vector<NeuronModel>& source_pop,
        const std::vector<NeuronModel>& target_pop,
        const std::vector<ConnectionRule>& rules
    );
    
    // ID management methods
    void setNextNeuronId(int id);
    void setNextSynapseId(int id);
    
    // Get current IDs
    int getCurrentNeuronId() const { return next_neuron_id_; }
    int getCurrentSynapseId() const { return next_synapse_id_; }

private:
    // Spatial hashing grid for efficient distance-based connection generation
    using SpatialGrid = std::vector<std::vector<std::vector<std::vector<int>>>>;

    // Initialize spatial grid for efficient neighbor searches
    void initializeSpatialGrid(const std::vector<NeuronModel>& all_neurons);
    
    // Get neurons within a certain radius
    std::vector<int> getNearbyNeurons(const NeuronModel& neuron, float radius);

    // Rule-specific generation methods
    void applyDistanceDecayRule(
        const ConnectionRule& rule,
        const std::vector<NeuronModel>& source_pop,
        const std::vector<NeuronModel>& target_pop,
        std::vector<GPUSynapse>& synapses
    );

    void applyProbabilisticRule(
        const ConnectionRule& rule,
        const std::vector<NeuronModel>& source_pop,
        const std::vector<NeuronModel>& target_pop,
        std::vector<GPUSynapse>& synapses
    );
    
    void applyFixedDegreeRule(
        const ConnectionRule& rule,
        const std::vector<NeuronModel>& source_pop,
        const std::vector<NeuronModel>& target_pop,
        std::vector<GPUSynapse>& synapses,
        bool is_in_degree
    );
    
    // Helper to calculate 3D distance
    float calculateDistance(const Position3D& p1, const Position3D& p2);
    
    // Helper to create a single synapse
    GPUSynapse createSynapse(int pre_id, int post_id, const ConnectionRule& rule);
    
    // Generate weight from normal distribution
    float generateWeight(float mean, float std_dev);
    
    // Generate delay within specified range
    float generateDelay(float min_delay, float max_delay);

    // Member variables
    const NetworkConfig& config_;
    std::mt19937 random_engine_;
    int next_neuron_id_;
    int next_synapse_id_;
    
    // Spatial organization
    SpatialGrid spatial_grid_;
    std::vector<NeuronModel> grid_neurons_;
    float grid_bin_size_ = 50.0f; // Spatial bin size in micrometers
    
    // Module-specific connectivity parameters
    float inter_module_connection_prob_ = 0.05f;  // Probability of inter-module connections
    float hub_neuron_connection_boost_ = 2.0f;    // Boost factor for hub neuron connections
};

#endif // TOPOLOGY_GENERATOR_H