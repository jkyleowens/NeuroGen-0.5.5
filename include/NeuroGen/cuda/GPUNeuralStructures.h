#ifndef GPU_NEURAL_STRUCTURES_H
#define GPU_NEURAL_STRUCTURES_H

#include "NeuroGen/NeuralConstants.h"

// ============================================================================
// COMPLETE GPU NEURON STATE STRUCTURE
// ============================================================================

struct GPUNeuronState {
    // === CORE MEMBRANE DYNAMICS ===
    float V;                            // Membrane potential (mV)
    float u;                            // Recovery variable (Izhikevich)
    float I_syn[MAX_COMPARTMENTS];      // Synaptic currents per compartment
    float ca_conc[MAX_COMPARTMENTS];    // Calcium concentrations
    
    // === TIMING AND ACTIVITY ===
    float last_spike_time;              // Time of last spike
    float previous_spike_time;          // Previous spike time
    float average_firing_rate;          // Running average firing rate
    float average_activity;             // Average activity level
    float activity_level;               // CRITICAL FIX: Added missing member
    
    // === PLASTICITY AND ADAPTATION ===
    float excitability;                 // Intrinsic excitability
    float synaptic_scaling_factor;      // Global synaptic scaling
    float bcm_threshold;                // BCM learning threshold
    float plasticity_threshold;         // Plasticity induction threshold
    
    // === NEUROMODULATION ===
    float dopamine_concentration;       // Local dopamine level
    float acetylcholine_level;          // Local acetylcholine level
    float serotonin_level;              // Local serotonin level
    float norepinephrine_level;         // Local norepinephrine level
    
    // === ION CHANNELS ===
    float na_m, na_h;                   // Sodium channel states
    float k_n;                          // Potassium channel state
    float ca_channel_state;             // Calcium channel state
    float channel_expression[NUM_RECEPTOR_TYPES]; // CRITICAL FIX: Added missing array
    float channel_maturation[NUM_RECEPTOR_TYPES]; // CRITICAL FIX: Added missing array
    
    // === MULTI-COMPARTMENT SUPPORT ===
    float V_compartments[MAX_COMPARTMENTS];        // Compartment voltages
    int compartment_types[MAX_COMPARTMENTS];       // Compartment types
    int num_compartments;                          // Number of active compartments
    bool dendritic_spike[MAX_DENDRITIC_SPIKES];    // Dendritic spike states
    float dendritic_spike_time[MAX_DENDRITIC_SPIKES]; // Dendritic spike timing
    
    // === NETWORK PROPERTIES ===
    int neuron_type;                    // Neuron type (excitatory/inhibitory)
    int layer_id;                       // Cortical layer
    int column_id;                      // Cortical column
    int active;                         // Activity flag
    bool is_principal_cell;             // Principal vs interneuron
    
    // === SPATIAL PROPERTIES ===
    float position_x, position_y, position_z;     // 3D coordinates
    float orientation_theta;            // Orientation
    
    // === DEVELOPMENT ===
    int developmental_stage;            // Current development stage
    float maturation_factor;            // Maturation level [0,1]
    float birth_time;                   // Time of neurogenesis
    
    // === METABOLISM ===
    float energy_level;                 // Cellular energy
    float metabolic_demand;             // Energy demand
    float glucose_uptake;               // Glucose consumption rate
};

// ============================================================================
// ENHANCED GPU SYNAPSE STRUCTURE
// ============================================================================

struct GPUSynapse {
    // === CONNECTIVITY ===
    int pre_neuron_idx;                 // Presynaptic neuron index
    int post_neuron_idx;                // Postsynaptic neuron index
    int post_compartment;               // Target compartment
    int receptor_index;                 // Receptor type
    int active;                         // Activity flag
    
    // === SYNAPTIC PROPERTIES ===
    float weight;                       // Current weight
    float max_weight, min_weight;       // Weight bounds
    float delay;                        // Synaptic delay
    float effective_weight;             // Modulated weight
    
    // === PLASTICITY ===
    float eligibility_trace;            // Eligibility trace
    float plasticity_modulation;        // Plasticity modulation
    bool is_plastic;                    // CRITICAL FIX: Added missing member
    float learning_rate;                // Synapse-specific learning rate
    float metaplasticity_factor;        // Meta-plasticity scaling
    
    // === TIMING ===
    float last_pre_spike_time;          // Last presynaptic spike
    float last_post_spike_time;         // Last postsynaptic spike
    float last_active_time;             // Last activation time
    float activity_metric;              // Activity measure
    float last_potentiation;            // Last potentiation time
    
    // === NEUROMODULATION ===
    float dopamine_sensitivity;         // Dopamine sensitivity
    float acetylcholine_sensitivity;    // ACh sensitivity
    float serotonin_sensitivity;        // Serotonin sensitivity
    float dopamine_level;               // Local dopamine
    
    // === VESICLE DYNAMICS ===
    int vesicle_count;                  // Available vesicles
    float release_probability;          // Release probability
    float facilitation_factor;          // Short-term facilitation
    float depression_factor;            // Short-term depression
    
    // === CALCIUM DYNAMICS ===
    float presynaptic_calcium;          // Pre-synaptic calcium
    float postsynaptic_calcium;         // Post-synaptic calcium
    
    // === HOMEOSTASIS ===
    float homeostatic_scaling;          // Homeostatic scaling
    float target_activity;              // Target activity level
    
    // === BIOPHYSICS ===
    float conductance;                  // Synaptic conductance
    float reversal_potential;           // Reversal potential
    float time_constant_rise;           // Rise time constant
    float time_constant_decay;          // Decay time constant
    
    // === DEVELOPMENT ===
    int developmental_stage;            // Development stage
    float structural_stability;         // Resistance to pruning
    float growth_factor;                // Growth tendency
};

// ============================================================================
// FORWARD DECLARATIONS FOR COMPLEX TYPES
// ============================================================================

// Value function approximation structure
struct ValueFunction {
    float state_features[64];           // State feature representation
    float value_weights[64];            // Value function weights
    float state_value;                  // Current state value estimate
    float td_error;                     // Temporal difference error
    float learning_rate;                // Value function learning rate
    float eligibility_trace;            // Eligibility trace for TD learning
    int feature_dimensions;             // Number of active features
    bool is_active;                     // Whether this function is active
};

// Actor-critic learning structure
struct ActorCriticState {
    float policy_parameters[32];        // Policy parameters
    float action_probabilities[32];     // Action probabilities
    float action_preferences[32];       // Action preferences
    float action_eligibility[32];       // Action eligibility traces
    float state_value;                  // State value estimate
    float baseline_estimate;            // Baseline for advantage
    float advantage_estimate;           // Advantage estimate
    float exploration_bonus;            // Exploration bonus
    float uncertainty_estimate;         // Epistemic uncertainty
    int num_actions;                    // Number of possible actions
    bool is_learning;                   // Learning flag
};

// Curiosity-driven exploration system
struct CuriosityState {
    float novelty_detector[32];         // Novelty detection features
    float surprise_level;               // Current surprise level
    float familiarity_level;            // Familiarity with current state
    float information_gain;             // Expected information gain
    float competence_progress;          // Learning progress measure
    float mastery_level;                // Current mastery level
    float random_exploration;           // Random exploration drive
    float directed_exploration;         // Directed exploration drive
    float goal_exploration;             // Goal-directed exploration
    bool is_exploring;                  // Exploration flag
};

// Neural progenitor cell structure
struct NeuralProgenitor {
    // === PROGENITOR IDENTITY ===
    int progenitor_id;                  // Unique identifier
    int progenitor_type;                // Type of progenitor
    int developmental_stage;            // Current stage
    
    // === SPATIAL PROPERTIES ===
    float position_x, position_y, position_z;     // 3D coordinates
    float migration_vector_x, migration_vector_y, migration_vector_z; // Migration direction
    
    // === TEMPORAL PROPERTIES ===
    float birth_time;                   // Time of creation
    float differentiation_time;         // Time of differentiation
    float last_division_time;           // Last division time
    
    // === PROLIFERATION ===
    int division_count;                 // Number of divisions
    int max_divisions;                  // Maximum allowed divisions
    float division_probability;         // Probability of division
    bool can_divide;                    // Division capability
    
    // === DIFFERENTIATION ===
    float differentiation_probability;  // Probability of differentiation
    float excitatory_bias;              // Bias toward excitatory fate
    float inhibitory_bias;              // Bias toward inhibitory fate
    float interneuron_probability;      // Probability of interneuron fate
    
    // === ENVIRONMENTAL SENSING ===
    float local_activity_level;         // Local network activity
    float local_neuron_density;         // Local neuron density
    float growth_factor_concentration;  // Growth factor levels
    float competition_pressure;         // Competition from other cells
    
    // === MOLECULAR STATE ===
    float transcription_factors[8];     // Key transcription factors
    float growth_signals[4];            // Growth signaling molecules
    float apoptosis_signals[4];         // Cell death signals
    
    // === FATE SPECIFICATION ===
    int target_layer;                   // Target cortical layer
    int target_column;                  // Target cortical column
    int target_neuron_type;             // Target neuron type
    bool fate_committed;                // Whether fate is determined
    
    // === ACTIVITY STATE ===
    bool is_active;                     // Whether progenitor is active
    bool is_migrating;                  // Currently migrating
    bool is_differentiating;            // Currently differentiating
    bool marked_for_deletion;           // Scheduled for removal
};

// Additional forward declarations
struct DevelopmentalTrajectory;
struct SynapticProgenitor;
struct SynapticCompetition;
struct PruningAssessment;
struct CompetitiveElimination;
struct NeuralHomeostasis;
struct SynapticHomeostasis;
struct NetworkHomeostasis;
struct STDPRuleConfig;
struct NeurogenesisController;
struct SynaptogenesisController;
struct PruningController;
struct CoordinationController;
struct PlasticityState;
struct DopamineNeuron;

#endif // GPU_NEURAL_STRUCTURES_H