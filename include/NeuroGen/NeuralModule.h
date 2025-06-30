#ifndef NEURAL_MODULE_H
#define NEURAL_MODULE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map> // Include for std::unordered_map
#include <NeuroGen/Network.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/NetworkStats.h>

class NeuralModule {
public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Construct neural module with configuration
     * @param name Module name for identification
     * @param config Network configuration parameters
     */
    NeuralModule(const std::string& name, const NetworkConfig& config);
    
    /**
     * @brief Virtual destructor for polymorphic inheritance
     */
    virtual ~NeuralModule();
    
    /**
     * @brief Initialize the neural module with biological parameters
     * @return Success status of initialization
     */
    bool initialize();
    
    /**
     * @brief Validate module configuration and state
     * @return Validation success status
     */
    bool validate_configuration() const;
    
    // ========================================================================
    // CORE PROCESSING INTERFACE
    // ========================================================================
    
    /**
     * @brief Process input through the neural module
     * @param input Input vector to process
     * @return Processed output vector
     */
    virtual std::vector<float> process(const std::vector<float>& input);
    
    /**
     * @brief Update internal state with temporal dynamics
     * @param dt Time step for integration
     */
    virtual void update_state(float dt);
    
    /**
     * @brief Apply learning rules based on activity patterns
     * @param target_output Target output for supervised learning
     * @param learning_signal Learning modulation signal
     */
    virtual void apply_learning(const std::vector<float>& target_output, 
                               float learning_signal = 1.0f);
    
    // ========================================================================
    // BIOLOGICAL PARAMETER CONTROL
    // ========================================================================
    
    /**
     * @brief Set learning rate for synaptic plasticity
     * @param rate Learning rate value [0.0, 1.0]
     */
    void set_learning_rate(float rate);
    
    /**
     * @brief Enable or disable synaptic plasticity
     * @param enable Plasticity enable flag
     */
    void enable_plasticity(bool enable);
    
    /**
     * @brief Set homeostatic target activity level
     * @param target Target activity level (Hz)
     */
    void set_homeostatic_target(float target);
    
    /**
     * @brief Set intrinsic excitability level
     * @param excitability Excitability parameter [0.0, 2.0]
     */
    void set_excitability(float excitability);
    
    /**
     * @brief Configure noise parameters for biological realism
     * @param noise_amplitude Amplitude of background noise
     */
    void set_background_noise(float noise_amplitude);
    
    /**
     * @brief Set refractory period for spike generation
     * @param period Refractory period in milliseconds
     */
    void set_refractory_period(float period);
    
    // ========================================================================
    // STATE ACCESS AND MONITORING
    // ========================================================================
    
    /**
     * @brief Get current internal state vector
     * @return Copy of internal state
     */
    std::vector<float> get_internal_state() const;
    
    /**
     * @brief Get module name identifier
     * @return Module name string
     */
    const std::string& get_name() const;
    
    /**
     * @brief Get current average activity level
     * @return Average activity over recent time window
     */
    float get_average_activity() const;
    
    /**
     * @brief Get current firing rate
     * @return Instantaneous firing rate (Hz)
     */
    float get_firing_rate() const;
    
    /**
     * @brief Get learning rate parameter
     * @return Current learning rate
     */
    float get_learning_rate() const;
    
    /**
     * @brief Check if plasticity is enabled
     * @return Plasticity enable status
     */
    bool is_plasticity_enabled() const;
    
    /**
     * @brief Get detailed module statistics
     * @param stats Output vector for statistics
     */
    void get_module_statistics(std::vector<float>& stats) const;
    
    // ========================================================================
    // ADVANCED FEATURES
    // ========================================================================
    
    /**
     * @brief Apply neuromodulation (dopamine, acetylcholine, etc.)
     * @param dopamine_level Dopamine concentration
     * @param acetylcholine_level Acetylcholine concentration
     * @param serotonin_level Serotonin concentration
     */
    void apply_neuromodulation(float dopamine_level, float acetylcholine_level, 
                              float serotonin_level);
    
    /**
     * @brief Execute homeostatic regulation
     * @param dt Time step for regulation
     */
    void apply_homeostatic_regulation(float dt);
    
    /**
     * @brief Implement structural plasticity (connection growth/pruning)
     * @param growth_factor Growth signal strength
     * @param pruning_threshold Threshold for connection removal
     */
    void apply_structural_plasticity(float growth_factor, float pruning_threshold);
    
    /**
     * @brief Reset module to baseline state
     */
    virtual void reset_to_baseline();
    
    // ========================================================================
    // STATE PERSISTENCE
    // ========================================================================
    
    /**
     * @brief Save module state to file
     * @param filename Output filename
     * @return Success status
     */
    bool save_state(const std::string& filename) const;
    
    /**
     * @brief Load module state from file
     * @param filename Input filename
     * @return Success status
     */
    bool load_state(const std::string& filename);

private:
    // Core module properties
    std::string module_name_;
    NetworkConfig config_;
    
    // Neural state management
    std::vector<float> internal_state_;
    std::vector<float> activation_history_;
    std::vector<float> synaptic_weights_;
    std::vector<float> neuron_outputs_;
    
    // Learning and plasticity
    float learning_rate_;
    float plasticity_strength_;
    float homeostatic_target_;
    bool plasticity_enabled_;
    
    // Biological parameters
    float excitability_level_;
    float adaptation_current_;
    float background_noise_;
    float refractory_period_;
    
    // Performance metrics
    float average_activity_;
    float firing_rate_;
    float connection_strength_;
    float plasticity_events_;
    
    // Thread safety
    mutable std::mutex module_mutex_;
    
    // Initialization state
    bool is_initialized_;

protected:
    // ========================================================================
    // INTERNAL PROCESSING METHODS
    // ========================================================================
    
    /**
     * @brief Apply nonlinear activation function
     * @param input Raw input value
     * @return Activated output
     */
    virtual float apply_activation(float input) const;
    
    /**
     * @brief Update synaptic weights using STDP
     * @param pre_activity Presynaptic activity
     * @param post_activity Postsynaptic activity
     * @param dt Time step
     */
    virtual void update_synaptic_weights(const std::vector<float>& pre_activity,
                                       const std::vector<float>& post_activity, float dt);
    
    /**
     * @brief Initialize synaptic weights with biological distribution
     */
    virtual void initialize_synaptic_weights();
    
    /**
     * @brief Update activity history for temporal processing
     * @param current_activity Current activity level
     */
    virtual void update_activity_history(float current_activity);
    
    /**
     * @brief Compute firing rate from activity history
     * @return Computed firing rate
     */
    virtual float compute_firing_rate() const;
    
    /**
     * @brief Apply noise for biological realism
     * @param signal Input signal
     * @return Noisy signal
     */
    virtual float apply_biological_noise(float signal) const;
};

#endif // NEURAL_MODULE_H