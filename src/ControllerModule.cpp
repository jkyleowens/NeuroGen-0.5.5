// ============================================================================
// NEUROMODULATORY CONTROLLER MODULE IMPLEMENTATION
// File: src/ControllerModule.cpp
// ============================================================================

#include "NeuroGen/ControllerModule.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>

// ============================================================================
// NEUROMODULATOR STATE IMPLEMENTATION
// ============================================================================

void NeuromodulatorState::reset_to_defaults() {
    concentration = 0.3f;
    baseline_level = 0.3f;
    production_rate = 0.1f;
    degradation_rate = 0.05f;
    half_life = 10.0f;
    target_modules_sensitivity = 1.0f;
    peak_concentration = 0.3f;
    time_since_peak = 0.0f;
    concentration_history.clear();
    concentration_history.resize(100, baseline_level); // 100-step history
    
    // Set type-specific defaults
    switch (type) {
        case NeuromodulatorType::DOPAMINE:
            baseline_level = 0.25f;
            production_rate = 0.15f;
            half_life = 8.0f;
            break;
        case NeuromodulatorType::SEROTONIN:
            baseline_level = 0.4f;
            production_rate = 0.08f;
            half_life = 12.0f;
            break;
        case NeuromodulatorType::NOREPINEPHRINE:
            baseline_level = 0.2f;
            production_rate = 0.2f;
            half_life = 5.0f;
            break;
        case NeuromodulatorType::ACETYLCHOLINE:
            baseline_level = 0.3f;
            production_rate = 0.12f;
            half_life = 6.0f;
            break;
        case NeuromodulatorType::GABA:
            baseline_level = 0.5f;
            production_rate = 0.1f;
            half_life = 15.0f;
            break;
        case NeuromodulatorType::GLUTAMATE:
            baseline_level = 0.4f;
            production_rate = 0.14f;
            half_life = 7.0f;
            break;
        default:
            // Keep defaults
            break;
    }
    
    concentration = baseline_level;
    peak_concentration = baseline_level;
}

void NeuromodulatorState::update_dynamics(float dt) {
    // Update concentration history
    concentration_history.push_back(concentration);
    if (concentration_history.size() > 100) {
        concentration_history.erase(concentration_history.begin());
    }
    
    // Natural degradation
    float degradation = concentration * degradation_rate * dt;
    
    // Homeostatic production (stronger when below baseline)
    float homeostatic_drive = std::max(0.0f, baseline_level - concentration);
    float production = production_rate * (1.0f + homeostatic_drive * 2.0f) * dt;
    
    // Update concentration
    concentration = std::max(0.0f, std::min(1.0f, concentration - degradation + production));
    
    // Track peaks
    if (concentration > peak_concentration) {
        peak_concentration = concentration;
        time_since_peak = 0.0f;
    } else {
        time_since_peak += dt;
    }
}

void NeuromodulatorState::apply_stimulus(float intensity, float duration) {
    // Immediate concentration increase
    float boost = intensity * target_modules_sensitivity;
    concentration = std::min(1.0f, concentration + boost);
    
    // Temporary increase in production rate
    production_rate = std::min(1.0f, production_rate + intensity * 0.1f);
}

// ============================================================================
// CONTROLLER MODULE IMPLEMENTATION
// ============================================================================

ControllerModule::ControllerModule(const ControllerConfig& config)
    : config_(config)
    , simulation_time_(0.0f)
    , global_performance_trend_(0.0f)
    , system_coherence_level_(0.5f)
    , is_running_(false)
    , detailed_logging_enabled_(false) {
    
    initialize_neuromodulators();
    last_update_time_ = std::chrono::high_resolution_clock::now();
    
    if (detailed_logging_enabled_) {
        log_action("ControllerModule initialized");
    }
}

ControllerModule::~ControllerModule() {
    emergency_stop();
}

void ControllerModule::initialize_neuromodulators() {
    // Create all neuromodulator states
    neuromodulators_[NeuromodulatorType::DOPAMINE] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::DOPAMINE);
    neuromodulators_[NeuromodulatorType::SEROTONIN] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::SEROTONIN);
    neuromodulators_[NeuromodulatorType::NOREPINEPHRINE] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::NOREPINEPHRINE);
    neuromodulators_[NeuromodulatorType::ACETYLCHOLINE] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::ACETYLCHOLINE);
    neuromodulators_[NeuromodulatorType::GABA] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::GABA);
    neuromodulators_[NeuromodulatorType::GLUTAMATE] = 
        std::make_unique<NeuromodulatorState>(NeuromodulatorType::GLUTAMATE);
    
    // Set initial concentrations from config
    neuromodulators_[NeuromodulatorType::DOPAMINE]->concentration = config_.initial_dopamine_level;
    neuromodulators_[NeuromodulatorType::SEROTONIN]->concentration = config_.initial_serotonin_level;
    neuromodulators_[NeuromodulatorType::NOREPINEPHRINE]->concentration = config_.initial_norepinephrine_level;
    neuromodulators_[NeuromodulatorType::ACETYLCHOLINE]->concentration = config_.initial_acetylcholine_level;
    neuromodulators_[NeuromodulatorType::GABA]->concentration = config_.initial_gaba_level;
    neuromodulators_[NeuromodulatorType::GLUTAMATE]->concentration = config_.initial_glutamate_level;
}

// ============================================================================
// MODULE MANAGEMENT
// ============================================================================

void ControllerModule::register_module(const std::string& name, std::shared_ptr<NeuralModule> module) {
    std::lock_guard<std::mutex> lock(modules_mutex_);
    registered_modules_[name] = module;
    module_performance_history_[name] = 0.5f; // Start with neutral performance
    
    if (detailed_logging_enabled_) {
        log_action("Registered module: " + name);
    }
}

void ControllerModule::unregister_module(const std::string& name) {
    std::lock_guard<std::mutex> lock(modules_mutex_);
    registered_modules_.erase(name);
    module_performance_history_.erase(name);
    
    if (detailed_logging_enabled_) {
        log_action("Unregistered module: " + name);
    }
}

std::shared_ptr<NeuralModule> ControllerModule::get_module(const std::string& name) const {
    std::lock_guard<std::mutex> lock(modules_mutex_);
    auto it = registered_modules_.find(name);
    return (it != registered_modules_.end()) ? it->second : nullptr;
}

std::vector<std::string> ControllerModule::get_registered_modules() const {
    std::lock_guard<std::mutex> lock(modules_mutex_);
    std::vector<std::string> names;
    for (const auto& pair : registered_modules_) {
        names.push_back(pair.first);
    }
    return names;
}

// ============================================================================
// CORE CONTROL FUNCTIONS
// ============================================================================

void ControllerModule::update(float dt) {
    if (!is_running_) {
        is_running_ = true;
    }
    
    simulation_time_ += dt;
    
    // Update neuromodulator dynamics
    update_neuromodulator_dynamics(dt);
    
    // Process pending rewards and commands
    process_pending_rewards();
    execute_pending_commands();
    
    // Assess current system state
    assess_system_state();
    
    // Generate automatic responses
    generate_automatic_responses();
    
    // Update performance metrics
    update_performance_metrics();
    
    // Specialized neuromodulator functions
    dopamine_reward_prediction_update(RewardSignal{});
    serotonin_mood_regulation();
    norepinephrine_attention_modulation();
    acetylcholine_learning_enhancement();
    gaba_inhibitory_balance();
    glutamate_excitatory_drive();
    
    last_update_time_ = std::chrono::high_resolution_clock::now();
}

void ControllerModule::reset() {
    simulation_time_ = 0.0f;
    global_performance_trend_ = 0.0f;
    system_coherence_level_ = 0.5f;
    
    // Reset all neuromodulators
    for (auto& pair : neuromodulators_) {
        pair.second->reset_to_defaults();
    }
    
    // Clear queues
    while (!pending_rewards_.empty()) pending_rewards_.pop();
    while (!pending_commands_.empty()) pending_commands_.pop();
    
    action_history_.clear();
    
    log_action("ControllerModule reset");
}

void ControllerModule::emergency_stop() {
    is_running_ = false;
    
    // Apply immediate global inhibition
    apply_global_inhibition(0.9f);
    
    // Set all neuromodulators to safe baseline levels
    for (auto& pair : neuromodulators_) {
        pair.second->concentration = pair.second->baseline_level;
    }
    
    log_action("Emergency stop executed");
}

// ============================================================================
// NEUROMODULATOR MANAGEMENT
// ============================================================================

void ControllerModule::release_neuromodulator(NeuromodulatorType type, float intensity, 
                                             const std::string& target_module) {
    auto it = neuromodulators_.find(type);
    if (it != neuromodulators_.end()) {
        it->second->apply_stimulus(intensity);
        
        // Apply to specific module if specified
        if (!target_module.empty()) {
            auto module = get_module(target_module);
            if (module) {
                // Apply neuromodulator effects to the target module
                // This would involve calling module-specific neuromodulation methods
                apply_neuromodulator_to_module(module, type, intensity);
            }
        } else {
            // Apply to all modules
            std::lock_guard<std::mutex> lock(modules_mutex_);
            for (const auto& pair : registered_modules_) {
                apply_neuromodulator_to_module(pair.second, type, intensity);
            }
        }
        
        if (detailed_logging_enabled_) {
            log_action("Released " + to_string(type) + " (intensity: " + 
                      std::to_string(intensity) + ") to " + 
                      (target_module.empty() ? "all modules" : target_module));
        }
    }
}

void ControllerModule::set_baseline_level(NeuromodulatorType type, float level) {
    auto it = neuromodulators_.find(type);
    if (it != neuromodulators_.end()) {
        it->second->baseline_level = std::max(0.0f, std::min(1.0f, level));
        
        if (detailed_logging_enabled_) {
            log_action("Set " + to_string(type) + " baseline to " + std::to_string(level));
        }
    }
}

float ControllerModule::get_concentration(NeuromodulatorType type) const {
    auto it = neuromodulators_.find(type);
    return (it != neuromodulators_.end()) ? it->second->concentration : 0.0f;
}

std::unordered_map<NeuromodulatorType, float> ControllerModule::get_all_concentrations() const {
    std::unordered_map<NeuromodulatorType, float> concentrations;
    for (const auto& pair : neuromodulators_) {
        concentrations[pair.first] = pair.second->concentration;
    }
    return concentrations;
}

// ============================================================================
// REWARD SYSTEM
// ============================================================================

void ControllerModule::process_reward_signal(const RewardSignal& signal) {
    pending_rewards_.push(signal);
    
    // Immediate dopamine response for significant rewards
    if (signal.magnitude > 0.5f) {
        release_neuromodulator(NeuromodulatorType::DOPAMINE, 
                              signal.magnitude * 0.3f, 
                              signal.target_module);
    }
    
    if (detailed_logging_enabled_) {
        log_action("Received reward signal: " + to_string(signal.type) + 
                  " (magnitude: " + std::to_string(signal.magnitude) + ")");
    }
}

void ControllerModule::generate_intrinsic_reward(const std::string& module_name, float curiosity_level) {
    RewardSignal signal;
    signal.type = RewardSignalType::INTRINSIC_CURIOSITY;
    signal.magnitude = curiosity_level * config_.curiosity_drive_strength;
    signal.confidence = 0.8f; // High confidence in intrinsic rewards
    signal.temporal_delay = 0.0f; // Immediate
    signal.source_module = "ControllerModule";
    signal.target_module = module_name;
    signal.timestamp = std::chrono::high_resolution_clock::now();
    
    process_reward_signal(signal);
}

void ControllerModule::distribute_global_reward(float reward_magnitude, RewardSignalType type) {
    std::lock_guard<std::mutex> lock(modules_mutex_);
    
    for (const auto& pair : registered_modules_) {
        RewardSignal signal;
        signal.type = type;
        signal.magnitude = reward_magnitude;
        signal.confidence = 0.7f;
        signal.temporal_delay = 0.0f;
        signal.source_module = "ControllerModule";
        signal.target_module = pair.first;
        signal.timestamp = std::chrono::high_resolution_clock::now();
        
        process_reward_signal(signal);
    }
    
    if (detailed_logging_enabled_) {
        log_action("Distributed global reward: " + to_string(type) + 
                  " (magnitude: " + std::to_string(reward_magnitude) + ")");
    }
}

// ============================================================================
// ATTENTION AND COORDINATION
// ============================================================================

void ControllerModule::allocate_attention(const std::unordered_map<std::string, float>& attention_weights) {
    std::lock_guard<std::mutex> lock(modules_mutex_);
    
    for (const auto& weight_pair : attention_weights) {
        auto module = get_module(weight_pair.first);
        if (module) {
            // Increase norepinephrine and acetylcholine for focused modules
            float attention_strength = weight_pair.second;
            
            if (attention_strength > 0.5f) {
                release_neuromodulator(NeuromodulatorType::NOREPINEPHRINE, 
                                      attention_strength * 0.4f, weight_pair.first);
                release_neuromodulator(NeuromodulatorType::ACETYLCHOLINE, 
                                      attention_strength * 0.3f, weight_pair.first);
            }
        }
    }
    
    if (detailed_logging_enabled_) {
        log_action("Allocated attention across " + std::to_string(attention_weights.size()) + " modules");
    }
}

void ControllerModule::coordinate_module_activities() {
    std::lock_guard<std::mutex> lock(modules_mutex_);
    
    if (registered_modules_.size() < 2) return;
    
    // Calculate coordination needs
    float coordination_demand = calculate_attention_demand();
    
    if (coordination_demand > 0.6f) {
        // Increase serotonin for better coordination
        release_neuromodulator(NeuromodulatorType::SEROTONIN, coordination_demand * 0.3f);
        
        // Moderate GABA to reduce excessive competition
        release_neuromodulator(NeuromodulatorType::GABA, coordination_demand * 0.2f);
    }
    
    if (detailed_logging_enabled_) {
        log_action("Coordinated module activities (demand: " + std::to_string(coordination_demand) + ")");
    }
}

void ControllerModule::apply_global_inhibition(float strength) {
    release_neuromodulator(NeuromodulatorType::GABA, strength);
    
    if (detailed_logging_enabled_) {
        log_action("Applied global inhibition (strength: " + std::to_string(strength) + ")");
    }
}

// ============================================================================
// SPECIALIZED NEUROMODULATOR FUNCTIONS
// ============================================================================

void ControllerModule::dopamine_reward_prediction_update(const RewardSignal& signal) {
    // Implement reward prediction error learning
    float current_dopamine = get_concentration(NeuromodulatorType::DOPAMINE);
    float baseline_dopamine = neuromodulators_[NeuromodulatorType::DOPAMINE]->baseline_level;
    
    // Update baseline based on recent performance
    float avg_performance = calculate_overall_system_performance();
    if (avg_performance > 0.7f) {
        // Increase baseline for consistently good performance
        float new_baseline = std::min(0.5f, baseline_dopamine + 0.01f);
        set_baseline_level(NeuromodulatorType::DOPAMINE, new_baseline);
    } else if (avg_performance < 0.3f) {
        // Decrease baseline for poor performance
        float new_baseline = std::max(0.1f, baseline_dopamine - 0.01f);
        set_baseline_level(NeuromodulatorType::DOPAMINE, new_baseline);
    }
}

void ControllerModule::serotonin_mood_regulation() {
    float stress_level = calculate_stress_level();
    float current_serotonin = get_concentration(NeuromodulatorType::SEROTONIN);
    
    // Increase serotonin if stress is high or if coordination is needed
    if (stress_level > 0.6f || system_coherence_level_ < 0.4f) {
        float boost = (stress_level - 0.6f) * 0.2f;
        release_neuromodulator(NeuromodulatorType::SEROTONIN, boost);
    }
}

void ControllerModule::norepinephrine_attention_modulation() {
    float attention_demand = calculate_attention_demand();
    float learning_opportunity = calculate_learning_opportunity();
    
    // Increase norepinephrine for high attention demand or learning opportunities
    if (attention_demand > 0.5f || learning_opportunity > 0.6f) {
        float boost = std::max(attention_demand, learning_opportunity) * 0.3f;
        release_neuromodulator(NeuromodulatorType::NOREPINEPHRINE, boost);
    }
}

void ControllerModule::acetylcholine_learning_enhancement() {
    float learning_opportunity = calculate_learning_opportunity();
    
    // Increase acetylcholine when learning opportunities are high
    if (learning_opportunity > 0.4f) {
        release_neuromodulator(NeuromodulatorType::ACETYLCHOLINE, learning_opportunity * 0.4f);
    }
}

void ControllerModule::gaba_inhibitory_balance() {
    float current_gaba = get_concentration(NeuromodulatorType::GABA);
    float current_glutamate = get_concentration(NeuromodulatorType::GLUTAMATE);
    
    // Maintain excitation-inhibition balance
    float target_ratio = 0.8f; // GABA should be slightly lower than glutamate
    float current_ratio = (current_glutamate > 0.0f) ? current_gaba / current_glutamate : 1.0f;
    
    if (current_ratio < target_ratio - 0.2f) {
        // Need more inhibition
        release_neuromodulator(NeuromodulatorType::GABA, 0.1f);
    }
}

void ControllerModule::glutamate_excitatory_drive() {
    float system_activity = calculate_overall_system_performance();
    
    // Maintain appropriate excitatory drive
    if (system_activity < 0.3f) {
        // Boost excitation if system is too quiet
        release_neuromodulator(NeuromodulatorType::GLUTAMATE, 0.2f);
    }
}

// ============================================================================
// INTERNAL HELPER METHODS
// ============================================================================

void ControllerModule::update_neuromodulator_dynamics(float dt) {
    for (auto& pair : neuromodulators_) {
        pair.second->update_dynamics(dt);
    }
}

void ControllerModule::process_pending_rewards() {
    while (!pending_rewards_.empty()) {
        RewardSignal signal = pending_rewards_.front();
        pending_rewards_.pop();
        
        // Update module performance tracking
        if (!signal.target_module.empty()) {
            module_performance_history_[signal.target_module] = 
                0.9f * module_performance_history_[signal.target_module] + 0.1f * signal.magnitude;
        }
        
        // Update global performance trend
        global_performance_trend_ = 0.95f * global_performance_trend_ + 0.05f * signal.magnitude;
    }
}

void ControllerModule::execute_pending_commands() {
    while (!pending_commands_.empty()) {
        NeuromodulationCommand cmd = pending_commands_.front();
        pending_commands_.pop();
        
        release_neuromodulator(cmd.modulator_type, cmd.intensity, cmd.target_module);
        
        if (detailed_logging_enabled_) {
            log_action("Executed command: " + cmd.reasoning);
        }
    }
}

void ControllerModule::assess_system_state() {
    // Calculate system coherence
    std::lock_guard<std::mutex> lock(modules_mutex_);
    
    if (registered_modules_.empty()) {
        system_coherence_level_ = 0.0f;
        return;
    }
    
    float total_coherence = 0.0f;
    int active_modules = 0;
    
    for (const auto& pair : registered_modules_) {
        auto stats = pair.second->get_stats();
        if (stats.active_neuron_count > 0) {
            float module_coherence = std::min(1.0f, 
                static_cast<float>(stats.active_neuron_count) / stats.total_neurons * 2.0f);
            total_coherence += module_coherence;
            active_modules++;
        }
    }
    
    system_coherence_level_ = (active_modules > 0) ? total_coherence / active_modules : 0.0f;
}

void ControllerModule::generate_automatic_responses() {
    // Generate homeostatic responses
    if (should_trigger_homeostatic_response()) {
        auto commands = generate_coordinated_response();
        for (const auto& cmd : commands) {
            pending_commands_.push(cmd);
        }
    }
}

float ControllerModule::calculate_stress_level() const {
    // Stress based on imbalance and poor performance
    float performance_stress = std::max(0.0f, 0.5f - global_performance_trend_);
    float coherence_stress = std::max(0.0f, 0.5f - system_coherence_level_);
    return std::min(1.0f, performance_stress + coherence_stress);
}

float ControllerModule::calculate_attention_demand() const {
    // High demand when multiple modules are active
    std::lock_guard<std::mutex> lock(modules_mutex_);
    
    int active_modules = 0;
    for (const auto& pair : registered_modules_) {
        auto stats = pair.second->get_stats();
        if (stats.active_neuron_count > 5) {
            active_modules++;
        }
    }
    
    return std::min(1.0f, static_cast<float>(active_modules) / 3.0f);
}

float ControllerModule::calculate_learning_opportunity() const {
    // High opportunity when novel patterns or errors are detected
    float avg_reward_prediction_error = 0.0f;
    for (const auto& pair : reward_prediction_errors_) {
        avg_reward_prediction_error += std::abs(pair.second);
    }
    
    if (!reward_prediction_errors_.empty()) {
        avg_reward_prediction_error /= reward_prediction_errors_.size();
    }
    
    return std::min(1.0f, avg_reward_prediction_error * 2.0f);
}

bool ControllerModule::should_trigger_homeostatic_response() const {
    // Trigger if any neuromodulator is far from baseline
    for (const auto& pair : neuromodulators_) {
        float deviation = std::abs(pair.second->concentration - pair.second->baseline_level);
        if (deviation > 0.3f) {
            return true;
        }
    }
    
    // Trigger if system performance is poor
    return global_performance_trend_ < 0.3f || system_coherence_level_ < 0.3f;
}

void ControllerModule::log_action(const std::string& action) {
    action_history_.push_back("[" + std::to_string(simulation_time_) + "s] " + action);
    
    // Keep only recent history
    if (action_history_.size() > 1000) {
        action_history_.erase(action_history_.begin());
    }
}

// ============================================================================
// PRIVATE HELPER METHOD DECLARATIONS (to be implemented)
// ============================================================================

void ControllerModule::apply_neuromodulator_to_module(std::shared_ptr<NeuralModule> module, 
                                                     NeuromodulatorType type, float intensity) {
    // This method would apply neuromodulator effects to a specific module
    // Implementation depends on the NeuralModule interface
    // For now, we'll implement a placeholder
    
    if (!module) return;
    
    // Apply effects based on neuromodulator type
    switch (type) {
        case NeuromodulatorType::DOPAMINE:
            // Enhance learning rate and reward sensitivity
            break;
        case NeuromodulatorType::SEROTONIN:
            // Improve stability and reduce impulsivity
            break;
        case NeuromodulatorType::NOREPINEPHRINE:
            // Increase attention and arousal
            break;
        case NeuromodulatorType::ACETYLCHOLINE:
            // Enhance plasticity and memory formation
            break;
        case NeuromodulatorType::GABA:
            // Increase inhibition and reduce activity
            break;
        case NeuromodulatorType::GLUTAMATE:
            // Increase excitation and activity
            break;
        default:
            break;
    }
}

void ControllerModule::update_performance_metrics() {
    // Update module statistics history
    std::lock_guard<std::mutex> lock(modules_mutex_);
    
    for (const auto& pair : registered_modules_) {
        module_stats_history_[pair.first] = pair.second->get_stats();
    }
}

std::vector<NeuromodulationCommand> ControllerModule::generate_coordinated_response() {
    std::vector<NeuromodulationCommand> commands;
    
    // Generate commands to restore homeostasis
    for (const auto& pair : neuromodulators_) {
        float deviation = pair.second->concentration - pair.second->baseline_level;
        
        if (std::abs(deviation) > 0.2f) {
            NeuromodulationCommand cmd;
            cmd.target_module = ""; // Apply to all modules
            cmd.modulator_type = pair.first;
            cmd.intensity = -deviation * 0.5f; // Correct half the deviation
            cmd.duration = 5.0f;
            cmd.urgency = std::abs(deviation);
            cmd.reasoning = "Homeostatic correction for " + to_string(pair.first);
            
            commands.push_back(cmd);
        }
    }
    
    return commands;
}

// ============================================================================
// PUBLIC API METHODS (continued)
// ============================================================================

float ControllerModule::calculate_module_performance(const std::string& module_name) {
    auto it = module_performance_history_.find(module_name);
    return (it != module_performance_history_.end()) ? it->second : 0.0f;
}

float ControllerModule::calculate_overall_system_performance() {
    if (module_performance_history_.empty()) return 0.0f;
    
    float total = 0.0f;
    for (const auto& pair : module_performance_history_) {
        total += pair.second;
    }
    
    return total / module_performance_history_.size();
}

std::string ControllerModule::generate_status_report() {
    std::stringstream ss;
    
    ss << "=== ControllerModule Status Report ===\n";
    ss << "Simulation Time: " << std::fixed << std::setprecision(2) << simulation_time_ << "s\n";
    ss << "System Coherence: " << std::setprecision(3) << system_coherence_level_ << "\n";
    ss << "Performance Trend: " << global_performance_trend_ << "\n\n";
    
    ss << "Neuromodulator Concentrations:\n";
    for (const auto& pair : neuromodulators_) {
        ss << "  " << to_string(pair.first) << ": " 
           << std::setprecision(3) << pair.second->concentration 
           << " (baseline: " << pair.second->baseline_level << ")\n";
    }
    
    ss << "\nModule Performance:\n";
    for (const auto& pair : module_performance_history_) {
        ss << "  " << pair.first << ": " << std::setprecision(3) << pair.second << "\n";
    }
    
    ss << "\nRecent Actions:\n";
    int recent_count = std::min(5, static_cast<int>(action_history_.size()));
    for (int i = action_history_.size() - recent_count; i < action_history_.size(); ++i) {
        ss << "  " << action_history_[i] << "\n";
    }
    
    return ss.str();
}

// Advanced mode methods
void ControllerModule::enable_creative_mode(float intensity) {
    release_neuromodulator(NeuromodulatorType::DOPAMINE, intensity * 0.4f);
    release_neuromodulator(NeuromodulatorType::NOREPINEPHRINE, intensity * 0.3f);
    set_baseline_level(NeuromodulatorType::SEROTONIN, 0.6f); // Higher baseline for openness
    
    log_action("Creative mode enabled (intensity: " + std::to_string(intensity) + ")");
}

void ControllerModule::enable_focus_mode(const std::string& target_module, float intensity) {
    // Increase attention modulators for target
    release_neuromodulator(NeuromodulatorType::NOREPINEPHRINE, intensity * 0.5f, target_module);
    release_neuromodulator(NeuromodulatorType::ACETYLCHOLINE, intensity * 0.4f, target_module);
    
    // Reduce activity in other modules
    std::lock_guard<std::mutex> lock(modules_mutex_);
    for (const auto& pair : registered_modules_) {
        if (pair.first != target_module) {
            release_neuromodulator(NeuromodulatorType::GABA, intensity * 0.3f, pair.first);
        }
    }
    
    log_action("Focus mode enabled for " + target_module + " (intensity: " + std::to_string(intensity) + ")");
}

void ControllerModule::enable_exploration_mode(float curiosity_boost) {
    release_neuromodulator(NeuromodulatorType::DOPAMINE, curiosity_boost * 0.3f);
    release_neuromodulator(NeuromodulatorType::NOREPINEPHRINE, curiosity_boost * 0.4f);
    
    // Generate intrinsic rewards for all modules
    std::lock_guard<std::mutex> lock(modules_mutex_);
    for (const auto& pair : registered_modules_) {
        generate_intrinsic_reward(pair.first, curiosity_boost);
    }
    
    log_action("Exploration mode enabled (curiosity boost: " + std::to_string(curiosity_boost) + ")");
}

void ControllerModule::enable_consolidation_mode(float memory_strength) {
    release_neuromodulator(NeuromodulatorType::ACETYLCHOLINE, memory_strength * 0.5f);
    release_neuromodulator(NeuromodulatorType::SEROTONIN, memory_strength * 0.3f);
    
    // Reduce overall activity for consolidation
    apply_global_inhibition(0.2f);
    
    log_action("Consolidation mode enabled (memory strength: " + std::to_string(memory_strength) + ")");
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

std::string to_string(NeuromodulatorType type) {
    switch (type) {
        case NeuromodulatorType::DOPAMINE: return "Dopamine";
        case NeuromodulatorType::SEROTONIN: return "Serotonin";
        case NeuromodulatorType::NOREPINEPHRINE: return "Norepinephrine";
        case NeuromodulatorType::ACETYLCHOLINE: return "Acetylcholine";
        case NeuromodulatorType::GABA: return "GABA";
        case NeuromodulatorType::GLUTAMATE: return "Glutamate";
        case NeuromodulatorType::OXYTOCIN: return "Oxytocin";
        case NeuromodulatorType::ENDORPHINS: return "Endorphins";
        case NeuromodulatorType::CORTISOL: return "Cortisol";
        case NeuromodulatorType::ADENOSINE: return "Adenosine";
        default: return "Unknown";
    }
}

std::string to_string(RewardSignalType type) {
    switch (type) {
        case RewardSignalType::INTRINSIC_CURIOSITY: return "Intrinsic Curiosity";
        case RewardSignalType::EXTRINSIC_TASK: return "Extrinsic Task";
        case RewardSignalType::SOCIAL_COOPERATION: return "Social Cooperation";
        case RewardSignalType::EFFICIENCY_BONUS: return "Efficiency Bonus";
        case RewardSignalType::NOVELTY_DETECTION: return "Novelty Detection";
        case RewardSignalType::PREDICTION_ACCURACY: return "Prediction Accuracy";
        case RewardSignalType::HOMEOSTATIC_BALANCE: return "Homeostatic Balance";
        case RewardSignalType::CREATIVITY_BURST: return "Creativity Burst";
        default: return "Unknown";
    }
}
