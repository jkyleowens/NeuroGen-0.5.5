#include <NeuroGen/ControllerModule.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <chrono>
#include <thread>
#include <cmath>

// ============================================================================
// NEUROMODULATOR CONTROLLER IMPLEMENTATION
// ============================================================================

NeuromodulatorController::NeuromodulatorController(const NetworkConfig& config)
    : global_arousal_(0.5f), cognitive_load_(0.0f) {
    
    // Initialize neuromodulator baseline states
    modulators_[DOPAMINE] = {0.4f, 0.1f, 0.05f, 0.4f, 0.1f, {}, {}};
    modulators_[SEROTONIN] = {0.6f, 0.08f, 0.04f, 0.6f, 0.12f, {}, {}};
    modulators_[NOREPINEPHRINE] = {0.3f, 0.15f, 0.08f, 0.3f, 0.15f, {}, {}};
    modulators_[ACETYLCHOLINE] = {0.5f, 0.12f, 0.06f, 0.5f, 0.1f, {}, {}};
    modulators_[GABA] = {0.7f, 0.05f, 0.02f, 0.7f, 0.05f, {}, {}};
    
    // Create specialized controller network
    NetworkConfig controller_config = config;
    controller_config.hidden_size = 128;  // Smaller controller network
    controller_config.num_layers = 3;
    controller_network_ = std::make_unique<Network>(controller_config);
}

void NeuromodulatorController::initialize() {
    if (controller_network_) {
        controller_network_->initialize();
        std::cout << "NeuromodulatorController: Initialized controller network with " 
                  << controller_network_->getNeuronCount() << " neurons" << std::endl;
    }
    
    // Initialize attention weights
    attention_weights_.resize(32, 1.0f);  // Equal initial attention
}

void NeuromodulatorController::update(float dt) {
    // Update neuromodulator dynamics
    for (int i = 0; i < NUM_MODULATORS; ++i) {
        ModulatorState& mod = modulators_[i];
        
        // Decay towards baseline
        float decay_force = (mod.baseline_level - mod.concentration) * mod.decay_constant * dt;
        
        // Release and reuptake dynamics
        float net_change = (mod.release_rate - mod.reuptake_rate) * dt + decay_force;
        mod.concentration += net_change;
        
        // Clamp to physiological range
        mod.concentration = std::clamp(mod.concentration, 0.0f, 2.0f);
    }
    
    // Update controller network
    if (controller_network_) {
        controller_network_->update(dt);
    }
    
    // Update global states
    global_arousal_ = (modulators_[NOREPINEPHRINE].concentration + 
                       modulators_[DOPAMINE].concentration) * 0.5f;
    cognitive_load_ = std::min(1.0f, cognitive_load_ * 0.99f);  // Decay cognitive load
}

void NeuromodulatorController::setTargetModulation(ModulatorType type, float target_level) {
    if (type >= 0 && type < NUM_MODULATORS) {
        modulators_[type].release_rate = std::max(0.0f, target_level - modulators_[type].concentration) * 0.5f;
    }
}

float NeuromodulatorController::getModulatorLevel(ModulatorType type) const {
    if (type >= 0 && type < NUM_MODULATORS) {
        return modulators_[type].concentration;
    }
    return 0.0f;
}

void NeuromodulatorController::setModuleActivation(const std::string& module_name, float activation) {
    module_activation_levels_[module_name] = std::clamp(activation, 0.0f, 1.0f);
}

float NeuromodulatorController::getModuleActivation(const std::string& module_name) const {
    auto it = module_activation_levels_.find(module_name);
    return (it != module_activation_levels_.end()) ? it->second : 0.0f;
}

void NeuromodulatorController::updateAttentionWeights(const std::vector<float>& context_input) {
    if (context_input.size() <= attention_weights_.size()) {
        // Simple attention mechanism based on context salience
        for (size_t i = 0; i < context_input.size(); ++i) {
            float salience = std::abs(context_input[i]);
            float modulation = getModulatorLevel(ACETYLCHOLINE);  // ACh modulates attention
            attention_weights_[i] = attention_weights_[i] * 0.9f + (salience * modulation) * 0.1f;
        }
    }
}

std::vector<float> NeuromodulatorController::generateModulationOutput() const {
    std::vector<float> output(NUM_MODULATORS);
    for (int i = 0; i < NUM_MODULATORS; ++i) {
        output[i] = modulators_[i].concentration;
    }
    return output;
}

// ============================================================================
// CONTROLLER MODULE IMPLEMENTATION
// ============================================================================

ControllerModule::ControllerModule(const std::string& name, const NetworkConfig& config)
    : EnhancedNeuralModule(name, config),
      cognitive_load_threshold_(0.8f),
      adaptive_processing_enabled_(true),
      is_processing_(false),
      shutdown_requested_(false) {
    
    // Initialize performance metrics
    performance_metrics_ = {};
    performance_metrics_.last_update = std::chrono::steady_clock::now();
    
    // Initialize memory system
    memory_system_ = std::make_unique<ContextualMemory>();
    memory_system_->working_memory_capacity = 7;  // Miller's magic number
    
    std::cout << "ControllerModule: Created with name '" << name << "'" << std::endl;
}

ControllerModule::~ControllerModule() {
    shutdown();
}

void ControllerModule::initialize() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Initialize base neural module
    EnhancedNeuralModule::initialize();
    
    // Initialize core components
    initializeNeuromodulatorController();
    initializeModularNetwork();
    initializeMemorySystem();
    setupDefaultModuleConnections();
    
    std::cout << "ControllerModule: Initialization complete" << std::endl;
}

void ControllerModule::update(double dt) {
    if (shutdown_requested_.load()) {
        return;
    }
    
    is_processing_.store(true);
    
    try {
        // Update base neural module
        EnhancedNeuralModule::update(dt);
        
        // Update core controller components
        processModuleUpdates(dt);
        processAttentionAllocation();
        processNeuromodulation();
        processInterModuleCommunication();
        processMemoryConsolidation();
        
        // Update performance metrics
        updatePerformanceMetrics();
        
        // Perform periodic system health checks
        static int health_check_counter = 0;
        if (++health_check_counter % 100 == 0) {
            performSystemHealthCheck();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ControllerModule::update error: " << e.what() << std::endl;
    }
    
    is_processing_.store(false);
}

void ControllerModule::shutdown() {
    shutdown_requested_.store(true);
    
    // Wait for current processing to complete
    while (is_processing_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Cleanup resources
    registered_modules_.clear();
    module_states_.clear();
    inter_module_signals_.clear();
    
    std::cout << "ControllerModule: Shutdown complete" << std::endl;
}

// ============================================================================
// MODULE MANAGEMENT AND ORCHESTRATION
// ============================================================================

void ControllerModule::registerModule(std::unique_ptr<NeuralModule> module) {
    if (!module) {
        std::cerr << "ControllerModule: Cannot register null module" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    const std::string module_name = module->get_name();
    
    // Initialize module state
    ModuleState state;
    state.module_name = module_name;
    state.is_active = false;
    state.activation_level = 0.0f;
    state.attention_weight = 1.0f;
    state.processing_load = 0.0f;
    state.specialization_index = 0.0f;
    
    module_states_[module_name] = state;
    registered_modules_[module_name] = std::move(module);
    
    // Register with modular network if available
    if (modular_network_) {
        modular_network_->add_module(std::make_unique<NeuralModule>(*registered_modules_[module_name]));
    }
    
    std::cout << "ControllerModule: Registered module '" << module_name << "'" << std::endl;
}

void ControllerModule::activateModule(const std::string& module_name, float activation_level) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    auto it = module_states_.find(module_name);
    if (it != module_states_.end()) {
        it->second.is_active = true;
        it->second.activation_level = std::clamp(activation_level, 0.0f, 1.0f);
        
        // Update neuromodulator controller
        if (neuromodulator_controller_) {
            neuromodulator_controller_->setModuleActivation(module_name, activation_level);
        }
        
        std::cout << "ControllerModule: Activated module '" << module_name 
                  << "' with level " << activation_level << std::endl;
    }
}

std::vector<float> ControllerModule::collect_inter_module_signals(const std::string& target_module) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    std::vector<float> collected_signals;
    collected_signals.reserve(64);  // Reserve space for efficiency
    
    // Collect signals directed to target module
    for (const auto& signal : inter_module_signals_) {
        if (signal.target_module == target_module || signal.target_module == "broadcast") {
            // Weight signals by strength and priority
            float signal_weight = signal.strength * (1.0f + 0.1f * signal.priority);
            
            for (float value : signal.data) {
                collected_signals.push_back(value * signal_weight);
            }
        }
    }
    
    // If no signals found, return baseline activation
    if (collected_signals.empty()) {
        collected_signals.resize(32, 0.1f);  // Baseline activation
    }
    
    return collected_signals;
}

void ControllerModule::distribute_module_output(const std::string& source_module, 
                                               const std::vector<float>& output_data) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Update module state
    auto it = module_states_.find(source_module);
    if (it != module_states_.end()) {
        it->second.output_signals = output_data;
    }
    
    // Create inter-module signals for connected modules
    for (const auto& [module_name, state] : module_states_) {
        if (module_name != source_module && state.is_active) {
            InterModuleSignal signal;
            signal.source_module = source_module;
            signal.target_module = module_name;
            signal.signal_type = "activation";
            signal.data = output_data;
            signal.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
            signal.strength = state.attention_weight;
            signal.priority = 1;
            
            inter_module_signals_.push_back(signal);
        }
    }
}

// ============================================================================
// DECISION AND ACTION SYSTEMS
// ============================================================================

std::vector<ControllerModule::BrowsingAction> ControllerModule::generate_action_candidates() {
    std::vector<BrowsingAction> candidates;
    candidates.reserve(10);
    
    // Generate click actions
    for (int i = 0; i < 3; ++i) {
        BrowsingAction action;
        action.type = BrowsingAction::CLICK;
        action.parameters.x = 100 + i * 50;
        action.parameters.y = 200 + i * 30;
        action.confidence = 0.7f + (i * 0.1f);
        action.expected_reward = 0.5f;
        action.element_id = i + 1;
        action.reasoning = "Click candidate " + std::to_string(i);
        candidates.push_back(action);
    }
    
    // Generate scroll actions
    BrowsingAction scroll_action;
    scroll_action.type = BrowsingAction::SCROLL;
    scroll_action.parameters.scroll_amount = 100.0f;
    scroll_action.confidence = 0.6f;
    scroll_action.expected_reward = 0.2f;
    scroll_action.reasoning = "Scroll to explore more content";
    candidates.push_back(scroll_action);
    
    // Generate wait action
    BrowsingAction wait_action;
    wait_action.type = BrowsingAction::WAIT;
    wait_action.parameters.wait_duration = 1.0f;
    wait_action.confidence = 0.8f;
    wait_action.expected_reward = 0.1f;
    wait_action.reasoning = "Wait for page to load";
    candidates.push_back(wait_action);
    
    return candidates;
}

std::vector<float> ControllerModule::evaluate_action_candidates(
    const std::vector<BrowsingAction>& candidates,
    const std::vector<MemoryTrace>& similar_episodes) {
    
    std::vector<float> action_values;
    action_values.reserve(candidates.size());
    
    for (const auto& action : candidates) {
        float base_value = action.expected_reward * action.confidence;
        
        // Adjust based on similar episodes
        float episodic_bonus = 0.0f;
        for (const auto& episode : similar_episodes) {
            if (episode.action_taken.type == action.type) {
                episodic_bonus += episode.reward_received * 0.1f;
            }
        }
        
        // Apply exploration bonus
        float exploration_bonus = 0.0f;
        if (neuromodulator_controller_) {
            float dopamine_level = neuromodulator_controller_->getModulatorLevel(
                NeuromodulatorController::DOPAMINE);
            exploration_bonus = dopamine_level * 0.2f;
        }
        
        float total_value = base_value + episodic_bonus + exploration_bonus;
        action_values.push_back(total_value);
    }
    
    return action_values;
}

ControllerModule::BrowsingAction ControllerModule::select_action_with_exploration(
    const std::vector<BrowsingAction>& candidates,
    const std::vector<float>& action_values) {
    
    if (candidates.empty()) {
        // Return default wait action
        BrowsingAction default_action;
        default_action.type = BrowsingAction::WAIT;
        default_action.parameters.wait_duration = 0.5f;
        default_action.confidence = 0.5f;
        default_action.reasoning = "Default fallback action";
        return default_action;
    }
    
    // Softmax selection with temperature based on exploration level
    float temperature = 1.0f;
    if (neuromodulator_controller_) {
        float norepinephrine = neuromodulator_controller_->getModulatorLevel(
            NeuromodulatorController::NOREPINEPHRINE);
        temperature = 0.5f + norepinephrine;  // Higher NE = more focused (lower temp)
    }
    
    std::vector<float> probabilities(action_values.size());
    float sum_exp = 0.0f;
    
    // Compute softmax probabilities
    for (size_t i = 0; i < action_values.size(); ++i) {
        probabilities[i] = std::exp(action_values[i] / temperature);
        sum_exp += probabilities[i];
    }
    
    // Normalize probabilities
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }
    
    // Sample from distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    float random_value = dis(gen);
    float cumulative_prob = 0.0f;
    
    for (size_t i = 0; i < probabilities.size(); ++i) {
        cumulative_prob += probabilities[i];
        if (random_value <= cumulative_prob) {
            return candidates[i];
        }
    }
    
    // Fallback to last candidate
    return candidates.back();
}

void ControllerModule::execute_action() {
    switch (selected_action_.type) {
        case BrowsingAction::CLICK:
            execute_click_action();
            break;
        case BrowsingAction::SCROLL:
            execute_scroll_action();
            break;
        case BrowsingAction::TYPE:
            execute_type_action();
            break;
        case BrowsingAction::NAVIGATE:
            execute_navigate_action();
            break;
        case BrowsingAction::WAIT:
            execute_wait_action();
            break;
        default:
            std::cerr << "ControllerModule: Unknown action type" << std::endl;
            break;
    }
    
    // Convert action to motor command and send
    std::vector<float> motor_command = convert_action_to_motor_command(selected_action_);
    sendMotorCommand(motor_command);
    
    // Update performance metrics
    performance_metrics_.total_actions++;
}

void ControllerModule::execute_click_action() {
    std::cout << "Executing CLICK action at (" << selected_action_.parameters.x 
              << ", " << selected_action_.parameters.y << ")" << std::endl;
    
    // Simulate click execution with motor module
    if (auto motor_module = getModule("motor")) {
        std::vector<float> click_input = {
            static_cast<float>(selected_action_.parameters.x),
            static_cast<float>(selected_action_.parameters.y),
            1.0f  // Click signal
        };
        motor_module->setInput("click_command", click_input);
    }
}

void ControllerModule::execute_scroll_action() {
    std::cout << "Executing SCROLL action with amount " 
              << selected_action_.parameters.scroll_amount << std::endl;
    
    if (auto motor_module = getModule("motor")) {
        std::vector<float> scroll_input = {
            0.0f,  // x position
            selected_action_.parameters.scroll_amount,
            2.0f   // Scroll signal
        };
        motor_module->setInput("scroll_command", scroll_input);
    }
}

void ControllerModule::execute_type_action() {
    std::cout << "Executing TYPE action: '" << selected_action_.parameters.text << "'" << std::endl;
    
    if (auto motor_module = getModule("motor")) {
        // Convert text to motor commands (simplified)
        std::vector<float> type_input;
        for (char c : selected_action_.parameters.text) {
            type_input.push_back(static_cast<float>(c));
        }
        motor_module->setInput("type_command", type_input);
    }
}

void ControllerModule::execute_navigate_action() {
    std::cout << "Executing NAVIGATE action to: " << selected_action_.parameters.url << std::endl;
    
    if (auto perception_module = getModule("perception")) {
        std::vector<float> nav_input = {4.0f};  // Navigation signal
        perception_module->setInput("navigation_command", nav_input);
    }
}

void ControllerModule::execute_wait_action() {
    std::cout << "Executing WAIT action for " << selected_action_.parameters.wait_duration 
              << " seconds" << std::endl;
    
    // Implement wait by reducing system activity
    if (neuromodulator_controller_) {
        neuromodulator_controller_->setTargetModulation(
            NeuromodulatorController::GABA, 0.8f);  // Increase inhibition
    }
}

std::vector<float> ControllerModule::convert_action_to_motor_command(const BrowsingAction& action) {
    std::vector<float> motor_command;
    motor_command.reserve(8);
    
    // Encode action type
    motor_command.push_back(static_cast<float>(action.type));
    
    // Encode parameters based on action type
    switch (action.type) {
        case BrowsingAction::CLICK:
            motor_command.push_back(static_cast<float>(action.parameters.x));
            motor_command.push_back(static_cast<float>(action.parameters.y));
            motor_command.push_back(1.0f);  // Click force
            break;
            
        case BrowsingAction::SCROLL:
            motor_command.push_back(0.0f);  // x component
            motor_command.push_back(action.parameters.scroll_amount);
            motor_command.push_back(0.5f);  // Scroll speed
            break;
            
        case BrowsingAction::WAIT:
            motor_command.push_back(action.parameters.wait_duration);
            motor_command.push_back(0.0f);
            motor_command.push_back(0.0f);
            break;
            
        default:
            motor_command.resize(4, 0.0f);
            break;
    }
    
    // Add confidence and expected reward
    motor_command.push_back(action.confidence);
    motor_command.push_back(action.expected_reward);
    
    return motor_command;
}

// ============================================================================
// ATTENTION AND CONTROL MECHANISMS
// ============================================================================

void ControllerModule::updateAttentionMechanism(const std::vector<float>& sensory_input) {
    if (neuromodulator_controller_) {
        neuromodulator_controller_->updateAttentionWeights(sensory_input);
    }
    
    // Update attention allocation based on module activity
    attention_allocation_.resize(module_states_.size());
    size_t idx = 0;
    
    for (const auto& [module_name, state] : module_states_) {
        float attention_demand = state.processing_load * state.activation_level;
        attention_allocation_[idx] = attention_demand;
        ++idx;
    }
    
    // Normalize attention weights
    float total_attention = std::accumulate(attention_allocation_.begin(), 
                                          attention_allocation_.end(), 0.0f);
    if (total_attention > 0.0f) {
        for (float& weight : attention_allocation_) {
            weight /= total_attention;
        }
    }
}

void ControllerModule::allocateAttention(const std::map<std::string, float>& attention_demands) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    float total_demand = 0.0f;
    for (const auto& [module_name, demand] : attention_demands) {
        total_demand += demand;
    }
    
    // Normalize and apply attention weights
    for (const auto& [module_name, demand] : attention_demands) {
        auto it = module_states_.find(module_name);
        if (it != module_states_.end()) {
            float normalized_attention = (total_demand > 0.0f) ? (demand / total_demand) : 0.0f;
            it->second.attention_weight = normalized_attention;
        }
    }
}

// ============================================================================
// LEARNING AND MEMORY SYSTEMS
// ============================================================================

void ControllerModule::updateWorkingMemory(const MemoryTrace& trace) {
    if (!memory_system_) return;
    
    // Add to working memory
    memory_system_->working_memory.push(trace);
    
    // Maintain capacity limit
    while (memory_system_->working_memory.size() > memory_system_->working_memory_capacity) {
        memory_system_->working_memory.pop();
    }
}

std::vector<ControllerModule::MemoryTrace> ControllerModule::retrieveSimilarEpisodes(
    const std::vector<float>& current_state) {
    
    std::vector<MemoryTrace> similar_episodes;
    
    if (!memory_system_) return similar_episodes;
    
    // Simple similarity search in episodic memory
    for (const auto& [context, episodes] : memory_system_->episodic_memories) {
        for (const auto& episode : episodes) {
            // Compute similarity based on state vector distance
            float similarity = 0.0f;
            if (episode.state_vector.size() == current_state.size()) {
                float distance = 0.0f;
                for (size_t i = 0; i < current_state.size(); ++i) {
                    float diff = current_state[i] - episode.state_vector[i];
                    distance += diff * diff;
                }
                similarity = 1.0f / (1.0f + std::sqrt(distance));
            }
            
            // Include if similarity is above threshold
            if (similarity > 0.7f) {
                similar_episodes.push_back(episode);
            }
        }
    }
    
    // Sort by similarity (importance weight) and limit results
    std::sort(similar_episodes.begin(), similar_episodes.end(),
              [](const MemoryTrace& a, const MemoryTrace& b) {
                  return a.importance_weight > b.importance_weight;
              });
    
    if (similar_episodes.size() > 10) {
        similar_episodes.resize(10);
    }
    
    return similar_episodes;
}

void ControllerModule::updateActionValues(const BrowsingAction& action, float reward) {
    std::string action_key = std::to_string(static_cast<int>(action.type));
    
    // Update action value with learning rate
    float learning_rate = 0.1f;
    if (neuromodulator_controller_) {
        learning_rate *= neuromodulator_controller_->getModulatorLevel(
            NeuromodulatorController::ACETYLCHOLINE);
    }
    
    float current_value = action_values_[action_key];
    float prediction_error = reward - current_value;
    action_values_[action_key] = current_value + learning_rate * prediction_error;
    
    // Update dopamine based on prediction error
    if (neuromodulator_controller_) {
        float dopamine_change = std::clamp(prediction_error * 0.5f, -0.2f, 0.2f);
        neuromodulator_controller_->setTargetModulation(
            NeuromodulatorController::DOPAMINE, 
            neuromodulator_controller_->getModulatorLevel(NeuromodulatorController::DOPAMINE) + dopamine_change);
    }
}

// ============================================================================
// INITIALIZATION HELPERS
// ============================================================================

void ControllerModule::initializeNeuromodulatorController() {
    neuromodulator_controller_ = std::make_unique<NeuromodulatorController>(getConfig());
    neuromodulator_controller_->initialize();
    
    std::cout << "ControllerModule: Initialized neuromodulator controller" << std::endl;
}

void ControllerModule::initializeModularNetwork() {
    modular_network_ = std::make_unique<ModularNeuralNetwork>();
    modular_network_->initialize();
    
    std::cout << "ControllerModule: Initialized modular network" << std::endl;
}

void ControllerModule::initializeMemorySystem() {
    if (!memory_system_) {
        memory_system_ = std::make_unique<ContextualMemory>();
    }
    
    memory_system_->working_memory_capacity = 7;
    
    std::cout << "ControllerModule: Initialized memory system" << std::endl;
}

void ControllerModule::setupDefaultModuleConnections() {
    // Setup default attention allocations
    attention_allocation_.resize(8, 0.125f);  // Equal initial allocation
    
    // Initialize module specializations
    module_specializations_["perception"] = 0.9f;
    module_specializations_["memory"] = 0.8f;
    module_specializations_["planning"] = 0.7f;
    module_specializations_["motor"] = 0.85f;
    
    std::cout << "ControllerModule: Setup default module connections" << std::endl;
}

// ============================================================================
// PROCESSING HELPERS
// ============================================================================

void ControllerModule::processModuleUpdates(double dt) {
    // Update all registered modules
    for (const auto& [module_name, module] : registered_modules_) {
        if (module && module_states_[module_name].is_active) {
            try {
                module->update(dt);
                
                // Update module state based on activity
                auto& state = module_states_[module_name];
                state.processing_load = std::min(1.0f, state.processing_load + 0.1f);
                
            } catch (const std::exception& e) {
                handleModuleError(module_name, e.what());
            }
        }
    }
}

void ControllerModule::processAttentionAllocation() {
    // Update attention based on current module states
    std::map<std::string, float> attention_demands;
    
    for (const auto& [module_name, state] : module_states_) {
        if (state.is_active) {
            float demand = state.activation_level * state.processing_load;
            attention_demands[module_name] = demand;
        }
    }
    
    allocateAttention(attention_demands);
}

void ControllerModule::processNeuromodulation() {
    if (neuromodulator_controller_) {
        neuromodulator_controller_->update(0.016f);  // ~60Hz update rate
        
        // Apply neuromodulation to modules
        auto modulation_output = neuromodulator_controller_->generateModulationOutput();
        
        for (const auto& [module_name, module] : registered_modules_) {
            if (module) {
                // Apply dopamine modulation for reward learning
                float dopamine_level = modulation_output[NeuromodulatorController::DOPAMINE];
                module->applyNeuromodulation("dopamine", dopamine_level);
                
                // Apply other neuromodulators as needed
                module->applyNeuromodulation("acetylcholine", 
                    modulation_output[NeuromodulatorController::ACETYLCHOLINE]);
            }
        }
    }
}

void ControllerModule::processInterModuleCommunication() {
    routeInterModuleSignals();
    updateSignalPriorities();
    manageSignalQueue();
    
    // Clear processed signals
    inter_module_signals_.clear();
}

void ControllerModule::processMemoryConsolidation() {
    // Perform periodic memory consolidation
    static int consolidation_counter = 0;
    if (++consolidation_counter % 1000 == 0) {  // Every ~16 seconds at 60Hz
        consolidateMemories();
    }
}

// ============================================================================
// PERFORMANCE AND STATE MANAGEMENT
// ============================================================================

void ControllerModule::updatePerformanceMetrics() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - performance_metrics_.last_update).count();
    
    if (elapsed >= 1000) {  // Update every second
        // Calculate success rate
        if (performance_metrics_.total_actions > 0) {
            float success_rate = static_cast<float>(performance_metrics_.successful_actions) / 
                               performance_metrics_.total_actions;
            performance_metrics_.decision_accuracy = success_rate;
        }
        
        // Update processing efficiency
        float active_modules = 0.0f;
        for (const auto& [name, state] : module_states_) {
            if (state.is_active) active_modules += 1.0f;
        }
        
        performance_metrics_.processing_efficiency = (active_modules > 0.0f) ? 
            (1.0f / active_modules) : 1.0f;
        
        performance_metrics_.last_update = now;
    }
}

bool ControllerModule::saveControllerState(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    std::ofstream file(filename + "_controller.json");
    if (!file.is_open()) {
        return false;
    }
    
    // Save controller state as JSON (simplified)
    file << "{\n";
    file << "  \"module_count\": " << registered_modules_.size() << ",\n";
    file << "  \"active_modules\": [";
    
    bool first = true;
    for (const auto& [name, state] : module_states_) {
        if (state.is_active) {
            if (!first) file << ", ";
            file << "\"" << name << "\"";
            first = false;
        }
    }
    
    file << "],\n";
    file << "  \"performance_metrics\": {\n";
    file << "    \"total_actions\": " << performance_metrics_.total_actions << ",\n";
    file << "    \"successful_actions\": " << performance_metrics_.successful_actions << ",\n";
    file << "    \"decision_accuracy\": " << performance_metrics_.decision_accuracy << "\n";
    file << "  }\n";
    file << "}\n";
    
    return true;
}

void ControllerModule::performSystemHealthCheck() {
    // Check for inactive modules that should be active
    for (const auto& [name, state] : module_states_) {
        if (!state.is_active && state.activation_level > 0.5f) {
            std::cout << "ControllerModule: Warning - Module '" << name 
                      << "' should be active but isn't" << std::endl;
        }
    }
    
    // Check memory usage
    if (memory_system_ && memory_system_->working_memory.size() > memory_system_->working_memory_capacity) {
        std::cout << "ControllerModule: Warning - Working memory overflow" << std::endl;
    }
    
    // Check neuromodulator levels
    if (neuromodulator_controller_) {
        for (int i = 0; i < NeuromodulatorController::NUM_MODULATORS; ++i) {
            float level = neuromodulator_controller_->getModulatorLevel(
                static_cast<NeuromodulatorController::ModulatorType>(i));
            if (level > 1.5f || level < 0.1f) {
                std::cout << "ControllerModule: Warning - Neuromodulator " << i 
                          << " at extreme level: " << level << std::endl;
            }
        }
    }
}

// ============================================================================
// AUTONOMOUS LEARNING AGENT IMPLEMENTATION
// ============================================================================

AutonomousLearningAgent::AutonomousLearningAgent(const NetworkConfig& config) {
    controller_module_ = std::make_unique<ControllerModule>("central_controller", config);
    
    // Create a simplified memory system interface
    // In a real implementation, this would be a more sophisticated memory system
    memory_system_ = nullptr;  // Will be handled by controller module
}

bool AutonomousLearningAgent::initialize() {
    if (!controller_module_) {
        return false;
    }
    
    controller_module_->initialize();
    std::cout << "AutonomousLearningAgent: Initialization complete" << std::endl;
    return true;
}

void AutonomousLearningAgent::update(double dt) {
    if (controller_module_) {
        controller_module_->update(dt);
    }
}

void AutonomousLearningAgent::shutdown() {
    if (controller_module_) {
        controller_module_->shutdown();
    }
}

// Delegate methods to controller module
std::vector<float> AutonomousLearningAgent::collect_inter_module_signals(const std::string& target_module) {
    return controller_module_ ? controller_module_->collect_inter_module_signals(target_module) : std::vector<float>();
}

void AutonomousLearningAgent::distribute_module_output(const std::string& source_module, 
                                                      const std::vector<float>& output_data) {
    if (controller_module_) {
        controller_module_->distribute_module_output(source_module, output_data);
    }
}

std::vector<AutonomousLearningAgent::BrowsingAction> AutonomousLearningAgent::generate_action_candidates() {
    return controller_module_ ? controller_module_->generate_action_candidates() : std::vector<BrowsingAction>();
}

std::vector<float> AutonomousLearningAgent::evaluate_action_candidates(
    const std::vector<BrowsingAction>& candidates,
    const std::vector<MemoryTrace>& similar_episodes) {
    return controller_module_ ? controller_module_->evaluate_action_candidates(candidates, similar_episodes) : std::vector<float>();
}

AutonomousLearningAgent::BrowsingAction AutonomousLearningAgent::select_action_with_exploration(
    const std::vector<BrowsingAction>& candidates,
    const std::vector<float>& action_values) {
    
    if (controller_module_) {
        selected_action_ = controller_module_->select_action_with_exploration(candidates, action_values);
        return selected_action_;
    }
    
    // Return default action if no controller
    BrowsingAction default_action;
    default_action.type = BrowsingAction::WAIT;
    default_action.parameters.wait_duration = 1.0f;
    return default_action;
}

void AutonomousLearningAgent::execute_action() {
    if (controller_module_) {
        controller_module_->execute_action();
    }
}

void AutonomousLearningAgent::execute_click_action() {
    if (controller_module_) {
        controller_module_->execute_click_action();
    }
}

void AutonomousLearningAgent::execute_scroll_action() {
    if (controller_module_) {
        controller_module_->execute_scroll_action();
    }
}

void AutonomousLearningAgent::execute_type_action() {
    if (controller_module_) {
        controller_module_->execute_type_action();
    }
}

void AutonomousLearningAgent::execute_navigate_action() {
    if (controller_module_) {
        controller_module_->execute_navigate_action();
    }
}

void AutonomousLearningAgent::execute_wait_action() {
    if (controller_module_) {
        controller_module_->execute_wait_action();
    }
}