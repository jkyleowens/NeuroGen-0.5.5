// ============================================================================
// DECISION MAKING AND ACTION EXECUTION SYSTEMS
// File: src/DecisionAndActionSystems.cpp
// ============================================================================

#include <NeuroGen/AutonomousLearningAgent.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>

// ============================================================================
// REMAINING SPECIALIZED MODULE PROCESSING METHODS
// ============================================================================

std::vector<float> SpecializedModule::process_motor_cortex(const std::vector<float>& motor_input) {
    // Motor cortex: Precise motor planning and execution with muscle synergies
    size_t input_size = std::min(motor_input.size(), internal_state_.size());
    
    // Update motor planning state
    for (size_t i = 0; i < input_size; ++i) {
        // Motor cortex integrates commands over time for smooth execution
        internal_state_[i] = internal_state_[i] * 0.85f + motor_input[i] * attention_weight_ * 0.15f;
    }
    
    // Generate motor commands with biological constraints
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float motor_command = 0.0f;
        
        // Integrate from corresponding internal state region
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            motor_command += internal_state_[j];
        }
        
        motor_command /= (state_end - state_start);
        
        // Apply motor activation function (sigmoidal for muscle-like response)
        output_buffer_[i] = std::tanh(motor_command * 2.0f - activation_threshold_);
        
        // Add motor noise for biological realism
        output_buffer_[i] += (rand() / float(RAND_MAX) - 0.5f) * 0.02f;
        
        // Bound motor outputs
        output_buffer_[i] = std::max(-1.0f, std::min(output_buffer_[i], 1.0f));
    }
    
    return output_buffer_;
}

std::vector<float> SpecializedModule::process_attention_system(const std::vector<float>& attention_input) {
    // Attention system: Dynamic resource allocation and focus control
    size_t input_size = std::min(attention_input.size(), internal_state_.size());
    
    // Rapid attention updates (high learning rate)
    for (size_t i = 0; i < input_size; ++i) {
        internal_state_[i] = internal_state_[i] * 0.8f + attention_input[i] * 0.2f;
    }
    
    // Compute attention weights with winner-take-all dynamics
    std::vector<float> raw_attention(output_buffer_.size());
    
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float attention_strength = 0.0f;
        
        // Compute attention from internal state
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            attention_strength += internal_state_[j] * internal_state_[j]; // Quadratic nonlinearity
        }
        
        raw_attention[i] = attention_strength / (state_end - state_start);
    }
    
    // Apply softmax for competitive attention
    float max_attention = *std::max_element(raw_attention.begin(), raw_attention.end());
    float sum_exp = 0.0f;
    
    for (float& att : raw_attention) {
        att = std::exp((att - max_attention) * 3.0f); // Temperature = 1/3
        sum_exp += att;
    }
    
    // Normalize and store in output buffer
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        output_buffer_[i] = raw_attention[i] / (sum_exp + 1e-8f);
    }
    
    return output_buffer_;
}

std::vector<float> SpecializedModule::process_reward_system(const std::vector<float>& reward_input) {
    // Reward system: Dopaminergic prediction error and value learning
    size_t input_size = std::min(reward_input.size(), internal_state_.size() / 2);
    
    // Update reward prediction state
    for (size_t i = 0; i < input_size; ++i) {
        internal_state_[i] = internal_state_[i] * 0.95f + reward_input[i] * 0.05f;
    }
    
    // Compute reward prediction and prediction error
    float predicted_reward = 0.0f;
    for (size_t i = 0; i < input_size; ++i) {
        predicted_reward += internal_state_[i];
    }
    predicted_reward /= input_size;
    
    // Current reward (simplified - would come from environment)
    float current_reward = (input_size > 0) ? reward_input[0] : 0.0f;
    float prediction_error = current_reward - predicted_reward;
    
    // Update value function using TD learning
    float td_learning_rate = learning_rate_ * 5.0f; // Fast reward learning
    for (size_t i = 0; i < input_size; ++i) {
        internal_state_[i] += td_learning_rate * prediction_error * internal_state_[i];
    }
    
    // Generate dopamine-like signals
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        if (i == 0) {
            // Primary dopamine signal (prediction error)
            output_buffer_[i] = std::tanh(prediction_error * 2.0f);
        } else if (i == 1) {
            // Predicted value signal
            output_buffer_[i] = std::tanh(predicted_reward);
        } else {
            // Modulated signals for different brain regions
            float modulation = prediction_error * (1.0f + 0.1f * std::sin(i * 0.5f));
            output_buffer_[i] = std::tanh(modulation);
        }
    }
    
    return output_buffer_;
}

std::vector<float> SpecializedModule::process_working_memory(const std::vector<float>& memory_input) {
    // Working memory: Temporary information maintenance and manipulation
    size_t input_size = std::min(memory_input.size(), internal_state_.size());
    
    // Fast working memory updates with gating
    for (size_t i = 0; i < input_size; ++i) {
        // Gating function - only update if input is strong enough
        float gate_value = std::abs(memory_input[i]) > 0.3f ? 1.0f : 0.1f;
        internal_state_[i] = internal_state_[i] * (1.0f - gate_value * 0.3f) + 
                            memory_input[i] * gate_value * 0.3f;
    }
    
    // Working memory decay (forgetting)
    for (float& state : internal_state_) {
        state *= 0.98f; // Gradual decay
    }
    
    // Output maintained information
    for (size_t i = 0; i < output_buffer_.size(); ++i) {
        float maintained_info = 0.0f;
        
        // Average over corresponding internal state
        size_t state_start = (i * internal_state_.size()) / output_buffer_.size();
        size_t state_end = ((i + 1) * internal_state_.size()) / output_buffer_.size();
        
        for (size_t j = state_start; j < state_end; ++j) {
            maintained_info += internal_state_[j];
        }
        
        output_buffer_[i] = maintained_info / (state_end - state_start);
    }
    
    return output_buffer_;
}

// ============================================================================
// AUTONOMOUS LEARNING AGENT - COGNITIVE PROCESSING METHODS
// ============================================================================

void AutonomousLearningAgent::process_visual_input() {
    if (!visual_interface_) return;
    
    // Capture and process current visual scene
    std::vector<float> visual_features = visual_interface_->capture_and_process_screen();
    
    // Send to visual cortex for processing
    if (modules_.count("visual_cortex")) {
        float visual_attention = attention_controller_->get_attention_weight("visual_cortex");
        auto visual_output = modules_["visual_cortex"]->process(visual_features, visual_attention);
        
        // Store visual features in environmental context
        size_t context_visual_size = std::min(visual_output.size(), environmental_context_.size() / 2);
        for (size_t i = 0; i < context_visual_size; ++i) {
            environmental_context_[i] = visual_output[i];
        }
    }
}

void AutonomousLearningAgent::update_working_memory() {
    if (!modules_.count("working_memory")) return;
    
    // Combine current sensory input with existing working memory
    std::vector<float> working_memory_input;
    working_memory_input.reserve(512);
    
    // Add visual context
    for (size_t i = 0; i < std::min(environmental_context_.size() / 2, size_t(256)); ++i) {
        working_memory_input.push_back(environmental_context_[i]);
    }
    
    // Add current goals
    for (size_t i = 0; i < std::min(current_goals_.size(), size_t(128)); ++i) {
        working_memory_input.push_back(current_goals_[i]);
    }
    
    // Add previous working memory content
    auto prev_working_memory = memory_system_->get_working_memory();
    for (size_t i = 0; i < std::min(prev_working_memory.size(), size_t(128)); ++i) {
        working_memory_input.push_back(prev_working_memory[i]);
    }
    
    // Process through working memory module
    float wm_attention = attention_controller_->get_attention_weight("working_memory");
    auto wm_output = modules_["working_memory"]->process(working_memory_input, wm_attention);
    
    // Update memory system
    memory_system_->update_working_memory(wm_output);
}

void AutonomousLearningAgent::update_attention_weights() {
    // Prepare context for attention computation
    std::vector<float> attention_context;
    attention_context.reserve(256);
    
    // Add visual saliency
    if (visual_interface_) {
        auto attention_map = visual_interface_->get_attention_map();
        for (size_t i = 0; i < std::min(attention_map.size(), size_t(64)); ++i) {
            attention_context.push_back(attention_map[i]);
        }
    }
    
    // Add goal relevance
    for (size_t i = 0; i < std::min(current_goals_.size(), size_t(64)); ++i) {
        attention_context.push_back(current_goals_[i]);
    }
    
    // Add environmental complexity
    float env_complexity = 0.0f;
    for (float val : environmental_context_) {
        env_complexity += val * val;
    }
    attention_context.push_back(std::tanh(env_complexity / 100.0f));
    
    // Add task urgency
    attention_context.push_back(exploration_rate_); // Use exploration rate as urgency proxy
    
    // Update attention controller
    attention_controller_->update_context(attention_context);
    
    // Apply attention to learning system
    if (learning_system_) {
        auto attention_weights = attention_controller_->get_all_attention_weights();
        learning_system_->update_attention_learning(attention_weights, 0.05f); // 50ms timestep
    }
}

void AutonomousLearningAgent::coordinate_modules() {
    // Get current attention weights
    auto attention_weights = attention_controller_->get_all_attention_weights();
    
    // Process each module with its attention weight and inter-module signals
    for (auto& [module_name, module] : modules_) {
        float attention_weight = attention_controller_->get_attention_weight(module_name);
        
        // Collect input from connected modules
        std::vector<float> module_input = collect_inter_module_signals(module_name);
        
        // Process the module
        auto module_output = module->process(module_input, attention_weight);
        
        // Send output to connected modules
        distribute_module_output(module_name, module_output);
    }
}

std::vector<float> AutonomousLearningAgent::collect_inter_module_signals(const std::string& target_module) {
    std::vector<float> combined_input;
    combined_input.reserve(512);
    
    // Collect signals from modules connected to the target
    for (const auto& [source_module, module] : modules_) {
        if (source_module != target_module) {
            auto signal = module->get_output_for_module(target_module);
            for (float val : signal) {
                combined_input.push_back(val);
            }
        }
    }
    
    // Add environmental context relevant to the module
    if (target_module == "visual_cortex") {
        // Add raw visual features
        for (size_t i = 0; i < std::min(environmental_context_.size() / 2, size_t(256)); ++i) {
            combined_input.push_back(environmental_context_[i]);
        }
    } else if (target_module == "prefrontal_cortex") {
        // Add working memory and goals
        auto working_memory = memory_system_->get_working_memory();
        for (size_t i = 0; i < std::min(working_memory.size(), size_t(128)); ++i) {
            combined_input.push_back(working_memory[i]);
        }
        for (size_t i = 0; i < std::min(current_goals_.size(), size_t(64)); ++i) {
            combined_input.push_back(current_goals_[i]);
        }
    }
    
    return combined_input;
}

void AutonomousLearningAgent::distribute_module_output(const std::string& source_module, 
                                                      const std::vector<float>& output) {
    // Send output to all connected modules
    for (auto& [target_module, module] : modules_) {
        if (target_module != source_module) {
            module->receive_signal(source_module, output);
        }
    }
    
    // Update global state based on key modules
    if (source_module == "prefrontal_cortex") {
        // Executive decisions affect global state
        size_t update_size = std::min(output.size(), global_state_.size() / 4);
        for (size_t i = 0; i < update_size; ++i) {
            global_state_[i] = global_state_[i] * 0.9f + output[i] * 0.1f;
        }
    } else if (source_module == "reward_system") {
        // Reward signals modulate global learning
        if (!output.empty()) {
            global_reward_signal_ = global_reward_signal_ * 0.95f + output[0] * 0.05f;
        }
    }
}

void AutonomousLearningAgent::make_decision() {
    // Decision-making pipeline combining multiple cognitive systems
    
    // Step 1: Get current situation assessment from prefrontal cortex
    if (!modules_.count("prefrontal_cortex")) return;
    
    auto prefrontal_output = modules_["prefrontal_cortex"]->get_internal_state();
    
    // Step 2: Retrieve relevant memories
    std::vector<float> current_state_summary;
    current_state_summary.reserve(256);
    
    // Combine key state information
    for (size_t i = 0; i < std::min(global_state_.size(), size_t(128)); ++i) {
        current_state_summary.push_back(global_state_[i]);
    }
    for (size_t i = 0; i < std::min(environmental_context_.size(), size_t(128)); ++i) {
        current_state_summary.push_back(environmental_context_[i]);
    }
    
    // Retrieve similar past experiences
    auto similar_episodes = memory_system_->retrieve_similar_episodes(current_state_summary, 5);
    
    // Step 3: Generate action candidates
    auto action_candidates = generate_action_candidates();
    
    // Step 4: Evaluate actions using learned value function
    auto action_values = evaluate_action_candidates(action_candidates, similar_episodes);
    
    // Step 5: Select action with exploration
    selected_action_ = select_action_with_exploration(action_candidates, action_values);
    
    static int decision_count = 0;
    if (++decision_count % 100 == 0) {
        std::cout << "Decision System: Made decision #" << decision_count 
                  << " - Action type: " << static_cast<int>(selected_action_.type) 
                  << ", Confidence: " << selected_action_.confidence << std::endl;
    }
}

std::vector<AutonomousLearningAgent::BrowsingAction> AutonomousLearningAgent::generate_action_candidates() {
    std::vector<BrowsingAction> candidates;
    
    if (!visual_interface_) {
        // Generate basic movement actions as fallback
        candidates.push_back({BrowsingAction::WAIT, 0, 0, "", "", 0.8f});
        return candidates;
    }
    
    // Get current screen elements
    auto screen_elements = visual_interface_->detect_screen_elements();
    
    // Generate click actions for detected elements
    for (const auto& element : screen_elements) {
        if (element.is_clickable && element.confidence > 0.5f) {
            BrowsingAction action;
            action.type = BrowsingAction::CLICK;
            action.x = element.x + element.width / 2;
            action.y = element.y + element.height / 2;
            action.text = element.text;
            action.confidence = element.confidence * 0.8f; // Reduce confidence slightly
            
            candidates.push_back(action);
        }
    }
    
    // Generate exploration actions
    if (exploration_rate_ > 0.1f) {
        // Random click for exploration
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> x_dist(100, 1800);
        std::uniform_int_distribution<int> y_dist(100, 1000);
        
        BrowsingAction explore_action;
        explore_action.type = BrowsingAction::CLICK;
        explore_action.x = x_dist(gen);
        explore_action.y = y_dist(gen);
        explore_action.confidence = exploration_rate_ * 0.5f;
        
        candidates.push_back(explore_action);
        
        // Scroll action
        BrowsingAction scroll_action;
        scroll_action.type = BrowsingAction::SCROLL;
        scroll_action.x = 960; // Screen center
        scroll_action.y = 540;
        scroll_action.confidence = exploration_rate_ * 0.3f;
        
        candidates.push_back(scroll_action);
    }
    
    // Wait action (always available)
    BrowsingAction wait_action;
    wait_action.type = BrowsingAction::WAIT;
    wait_action.confidence = 0.6f;
    
    candidates.push_back(wait_action);
    
    return candidates;
}

std::vector<float> AutonomousLearningAgent::evaluate_action_candidates(
    const std::vector<BrowsingAction>& candidates,
    const std::vector<MemorySystem::MemoryTrace>& similar_episodes) {
    
    std::vector<float> values(candidates.size());
    
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& action = candidates[i];
        float value = 0.0f;
        
        // Base value from action type
        switch (action.type) {
            case BrowsingAction::CLICK:
                value = 0.6f; // Generally positive
                break;
            case BrowsingAction::SCROLL:
                value = 0.4f; // Moderate value
                break;
            case BrowsingAction::TYPE:
                value = 0.7f; // High value for input
                break;
            case BrowsingAction::NAVIGATE:
                value = 0.5f; // Navigation is risky but potentially valuable
                break;
            case BrowsingAction::WAIT:
                value = 0.2f; // Low value, but safe
                break;
        }
        
        // Adjust based on confidence
        value *= (0.5f + 0.5f * action.confidence);
        
        // Adjust based on similar episodes
        for (const auto& episode : similar_episodes) {
            if (!episode.action_vector.empty() && episode.action_vector.size() > 2) {
                // Simple action similarity check
                float action_similarity = 0.0f;
                if (static_cast<int>(action.type) < episode.action_vector.size()) {
                    action_similarity = episode.action_vector[static_cast<int>(action.type)];
                }
                
                // Weight by episode outcome
                value += action_similarity * episode.reward * 0.1f;
            }
        }
        
        // Add exploration bonus
        if (action.type != BrowsingAction::WAIT) {
            value += exploration_rate_ * 0.2f;
        }
        
        values[i] = value;
    }
    
    return values;
}

AutonomousLearningAgent::BrowsingAction AutonomousLearningAgent::select_action_with_exploration(
    const std::vector<BrowsingAction>& candidates, const std::vector<float>& values) {
    
    if (candidates.empty()) {
        return {BrowsingAction::WAIT, 0, 0, "", "", 0.5f};
    }
    
    // Softmax action selection with temperature
    float temperature = exploration_rate_ * 2.0f + 0.1f; // Higher exploration = higher temperature
    std::vector<float> probabilities(values.size());
    
    // Find max value for numerical stability
    float max_value = *std::max_element(values.begin(), values.end());
    
    float sum_exp = 0.0f;
    for (size_t i = 0; i < values.size(); ++i) {
        probabilities[i] = std::exp((values[i] - max_value) / temperature);
        sum_exp += probabilities[i];
    }
    
    // Normalize probabilities
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }
    
    // Sample action
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float random_value = dist(gen);
    float cumulative_prob = 0.0f;
    
    for (size_t i = 0; i < candidates.size(); ++i) {
        cumulative_prob += probabilities[i];
        if (random_value <= cumulative_prob) {
            return candidates[i];
        }
    }
    
    // Fallback to last action
    return candidates.back();
}

void AutonomousLearningAgent::execute_action() {
    // Simulate action execution (in a real system, this would interface with OS)
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
    }
    
    // Update action statistics
    metrics_.total_actions++;
    
    // Motor cortex processes the action
    if (modules_.count("motor_cortex")) {
        std::vector<float> motor_command = convert_action_to_motor_command(selected_action_);
        float motor_attention = attention_controller_->get_attention_weight("motor_cortex");
        modules_["motor_cortex"]->process(motor_command, motor_attention);
    }
}

void AutonomousLearningAgent::execute_click_action() {
    std::cout << "Executing CLICK at (" << selected_action_.x << ", " << selected_action_.y 
              << ") with confidence " << selected_action_.confidence << std::endl;
    
    // In a real implementation, this would use platform-specific APIs:
    // - Windows: SetCursorPos() + mouse_event()
    // - Linux: XTestFakeButtonEvent()
    // - macOS: CGEventCreateMouseEvent()
    
    // Simulate click success based on confidence
    bool success = selected_action_.confidence > 0.5f;
    if (success) {
        metrics_.successful_actions++;
        global_reward_signal_ += 0.1f; // Small positive reward for successful clicks
    }
    
    // Add slight delay for realism
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void AutonomousLearningAgent::execute_scroll_action() {
    std::cout << "Executing SCROLL at (" << selected_action_.x << ", " << selected_action_.y 
              << ") with confidence " << selected_action_.confidence << std::endl;
    
    // Simulate scroll execution
    bool success = selected_action_.confidence > 0.3f;
    if (success) {
        metrics_.successful_actions++;
        global_reward_signal_ += 0.05f; // Small reward for exploration
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

void AutonomousLearningAgent::execute_type_action() {
    std::cout << "Executing TYPE: '" << selected_action_.text 
              << "' with confidence " << selected_action_.confidence << std::endl;
    
    // Simulate typing
    bool success = !selected_action_.text.empty() && selected_action_.confidence > 0.4f;
    if (success) {
        metrics_.successful_actions++;
        global_reward_signal_ += 0.15f; // Higher reward for meaningful input
    }
    
    // Simulate typing delay
    std::this_thread::sleep_for(std::chrono::milliseconds(selected_action_.text.length() * 50));
}

void AutonomousLearningAgent::execute_navigate_action() {
    std::cout << "Executing NAVIGATE to: '" << selected_action_.url 
              << "' with confidence " << selected_action_.confidence << std::endl;
    
    // Simulate navigation
    bool success = !selected_action_.url.empty() && selected_action_.confidence > 0.6f;
    if (success) {
        metrics_.successful_actions++;
        global_reward_signal_ += 0.2f; // High reward for successful navigation
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(2000)); // Page load simulation
}

void AutonomousLearningAgent::execute_wait_action() {
    std::cout << "Executing WAIT with confidence " << selected_action_.confidence << std::endl;
    
    // Wait actions are always "successful" but provide minimal reward
    metrics_.successful_actions++;
    global_reward_signal_ += 0.01f;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

std::vector<float> AutonomousLearningAgent::convert_action_to_motor_command(const BrowsingAction& action) {
    std::vector<float> motor_command(64, 0.0f); // Motor cortex expects 64-dimensional input
    
    // Encode action type
    if (static_cast<size_t>(action.type) < motor_command.size()) {
        motor_command[static_cast<size_t>(action.type)] = 1.0f;
    }
    
    // Encode spatial coordinates (normalized)
    if (motor_command.size() > 10) {
        motor_command[5] = action.x / 1920.0f; // Normalize x coordinate
        motor_command[6] = action.y / 1080.0f; // Normalize y coordinate
        motor_command[7] = action.confidence;   // Action confidence
    }
    
    // Add motor noise for biological realism
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.02f);
    
    for (size_t i = 8; i < motor_command.size(); ++i) {
        motor_command[i] = noise(gen);
    }
    
    return motor_command;
}

void AutonomousLearningAgent::learn_from_feedback() {
    // Comprehensive learning from action outcomes
    
    // Prepare experience for storage
    std::vector<float> current_state = global_state_;
    std::vector<float> action_vector(5, 0.0f);
    action_vector[static_cast<int>(selected_action_.type)] = 1.0f;
    
    // Compute reward signal based on action outcome
    float action_reward = compute_action_reward();
    
    // Store episode in memory
    memory_system_->store_episode(current_state, action_vector, action_reward, 
                                 selected_action_.confidence);
    
    // Update learning system
    if (learning_system_) {
        learning_system_->update_learning(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count(),
            50.0f, // 50ms timestep
            action_reward
        );
    }
    
    // Update performance metrics
    metrics_.average_reward = metrics_.average_reward * 0.99f + action_reward * 0.01f;
    
    // Adapt exploration rate based on recent performance
    adapt_exploration_rate();
    
    // Apply learning to specific modules
    apply_modular_learning(action_reward);
}

float AutonomousLearningAgent::compute_action_reward() {
    float reward = 0.0f;
    
    // Base reward from action success
    bool action_successful = (metrics_.total_actions > 0) ? 
        (static_cast<float>(metrics_.successful_actions) / metrics_.total_actions > 0.5f) : false;
    
    if (action_successful) {
        reward += 0.1f;
    } else {
        reward -= 0.05f;
    }
    
    // Reward based on action type effectiveness
    switch (selected_action_.type) {
        case BrowsingAction::CLICK:
            reward += selected_action_.confidence * 0.1f;
            break;
        case BrowsingAction::TYPE:
            reward += selected_action_.confidence * 0.15f;
            break;
        case BrowsingAction::NAVIGATE:
            reward += selected_action_.confidence * 0.2f;
            break;
        case BrowsingAction::SCROLL:
            reward += selected_action_.confidence * 0.05f;
            break;
        case BrowsingAction::WAIT:
            reward += 0.01f; // Minimal reward for waiting
            break;
    }
    
    // Penalty for excessive exploration
    if (exploration_rate_ > 0.5f) {
        reward -= 0.02f;
    }
    
    // Reward for learning progress
    if (learning_system_) {
        float learning_progress = learning_system_->get_learning_progress();
        reward += learning_progress * 0.05f;
    }
    
    return std::max(-0.5f, std::min(reward, 0.5f)); // Bound reward
}

void AutonomousLearningAgent::adapt_exploration_rate() {
    float success_rate = (metrics_.total_actions > 0) ? 
        static_cast<float>(metrics_.successful_actions) / metrics_.total_actions : 0.5f;
    
    // Decrease exploration if doing well, increase if struggling
    if (success_rate > 0.7f) {
        exploration_rate_ *= 0.995f; // Slowly decrease exploration
    } else if (success_rate < 0.3f) {
        exploration_rate_ *= 1.005f; // Slowly increase exploration
    }
    
    // Bound exploration rate
    exploration_rate_ = std::max(0.05f, std::min(exploration_rate_, 0.8f));
}

void AutonomousLearningAgent::apply_modular_learning(float reward) {
    if (!learning_system_) return;
    
    // Apply learning to each module based on its contribution to the action
    std::vector<std::string> relevant_modules;
    
    switch (selected_action_.type) {
        case BrowsingAction::CLICK:
        case BrowsingAction::SCROLL:
            relevant_modules = {"visual_cortex", "motor_cortex", "attention_system"};
            break;
        case BrowsingAction::TYPE:
            relevant_modules = {"prefrontal_cortex", "working_memory", "motor_cortex"};
            break;
        case BrowsingAction::NAVIGATE:
            relevant_modules = {"prefrontal_cortex", "hippocampus", "motor_cortex"};
            break;
        case BrowsingAction::WAIT:
            relevant_modules = {"prefrontal_cortex", "attention_system"};
            break;
    }
    
    // Apply module-specific learning
    for (size_t i = 0; i < relevant_modules.size(); ++i) {
        if (i < modules_.size()) {
            learning_system_->update_modular_learning(i, reward, 0.05f);
        }
    }
}

void AutonomousLearningAgent::update_global_state() {
    // Integrate information from all cognitive systems into global state
    
    // Decay existing state
    for (float& state : global_state_) {
        state *= 0.98f;
    }
    
    // Add environmental context
    for (size_t i = 0; i < std::min(environmental_context_.size(), global_state_.size() / 4); ++i) {
        global_state_[i] += environmental_context_[i] * 0.02f;
    }
    
    // Add working memory content
    auto working_memory = memory_system_->get_working_memory();
    size_t wm_offset = global_state_.size() / 4;
    for (size_t i = 0; i < std::min(working_memory.size(), global_state_.size() / 4); ++i) {
        if (wm_offset + i < global_state_.size()) {
            global_state_[wm_offset + i] += working_memory[i] * 0.03f;
        }
    }
    
    // Add attention state
    auto attention_weights = attention_controller_->get_all_attention_weights();
    size_t att_offset = global_state_.size() / 2;
    for (size_t i = 0; i < std::min(attention_weights.size(), global_state_.size() / 8); ++i) {
        if (att_offset + i < global_state_.size()) {
            global_state_[att_offset + i] = attention_weights[i];
        }
    }
    
    // Add reward history
    size_t reward_offset = 3 * global_state_.size() / 4;
    if (reward_offset < global_state_.size()) {
        global_state_[reward_offset] = global_reward_signal_;
    }
    
    // Bound global state values
    for (float& state : global_state_) {
        state = std::max(-2.0f, std::min(state, 2.0f));
    }
}

void AutonomousLearningAgent::consolidate_learning() {
    // Periodic learning consolidation and memory management
    
    std::cout << "Learning consolidation: Integrating recent experiences..." << std::endl;
    
    // Consolidate memories
    memory_system_->consolidate_memories();
    
    // Transfer learning between modules
    transfer_knowledge_between_modules();
    
    // Update long-term learning parameters
    if (learning_system_) {
        // Adjust learning rates based on performance
        float performance = static_cast<float>(metrics_.successful_actions) / 
                          std::max(metrics_.total_actions, 1);
        
        float lr_adjustment = (performance > 0.6f) ? 0.95f : 1.05f;
        learning_system_->configure_learning_parameters(
            learning_rate_ * lr_adjustment, 0.995f, 1.0f);
    }
    
    std::cout << "Learning consolidation complete." << std::endl;
}

void AutonomousLearningAgent::transfer_knowledge_between_modules() {
    // Simple knowledge transfer between related modules
    
    // Visual-Prefrontal transfer (visual attention patterns)
    if (modules_.count("visual_cortex") && modules_.count("prefrontal_cortex")) {
        auto visual_state = modules_["visual_cortex"]->get_internal_state();
        std::vector<float> visual_patterns(visual_state.begin(), 
                                         visual_state.begin() + std::min(visual_state.size(), size_t(128)));
        modules_["prefrontal_cortex"]->receive_signal("visual_transfer", visual_patterns);
    }
    
    // Hippocampus-Prefrontal transfer (memory-guided decisions)
    if (modules_.count("hippocampus") && modules_.count("prefrontal_cortex")) {
        auto memory_state = modules_["hippocampus"]->get_internal_state();
        std::vector<float> memory_patterns(memory_state.begin(),
                                         memory_state.begin() + std::min(memory_state.size(), size_t(256)));
        modules_["prefrontal_cortex"]->receive_signal("memory_transfer", memory_patterns);
    }
}

void AutonomousLearningAgent::stop_autonomous_operation() {
    if (!is_running_) return;
    
    std::cout << "\nStopping autonomous operation..." << std::endl;
    is_running_ = false;
    
    // Stop visual capture
    if (visual_interface_) {
        visual_interface_->stop_capture();
    }
    
    // Wait for main loop to finish
    if (main_loop_thread_.joinable()) {
        main_loop_thread_.join();
    }
    
    std::cout << "✓ Autonomous operation stopped successfully" << std::endl;
}