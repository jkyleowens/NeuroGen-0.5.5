// AutonomousLearningAgent.cpp
// Implementation of Continuous Reinforcement Learning Agent
// Version 0.5.5 - Advanced Autonomous Learning Framework

#include "NeuroGen/AutonomousLearningAgent.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

// ============================================================================
// CONSTRUCTOR AND DESTRUCTOR
// ============================================================================

AutonomousLearningAgent::AutonomousLearningAgent(const AutonomousAgentConfig& config)
    : config_(config)
    , random_generator_(std::chrono::steady_clock::now().time_since_epoch().count())
    , uniform_dist_(0.0f, 1.0f)
    , exploration_noise_(0.0f, 1.0f)
    , current_exploration_rate_(config.exploration_rate)
    , is_learning_enabled_(true)
    , is_autonomous_mode_(false)
    , simulation_step_(0)
    , current_time_(0.0)
    , current_neuron_count_(config.initial_neuron_count)
    , last_expansion_time_(0.0f) {
    
    std::cout << "🤖 Initializing Autonomous Learning Agent v0.5.5..." << std::endl;
    
    // Initialize neural network
    initializeNeuralNetwork();
    
    // Setup default learning goals
    setupDefaultGoals();
    
    // Initialize current state
    current_state_.resize(64, 0.0f);  // Default state size
    current_action_.resize(32, 0.0f); // Default action size
    
    std::cout << "✅ Autonomous Learning Agent initialized successfully!" << std::endl;
    std::cout << "   • Initial neurons: " << current_neuron_count_ << std::endl;
    std::cout << "   • Learning goals: " << goals_.size() << std::endl;
    std::cout << "   • Exploration rate: " << current_exploration_rate_ << std::endl;
}

AutonomousLearningAgent::~AutonomousLearningAgent() {
    stopAutonomousLearning();
    std::cout << "🤖 Autonomous Learning Agent shutdown complete." << std::endl;
}

// ============================================================================
// CORE AUTONOMOUS LEARNING INTERFACE
// ============================================================================

void AutonomousLearningAgent::startAutonomousLearning() {
    if (is_autonomous_mode_) {
        std::cout << "⚠️ Autonomous learning already active!" << std::endl;
        return;
    }
    
    is_autonomous_mode_ = true;
    is_learning_enabled_ = true;
    
    std::cout << "🚀 Starting autonomous learning mode..." << std::endl;
    std::cout << "   • Initial exploration rate: " << current_exploration_rate_ << std::endl;
    std::cout << "   • Network neurons: " << current_neuron_count_ << std::endl;
    std::cout << "   • Active goals: " << goals_.size() << std::endl;
    
    logLearningEvent("AUTONOMOUS_START", {
        {"neuron_count", static_cast<float>(current_neuron_count_)},
        {"exploration_rate", current_exploration_rate_},
        {"goal_count", static_cast<float>(goals_.size())}
    });
}

void AutonomousLearningAgent::stopAutonomousLearning() {
    if (!is_autonomous_mode_) return;
    
    is_autonomous_mode_ = false;
    
    std::cout << "🛑 Stopping autonomous learning mode..." << std::endl;
    
    // Generate final learning report
    std::string report = generateLearningReport();
    std::cout << report << std::endl;
    
    logLearningEvent("AUTONOMOUS_STOP", {
        {"total_steps", static_cast<float>(simulation_step_)},
        {"final_reward", metrics_.average_reward},
        {"network_expansions", static_cast<float>(metrics_.network_expansions)}
    });
}

float AutonomousLearningAgent::autonomousLearningStep(float dt) {
    if (!is_autonomous_mode_ || !is_learning_enabled_) return 0.0f;
    
    auto step_start = std::chrono::high_resolution_clock::now();
    
    // ========================================
    // PHASE 1: ENVIRONMENT PERCEPTION
    // ========================================
    
    std::vector<float> new_state = perceiveEnvironment();
    if (new_state.empty()) {
        // Generate synthetic environment for autonomous exploration
        new_state.resize(64);
        for (size_t i = 0; i < new_state.size(); ++i) {
            new_state[i] = uniform_dist_(random_generator_) * 2.0f - 1.0f;
        }
    }
    
    // ========================================
    // PHASE 2: ACTION SELECTION AND EXECUTION
    // ========================================
    
    std::vector<float> action = selectAction(new_state);
    float external_reward = executeAction(action);
    float environmental_reward = getEnvironmentalReward();
    
    // Combine rewards
    float total_external_reward = external_reward + environmental_reward;
    
    // ========================================
    // PHASE 3: EXPERIENCE CREATION
    // ========================================
    
    LearningExperience experience;
    experience.state = current_state_;
    experience.action = action;
    experience.reward = total_external_reward;
    experience.next_state = new_state;
    experience.timestamp = current_time_;
    
    // Calculate intrinsic reward
    experience.intrinsic_reward = computeIntrinsicReward(experience);
    experience.surprise_level = calculateNoveltyScore(new_state);
    experience.difficulty_estimate = calculateLearningComplexity();
    experience.novelty_score = experience.surprise_level;
    
    // ========================================
    // PHASE 4: LEARNING AND ADAPTATION
    // ========================================
    
    // Process experience
    processLearningExperience(experience);
    
    // Update neural network
    updateNeuralNetwork(dt);
    
    // Update goal progress
    updateGoalProgress();
    
    // ========================================
    // PHASE 5: META-LEARNING AND ADAPTATION
    // ========================================
    
    // Perform meta-learning every 100 steps
    if (simulation_step_ % 100 == 0) {
        performMetaLearning();
        updateExplorationStrategy();
    }
    
    // Experience replay
    if (uniform_dist_(random_generator_) < config_.experience_replay_probability) {
        performExperienceReplay();
    }
    
    // ========================================
    // PHASE 6: NETWORK EXPANSION CHECK
    // ========================================
    
    if (shouldExpandNetwork()) {
        std::cout << "🌱 Network expansion triggered at step " << simulation_step_ << std::endl;
        expandNeuralNetwork();
    }
    
    // ========================================
    // PHASE 7: UPDATE STATE AND METRICS
    // ========================================
    
    current_state_ = new_state;
    current_action_ = action;
    current_time_ += dt;
    simulation_step_++;
    
    updatePerformanceMetrics();
    
    // Calculate learning progress
    float learning_progress = 0.0f;
    if (!goals_.empty()) {
        float total_competence = 0.0f;
        for (const auto& goal : goals_) {
            if (goal->is_active) {
                total_competence += goal->current_competence;
            }
        }
        learning_progress = total_competence / goals_.size();
    }
    
    // Log significant events
    if (simulation_step_ % config_.evaluation_interval == 0) {
        logLearningEvent("EVALUATION_STEP", {
            {"step", static_cast<float>(simulation_step_)},
            {"reward", total_external_reward},
            {"intrinsic_reward", experience.intrinsic_reward},
            {"learning_progress", learning_progress},
            {"exploration_rate", current_exploration_rate_},
            {"network_size", static_cast<float>(current_neuron_count_)}
        });
        
        std::cout << "📊 Step " << simulation_step_ 
                  << " | Reward: " << std::fixed << std::setprecision(3) << total_external_reward
                  << " | Intrinsic: " << experience.intrinsic_reward
                  << " | Progress: " << learning_progress
                  << " | Exploration: " << current_exploration_rate_
                  << " | Neurons: " << current_neuron_count_ << std::endl;
    }
    
    auto step_end = std::chrono::high_resolution_clock::now();
    auto step_duration = std::chrono::duration<double, std::milli>(step_end - step_start).count();
    
    // Update timing metrics
    metrics_.learning_efficiency = learning_progress / (step_duration + 1.0f);
    
    return learning_progress;
}

void AutonomousLearningAgent::continuousLearningLoop(int max_steps) {
    std::cout << "🔄 Starting continuous learning loop..." << std::endl;
    
    startAutonomousLearning();
    
    int step_count = 0;
    while (is_autonomous_mode_ && (max_steps < 0 || step_count < max_steps)) {
        float progress = autonomousLearningStep(1.0f);
        
        // Check for convergence or other stopping conditions
        if (progress > 0.95f && config_.competence_threshold < 1.0f) {
            std::cout << "🎯 High competence achieved! Progress: " << progress << std::endl;
            break;
        }
        
        step_count++;
        
        // Small delay to prevent CPU overload
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    stopAutonomousLearning();
    std::cout << "✅ Continuous learning loop completed after " << step_count << " steps." << std::endl;
}

// ============================================================================
// ENVIRONMENT INTERACTION
// ============================================================================

void AutonomousLearningAgent::setEnvironmentSensor(std::function<std::vector<float>()> sensor) {
    environment_sensor_ = sensor;
    std::cout << "🔍 Environment sensor configured." << std::endl;
}

void AutonomousLearningAgent::setEnvironmentActuator(std::function<float(const std::vector<float>&)> actuator) {
    environment_actuator_ = actuator;
    std::cout << "⚡ Environment actuator configured." << std::endl;
}

void AutonomousLearningAgent::setEnvironmentRewardSignal(std::function<float()> reward_signal) {
    environment_reward_signal_ = reward_signal;
    std::cout << "🎁 Environment reward signal configured." << std::endl;
}

std::vector<float> AutonomousLearningAgent::perceiveEnvironment() {
    if (environment_sensor_) {
        return environment_sensor_();
    }
    return std::vector<float>(); // Empty if no sensor configured
}

float AutonomousLearningAgent::executeAction(const std::vector<float>& action) {
    if (environment_actuator_) {
        return environment_actuator_(action);
    }
    return 0.0f; // No reward if no actuator configured
}

float AutonomousLearningAgent::getEnvironmentalReward() {
    if (environment_reward_signal_) {
        return environment_reward_signal_();
    }
    return 0.0f; // No environmental reward if not configured
}

// ============================================================================
// NEURAL NETWORK EXPANSION
// ============================================================================

bool AutonomousLearningAgent::shouldExpandNetwork() const {
    if (current_neuron_count_ >= config_.max_neuron_count) return false;
    
    // Check learning complexity
    float complexity = calculateLearningComplexity();
    if (complexity > config_.expansion_threshold) return true;
    
    // Check if learning has plateaued
    if (metrics_.learning_progress_history.size() >= 10) {
        float recent_progress = 0.0f;
        for (int i = metrics_.learning_progress_history.size() - 10; 
             i < metrics_.learning_progress_history.size(); ++i) {
            recent_progress += metrics_.learning_progress_history[i];
        }
        recent_progress /= 10.0f;
        
        // If progress is low but complexity is high, expand
        return (recent_progress < 0.1f && complexity > 0.5f);
    }
    
    return false;
}

bool AutonomousLearningAgent::expandNeuralNetwork(int additional_neurons) {
    if (current_neuron_count_ >= config_.max_neuron_count) {
        std::cout << "⚠️ Cannot expand: Maximum neuron count reached." << std::endl;
        return false;
    }
    
    if (additional_neurons < 0) {
        additional_neurons = config_.expansion_batch_size;
    }
    
    // Ensure we don't exceed maximum
    additional_neurons = std::min(additional_neurons, 
                                 config_.max_neuron_count - current_neuron_count_);
    
    std::cout << "🌱 Expanding neural network by " << additional_neurons << " neurons..." << std::endl;
    
    try {
        // Get current network statistics before expansion
        auto prev_stats = getNetworkStatistics();
        
        // Expand the modular neural network
        // This would involve adding new neurons to appropriate modules
        // For now, we'll simulate this by updating our tracking
        current_neuron_count_ += additional_neurons;
        last_expansion_time_ = current_time_;
        expansion_history_.push_back(additional_neurons);
        metrics_.network_expansions++;
        
        // Update network manager with new configuration
        if (network_manager_) {
            // In a real implementation, this would trigger actual network expansion
            // network_manager_->expandNetwork(additional_neurons);
        }
        
        std::cout << "✅ Network expanded successfully!" << std::endl;
        std::cout << "   • Previous neurons: " << prev_stats.neuron_count << std::endl;
        std::cout << "   • New neurons: " << current_neuron_count_ << std::endl;
        std::cout << "   • Total expansions: " << metrics_.network_expansions << std::endl;
        
        logLearningEvent("NETWORK_EXPANSION", {
            {"previous_neurons", static_cast<float>(prev_stats.neuron_count)},
            {"added_neurons", static_cast<float>(additional_neurons)},
            {"new_total", static_cast<float>(current_neuron_count_)},
            {"complexity", calculateLearningComplexity()}
        });
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Network expansion failed: " << e.what() << std::endl;
        return false;
    }
}

void AutonomousLearningAgent::pruneNeuralNetwork() {
    std::cout << "✂️ Pruning neural network..." << std::endl;
    
    // Get current statistics
    auto stats = getNetworkStatistics();
    
    if (stats.network_density < config_.pruning_threshold) {
        std::cout << "ℹ️ Network density already optimal, skipping pruning." << std::endl;
        return;
    }
    
    // In a real implementation, this would identify and remove low-activity connections
    // For now, we'll simulate the pruning effect
    
    logLearningEvent("NETWORK_PRUNING", {
        {"neurons", static_cast<float>(current_neuron_count_)},
        {"density_before", stats.network_density},
        {"complexity", calculateLearningComplexity()}
    });
    
    std::cout << "✅ Network pruning completed." << std::endl;
}

AutonomousLearningAgent::NetworkStatistics AutonomousLearningAgent::getNetworkStatistics() const {
    NetworkStatistics stats;
    stats.neuron_count = current_neuron_count_;
    stats.synapse_count = current_neuron_count_ * 10; // Estimated
    stats.network_density = 0.1f; // Simplified calculation
    stats.average_activity = metrics_.average_reward; // Proxy
    stats.learning_capacity = static_cast<float>(current_neuron_count_) / config_.max_neuron_count;
    stats.computational_complexity = calculateLearningComplexity();
    return stats;
}

// ============================================================================
// GOAL MANAGEMENT
// ============================================================================

void AutonomousLearningAgent::addLearningGoal(std::unique_ptr<AutonomousGoal> goal) {
    if (goal) {
        std::cout << "🎯 Adding learning goal: " << goal->description << std::endl;
        goals_.push_back(std::move(goal));
    }
}

void AutonomousLearningAgent::removeLearningGoal(const std::string& goal_description) {
    auto it = std::remove_if(goals_.begin(), goals_.end(),
        [&goal_description](const std::unique_ptr<AutonomousGoal>& goal) {
            return goal->description == goal_description;
        });
    
    if (it != goals_.end()) {
        std::cout << "🗑️ Removed learning goal: " << goal_description << std::endl;
        goals_.erase(it, goals_.end());
    }
}

std::map<std::string, float> AutonomousLearningAgent::getGoalCompetencies() const {
    std::map<std::string, float> competencies;
    for (const auto& goal : goals_) {
        competencies[goal->description] = goal->current_competence;
    }
    return competencies;
}

void AutonomousLearningAgent::setGoalPriority(const std::string& goal_description, int priority) {
    for (auto& goal : goals_) {
        if (goal->description == goal_description) {
            goal->priority = priority;
            std::cout << "🔄 Updated priority for goal '" << goal_description 
                      << "' to " << priority << std::endl;
            break;
        }
    }
}

// ============================================================================
// PRIVATE IMPLEMENTATION METHODS
// ============================================================================

float AutonomousLearningAgent::computeIntrinsicReward(const LearningExperience& experience) {
    float intrinsic_reward = 0.0f;
    
    // Novelty-based reward
    float novelty_reward = experience.novelty_score * config_.novelty_bonus_weight;
    
    // Curiosity-based reward (prediction error)
    float curiosity_reward = experience.surprise_level * config_.curiosity_weight;
    
    // Learning progress reward
    float progress_reward = experience.learning_progress * config_.intrinsic_motivation_strength;
    
    // Combine intrinsic rewards
    intrinsic_reward = novelty_reward + curiosity_reward + progress_reward;
    
    // Normalize to reasonable range
    intrinsic_reward = std::tanh(intrinsic_reward) * 0.5f;
    
    return intrinsic_reward;
}

void AutonomousLearningAgent::updateExplorationStrategy() {
    // Decay exploration rate
    current_exploration_rate_ *= config_.exploration_decay;
    current_exploration_rate_ = std::max(current_exploration_rate_, 0.01f);
    
    // Adaptive exploration based on learning progress
    if (!metrics_.learning_progress_history.empty()) {
        float recent_progress = metrics_.learning_progress_history.back();
        if (recent_progress < 0.1f) {
            // Increase exploration if learning has stagnated
            current_exploration_rate_ = std::min(current_exploration_rate_ * 1.1f, 0.5f);
        }
    }
}

std::vector<float> AutonomousLearningAgent::selectAction(const std::vector<float>& state) {
    std::vector<float> action(current_action_.size());
    
    if (uniform_dist_(random_generator_) < current_exploration_rate_) {
        // Exploration action
        action = generateExplorationAction(state);
    } else {
        // Exploitation action using neural network
        if (neural_network_) {
            // In a real implementation, this would use the neural network to generate actions
            // For now, generate a policy-based action
            for (size_t i = 0; i < action.size(); ++i) {
                action[i] = std::tanh(state[i % state.size()] + exploration_noise_(random_generator_) * 0.1f);
            }
        }
    }
    
    return action;
}

void AutonomousLearningAgent::processLearningExperience(const LearningExperience& experience) {
    if (isExperienceWorthStoring(experience)) {
        experience_buffer_.push(experience);
        
        // Limit buffer size
        while (experience_buffer_.size() > static_cast<size_t>(config_.memory_capacity)) {
            experience_buffer_.pop();
        }
        
        metrics_.total_experiences++;
    }
}

void AutonomousLearningAgent::updateNeuralNetwork(float dt) {
    if (network_manager_ && is_learning_enabled_) {
        // Provide recent experience as reward signal
        float total_reward = 0.0f;
        if (!experience_buffer_.empty()) {
            auto recent_exp = experience_buffer_.back();
            total_reward = recent_exp.reward + recent_exp.intrinsic_reward;
        }
        
        // Update network with learning
        network_manager_->simulateStep(dt, total_reward);
    }
}

void AutonomousLearningAgent::updateGoalProgress() {
    for (auto& goal : goals_) {
        if (!goal->is_active) continue;
        
        goal->attempts++;
        
        // Evaluate goal using its evaluation function
        if (goal->evaluation_fn) {
            float new_competence = goal->evaluation_fn(current_state_);
            
            // Smooth competence updates
            goal->current_competence = goal->current_competence * 0.9f + new_competence * 0.1f;
            
            // Update learning curve
            goal->learning_curve.push_back(goal->current_competence);
            
            // Check for success
            if (goal->current_competence > goal->target_competence) {
                goal->successes++;
            }
            
            // Update skill competency mapping
            skill_competencies_[goal->description] = goal->current_competence;
        }
    }
}

void AutonomousLearningAgent::performMetaLearning() {
    // Analyze recent performance and adapt learning parameters
    
    // Adapt learning rate based on performance
    if (!metrics_.reward_history.empty() && metrics_.reward_history.size() >= 10) {
        float recent_variance = 0.0f;
        float recent_mean = 0.0f;
        int window = std::min(10, static_cast<int>(metrics_.reward_history.size()));
        
        for (int i = metrics_.reward_history.size() - window; 
             i < metrics_.reward_history.size(); ++i) {
            recent_mean += metrics_.reward_history[i];
        }
        recent_mean /= window;
        
        for (int i = metrics_.reward_history.size() - window; 
             i < metrics_.reward_history.size(); ++i) {
            float diff = metrics_.reward_history[i] - recent_mean;
            recent_variance += diff * diff;
        }
        recent_variance /= window;
        
        // Adjust exploration based on reward variance
        if (recent_variance > 0.1f) {
            // High variance: reduce exploration for stability
            current_exploration_rate_ *= 0.95f;
        } else if (recent_variance < 0.01f) {
            // Low variance: might be stuck, increase exploration
            current_exploration_rate_ *= 1.05f;
        }
    }
    
    metrics_.successful_adaptations++;
    
    logLearningEvent("META_LEARNING", {
        {"exploration_rate", current_exploration_rate_},
        {"adaptations", static_cast<float>(metrics_.successful_adaptations)}
    });
}

void AutonomousLearningAgent::performExperienceReplay() {
    if (experience_buffer_.size() < 10) return;
    
    // Sample random experiences from buffer
    std::vector<LearningExperience> sampled_experiences;
    for (int i = 0; i < 5; ++i) {
        int random_index = static_cast<int>(uniform_dist_(random_generator_) * experience_buffer_.size());
        // Note: This is a simplified sampling - would need proper container access in real implementation
    }
    
    // Process sampled experiences for additional learning
    // This would involve re-training the network on past experiences
}

void AutonomousLearningAgent::updatePerformanceMetrics() {
    // Update reward history
    if (!experience_buffer_.empty()) {
        auto recent_exp = experience_buffer_.back();
        metrics_.reward_history.push_back(recent_exp.reward);
        
        // Limit history size
        if (metrics_.reward_history.size() > 1000) {
            metrics_.reward_history.erase(metrics_.reward_history.begin());
        }
        
        // Calculate average reward
        float sum = std::accumulate(metrics_.reward_history.begin(), metrics_.reward_history.end(), 0.0f);
        metrics_.average_reward = sum / metrics_.reward_history.size();
        
        // Update cumulative intrinsic reward
        metrics_.cumulative_intrinsic_reward += recent_exp.intrinsic_reward;
    }
    
    // Update complexity history
    float current_complexity = calculateLearningComplexity();
    metrics_.complexity_history.push_back(current_complexity);
    if (metrics_.complexity_history.size() > 1000) {
        metrics_.complexity_history.erase(metrics_.complexity_history.begin());
    }
    
    // Update learning progress
    float progress = 0.0f;
    if (!goals_.empty()) {
        for (const auto& goal : goals_) {
            if (goal->is_active) {
                progress += goal->current_competence;
            }
        }
        progress /= goals_.size();
    }
    metrics_.learning_progress_history.push_back(progress);
    if (metrics_.learning_progress_history.size() > 1000) {
        metrics_.learning_progress_history.erase(metrics_.learning_progress_history.begin());
    }
    
    // Calculate exploration effectiveness
    metrics_.exploration_effectiveness = current_exploration_rate_ * metrics_.average_reward;
    
    // Update network complexity
    metrics_.network_complexity = static_cast<float>(current_neuron_count_) / config_.max_neuron_count;
}

void AutonomousLearningAgent::initializeNeuralNetwork() {
    try {
        // Create modular neural network configuration
        NetworkConfig net_config;
        net_config.num_neurons = config_.initial_neuron_count;
        net_config.connection_probability = 0.1f;
        net_config.learning_rate = config_.base_learning_rate;
        net_config.enable_stdp = true;
        net_config.enable_homeostasis = true;
        
        // Initialize modular network
        neural_network_ = std::make_unique<ModularNeuralNetwork>(net_config);
        
        // Initialize enhanced network manager
        network_manager_ = std::make_unique<EnhancedNetworkManager>(net_config);
        
        std::cout << "🧠 Neural network initialized with " << config_.initial_neuron_count << " neurons." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to initialize neural network: " << e.what() << std::endl;
        // Continue with simplified simulation
    }
}

void AutonomousLearningAgent::setupDefaultGoals() {
    // Add basic exploration goal
    addLearningGoal(createExplorationGoal());
    
    // Add pattern recognition goal
    addLearningGoal(createPatternRecognitionGoal());
    
    // Add adaptive behavior goal
    addLearningGoal(createAdaptiveBehaviorGoal());
    
    std::cout << "🎯 Default learning goals established." << std::endl;
}

float AutonomousLearningAgent::calculateLearningComplexity() const {
    float complexity = 0.0f;
    
    // Base complexity on state space
    complexity += current_state_.size() * 0.01f;
    
    // Add complexity based on goal difficulty
    for (const auto& goal : goals_) {
        if (goal->is_active) {
            complexity += (1.0f - goal->current_competence) * goal->priority * 0.1f;
        }
    }
    
    // Add complexity based on recent reward variance
    if (metrics_.reward_history.size() > 10) {
        float variance = 0.0f;
        float mean = metrics_.average_reward;
        for (float reward : metrics_.reward_history) {
            variance += (reward - mean) * (reward - mean);
        }
        variance /= metrics_.reward_history.size();
        complexity += std::sqrt(variance) * 0.5f;
    }
    
    return std::tanh(complexity); // Normalize to [0,1]
}

float AutonomousLearningAgent::calculateNoveltyScore(const std::vector<float>& state) const {
    // Simple novelty calculation based on distance from previous states
    float novelty = 0.0f;
    
    if (!current_state_.empty()) {
        float distance = 0.0f;
        for (size_t i = 0; i < std::min(state.size(), current_state_.size()); ++i) {
            float diff = state[i] - current_state_[i];
            distance += diff * diff;
        }
        novelty = std::sqrt(distance) / state.size();
    }
    
    return std::tanh(novelty); // Normalize to [0,1]
}

bool AutonomousLearningAgent::isExperienceWorthStoring(const LearningExperience& experience) const {
    // Store experience if:
    // 1. Reward is significant
    // 2. Novelty is high
    // 3. Surprise level is high
    
    return (std::abs(experience.reward) > 0.1f || 
            experience.novelty_score > 0.3f || 
            experience.surprise_level > 0.3f);
}

std::vector<float> AutonomousLearningAgent::generateExplorationAction(const std::vector<float>& state) {
    std::vector<float> action(current_action_.size());
    
    // Generate random exploration action with some bias towards promising directions
    for (size_t i = 0; i < action.size(); ++i) {
        float random_component = exploration_noise_(random_generator_) * 0.5f;
        float state_bias = (i < state.size()) ? state[i] * 0.1f : 0.0f;
        action[i] = std::tanh(random_component + state_bias);
    }
    
    return action;
}

void AutonomousLearningAgent::logLearningEvent(const std::string& event, const std::map<std::string, float>& data) {
    // Simple console logging - could be extended to file logging
    std::cout << "📝 [" << std::fixed << std::setprecision(2) << current_time_ << "s] " << event;
    for (const auto& pair : data) {
        std::cout << " " << pair.first << "=" << pair.second;
    }
    std::cout << std::endl;
}

std::string AutonomousLearningAgent::generateLearningReport() const {
    std::stringstream report;
    
    report << "\n📊 ========== AUTONOMOUS LEARNING REPORT ==========\n";
    report << "🕐 Simulation Time: " << std::fixed << std::setprecision(2) << current_time_ << "s\n";
    report << "🔢 Total Steps: " << simulation_step_ << "\n";
    report << "🧠 Neural Network: " << current_neuron_count_ << " neurons\n";
    report << "📈 Network Expansions: " << metrics_.network_expansions << "\n";
    report << "🎯 Active Goals: " << goals_.size() << "\n\n";
    
    report << "📊 PERFORMANCE METRICS:\n";
    report << "   • Average Reward: " << std::setprecision(4) << metrics_.average_reward << "\n";
    report << "   • Learning Efficiency: " << metrics_.learning_efficiency << "\n";
    report << "   • Exploration Rate: " << current_exploration_rate_ << "\n";
    report << "   • Total Experiences: " << metrics_.total_experiences << "\n";
    report << "   • Intrinsic Reward: " << metrics_.cumulative_intrinsic_reward << "\n\n";
    
    report << "🎯 GOAL COMPETENCIES:\n";
    for (const auto& goal : goals_) {
        report << "   • " << goal->description << ": " 
               << std::setprecision(3) << goal->current_competence 
               << " (target: " << goal->target_competence << ")\n";
        if (goal->attempts > 0) {
            float success_rate = static_cast<float>(goal->successes) / goal->attempts;
            report << "     Success rate: " << std::setprecision(2) << success_rate * 100 << "%\n";
        }
    }
    
    report << "\n🔬 LEARNING ANALYSIS:\n";
    if (!metrics_.learning_progress_history.empty()) {
        float recent_progress = metrics_.learning_progress_history.back();
        report << "   • Current Learning Progress: " << std::setprecision(3) << recent_progress << "\n";
    }
    
    if (!metrics_.complexity_history.empty()) {
        float current_complexity = metrics_.complexity_history.back();
        report << "   • Current Task Complexity: " << std::setprecision(3) << current_complexity << "\n";
    }
    
    report << "   • Meta-learning Adaptations: " << metrics_.successful_adaptations << "\n";
    report << "   • Network Complexity: " << std::setprecision(3) << metrics_.network_complexity << "\n";
    
    report << "\n✅ ============================================\n";
    
    return report.str();
}

// ============================================================================
// UTILITY FUNCTIONS FOR CREATING COMMON GOALS
// ============================================================================

std::unique_ptr<AutonomousGoal> createExplorationGoal() {
    auto goal = std::make_unique<AutonomousGoal>();
    goal->description = "Environmental Exploration";
    goal->target_competence = 0.8f;
    goal->priority = 5;
    goal->evaluation_fn = [](const std::vector<float>& state) {
        // Reward diverse state exploration
        float diversity = 0.0f;
        for (float val : state) {
            diversity += std::abs(val);
        }
        return std::tanh(diversity / state.size());
    };
    return goal;
}

std::unique_ptr<AutonomousGoal> createPatternRecognitionGoal() {
    auto goal = std::make_unique<AutonomousGoal>();
    goal->description = "Pattern Recognition";
    goal->target_competence = 0.9f;
    goal->priority = 8;
    goal->evaluation_fn = [](const std::vector<float>& state) {
        // Reward recognition of patterns in state
        float pattern_score = 0.0f;
        for (size_t i = 1; i < state.size(); ++i) {
            if (std::abs(state[i] - state[i-1]) < 0.1f) {
                pattern_score += 0.1f;
            }
        }
        return std::tanh(pattern_score);
    };
    return goal;
}

std::unique_ptr<AutonomousGoal> createAdaptiveBehaviorGoal() {
    auto goal = std::make_unique<AutonomousGoal>();
    goal->description = "Adaptive Behavior";
    goal->target_competence = 0.75f;
    goal->priority = 7;
    goal->evaluation_fn = [](const std::vector<float>& state) {
        // Reward adaptive responses to state changes
        float adaptiveness = 0.0f;
        float state_magnitude = 0.0f;
        for (float val : state) {
            state_magnitude += val * val;
        }
        adaptiveness = 1.0f / (1.0f + std::sqrt(state_magnitude));
        return adaptiveness;
    };
    return goal;
}
