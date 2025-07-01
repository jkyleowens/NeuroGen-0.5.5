// ============================================================================
// AUTONOMOUS LEARNING AGENT IMPLEMENTATION - NO REDEFINITION
// File: src/AutonomousLearningAgent.cpp
// ============================================================================

#include <NeuroGen/ControllerModule.h>
#include <NeuroGen/Network.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/EnhancedNeuralModule.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>

// ============================================================================
// MEMORY SYSTEM IMPLEMENTATION
// ============================================================================

/**
 * @brief Memory system for the autonomous learning agent
 */
class MemorySystem {
public:
    // Memory trace structure (already defined in ControllerModule.h)
    using MemoryTrace = ::MemoryTrace;
    
    // Episode storage and retrieval
    std::vector<MemoryTrace> episodic_memory_;
    std::map<std::string, std::vector<float>> semantic_memory_;
    
    // Memory management
    size_t max_episodes_;
    float consolidation_threshold_;
    
    MemorySystem(size_t max_episodes = 10000) 
        : max_episodes_(max_episodes), consolidation_threshold_(0.7f) {}
    
    void storeEpisode(const MemoryTrace& trace) {
        episodic_memory_.push_back(trace);
        
        // Maintain memory capacity
        if (episodic_memory_.size() > max_episodes_) {
            // Remove oldest memories with low importance
            auto it = std::min_element(episodic_memory_.begin(), episodic_memory_.end(),
                [](const MemoryTrace& a, const MemoryTrace& b) {
                    return a.importance_weight < b.importance_weight;
                });
            if (it != episodic_memory_.end() && it->importance_weight < consolidation_threshold_) {
                episodic_memory_.erase(it);
            }
        }
    }
    
    std::vector<MemoryTrace> retrieveSimilarEpisodes(
        const BrowsingState& current_state, size_t max_results = 10) {
        std::vector<std::pair<float, MemoryTrace*>> similarities;
        
        // Extract features from current state for comparison
        std::vector<float> current_features = extractStateFeatures(current_state);
        
        for (auto& episode : episodic_memory_) {
            float similarity = computeCosineSimilarity(current_features, episode.state_vector);
            similarities.emplace_back(similarity, &episode);
        }
        
        // Sort by similarity and return top results
        std::sort(similarities.begin(), similarities.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        
        std::vector<MemoryTrace> results;
        size_t count = std::min(max_results, similarities.size());
        for (size_t i = 0; i < count; ++i) {
            results.push_back(*similarities[i].second);
        }
        
        return results;
    }
    
private:
    std::vector<float> extractStateFeatures(const BrowsingState& state) {
        std::vector<float> features;
        
        // URL features (simple hash-based encoding)
        features.push_back(static_cast<float>(std::hash<std::string>{}(state.current_url) % 1000) / 1000.0f);
        
        // Page element features
        features.push_back(static_cast<float>(state.page_elements.size()) / 100.0f);
        
        // Scroll position feature
        features.push_back(static_cast<float>(state.scroll_position) / 10000.0f);
        
        // Window dimensions
        features.push_back(static_cast<float>(state.window_width) / 2000.0f);
        features.push_back(static_cast<float>(state.window_height) / 2000.0f);
        
        // Loading state
        features.push_back(state.page_loading ? 1.0f : 0.0f);
        
        return features;
    }
    
    float computeCosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) return 0.0f;
        
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        
        for (size_t i = 0; i < a.size(); ++i) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
        
        return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
};

// ============================================================================
// PERFORMANCE STATISTICS STRUCTURE
// ============================================================================

/**
 * @brief Performance statistics structure
 */
struct PerformanceStats {
    float total_actions_taken;
    float successful_actions;
    float average_reward;
    float exploration_rate;
    std::map<ActionType, int> action_counts;
    std::map<std::string, float> context_performance;
    
    PerformanceStats() : 
        total_actions_taken(0.0f),
        successful_actions(0.0f), 
        average_reward(0.0f),
        exploration_rate(0.1f) {}
};

// ============================================================================
// AUTONOMOUS LEARNING AGENT METHODS IMPLEMENTATION
// ============================================================================

// Constructor implementation for AutonomousLearningAgent
AutonomousLearningAgent::AutonomousLearningAgent(const NetworkConfig& config) {
    // Initialize controller module
    controller_module_ = std::make_unique<ControllerModule>("controller", config);
    
    // Initialize memory system
    memory_system_ = std::make_unique<MemorySystem>();
    
    std::cout << "AutonomousLearningAgent created with configuration" << std::endl;
}

// Initialize method implementation
bool AutonomousLearningAgent::initialize() {
    if (!controller_module_) {
        std::cerr << "Error: Controller module not created" << std::endl;
        return false;
    }
    
    if (!controller_module_->initialize()) {
        std::cerr << "Error: Failed to initialize controller module" << std::endl;
        return false;
    }
    
    std::cout << "AutonomousLearningAgent initialized successfully" << std::endl;
    return true;
}

// Update method implementation
void AutonomousLearningAgent::update(float dt) {
    if (controller_module_) {
        controller_module_->update(dt);
    }
}

// Shutdown method implementation
void AutonomousLearningAgent::shutdown() {
    if (controller_module_) {
        controller_module_->shutdown();
    }
    std::cout << "AutonomousLearningAgent shutdown complete" << std::endl;
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
    return controller_module_ ? controller_module_->select_action_with_exploration(candidates, action_values) : BrowsingAction{};
}

void AutonomousLearningAgent::execute_action() {
    if (controller_module_) {
        controller_module_->execute_action(selected_action_);
    }
}

void AutonomousLearningAgent::execute_click_action() {
    if (controller_module_) {
        controller_module_->execute_click_action(selected_action_);
    }
}

void AutonomousLearningAgent::execute_scroll_action() {
    if (controller_module_) {
        controller_module_->execute_scroll_action(selected_action_);
    }
}

void AutonomousLearningAgent::execute_type_action() {
    if (controller_module_) {
        controller_module_->execute_type_action(selected_action_);
    }
}

void AutonomousLearningAgent::execute_navigate_action() {
    if (controller_module_) {
        controller_module_->execute_navigate_action(selected_action_);
    }
}

void AutonomousLearningAgent::execute_wait_action() {
    if (controller_module_) {
        controller_module_->execute_wait_action(selected_action_);
    }
}

// ============================================================================
// UTILITY FUNCTION IMPLEMENTATIONS - FIX: Use global types
// ============================================================================

/**
 * @brief Converts ActionType enum to string representation
 */
std::string actionTypeToString(ActionType type) {
    switch (type) {
        case ActionType::CLICK: return "CLICK";
        case ActionType::SCROLL: return "SCROLL";
        case ActionType::TYPE: return "TYPE";
        case ActionType::NAVIGATE: return "NAVIGATE";
        case ActionType::WAIT: return "WAIT";
        case ActionType::OBSERVE: return "OBSERVE";
        case ActionType::BACK: return "BACK";
        case ActionType::FORWARD: return "FORWARD";
        case ActionType::REFRESH: return "REFRESH";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Converts string to ActionType enum
 * FIX: Use global ActionType, not AutonomousLearningAgent::ActionType
 */
ActionType stringToActionType(const std::string& type_str) {
    if (type_str == "CLICK") return ActionType::CLICK;
    if (type_str == "SCROLL") return ActionType::SCROLL;
    if (type_str == "TYPE") return ActionType::TYPE;
    if (type_str == "NAVIGATE") return ActionType::NAVIGATE;
    if (type_str == "WAIT") return ActionType::WAIT;
    if (type_str == "OBSERVE") return ActionType::OBSERVE;
    if (type_str == "BACK") return ActionType::BACK;
    if (type_str == "FORWARD") return ActionType::FORWARD;
    if (type_str == "REFRESH") return ActionType::REFRESH;
    return ActionType::OBSERVE; // Default fallback
}

/**
 * @brief Computes similarity between two browsing states
 * FIX: Use global BrowsingState, not AutonomousLearningAgent::BrowsingState
 */
float computeBrowsingStateSimilarity(const BrowsingState& state1,
                                   const BrowsingState& state2) {
    float similarity = 0.0f;
    float weight_sum = 0.0f;
    
    // URL similarity (exact match or domain match)
    float url_weight = 0.3f;
    if (state1.current_url == state2.current_url) {
        similarity += url_weight * 1.0f;
    } else {
        // Extract domain and compare
        auto extractDomain = [](const std::string& url) {
            size_t start = url.find("://");
            if (start != std::string::npos) {
                start += 3;
                size_t end = url.find("/", start);
                return (end != std::string::npos) ? url.substr(start, end - start) : url.substr(start);
            }
            return url;
        };
        
        if (extractDomain(state1.current_url) == extractDomain(state2.current_url)) {
            similarity += url_weight * 0.5f;
        }
    }
    weight_sum += url_weight;
    
    // Page structure similarity (number of elements)
    float structure_weight = 0.2f;
    float element_diff = std::abs(static_cast<float>(state1.page_elements.size()) - 
                                 static_cast<float>(state2.page_elements.size()));
    float max_elements = std::max(state1.page_elements.size(), state2.page_elements.size());
    if (max_elements > 0) {
        similarity += structure_weight * (1.0f - element_diff / max_elements);
    }
    weight_sum += structure_weight;
    
    // Scroll position similarity
    float scroll_weight = 0.1f;
    float scroll_diff = std::abs(state1.scroll_position - state2.scroll_position);
    float max_scroll = std::max(std::abs(state1.scroll_position), std::abs(state2.scroll_position));
    if (max_scroll > 0) {
        similarity += scroll_weight * (1.0f - std::min(1.0f, scroll_diff / max_scroll));
    } else {
        similarity += scroll_weight; // Both at same position
    }
    weight_sum += scroll_weight;
    
    // Window dimensions similarity
    float window_weight = 0.1f;
    float width_diff = std::abs(state1.window_width - state2.window_width);
    float height_diff = std::abs(state1.window_height - state2.window_height);
    float max_width = std::max(state1.window_width, state2.window_width);
    float max_height = std::max(state1.window_height, state2.window_height);
    
    float window_sim = 0.0f;
    if (max_width > 0) window_sim += 0.5f * (1.0f - width_diff / max_width);
    if (max_height > 0) window_sim += 0.5f * (1.0f - height_diff / max_height);
    similarity += window_weight * window_sim;
    weight_sum += window_weight;
    
    // Visual features similarity (if available)
    float visual_weight = 0.3f;
    if (!state1.visual_features.empty() && !state2.visual_features.empty() &&
        state1.visual_features.size() == state2.visual_features.size()) {
        
        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        
        for (size_t i = 0; i < state1.visual_features.size(); ++i) {
            dot_product += state1.visual_features[i] * state2.visual_features[i];
            norm1 += state1.visual_features[i] * state1.visual_features[i];
            norm2 += state2.visual_features[i] * state2.visual_features[i];
        }
        
        if (norm1 > 0.0f && norm2 > 0.0f) {
            float cosine_sim = dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
            similarity += visual_weight * std::max(0.0f, cosine_sim);
        }
    }
    weight_sum += visual_weight;
    
    return (weight_sum > 0.0f) ? similarity / weight_sum : 0.0f;
}

/**
 * @brief Extracts features from a browsing action for learning
 */
std::vector<float> extractActionFeatures(const BrowsingAction& action) {
    std::vector<float> features;
    
    // Action type (one-hot encoding)
    for (int i = 0; i <= static_cast<int>(ActionType::REFRESH); ++i) {
        features.push_back((static_cast<int>(action.type) == i) ? 1.0f : 0.0f);
    }
    
    // Spatial features (normalized coordinates)
    features.push_back(static_cast<float>(action.parameters.x) / 1920.0f); // Assume max 1920px
    features.push_back(static_cast<float>(action.parameters.y) / 1080.0f); // Assume max 1080px
    
    // Scroll amount (normalized)
    features.push_back(std::tanh(action.parameters.scroll_amount / 1000.0f));
    
    // Wait duration (normalized)
    features.push_back(std::tanh(action.parameters.wait_duration / 10.0f));
    
    // Text length (if applicable)
    features.push_back(std::tanh(static_cast<float>(action.parameters.text.length()) / 100.0f));
    
    // Confidence and expected reward
    features.push_back(action.confidence);
    features.push_back(std::tanh(action.expected_reward));
    
    // Priority (normalized)
    features.push_back(std::tanh(static_cast<float>(action.priority) / 10.0f));
    
    // Binary features
    features.push_back(action.requires_confirmation ? 1.0f : 0.0f);
    features.push_back(action.followup_actions.empty() ? 0.0f : 1.0f);
    
    return features;
}

// ============================================================================
// PERFORMANCE OPTIMIZATION HELPERS
// ============================================================================

namespace PerformanceUtils {
    /**
     * @brief Compute action value using temporal difference learning
     */
    float computeActionValue(const BrowsingAction& action, 
                           const std::vector<MemoryTrace>& similar_episodes,
                           float discount_factor = 0.95f) {
        if (similar_episodes.empty()) {
            return action.expected_reward;
        }
        
        float total_value = 0.0f;
        float weight_sum = 0.0f;
        
        for (const auto& episode : similar_episodes) {
            // Compute similarity weight based on action features
            std::vector<float> action_features = extractActionFeatures(action);
            float similarity = 1.0f; // Simplified - would need action comparison
            
            // Discounted future reward
            float discounted_reward = episode.reward_received * 
                std::pow(discount_factor, episode.temporal_discount);
            
            total_value += similarity * discounted_reward;
            weight_sum += similarity;
        }
        
        return (weight_sum > 0.0f) ? total_value / weight_sum : action.expected_reward;
    }
    
    /**
     * @brief Update exploration rate based on performance
     */
    float updateExplorationRate(float current_rate, float recent_performance, 
                              float target_performance = 0.8f) {
        const float min_rate = 0.01f;
        const float max_rate = 0.5f;
        const float adaptation_speed = 0.01f;
        
        if (recent_performance < target_performance) {
            // Increase exploration if performance is low
            return std::min(max_rate, current_rate + adaptation_speed);
        } else {
            // Decrease exploration if performance is good
            return std::max(min_rate, current_rate - adaptation_speed * 0.5f);
        }
    }
}

// ============================================================================
// NETWORK INTEGRATION HELPERS
// ============================================================================

namespace NetworkIntegration {
    /**
     * @brief Create a specialized neural module for specific tasks
     */
    std::unique_ptr<EnhancedNeuralModule> createSpecializedModule(
        const std::string& module_name,
        const NetworkConfig& config,
        const std::string& specialization) {
        
        auto module = std::make_unique<EnhancedNeuralModule>(module_name, config);
        
        if (!module->initialize()) {
            std::cerr << "Failed to initialize specialized module: " << module_name << std::endl;
            return nullptr;
        }
        
        // Set up specialization-specific parameters
        if (specialization == "vision") {
            module->setDevelopmentalStage(3); // Mature visual processing
            module->setAttentionWeight(0.8f); // High attention for vision
        } else if (specialization == "memory") {
            module->setDevelopmentalStage(2); // Developing memory systems
            module->setAttentionWeight(0.6f); // Moderate attention for memory
        } else if (specialization == "motor") {
            module->setDevelopmentalStage(3); // Mature motor control
            module->setAttentionWeight(0.9f); // Very high attention for motor actions
        }
        
        std::cout << "Created specialized " << specialization 
                  << " module: " << module_name << std::endl;
        
        return module;
    }
    
    /**
     * @brief Establish connections between neural modules
     */
    void connectModules(EnhancedNeuralModule& source_module,
                       EnhancedNeuralModule& target_module,
                       const std::string& source_port,
                       const std::string& target_port,
                       float connection_strength = 1.0f) {
        
        EnhancedNeuralModule::InterModuleConnection connection;
        connection.source_port = source_port;
        connection.target_module = target_module.get_name();
        connection.target_port = target_port;
        connection.connection_strength = connection_strength;
        connection.is_feedback = false;
        connection.delay_ms = 5.0f; // 5ms transmission delay
        
        source_module.registerInterModuleConnection(connection);
        
        std::cout << "Connected " << source_module.get_name() 
                  << ":" << source_port << " -> " 
                  << target_module.get_name() << ":" << target_port
                  << " (strength: " << connection_strength << ")" << std::endl;
    }
}