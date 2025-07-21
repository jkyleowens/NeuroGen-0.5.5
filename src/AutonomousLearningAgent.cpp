// ============================================================================
// AUTONOMOUS LEARNING AGENT - NATURAL LANGUAGE PROCESSING FOCUSED
// File: src/AutonomousLearningAgent.cpp
// ============================================================================

#include <NeuroGen/AutonomousLearningAgent.h>
#include <NeuroGen/BrainModuleArchitecture.h>
#include <NeuroGen/SafetyManager.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <sstream>
#include <fstream>
#include <iomanip>

// ============================================================================
// AUTONOMOUS LEARNING AGENT - LANGUAGE PROCESSING METHODS
// ============================================================================

void AutonomousLearningAgent::process_language_input() {
    if (!language_interface_) return;
    
    // Capture and process current language context
    std::string current_text = language_interface_->getCurrentTextContext();
    std::vector<float> language_features = extractLanguageFeatures(current_text);
    
    // Send to language comprehension module for processing
    if (modules_.count("language_comprehension")) {
        float language_attention = attention_controller_->get_attention_weight("language_comprehension");
        
        // Apply attention to input before processing
        std::vector<float> attended_language_features = language_features;
        for (size_t i = 0; i < attended_language_features.size(); ++i) {
            attended_language_features[i] *= language_attention;
        }
        
        auto language_output = modules_["language_comprehension"]->process(attended_language_features);
        
        // Store language features in environmental context
        size_t context_language_size = std::min(language_output.size(), environmental_context_.size() / 2);
        for (size_t i = 0; i < context_language_size; ++i) {
            environmental_context_[i] = language_output[i];
        }
        
        // Process through semantic memory for deeper understanding
        if (modules_.count("semantic_memory")) {
            auto semantic_output = modules_["semantic_memory"]->process(language_output);
            
            // Store semantic representation
            size_t semantic_offset = environmental_context_.size() / 4;
            size_t semantic_size = std::min(semantic_output.size(), environmental_context_.size() / 4);
            for (size_t i = 0; i < semantic_size; ++i) {
                if (semantic_offset + i < environmental_context_.size()) {
                    environmental_context_[semantic_offset + i] = semantic_output[i];
                }
            }
        }
    }
}

void AutonomousLearningAgent::process_language_generation() {
    if (!modules_.count("language_production")) return;
    
    // Gather context for language generation
    std::vector<float> generation_context;
    generation_context.reserve(1024);
    
    // Add current language understanding
    if (modules_.count("language_comprehension")) {
        auto comprehension_state = modules_["language_comprehension"]->get_output();
        size_t comp_size = std::min(comprehension_state.size(), size_t(256));
        generation_context.insert(generation_context.end(), 
                                comprehension_state.begin(), 
                                comprehension_state.begin() + comp_size);
    }
    
    // Add semantic memory context
    if (modules_.count("semantic_memory")) {
        auto semantic_state = modules_["semantic_memory"]->get_output();
        size_t sem_size = std::min(semantic_state.size(), size_t(256));
        generation_context.insert(generation_context.end(),
                                semantic_state.begin(),
                                semantic_state.begin() + sem_size);
    }
    
    // Add working memory content
    auto working_memory = memory_system_->get_working_memory();
    size_t wm_size = std::min(working_memory.size(), size_t(256));
    generation_context.insert(generation_context.end(),
                            working_memory.begin(),
                            working_memory.begin() + wm_size);
    
    // Add current goals and context
    size_t goal_size = std::min(current_goals_.size(), size_t(128));
    generation_context.insert(generation_context.end(),
                            current_goals_.begin(),
                            current_goals_.begin() + goal_size);
    
    // Process through language production module
    float generation_attention = attention_controller_->get_attention_weight("language_production");
    
    // Apply attention to generation context
    for (size_t i = 0; i < generation_context.size(); ++i) {
        generation_context[i] *= generation_attention;
    }
    
    auto generation_output = modules_["language_production"]->process(generation_context);
    
    // Convert neural output to language and store for response
    current_language_response_ = convertNeuralToLanguage(generation_output);
}

void AutonomousLearningAgent::update_working_memory() {
    if (!modules_.count("working_memory")) return;
    
    // Combine current language input with existing working memory
    std::vector<float> working_memory_input;
    working_memory_input.reserve(1024);
    
    // Add language context (no visual context)
    for (size_t i = 0; i < std::min(environmental_context_.size() / 2, size_t(384)); ++i) {
        working_memory_input.push_back(environmental_context_[i]);
    }
    
    // Add current goals
    for (size_t i = 0; i < std::min(current_goals_.size(), size_t(192)); ++i) {
        working_memory_input.push_back(current_goals_[i]);
    }
    
    // Add previous working memory content
    auto prev_working_memory = memory_system_->get_working_memory();
    for (size_t i = 0; i < std::min(prev_working_memory.size(), size_t(192)); ++i) {
        working_memory_input.push_back(prev_working_memory[i]);
    }
    
    // Add syntactic processing state if available
    if (modules_.count("syntactic_processor")) {
        auto syntactic_state = modules_["syntactic_processor"]->get_output();
        size_t synt_size = std::min(syntactic_state.size(), size_t(128));
        working_memory_input.insert(working_memory_input.end(),
                                   syntactic_state.begin(),
                                   syntactic_state.begin() + synt_size);
    }
    
    // Process through working memory module
    float wm_attention = attention_controller_->get_attention_weight("working_memory");
    
    // Apply attention to input before processing
    std::vector<float> attended_wm_input = working_memory_input;
    for (size_t i = 0; i < attended_wm_input.size(); ++i) {
        attended_wm_input[i] *= wm_attention;
    }
    
    auto wm_output = modules_["working_memory"]->process(attended_wm_input);
    
    // Update memory system
    memory_system_->update_working_memory(wm_output);
}

void AutonomousLearningAgent::update_attention_weights() {
    // Prepare context for attention computation (language-focused)
    std::vector<float> attention_context;
    attention_context.reserve(512);
    
    // Add language processing saliency
    if (language_interface_) {
        auto language_importance = language_interface_->getLanguageImportanceMap();
        size_t imp_size = std::min(language_importance.size(), size_t(128));
        for (size_t i = 0; i < imp_size; ++i) {
            attention_context.push_back(language_importance[i]);
        }
    }
    
    // Add current comprehension state
    if (modules_.count("language_comprehension")) {
        auto comp_state = modules_["language_comprehension"]->get_output();
        size_t comp_size = std::min(comp_state.size(), size_t(128));
        for (size_t i = 0; i < comp_size; ++i) {
            attention_context.push_back(comp_state[i]);
        }
    }
    
    // Add working memory load
    auto working_memory = memory_system_->get_working_memory();
    size_t wm_size = std::min(working_memory.size(), size_t(128));
    for (size_t i = 0; i < wm_size; ++i) {
        attention_context.push_back(working_memory[i]);
    }
    
    // Add current task demands
    size_t goal_size = std::min(current_goals_.size(), size_t(64));
    for (size_t i = 0; i < goal_size; ++i) {
        attention_context.push_back(current_goals_[i]);
    }
    
    // Add global reward signal for attention modulation
    attention_context.push_back(global_reward_signal_);
    attention_context.push_back(simulation_time_ / 1000.0f); // Normalized time
    
    // Process through attention system
    if (modules_.count("attention_system")) {
        auto attention_output = modules_["attention_system"]->process(attention_context);
        
        // Update attention weights for language modules using individual weight setting
        std::vector<float> attention_weights = attention_output;
        updateAttentionWeights(attention_weights);
        
        // Set module-specific attention weights
        if (attention_weights.size() >= 6) {
            attention_controller_->set_attention_weight("language_comprehension", 
                                                      std::max(0.1f, std::min(1.0f, attention_weights[0])));
            attention_controller_->set_attention_weight("language_production", 
                                                      std::max(0.1f, std::min(1.0f, attention_weights[1])));
            attention_controller_->set_attention_weight("semantic_memory", 
                                                      std::max(0.1f, std::min(1.0f, attention_weights[2])));
            attention_controller_->set_attention_weight("syntactic_processor", 
                                                      std::max(0.1f, std::min(1.0f, attention_weights[3])));
            attention_controller_->set_attention_weight("working_memory", 
                                                      std::max(0.1f, std::min(1.0f, attention_weights[4])));
            attention_controller_->set_attention_weight("executive_function", 
                                                      std::max(0.1f, std::min(1.0f, attention_weights[5])));
        }
    }
}

std::vector<float> AutonomousLearningAgent::gather_module_input(const std::string& target_module) {
    std::vector<float> combined_input;
    combined_input.reserve(2048);
    
    // Add signals from connected modules (excluding visual cortex)
    for (auto& [module_name, module] : modules_) {
        if (module_name != target_module && module_name != "visual_cortex") {
            // Use general get_output method instead of specific get_output_for_module
            auto signal = module->get_output();
            if (!signal.empty()) {
                // Limit input size to prevent overwhelming
                size_t signal_size = std::min(signal.size(), size_t(256));
                combined_input.insert(combined_input.end(), signal.begin(), signal.begin() + signal_size);
            }
        }
    }
    
    // Add environmental context relevant to the module (language-focused)
    if (target_module == "language_comprehension") {
        // Add raw language features
        for (size_t i = 0; i < std::min(environmental_context_.size() / 2, size_t(384)); ++i) {
            combined_input.push_back(environmental_context_[i]);
        }
    } else if (target_module == "language_production") {
        // Add semantic context for generation
        size_t semantic_offset = environmental_context_.size() / 4;
        for (size_t i = 0; i < std::min(environmental_context_.size() / 4, size_t(256)); ++i) {
            if (semantic_offset + i < environmental_context_.size()) {
                combined_input.push_back(environmental_context_[semantic_offset + i]);
            }
        }
    } else if (target_module == "executive_function") {
        // Add working memory and goals
        auto working_memory = memory_system_->get_working_memory();
        for (size_t i = 0; i < std::min(working_memory.size(), size_t(192)); ++i) {
            combined_input.push_back(working_memory[i]);
        }
        for (size_t i = 0; i < std::min(current_goals_.size(), size_t(96)); ++i) {
            combined_input.push_back(current_goals_[i]);
        }
    } else if (target_module == "semantic_memory") {
        // Add language comprehension output for semantic processing
        if (modules_.count("language_comprehension")) {
            auto comp_output = modules_["language_comprehension"]->get_output();
            size_t comp_size = std::min(comp_output.size(), size_t(256));
            combined_input.insert(combined_input.end(), 
                                comp_output.begin(), 
                                comp_output.begin() + comp_size);
        }
    }
    
    return combined_input;
}

void AutonomousLearningAgent::distribute_module_output(const std::string& source_module, 
                                                      const std::vector<float>& output) {
    // Send output to all connected modules (excluding visual cortex)
    for (auto& [target_module, module] : modules_) {
        if (target_module != source_module && target_module != "visual_cortex") {
            // Limit signal size for efficient processing
            size_t signal_size = std::min(output.size(), size_t(256));
            std::vector<float> limited_output(output.begin(), output.begin() + signal_size);
            module->receive_signal(limited_output, source_module, "semantic");
        }
    }
    
    // Update global state based on key language modules
    if (source_module == "executive_function") {
        // Executive decisions affect global state
        size_t update_size = std::min(output.size(), global_state_.size() / 4);
        for (size_t i = 0; i < update_size; ++i) {
            global_state_[i] = global_state_[i] * 0.9f + output[i] * 0.1f;
        }
    } else if (source_module == "language_comprehension") {
        // Language understanding affects cognitive state
        size_t update_size = std::min(output.size(), global_state_.size() / 6);
        size_t offset = global_state_.size() / 6;
        for (size_t i = 0; i < update_size && (offset + i) < global_state_.size(); ++i) {
            global_state_[offset + i] = global_state_[offset + i] * 0.85f + output[i] * 0.15f;
        }
    } else if (source_module == "reward_system") {
        // Reward signals modulate global learning
        if (!output.empty()) {
            global_reward_signal_ = global_reward_signal_ * 0.95f + output[0] * 0.05f;
        }
    }
}

void AutonomousLearningAgent::make_decision() {
    // Decision making based on language understanding and goals
    
    if (!modules_.count("executive_function")) return;
    
    // Gather decision context from language processing modules
    std::vector<float> decision_context;
    decision_context.reserve(1024);
    
    // Add language comprehension state
    if (modules_.count("language_comprehension")) {
        auto comp_state = modules_["language_comprehension"]->get_output();
        size_t comp_size = std::min(comp_state.size(), size_t(256));
        decision_context.insert(decision_context.end(), 
                              comp_state.begin(), 
                              comp_state.begin() + comp_size);
    }
    
    // Add semantic memory context
    if (modules_.count("semantic_memory")) {
        auto sem_state = modules_["semantic_memory"]->get_output();
        size_t sem_size = std::min(sem_state.size(), size_t(256));
        decision_context.insert(decision_context.end(),
                              sem_state.begin(),
                              sem_state.begin() + sem_size);
    }
    
    // Add working memory
    auto working_memory = memory_system_->get_working_memory();
    size_t wm_size = std::min(working_memory.size(), size_t(256));
    decision_context.insert(decision_context.end(),
                          working_memory.begin(),
                          working_memory.begin() + wm_size);
    
    // Add current goals and constraints
    size_t goal_size = std::min(current_goals_.size(), size_t(128));
    decision_context.insert(decision_context.end(),
                          current_goals_.begin(),
                          current_goals_.begin() + goal_size);
    
    // Add global context
    decision_context.push_back(global_reward_signal_);
    decision_context.push_back(simulation_time_ / 1000.0f);
    
    // Process through executive function module
    float exec_attention = attention_controller_->get_attention_weight("executive_function");
    
    // Apply attention weighting
    for (size_t i = 0; i < decision_context.size(); ++i) {
        decision_context[i] *= exec_attention;
    }
    
    auto decision_output = modules_["executive_function"]->process(decision_context);
    
    // Extract decision from neural output
    if (!decision_output.empty()) {
        // Simple decision extraction - could be more sophisticated
        float decision_value = decision_output[0];
        
        if (decision_value > 0.7f) {
            current_decision_ = "generate_response";
        } else if (decision_value > 0.4f) {
            current_decision_ = "seek_more_information";
        } else if (decision_value > 0.1f) {
            current_decision_ = "consolidate_memory";
        } else {
            current_decision_ = "wait_and_observe";
        }
        
        // Update decision confidence
        decision_confidence_ = std::abs(decision_value);
    }
}

void AutonomousLearningAgent::execute_action() {
    // Execute action based on current decision (language-focused)
    
    if (current_decision_ == "generate_response") {
        process_language_generation();
        
        // Output response if available
        if (!current_language_response_.empty() && language_interface_) {
            language_interface_->outputResponse(current_language_response_);
            std::cout << "Generated Response: " << current_language_response_ << std::endl;
        }
        
    } else if (current_decision_ == "seek_more_information") {
        // Request more input or context
        if (language_interface_) {
            language_interface_->requestMoreInput();
        }
        
    } else if (current_decision_ == "consolidate_memory") {
        // Trigger memory consolidation
        if (memory_system_) {
            memory_system_->consolidateMemories();
        }
        
    } else if (current_decision_ == "wait_and_observe") {
        // Continue processing current context
        // No immediate action needed
    }
    
    // Log action for learning (using a simple logging approach since SafetyManager doesn't have logAction)
    if (safety_manager_) {
        // Note: SafetyManager uses recordAction for BrowsingAction, but we're doing language actions
        // For now, just log to console or implement a simple logging mechanism
        std::cout << "Action Decision: " << current_decision_ 
                  << " (confidence: " << decision_confidence_ << ")" << std::endl;
    }
}

void AutonomousLearningAgent::transfer_knowledge_between_modules() {
    // Language-focused knowledge transfer between related modules
    
    // Language Comprehension -> Semantic Memory transfer
    if (modules_.count("language_comprehension") && modules_.count("semantic_memory")) {
        auto comp_state = modules_["language_comprehension"]->get_output();
        std::vector<float> semantic_patterns(comp_state.begin(), 
                                           comp_state.begin() + std::min(comp_state.size(), size_t(256)));
        modules_["semantic_memory"]->receive_signal(semantic_patterns, "language_comprehension", "semantic");
    }
    
    // Semantic Memory -> Language Production transfer
    if (modules_.count("semantic_memory") && modules_.count("language_production")) {
        auto semantic_state = modules_["semantic_memory"]->get_output();
        std::vector<float> generation_patterns(semantic_state.begin(),
                                             semantic_state.begin() + std::min(semantic_state.size(), size_t(256)));
        modules_["language_production"]->receive_signal(generation_patterns, "semantic_memory", "semantic");
    }
    
    // Working Memory -> Executive Function transfer
    if (modules_.count("working_memory") && modules_.count("executive_function")) {
        auto wm_state = modules_["working_memory"]->get_output();
        std::vector<float> executive_patterns(wm_state.begin(),
                                            wm_state.begin() + std::min(wm_state.size(), size_t(192)));
        modules_["executive_function"]->receive_signal(executive_patterns, "working_memory", "control");
    }
    
    // Syntactic Processor -> Language Production transfer
    if (modules_.count("syntactic_processor") && modules_.count("language_production")) {
        auto syntactic_state = modules_["syntactic_processor"]->get_output();
        std::vector<float> syntax_patterns(syntactic_state.begin(),
                                         syntactic_state.begin() + std::min(syntactic_state.size(), size_t(128)));
        modules_["language_production"]->receive_signal(syntax_patterns, "syntactic_processor", "syntactic");
    }
}

// ============================================================================
// LANGUAGE PROCESSING UTILITIES
// ============================================================================

std::vector<float> AutonomousLearningAgent::extractLanguageFeatures(const std::string& text) const {
    // Enhanced language feature extraction
    std::vector<float> features(768, 0.0f); // Increased to 768 dimensions for richer representation
    
    if (text.empty()) return features;
    
    // Basic text statistics
    features[0] = std::min(1.0f, text.length() / 500.0f); // Normalized length
    features[1] = std::min(1.0f, std::count(text.begin(), text.end(), ' ') / 50.0f); // Word count
    features[2] = std::min(1.0f, std::count(text.begin(), text.end(), '.') / 10.0f); // Sentence count
    features[3] = std::min(1.0f, std::count(text.begin(), text.end(), '?') / 5.0f); // Question count
    features[4] = std::min(1.0f, std::count(text.begin(), text.end(), '!') / 5.0f); // Exclamation count
    
    // Character-level features (first 500 characters)
    for (size_t i = 0; i < text.length() && i < 500; ++i) {
        if (i + 10 < features.size()) {
            features[i + 10] = static_cast<float>(text[i]) / 255.0f;
        }
    }
    
    // Simple word-level features (frequency of common words)
    std::vector<std::string> common_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "are", "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "shall", "must",
        "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "this", "that", "these", "those", "here", "there", "where", "when", "why", "how", "what", "who"
    };
    
    for (size_t i = 0; i < common_words.size() && (i + 520) < features.size(); ++i) {
        std::string word = common_words[i];
        size_t count = 0;
        size_t pos = 0;
        while ((pos = text.find(word, pos)) != std::string::npos) {
            count++;
            pos += word.length();
        }
        features[i + 520] = std::min(1.0f, count / 10.0f);
    }
    
    return features;
}

std::string AutonomousLearningAgent::convertNeuralToLanguage(const std::vector<float>& neural_features) const {
    // Enhanced neural output to language conversion
    if (neural_features.empty()) return "No response generated.";
    
    // Calculate various activation patterns
    float avg_activation = 0.0f;
    float max_activation = -1.0f;
    float min_activation = 1.0f;
    float variance = 0.0f;
    
    for (float value : neural_features) {
        avg_activation += value;
        max_activation = std::max(max_activation, value);
        min_activation = std::min(min_activation, value);
    }
    avg_activation /= neural_features.size();
    
    // Calculate variance
    for (float value : neural_features) {
        variance += (value - avg_activation) * (value - avg_activation);
    }
    variance /= neural_features.size();
    
    // Generate response based on activation patterns
    std::string response;
    
    if (avg_activation > 0.7f) {
        if (variance > 0.3f) {
            response = "I have a clear understanding of your request and can provide a comprehensive response. ";
        } else {
            response = "I understand your request with high confidence. ";
        }
    } else if (avg_activation > 0.4f) {
        if (variance > 0.2f) {
            response = "I'm processing your request and working to provide an appropriate response. ";
        } else {
            response = "I'm analyzing your input to better understand what you're asking. ";
        }
    } else if (avg_activation > 0.1f) {
        response = "I'm working to understand your request. Could you provide more context? ";
    } else {
        response = "I need more information to provide a meaningful response. ";
    }
    
    // Add response type based on activation distribution
    if (max_activation > 0.8f) {
        response += "This appears to be a clear and specific query.";
    } else if (max_activation > 0.5f) {
        response += "Let me help you with this.";
    } else {
        response += "Please feel free to clarify your request.";
    }
    
    return response;
}

void AutonomousLearningAgent::initialize_attention_system() {
    // Set up attention priorities for language processing contexts
    if (attention_controller_) {
        attention_controller_->set_priority("language_comprehension", 0.9f);
        attention_controller_->set_priority("language_production", 0.8f);
        attention_controller_->set_priority("semantic_memory", 0.7f);
        attention_controller_->set_priority("syntactic_processor", 0.6f);
        attention_controller_->set_priority("working_memory", 0.7f);
        attention_controller_->set_priority("executive_function", 0.8f);
        attention_controller_->set_priority("memory_consolidation", 0.5f);
    }
}

std::map<std::string, float> AutonomousLearningAgent::getAttentionWeights() const {
    std::map<std::string, float> weights;
    
    if (attention_controller_) {
        // Get weights for all language processing modules
        weights["language_comprehension"] = attention_controller_->get_attention_weight("language_comprehension");
        weights["language_production"] = attention_controller_->get_attention_weight("language_production");
        weights["semantic_memory"] = attention_controller_->get_attention_weight("semantic_memory");
        weights["syntactic_processor"] = attention_controller_->get_attention_weight("syntactic_processor");
        weights["working_memory"] = attention_controller_->get_attention_weight("working_memory");
        weights["executive_function"] = attention_controller_->get_attention_weight("executive_function");
        weights["episodic_memory"] = attention_controller_->get_attention_weight("episodic_memory");
        weights["reward_system"] = attention_controller_->get_attention_weight("reward_system");
    }
    
    return weights;
}

int AutonomousLearningAgent::getModuleNeuronCount(const std::string& module_name) const {
    // Return neuron counts based on language-focused neural architecture
    if (module_name == "language_comprehension") return 20480;  // Increased for language complexity
    if (module_name == "language_production") return 16384;
    if (module_name == "semantic_memory") return 24576;        // Large semantic representation space
    if (module_name == "syntactic_processor") return 8192;
    if (module_name == "executive_function") return 12288;
    if (module_name == "working_memory") return 8192;
    if (module_name == "episodic_memory") return 6144;
    if (module_name == "reward_system") return 4096;
    if (module_name == "attention_system") return 3072;
    if (module_name == "pragmatic_processor") return 4096;
    if (module_name == "lexical_access") return 6144;
    return 1024; // Default for unknown modules
}

// ============================================================================
// LANGUAGE INTERFACE MANAGEMENT
// ============================================================================

bool AutonomousLearningAgent::initializeLanguageInterface() {
    try {
        // Initialize language interface instead of visual interface
        language_interface_ = std::make_unique<LanguageInterface>();
        
        if (!language_interface_->initialize()) {
            std::cerr << "Failed to initialize language interface" << std::endl;
            return false;
        }
        
        std::cout << "Language interface initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during language interface initialization: " << e.what() << std::endl;
        return false;
    }
}

void AutonomousLearningAgent::shutdownLanguageInterface() {
    if (language_interface_) {
        language_interface_->shutdown();
        language_interface_.reset();
    }
}

// ============================================================================
// ADDITIONAL HELPER METHODS FOR ATTENTION CONTROLLER COMPATIBILITY
// ============================================================================

void AutonomousLearningAgent::updateAttentionWeights(const std::vector<float>& weights) {
    if (!attention_controller_ || weights.empty()) return;
    
    // Map weights to specific language modules
    std::vector<std::string> language_modules = {
        "language_comprehension", "language_production", "semantic_memory", 
        "syntactic_processor", "working_memory", "executive_function"
    };
    
    for (size_t i = 0; i < std::min(weights.size(), language_modules.size()); ++i) {
        attention_controller_->set_attention_weight(language_modules[i], weights[i]);
    }
}

// ============================================================================
// CONSOLIDATION AND ATTENTION HELPER METHODS
// ============================================================================

void AutonomousLearningAgent::consolidateAttentionPatterns() {
    if (!attention_controller_) return;
    
    // Simple attention pattern consolidation
    // In a full implementation, this would analyze attention patterns and strengthen useful ones
    std::cout << "Consolidating attention patterns for language processing..." << std::endl;
}