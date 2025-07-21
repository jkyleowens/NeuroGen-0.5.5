// ============================================================================
// BRAIN MODULE ARCHITECTURE IMPLEMENTATION - LANGUAGE-FOCUSED
// File: src/BrainModuleArchitecture.cpp
// ============================================================================

#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/EnhancedNeuralModule.h"
#include "NeuroGen/NetworkConfig.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <chrono>

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

BrainModuleArchitecture::BrainModuleArchitecture(const ArchitectureConfig& config)
    : config_(config)
    , is_processing_(false)
    , is_learning_enabled_(true)
    , last_update_time_(std::chrono::high_resolution_clock::now()) {
    
    // Initialize global linguistic state
    global_linguistic_state_.resize(1024, 0.0f);
    
    std::cout << "Brain Module Architecture created with language-focused configuration" << std::endl;
}

BrainModuleArchitecture::~BrainModuleArchitecture() {
    // Cleanup will be handled automatically by smart pointers
    std::cout << "Brain Module Architecture destroyed" << std::endl;
}

bool BrainModuleArchitecture::initialize(size_t vocabulary_size, size_t max_sequence_length) {
    try {
        std::cout << "Initializing language-focused brain architecture..." << std::endl;
        
        // Update configuration with provided parameters
        config_.vocabulary_size = vocabulary_size;
        config_.max_sequence_length = max_sequence_length;
        
        // Initialize language processing pipeline
        auto [success, message] = initializeLanguagePipeline();
        if (!success) {
            std::cerr << "Failed to initialize language pipeline: " << message << std::endl;
            return false;
        }
        
        // Initialize attention system (placeholder)
        // attention_controller_ = std::make_unique<AttentionController>();
        
        std::cout << "Language-focused brain architecture initialized successfully" << std::endl;
        std::cout << "  Vocabulary Size: " << vocabulary_size << std::endl;
        std::cout << "  Max Sequence Length: " << max_sequence_length << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        return false;
    }
}

bool BrainModuleArchitecture::initializeCustom(const std::vector<ModuleConfig>& module_configs,
                                              const std::vector<InterModuleConnection>& connections) {
    try {
        std::cout << "Initializing custom brain architecture with " << module_configs.size() 
                  << " modules and " << connections.size() << " connections" << std::endl;
        
        // Initialize modules from configurations
        for (const auto& config : module_configs) {
            auto [success, error_msg] = addLanguageModule(config);
            if (!success) {
                std::cerr << "Failed to add module " << config.name << ": " << error_msg << std::endl;
                return false;
            }
        }
        
        // Setup connections
        connections_ = connections;
        
        // Validate the architecture
        auto [valid, validation_msg] = validateConfiguration();
        if (!valid) {
            std::cerr << "Architecture validation failed: " << validation_msg << std::endl;
            return false;
        }
        
        std::cout << "Custom brain architecture initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during custom initialization: " << e.what() << std::endl;
        return false;
    }
}

std::pair<bool, std::string> BrainModuleArchitecture::initializeLanguagePipeline() {
    try {
        // Create default language processing modules
        std::vector<ModuleConfig> default_configs;
        
        // Language Comprehension Module
        ModuleConfig comprehension_config;
        comprehension_config.type = ModuleType::LANGUAGE_COMPREHENSION;
        comprehension_config.name = "language_comprehension";
        comprehension_config.description = "Primary language understanding and semantic processing";
        comprehension_config.input_size = 768;
        comprehension_config.output_size = 512;
        comprehension_config.internal_neurons = 2048;
        comprehension_config.linguistic_layers = 6;
        comprehension_config.semantic_dimensions = 300;
        comprehension_config.learning_rate = 0.001f;
        comprehension_config.supports_sequential = true;
        comprehension_config.supports_hierarchical = true;
        default_configs.push_back(comprehension_config);
        
        // Language Production Module
        ModuleConfig production_config;
        production_config.type = ModuleType::LANGUAGE_PRODUCTION;
        production_config.name = "language_production";
        production_config.description = "Text generation and language output";
        production_config.input_size = 512;
        production_config.output_size = 768;
        production_config.internal_neurons = 1536;
        production_config.linguistic_layers = 4;
        production_config.semantic_dimensions = 300;
        production_config.learning_rate = 0.0008f;
        production_config.supports_sequential = true;
        production_config.bidirectional = true;
        default_configs.push_back(production_config);
        
        // Semantic Memory Module
        ModuleConfig semantic_config;
        semantic_config.type = ModuleType::SEMANTIC_MEMORY;
        semantic_config.name = "semantic_memory";
        semantic_config.description = "Conceptual knowledge and word meanings";
        semantic_config.input_size = 512;
        semantic_config.output_size = 512;
        semantic_config.internal_neurons = 3072;
        semantic_config.linguistic_layers = 8;
        semantic_config.semantic_dimensions = 512;
        semantic_config.learning_rate = 0.0005f;
        semantic_config.semantic_decay_rate = 0.995f;
        default_configs.push_back(semantic_config);
        
        // Working Memory Module
        ModuleConfig working_memory_config;
        working_memory_config.type = ModuleType::WORKING_MEMORY;
        working_memory_config.name = "working_memory";
        working_memory_config.description = "Temporary linguistic information storage";
        working_memory_config.input_size = 512;
        working_memory_config.output_size = 256;
        working_memory_config.internal_neurons = 1024;
        working_memory_config.linguistic_layers = 3;
        working_memory_config.learning_rate = 0.002f;
        default_configs.push_back(working_memory_config);
        
        // Executive Function Module
        ModuleConfig executive_config;
        executive_config.type = ModuleType::EXECUTIVE_FUNCTION;
        executive_config.name = "executive_function";
        executive_config.description = "High-level language control and planning";
        executive_config.input_size = 768;
        executive_config.output_size = 512;
        executive_config.internal_neurons = 1536;
        executive_config.linguistic_layers = 5;
        executive_config.learning_rate = 0.001f;
        executive_config.supports_hierarchical = true;
        default_configs.push_back(executive_config);
        
        // Initialize modules
        for (const auto& config : default_configs) {
            auto [success, error_msg] = addLanguageModule(config);
            if (!success) {
                return {false, "Failed to add module " + config.name + ": " + error_msg};
            }
        }
        
        // Create default connections
        connections_.clear();
        connections_.emplace_back("language_comprehension", "semantic_memory", 0.8f, "semantic");
        connections_.emplace_back("semantic_memory", "language_production", 0.7f, "semantic");
        connections_.emplace_back("language_comprehension", "working_memory", 0.6f, "attentional");
        connections_.emplace_back("working_memory", "executive_function", 0.8f, "control");
        connections_.emplace_back("executive_function", "language_production", 0.7f, "control");
        
        return {true, "Language pipeline initialized successfully"};
        
    } catch (const std::exception& e) {
        return {false, "Exception during language pipeline initialization: " + std::string(e.what())};
    }
}

std::pair<bool, std::string> BrainModuleArchitecture::validateConfiguration() const {
    // Basic validation checks
    if (modules_.empty()) {
        return {false, "No modules configured"};
    }
    
    if (config_.vocabulary_size == 0) {
        return {false, "Vocabulary size not set"};
    }
    
    if (config_.max_sequence_length == 0) {
        return {false, "Maximum sequence length not set"};
    }
    
    // Check for essential language modules
    bool has_comprehension = modules_.count("language_comprehension") > 0;
    bool has_production = modules_.count("language_production") > 0;
    
    if (!has_comprehension) {
        return {false, "Missing language comprehension module"};
    }
    
    if (!has_production) {
        return {false, "Missing language production module"};
    }
    
    return {true, "Configuration valid"};
}

void BrainModuleArchitecture::reset(bool preserve_language_knowledge) {
    std::cout << "Resetting brain architecture (preserve knowledge: " 
              << (preserve_language_knowledge ? "true" : "false") << ")" << std::endl;
    
    if (!preserve_language_knowledge) {
        // Reset all modules
        for (auto& [name, module] : modules_) {
            if (module) {
                // Reset module state (would need to implement in EnhancedNeuralModule)
                std::cout << "Resetting module: " << name << std::endl;
            }
        }
        
        // Reset global state
        std::fill(global_linguistic_state_.begin(), global_linguistic_state_.end(), 0.0f);
        module_outputs_.clear();
        processing_times_.clear();
        language_metrics_.clear();
    }
    
    is_processing_ = false;
    last_update_time_ = std::chrono::high_resolution_clock::now();
}

// ============================================================================
// LANGUAGE PROCESSING INTERFACE
// ============================================================================

BrainModuleArchitecture::LanguageOutput BrainModuleArchitecture::processLanguage(
    const LanguageInput& input, bool learning_enabled) {
    
    LanguageOutput output;
    
    try {
        if (input.text.empty()) {
            output.confidence = 0.0f;
            output.generated_text = "";
            return output;
        }
        
        is_processing_ = true;
        
        // Extract features from input
        std::vector<float> language_features = extractLanguageFeatures(input);
        
        // Process through language comprehension
        if (modules_.count("language_comprehension")) {
            auto comprehension_output = modules_["language_comprehension"]->process(language_features);
            module_outputs_["language_comprehension"] = comprehension_output;
            
            // Store semantic representation
            if (comprehension_output.size() >= 256) {
                output.semantic_representation.assign(comprehension_output.begin(), 
                                                     comprehension_output.begin() + 256);
            }
        }
        
        // Process through semantic memory
        if (modules_.count("semantic_memory") && module_outputs_.count("language_comprehension")) {
            auto semantic_output = modules_["semantic_memory"]->process(module_outputs_["language_comprehension"]);
            module_outputs_["semantic_memory"] = semantic_output;
        }
        
        // Generate response through language production
        if (modules_.count("language_production")) {
            std::vector<float> production_input;
            
            // Combine inputs from comprehension and semantic memory
            if (module_outputs_.count("language_comprehension")) {
                auto& comp_output = module_outputs_["language_comprehension"];
                production_input.insert(production_input.end(), 
                                      comp_output.begin(), 
                                      comp_output.begin() + std::min(comp_output.size(), size_t(256)));
            }
            
            if (module_outputs_.count("semantic_memory")) {
                auto& sem_output = module_outputs_["semantic_memory"];
                production_input.insert(production_input.end(),
                                      sem_output.begin(),
                                      sem_output.begin() + std::min(sem_output.size(), size_t(256)));
            }
            
            auto production_output = modules_["language_production"]->process(production_input);
            module_outputs_["language_production"] = production_output;
            
            // Convert to text (simplified)
            output.generated_text = convertNeuralToText(production_output);
            output.confidence = calculateOutputConfidence(production_output);
        }
        
        // Update linguistic scores
        output.linguistic_scores["comprehension"] = 0.8f; // Placeholder
        output.linguistic_scores["coherence"] = 0.7f;     // Placeholder
        output.linguistic_scores["relevance"] = 0.75f;    // Placeholder
        
        is_processing_ = false;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during language processing: " << e.what() << std::endl;
        output.confidence = 0.0f;
        output.generated_text = "Error processing language input";
        is_processing_ = false;
    }
    
    return output;
}

std::string BrainModuleArchitecture::processText(const std::string& text, bool learning_enabled) {
    LanguageInput input(text);
    auto output = processLanguage(input, learning_enabled);
    return output.generated_text;
}

BrainModuleArchitecture::LanguageOutput BrainModuleArchitecture::generateResponse(
    const LanguageInput& context, size_t max_length, float temperature) {
    
    // For now, process the context and generate a simple response
    auto response = processLanguage(context, false);
    
    // Apply temperature and max_length constraints
    if (response.generated_text.length() > max_length) {
        response.generated_text = response.generated_text.substr(0, max_length - 3) + "...";
    }
    
    // Temperature affects randomness (placeholder implementation)
    if (temperature > 0.8f) {
        response.generation_strategy = "sampling";
    } else if (temperature > 0.5f) {
        response.generation_strategy = "beam_search";
    } else {
        response.generation_strategy = "greedy";
    }
    
    return response;
}

// ============================================================================
// MODULE MANAGEMENT
// ============================================================================

std::pair<bool, std::string> BrainModuleArchitecture::addLanguageModule(const ModuleConfig& config) {
    try {
        if (config.name.empty()) {
            return {false, "Module name cannot be empty"};
        }
        
        if (modules_.count(config.name) > 0) {
            return {false, "Module with name '" + config.name + "' already exists"};
        }
        
        // Create NetworkConfig for the module
        NetworkConfig net_config;
        net_config.num_neurons = config.internal_neurons;
        net_config.stdp_learning_rate = config.learning_rate; // Use stdp_learning_rate instead
        
        // Create the module
        auto module = std::make_shared<EnhancedNeuralModule>(config.name, net_config);
        
        if (!module->initialize()) {
            return {false, "Failed to initialize module: " + config.name};
        }
        
        // Store the module and its config
        modules_[config.name] = module;
        module_configs_[config.name] = config;
        
        std::cout << "Added language module: " << config.name 
                  << " (type: " << static_cast<int>(config.type) << ")" << std::endl;
        
        return {true, "Module added successfully"};
        
    } catch (const std::exception& e) {
        return {false, "Exception adding module: " + std::string(e.what())};
    }
}

bool BrainModuleArchitecture::removeModule(const std::string& module_name, bool cleanup_connections) {
    if (modules_.count(module_name) == 0) {
        return false;
    }
    
    if (cleanup_connections) {
        // Remove connections involving this module
        connections_.erase(
            std::remove_if(connections_.begin(), connections_.end(),
                [&module_name](const InterModuleConnection& conn) {
                    return conn.source_module == module_name || conn.target_module == module_name;
                }),
            connections_.end()
        );
    }
    
    modules_.erase(module_name);
    module_configs_.erase(module_name);
    module_outputs_.erase(module_name);
    
    std::cout << "Removed module: " << module_name << std::endl;
    return true;
}

std::vector<std::shared_ptr<EnhancedNeuralModule>> BrainModuleArchitecture::getModulesByType(ModuleType type) const {
    std::vector<std::shared_ptr<EnhancedNeuralModule>> result;
    
    for (const auto& [name, config] : module_configs_) {
        if (config.type == type && modules_.count(name) > 0) {
            result.push_back(modules_.at(name));
        }
    }
    
    return result;
}

// ============================================================================
// STATE MANAGEMENT AND PERSISTENCE
// ============================================================================

bool BrainModuleArchitecture::saveState(const std::string& filepath, bool include_language_knowledge) {
    try {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for saving: " << filepath << std::endl;
            return false;
        }
        
        // Save configuration
        file.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        
        // Save module count
        size_t module_count = modules_.size();
        file.write(reinterpret_cast<const char*>(&module_count), sizeof(module_count));
        
        // Save module configurations
        for (const auto& [name, config] : module_configs_) {
            size_t name_length = name.length();
            file.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
            file.write(name.c_str(), name_length);
            file.write(reinterpret_cast<const char*>(&config), sizeof(config));
        }
        
        // Save connections
        size_t connection_count = connections_.size();
        file.write(reinterpret_cast<const char*>(&connection_count), sizeof(connection_count));
        for (const auto& conn : connections_) {
            size_t src_len = conn.source_module.length();
            size_t tgt_len = conn.target_module.length();
            size_t type_len = conn.connection_type.length();
            
            file.write(reinterpret_cast<const char*>(&src_len), sizeof(src_len));
            file.write(conn.source_module.c_str(), src_len);
            file.write(reinterpret_cast<const char*>(&tgt_len), sizeof(tgt_len));
            file.write(conn.target_module.c_str(), tgt_len);
            file.write(reinterpret_cast<const char*>(&type_len), sizeof(type_len));
            file.write(conn.connection_type.c_str(), type_len);
            file.write(reinterpret_cast<const char*>(&conn.connection_strength), sizeof(conn.connection_strength));
            file.write(reinterpret_cast<const char*>(&conn.is_bidirectional), sizeof(conn.is_bidirectional));
            file.write(reinterpret_cast<const char*>(&conn.delay_ms), sizeof(conn.delay_ms));
        }
        
        file.close();
        std::cout << "Architecture state saved to: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving state: " << e.what() << std::endl;
        return false;
    }
}

bool BrainModuleArchitecture::loadState(const std::string& filepath, bool merge_with_current) {
    try {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for loading: " << filepath << std::endl;
            return false;
        }
        
        if (!merge_with_current) {
            // Clear current state
            modules_.clear();
            module_configs_.clear();
            connections_.clear();
        }
        
        // Load configuration
        ArchitectureConfig loaded_config;
        file.read(reinterpret_cast<char*>(&loaded_config), sizeof(loaded_config));
        config_ = loaded_config;
        
        // Load modules
        size_t module_count;
        file.read(reinterpret_cast<char*>(&module_count), sizeof(module_count));
        
        for (size_t i = 0; i < module_count; ++i) {
            size_t name_length;
            file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
            
            std::string name(name_length, '\0');
            file.read(&name[0], name_length);
            
            ModuleConfig config;
            file.read(reinterpret_cast<char*>(&config), sizeof(config));
            
            // Recreate module
            auto [success, error_msg] = addLanguageModule(config);
            if (!success) {
                std::cerr << "Failed to recreate module " << name << ": " << error_msg << std::endl;
            }
        }
        
        // Load connections
        size_t connection_count;
        file.read(reinterpret_cast<char*>(&connection_count), sizeof(connection_count));
        
        connections_.clear();
        for (size_t i = 0; i < connection_count; ++i) {
            size_t src_len, tgt_len, type_len;
            file.read(reinterpret_cast<char*>(&src_len), sizeof(src_len));
            
            std::string source_module(src_len, '\0');
            file.read(&source_module[0], src_len);
            
            file.read(reinterpret_cast<char*>(&tgt_len), sizeof(tgt_len));
            std::string target_module(tgt_len, '\0');
            file.read(&target_module[0], tgt_len);
            
            file.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
            std::string connection_type(type_len, '\0');
            file.read(&connection_type[0], type_len);
            
            float connection_strength;
            bool is_bidirectional;
            float delay_ms;
            file.read(reinterpret_cast<char*>(&connection_strength), sizeof(connection_strength));
            file.read(reinterpret_cast<char*>(&is_bidirectional), sizeof(is_bidirectional));
            file.read(reinterpret_cast<char*>(&delay_ms), sizeof(delay_ms));
            
            connections_.emplace_back(source_module, target_module, connection_strength, 
                                    connection_type, is_bidirectional, delay_ms);
        }
        
        file.close();
        std::cout << "Architecture state loaded from: " << filepath << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading state: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// METRICS AND STATUS
// ============================================================================

std::map<std::string, float> BrainModuleArchitecture::getPerformanceMetrics() const {
    std::map<std::string, float> stats;
    
    // Basic statistics
    stats["module_count"] = static_cast<float>(modules_.size());
    stats["connection_count"] = static_cast<float>(connections_.size());
    stats["is_processing"] = is_processing_ ? 1.0f : 0.0f;
    stats["learning_enabled"] = is_learning_enabled_ ? 1.0f : 0.0f;
    
    // Calculate average processing times
    if (!processing_times_.empty()) {
        float avg_time = 0.0f;
        for (const auto& [module, time] : processing_times_) {
            avg_time += time;
        }
        stats["avg_processing_time"] = avg_time / processing_times_.size();
    }
    
    // Language-specific metrics
    if (!language_metrics_.empty()) {
        for (const auto& [metric, value] : language_metrics_) {
            stats["lang_" + metric] = value;
        }
    }
    
    return stats;
}

std::string BrainModuleArchitecture::getDetailedStatus() const {
    std::ostringstream status;
    
    status << "Brain Module Architecture Status:\n";
    status << "================================\n";
    status << "Configuration:\n";
    status << "  Vocabulary Size: " << config_.vocabulary_size << "\n";
    status << "  Max Sequence Length: " << config_.max_sequence_length << "\n";
    status << "  Embedding Dimensions: " << config_.embedding_dimensions << "\n";
    status << "  Global Learning Rate: " << config_.global_learning_rate << "\n";
    status << "  GPU Acceleration: " << (config_.use_gpu_acceleration ? "Enabled" : "Disabled") << "\n";
    status << "\n";
    
    status << "Modules (" << modules_.size() << "):\n";
    for (const auto& [name, module] : modules_) {
        if (module_configs_.count(name)) {
            const auto& config = module_configs_.at(name);
            status << "  " << name << " (Type: " << static_cast<int>(config.type) << ")\n";
            status << "    Input: " << config.input_size 
                   << ", Output: " << config.output_size 
                   << ", Internal: " << config.internal_neurons << "\n";
        }
    }
    status << "\n";
    
    status << "Connections (" << connections_.size() << "):\n";
    for (const auto& conn : connections_) {
        status << "  " << conn.source_module << " -> " << conn.target_module 
               << " (strength: " << std::fixed << std::setprecision(2) << conn.connection_strength 
               << ", type: " << conn.connection_type << ")\n";
    }
    
    status << "\nStatus:\n";
    status << "  Processing: " << (is_processing_ ? "Active" : "Idle") << "\n";
    status << "  Learning: " << (is_learning_enabled_ ? "Enabled" : "Disabled") << "\n";
    
    return status.str();
}

void BrainModuleArchitecture::setPerformanceMonitoring(bool enable_monitoring) {
    std::cout << "Performance monitoring " << (enable_monitoring ? "enabled" : "disabled") << std::endl;
    // Implementation would track performance metrics
}

// ============================================================================
// HELPER METHODS
// ============================================================================

std::vector<float> BrainModuleArchitecture::extractLanguageFeatures(const LanguageInput& input) {
    // Simplified feature extraction - would be much more sophisticated in practice
    std::vector<float> features(768, 0.0f); // Standard embedding size
    
    // Basic word count and length features
    features[0] = static_cast<float>(input.text.length()) / 1000.0f; // Normalized text length
    
    // Count common words (simplified)
    size_t word_count = std::count(input.text.begin(), input.text.end(), ' ') + 1;
    features[1] = static_cast<float>(word_count) / 100.0f; // Normalized word count
    
    // Use provided embeddings if available
    if (!input.embeddings.empty()) {
        size_t copy_size = std::min(input.embeddings.size(), features.size() - 2);
        std::copy(input.embeddings.begin(), input.embeddings.begin() + copy_size, 
                 features.begin() + 2);
    }
    
    // Use linguistic features if available
    for (const auto& [feature_name, value] : input.linguistic_features) {
        // Simple hash-based feature mapping (would be more sophisticated in practice)
        size_t hash = std::hash<std::string>{}(feature_name);
        size_t index = (hash % (features.size() - 100)) + 100; // Leave space for other features
        features[index] = value;
    }
    
    return features;
}

std::string BrainModuleArchitecture::convertNeuralToText(const std::vector<float>& neural_output) {
    // Simplified neural-to-text conversion
    // In a real implementation, this would use a vocabulary and proper decoding
    
    if (neural_output.empty()) {
        return "Empty response";
    }
    
    // Simple approach: generate text based on neural activation patterns
    std::ostringstream text;
    
    // Find dominant activations
    float max_activation = *std::max_element(neural_output.begin(), neural_output.end());
    
    if (max_activation > 0.8f) {
        text << "High confidence response: ";
    } else if (max_activation > 0.5f) {
        text << "Moderate response: ";
    } else {
        text << "Low confidence response: ";
    }
    
    // Generate simple response based on activation pattern
    float avg_activation = 0.0f;
    for (float val : neural_output) {
        avg_activation += val;
    }
    avg_activation /= neural_output.size();
    
    if (avg_activation > 0.3f) {
        text << "The neural processing indicates strong semantic understanding.";
    } else if (avg_activation > 0.1f) {
        text << "Processing shows moderate semantic activation.";
    } else {
        text << "Minimal neural response detected.";
    }
    
    return text.str();
}

float BrainModuleArchitecture::calculateOutputConfidence(const std::vector<float>& output) {
    if (output.empty()) {
        return 0.0f;
    }
    
    // Calculate confidence based on activation statistics
    float max_val = *std::max_element(output.begin(), output.end());
    float min_val = *std::min_element(output.begin(), output.end());
    
    // Simple confidence metric based on dynamic range
    float range = max_val - min_val;
    float confidence = std::min(1.0f, range * 2.0f);
    
    return confidence;
}