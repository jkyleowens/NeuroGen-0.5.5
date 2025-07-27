// ============================================================================
// BRAIN MODULE ARCHITECTURE HEADER - UPDATED VERSION
// File: include/NeuroGen/BrainModuleArchitecture.h
// ============================================================================

#ifndef BRAIN_MODULE_ARCHITECTURE_H
#define BRAIN_MODULE_ARCHITECTURE_H

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <sstream>

// Include required headers
#include "NeuroGen/LearningState.h"

// Forward declarations
class EnhancedNeuralModule;
class NetworkCUDA;

// ============================================================================
// CORE STRUCTURES AND ENUMS
// ============================================================================

/**
 * @brief Module type enumeration
 */
enum class ModuleType {
    LANGUAGE_ENCODER,
    LANGUAGE_PROCESSOR, 
    LANGUAGE_DECODER,
    WORKING_MEMORY,
    ATTENTION,
    MOTOR,
    SENSORY,
    CUSTOM
};

/**
 * @brief Module configuration structure
 */
struct ModuleConfig {
    std::string name;
    ModuleType type = ModuleType::CUSTOM;
    size_t input_size = 128;
    size_t output_size = 128;
    float learning_rate = 0.01f;
    bool is_trainable = true;
    std::map<std::string, float> parameters;
    
    ModuleConfig() = default;
    ModuleConfig(const std::string& module_name, size_t in_size, size_t out_size) 
        : name(module_name), input_size(in_size), output_size(out_size) {}
};

/**
 * @brief Inter-module connection specification
 */
struct InterModuleConnection {
    std::string source_module;
    std::string target_module;
    std::string source_port = "output";
    std::string target_port = "input";
    float connection_strength = 0.5f;
    bool is_feedback = false;
    float delay_ms = 0.0f;
    
    InterModuleConnection() = default;
    InterModuleConnection(const std::string& src, const std::string& tgt, float strength = 0.5f)
        : source_module(src), target_module(tgt), connection_strength(strength) {}
};

/**
 * @brief Internal connection information
 */
struct ConnectionInfo {
    std::string source_module;
    std::string target_module;
    float connection_strength;
    bool is_active;
    std::chrono::steady_clock::time_point creation_time;
    std::chrono::steady_clock::time_point last_used;
    
    ConnectionInfo() : connection_strength(0.0f), is_active(false) {
        creation_time = std::chrono::steady_clock::now();
        last_used = creation_time;
    }
};

// ============================================================================
// BRAIN MODULE ARCHITECTURE CLASS
// ============================================================================

/**
 * @brief Brain Module Architecture for organizing neural modules
 * 
 * This class provides a framework for creating, managing, and coordinating
 * multiple neural modules in a brain-like architecture. It supports:
 * - Dynamic module creation and removal
 * - Inter-module connection management
 * - GPU acceleration (optional)
 * - State persistence and loading
 * - Memory consolidation across modules
 */
class BrainModuleArchitecture : public std::enable_shared_from_this<BrainModuleArchitecture> {
public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Default constructor
     */
    BrainModuleArchitecture();
    
    /**
     * @brief Destructor with proper cleanup
     */
    virtual ~BrainModuleArchitecture();
    
    /**
     * @brief Initialize the architecture with default brain modules
     * @param visual_input_width Width of input (adapted for language: vocab size factor)
     * @param visual_input_height Height of input (adapted for language: sequence length)
     * @return Success status
     */
    bool initialize(int visual_input_width = 128, int visual_input_height = 128);
    
    /**
     * @brief Update all modules and connections
     * @param dt Time step in seconds
     * @param global_reward Global reward signal
     */
    void update(float dt, float global_reward = 0.0f);
    
    // ========================================================================
    // MODULE MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Add a new module to the architecture
     * @param config Module configuration
     * @return Success status with error message if failed
     */
    std::pair<bool, std::string> addModule(const ModuleConfig& config);
    
    /**
     * @brief Remove a module from the architecture
     * @param module_name Name of module to remove
     * @param cleanup_connections Whether to remove associated connections
     * @return Success status
     */
    bool removeModule(const std::string& module_name, bool cleanup_connections = true);
    
    /**
     * @brief Get module by name
     * @param module_name Name of the module
     * @return Shared pointer to module (nullptr if not found)
     */
    std::shared_ptr<EnhancedNeuralModule> getModule(const std::string& module_name) const;
    
    /**
     * @brief Get list of all module names
     * @return Vector of module names sorted alphabetically
     */
    std::vector<std::string> getModuleNames() const;
    
    /**
     * @brief Get module count
     * @return Number of modules in the architecture
     */
    size_t getModuleCount() const;
    
    // ========================================================================
    // INTER-MODULE CONNECTION MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Add connection between modules
     * @param connection Connection specification
     * @return Success status with error message if failed
     */
    std::pair<bool, std::string> addConnection(const InterModuleConnection& connection);
    
    /**
     * @brief Remove connection between modules
     * @param source_module Source module name
     * @param target_module Target module name
     * @return Success status
     */
    bool removeConnection(const std::string& source_module, const std::string& target_module);
    
    /**
     * @brief Remove all connections for a module
     * @param module_name Module name
     * @return Number of connections removed
     */
    size_t removeAllConnections(const std::string& module_name);
    
    // ========================================================================
    // STATE PERSISTENCE
    // ========================================================================
    
    /**
     * @brief Save learning state to disk
     * @param save_directory Directory to save state
     * @param checkpoint_name Name for this checkpoint
     * @return Success status
     */
    bool saveLearningState(const std::string& save_directory, const std::string& checkpoint_name);
    
    /**
     * @brief Load learning state from disk 
     * @param save_directory Directory containing saved state
     * @param target_checkpoint Checkpoint name to load
     * @return Success status
     */
    bool loadLearningState(const std::string& save_directory, const std::string& target_checkpoint);
    
    // ========================================================================
    // MEMORY CONSOLIDATION
    // ========================================================================
    
    /**
     * @brief Perform global memory consolidation across all modules
     * @param consolidation_strength Strength of consolidation (0-1)
     * @return Number of consolidated memory items
     */
    size_t performGlobalMemoryConsolidation(float consolidation_strength);
    
    // ========================================================================
    // GPU ACCELERATION (CONDITIONAL)
    // ========================================================================
    
#ifdef CUDA_ENABLED
    /**
     * @brief Set CUDA network for GPU acceleration
     * @param cuda_network Shared pointer to NetworkCUDA instance
     */
    void setCUDANetwork(std::shared_ptr<NetworkCUDA> cuda_network);
    
    /**
     * @brief Enable or disable GPU acceleration
     * @param enable True to enable GPU acceleration
     * @return Success status with message
     */
    std::pair<bool, std::string> enableGPUAcceleration(bool enable);
#endif
    
    /**
     * @brief Check if GPU acceleration is enabled
     * @return True if GPU acceleration is active
     */
    bool isGPUEnabled() const;
    
    // ========================================================================
    // STATISTICS AND MONITORING
    // ========================================================================
    
    /**
     * @brief Get global learning state
     * @return Current global learning state
     */
    SessionLearningState getGlobalLearningState() const;
    
    /**
     * @brief Calculate architecture hash for compatibility checking
     * @return Architecture hash string
     */
    std::string calculateArchitectureHash() const;
    
    /**
     * @brief Get architecture statistics
     * @return Map of statistic names to values
     */
    std::map<std::string, uint32_t> getArchitectureStatistics() const;
    
    /**
     * @brief Get total number of neurons across all modules
     * @return Total neuron count
     */
    size_t getTotalNeurons() const;
    
    /**
     * @brief Get total number of synapses across all modules
     * @return Total synapse count
     */
    size_t getTotalSynapses() const;
    
    /**
     * @brief Get architecture information as string
     * @return Formatted architecture information
     */
    std::string getArchitectureInfo() const;

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    // Core state
    bool is_initialized_;
    bool gpu_enabled_;
    
    // Language processing parameters
    size_t vocab_size_;
    size_t max_sequence_length_;
    
    // Module management
    std::unordered_map<std::string, std::shared_ptr<EnhancedNeuralModule>> modules_;
    size_t total_modules_;
    size_t total_connections_;
    
    // Connection management
    std::map<std::pair<std::string, std::string>, ConnectionInfo> inter_module_connections_;
    
    // Learning parameters
    float learning_rate_;
    
    // GPU acceleration (conditional)
#ifdef CUDA_ENABLED
    std::shared_ptr<NetworkCUDA> cuda_network_;
#endif
    
    // Thread safety
    mutable std::mutex modules_mutex_;
    
    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================
    
    /**
     * @brief Initialize default language processing modules
     * @return Success status
     */
    bool initializeDefaultModules();
    
    /**
     * @brief Initialize inter-module connections
     */
    void initializeInterModuleConnections();
    
    /**
     * @brief Update inter-module connections (internal)
     * @param dt Time step
     */
    void updateInterModuleConnectionsInternal(float dt);
    
    /**
     * @brief Generate unique module name
     * @param base_name Base name for the module
     * @return Unique module name
     */
    std::string generateUniqueModuleName(const std::string& base_name) const;
    
    /**
     * @brief Validate module configuration
     * @param config Module configuration to validate
     * @return Validation result with error details
     */
    std::pair<bool, std::string> validateModuleConfig(const ModuleConfig& config) const;
    
    /**
     * @brief Validate connection configuration  
     * @param connection Connection configuration to validate
     * @return Validation result with error details
     */
    std::pair<bool, std::string> validateConnectionConfig(const InterModuleConnection& connection) const;
    
    /**
     * @brief Validate architecture compatibility with saved state
     * @param state_hash Hash from saved state
     * @return Compatibility result with details
     */
    std::pair<bool, std::string> validateArchitectureCompatibility(const std::string& state_hash) const;
};

#endif // BRAIN_MODULE_ARCHITECTURE_H