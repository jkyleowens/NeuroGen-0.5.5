#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <memory>
#include <cstdint>
#include "NeuroGen/cuda/NetworkCUDA.cuh" // Assumed header for NetworkCUDA
#include "NeuroGen/InterModuleConnection.h" // Assumed header for InterModuleConnection
#include "NeuroGen/LearningStateManager.h" // Assumed header for LearningStateManager
#include "NeuroGen/BrainState.h" // Assumed header for BrainState

// Forward declaration if needed
class NetworkCUDA;

// Enum for BrainType
enum class BrainType {
    SIMPLE,
    COMPLEX,
    LANGUAGE
};

class BrainModuleArchitecture : public std::enable_shared_from_this<BrainModuleArchitecture> {
public:
    BrainModuleArchitecture(std::string id, BrainType type);
    ~BrainModuleArchitecture();

    void initialize();
    void update(float dt, float global_reward);
    void updateGlobalContext(const std::vector<float>& new_context);
    
    // Getters
    BrainState getBrainState() const;
    std::vector<InterModuleConnection> getConnections() const;

    // State management
    void saveState(const std::string& path);
    void loadState(const std::string& path);

private:
    // --- Member Variable Declarations ---

    // Core properties
    std::string id_;
    BrainType type_;
    
    // Timestamps - Declaration order matters for the constructor
    std::chrono::steady_clock::time_point creation_time_;
    std::chrono::steady_clock::time_point last_update_time_;
    
    // Global State & Learning
    uint64_t global_learning_steps_ = 0;
    double global_reward_accumulator_ = 0.0;
    std::shared_ptr<LearningStateManager> learning_state_manager_;

    // Neuromodulator Levels
    float global_dopamine_level_ = 0.1f;
    float global_acetylcholine_level_ = 0.1f;
    float global_norepinephrine_level_ = 0.1f;
    float global_serotonin_level_ = 0.1f;

    // Attention Mechanism
    std::map<std::string, float> attention_weights_;
    std::map<std::string, std::vector<float>> attention_history_;

    // CUDA Network Handler
    std::shared_ptr<NetworkCUDA> cuda_network_;
    
    // Modules and connections (assuming these are defined elsewhere)
    std::map<std::string, std::shared_ptr<void>> modules_; // Using void for generality
    std::vector<InterModuleConnection> connections_;


    // --- Private Function Declarations ---

    // NLP-specific setup
    void createNLPModules();
    void setupNLPConnections();
    void initializeNLPAttentionSystem();

    // Core update logic
    void updateNeuromodulatorLevels(float reward, float dt);
    void updateGlobalAttention(float dt);
};