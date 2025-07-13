// ============================================================================
// LEARNING STATE MANAGER HEADER - FIXED
// File: include/NeuroGen/LearningStateManager.h
// ============================================================================

#ifndef LEARNING_STATE_MANAGER_H
#define LEARNING_STATE_MANAGER_H

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <filesystem>

// Forward declaration to avoid circular dependency
class BrainModuleArchitecture;

/**
 * @brief Learning State Manager for Neural Networks
 * 
 * Manages persistent learning states across training sessions
 */
class LearningStateManager {
public:
    /**
     * @brief Constructor with architecture and save path
     * @param architecture Shared pointer to brain architecture
     * @param base_save_path Base path for saving states
     */
    LearningStateManager(std::shared_ptr<BrainModuleArchitecture> architecture, 
                        const std::string& base_save_path);
    
    /**
     * @brief Destructor
     */
    ~LearningStateManager();
    
    /**
     * @brief Initialize the learning state manager
     * @return Success status
     */
    bool initialize();
    
    /**
     * @brief Save current learning state
     * @param checkpoint_name Name for the checkpoint
     * @return Success status
     */
    bool saveLearningState(const std::string& checkpoint_name = "latest");
    
    /**
     * @brief Load learning state from checkpoint
     * @param checkpoint_name Name of the checkpoint to load
     * @return Success status
     */
    bool loadLearningState(const std::string& checkpoint_name = "latest");
    
    /**
     * @brief Update learning statistics
     * @param reward Current reward signal
     * @param performance Current performance metric
     */
    void updateLearningStats(float reward, float performance);
    
    /**
     * @brief Get learning statistics
     * @return Map of statistic names to values
     */
    std::map<std::string, float> getLearningStats() const;
    
    /**
     * @brief Check if checkpoint exists
     * @param checkpoint_name Name of checkpoint to check
     * @return True if checkpoint exists
     */
    bool checkpointExists(const std::string& checkpoint_name) const;

private:
    // Use weak_ptr to avoid circular reference with BrainModuleArchitecture
    std::weak_ptr<BrainModuleArchitecture> architecture_;
    std::string base_save_path_;
    std::map<std::string, float> learning_stats_;
    std::chrono::steady_clock::time_point last_save_time_;
    bool initialized_ = false;
};

#endif // LEARNING_STATE_MANAGER_H