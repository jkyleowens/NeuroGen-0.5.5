#ifndef TASK_AUTOMATION_MODULES_H
#define TASK_AUTOMATION_MODULES_H

#include <NeuroGen/NeuralModule.h> //

/**
 * @brief An example module for processing sensory information.
 *
 * This module will define an internal network with an "INPUT" population to receive
 * raw sensory data (as injected current) and an "OUTPUT" population whose firing
 * patterns represent the processed sensory state.
 */
class SensoryProcessingModule : public NeuralModule {
public:
    SensoryProcessingModule(const std::string& name, const NetworkConfig& config);
    void initialize() override;
};


/**
 * @brief An example module for selecting an action.
 *
 * Receives input from other modules (like the SensoryProcessingModule) and makes
 * a decision, represented by the activity of its "ACTION_OUTPUT" neurons.
 */
class ActionSelectionModule : public NeuralModule {
public:
    ActionSelectionModule(const std::string& name, const NetworkConfig& config);
    void initialize() override;
};

#endif // TASK_AUTOMATION_MODULES_H