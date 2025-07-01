// ============================================================================
// CENTRAL CONTROLLER HEADER
// File: include/NeuroGen/CentralController.h
// ============================================================================

#pragma once

#include "NeuroGen/ScreenElement.h"
#include <memory>
#include <vector>
#include <string>

// Forward declarations to avoid circular dependencies
class ControllerModule;
class NeuralModule; 
class AutonomousLearningAgent;
class VisualInterface;
class CognitiveModule;
class MotorModule;

/**
 * @brief Central Controller for Task Automation System
 * 
 * This class serves as the main coordinator for the ANIMA-based
 * task automation simulation, integrating neural modules, screen
 * element processing, and autonomous decision making.
 */
class CentralController {
public:
    CentralController();
    ~CentralController();
    
    // Initialization and lifecycle
    bool initialize();
    void shutdown();
    
    // Main control interface
    void run(int cycles = 1);
    void simulateNewScreenData(const std::vector<ScreenElement>& screen_elements);
    
    // System status
    bool isInitialized() const { return is_initialized_; }
    std::string getSystemStatus() const;
    float getSystemPerformance() const;
    
private:
    // Internal components
    std::unique_ptr<ControllerModule> neuro_controller_;
    std::unique_ptr<AutonomousLearningAgent> learning_agent_;
    std::unique_ptr<VisualInterface> visual_interface_;
    
    // Neural modules
    std::shared_ptr<NeuralModule> perception_module_;
    std::shared_ptr<NeuralModule> planning_module_;
    std::shared_ptr<NeuralModule> motor_module_;
    
    // Task modules
    std::shared_ptr<CognitiveModule> cognitive_module_;
    std::shared_ptr<MotorModule> motor_task_module_;
    
    // Current state
    std::vector<ScreenElement> current_screen_elements_;
    bool is_initialized_;
    int cycle_count_;
    
    // Internal methods
    void initialize_neural_modules();
    void initialize_task_modules();
    void process_screen_elements();
    void execute_cognitive_cycle();
    void update_performance_metrics();
};
