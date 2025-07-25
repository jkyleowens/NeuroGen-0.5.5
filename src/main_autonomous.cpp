// Enhanced main.cpp with Autonomous Learning Agent Integration
// NeuroGen Version 0.5.5 - Advanced Autonomous Learning Framework

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <functional>
#include <iomanip>
#include <string>

// Core NeuroGen includes
#include "NeuroGen/TaskAutomationModules.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/NeuralModule.h"
#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/NetworkIntegration.h"
#include "NeuroGen/ControllerModule.h"

// Function to create a default configuration for a neural module
NetworkConfig create_default_config() {
    NetworkConfig config;
    config.num_neurons = 8192; // MASSIVE scale-up: 8K neurons per base module for free-thinking AI
    config.enable_neurogenesis = true;
    config.enable_stdp = true;
    config.enable_pruning = true;
    config.enable_structural_plasticity = true; // Enable dynamic synaptogenesis
    
    // Enhanced connectivity parameters for version 0.5.5 - optimized for large scale
    config.input_hidden_prob = 0.15f;  // Reduced for computational efficiency at scale
    config.hidden_hidden_prob = 0.08f; // Sparse connectivity for emergent patterns
    config.hidden_output_prob = 0.4f;  // Selective output connections
    config.exc_ratio = 0.8f;
    
    // Synaptic parameters optimized for large-scale networks
    config.min_weight = 0.001f;        // Finer resolution for large networks
    config.max_weight = 2.0f;          // Increased for stronger signal propagation
    config.weight_init_std = 0.15f;    // Reduced for stability at scale
    
    // Topology parameters - MASSIVE SCALE-UP for tens of thousands of neurons
    config.numColumns = 16;            // 4x increase: 16 cortical columns
    config.neuronsPerColumn = 512;     // 8x increase: 512 neurons per column = 8,192 total
    config.localFanOut = 40;           // Increased connectivity for richer dynamics
    config.localFanIn = 40;            // Increased fan-in for complex integration
    
    // Enhanced timing
    config.dt = 0.1;
    config.simulation_time = 1000.0f; // 1 second simulation
    
    config.finalizeConfig();
    return config;
}

// ============================================================================
// SIMULATION MODE SELECTION
// ============================================================================

enum class SimulationMode {
    BASIC_MODULAR,           // Original modular simulation
    AUTONOMOUS_LEARNING,     // New autonomous learning mode
    INTERACTIVE_TRAINING,    // Interactive training with user feedback
    BENCHMARK_SUITE         // Performance benchmarking
};

// ============================================================================
// BASIC MODULAR SIMULATION (Enhanced Version)
// ============================================================================

void runBasicModularSimulation() {
    std::cout << "🧠 Running Enhanced Modular Neural Network Simulation..." << std::endl;

    // --- Configuration ---
    auto cognitive_config = create_default_config();
    cognitive_config.neurogenesis_rate = 0.002; // Higher plasticity for cognitive tasks

    auto motor_config = create_default_config();
    motor_config.stdp_learning_rate = 0.005; // Lower learning rate for stable motor control

    // --- Module Creation ---
    auto perception_net = std::make_shared<NeuralModule>("PerceptionNet", cognitive_config);
    auto planning_net = std::make_shared<NeuralModule>("PlanningNet", cognitive_config);
    auto motor_control_net = std::make_shared<NeuralModule>("MotorControlNet", motor_config);

    // --- CONTROLLER MODULE INTEGRATION ---
    std::cout << "🎛️ Initializing Central Neuromodulatory Controller..." << std::endl;
    
    ControllerConfig controller_config;
    controller_config.initial_dopamine_level = 0.4f;    // Start with good motivation
    controller_config.initial_serotonin_level = 0.5f;   // Balanced mood
    controller_config.reward_learning_rate = 0.02f;     // Enhanced learning rate
    controller_config.enable_detailed_logging = true;   // Enable detailed logging
    controller_config.enable_auto_regulation = true;    // Enable auto regulation
    
    auto neuro_controller = std::make_unique<ControllerModule>(controller_config);
    
    // Register modules with the controller
    neuro_controller->register_module("PerceptionNet", perception_net);
    neuro_controller->register_module("PlanningNet", planning_net);
    neuro_controller->register_module("MotorControlNet", motor_control_net);
    
    std::cout << "✅ Neuromodulatory controller configured with 3 modules" << std::endl;

    // --- Task-Level Module Creation ---
    auto cognitive_module = std::make_shared<CognitiveModule>(perception_net, planning_net);
    auto motor_module = std::make_shared<MotorModule>(motor_control_net);

    // --- System Initialization ---
    std::vector<std::shared_ptr<TaskModule>> task_modules;
    task_modules.push_back(cognitive_module);
    task_modules.push_back(motor_module);

    std::cout << "\nInitializing all task modules..." << std::endl;
    for (const auto& module : task_modules) {
        module->initialize();
    }

    // Enable detailed controller logging
    neuro_controller->enable_detailed_logging(true);

    // --- Enhanced Simulation Loop ---
    std::cout << "\n🚀 Starting enhanced simulation loop..." << std::endl;
    
    float total_simulation_time = 1000.0f; // ms
    float dt = 0.1f; // ms
    int num_steps = static_cast<int>(total_simulation_time / dt);
    
    // Enhanced input patterns for testing modular responses
    std::vector<std::vector<float>> test_patterns = {
        std::vector<float>(256, 0.0f), // Baseline
        std::vector<float>(256, 0.0f), // Pattern 1: Visual-like input
        std::vector<float>(256, 0.0f), // Pattern 2: Motor command
        std::vector<float>(256, 0.0f)  // Pattern 3: Mixed pattern
    };
    
    // Configure test patterns
    for (int i = 0; i < 32; i++) {
        test_patterns[1][i] = 15.0f + (i % 3) * 5.0f; // Spatial pattern
    }
    
    for (int i = 64; i < 128; i += 4) {
        test_patterns[2][i] = 20.0f; // Rhythmic pattern
    }
    
    for (int i = 0; i < 256; i += 8) {
        test_patterns[3][i] = 12.0f + (i / 32) * 2.0f; // Gradient pattern
    }
    
    int current_pattern = 0;
    int pattern_duration = 2000; // Steps per pattern
    int structural_plasticity_interval = 1000; // Every 100ms
    
    for (int i = 0; i < num_steps; ++i) {
        float current_time = i * dt;
        
        // Update the neuromodulatory controller first
        neuro_controller->update(dt);
        
        // Switch input patterns periodically
        if (i % pattern_duration == 0) {
            current_pattern = (current_pattern + 1) % test_patterns.size();
            std::cout << "🔄 Switching to input pattern " << current_pattern 
                      << " at time " << current_time << "ms" << std::endl;
            
            // Notify controller of pattern change (novelty detection)
            RewardSignal novelty_signal(RewardSignalType::NOVELTY_DETECTION, 0.3f, "Environment");
            novelty_signal.context = "Pattern change detected";
            neuro_controller->apply_reward("PerceptionNet", 0.3f, RewardSignalType::NOVELTY_DETECTION);
        }
        
        std::vector<float> inputs = test_patterns[current_pattern];
        
        // Add noise for biological realism
        for (auto& input : inputs) {
            if (input > 0.0f) {
                input += ((rand() % 100) / 100.0f - 0.5f) * 2.0f;
            }
        }
        
        // Calculate reward based on network coordination
        float reward = 0.1f;
        auto perception_stats = perception_net->get_stats();
        auto planning_stats = planning_net->get_stats();
        auto motor_stats = motor_control_net->get_stats();
        
        // Enhanced reward calculation with controller feedback
        bool modules_coordinated = (perception_stats.active_neuron_count > 5 && 
                                  planning_stats.active_neuron_count > 3 &&
                                  motor_stats.active_neuron_count > 2);
        
        if (modules_coordinated) {
            reward = 0.5f;
            
            // Generate cooperation reward through controller
            neuro_controller->apply_reward("", 0.4f, RewardSignalType::SOCIAL_COOPERATION);
        }
        
        // Update modules with inter-modular communication
        perception_net->update(dt, inputs, reward);
        auto perception_output = perception_net->get_output();
        
        planning_net->update(dt, perception_output, reward);
        auto planning_output = planning_net->get_output();
        
        motor_control_net->update(dt, planning_output, reward);
        
        // Controller-mediated attention allocation
        if (i % 500 == 0) {
            // Use available controller methods for attention modulation
            float total_activity = perception_stats.active_neuron_count + 
                                 planning_stats.active_neuron_count + 
                                 motor_stats.active_neuron_count;
            
            if (total_activity > 0) {
                // Focus on the most active module
                if (perception_stats.active_neuron_count > planning_stats.active_neuron_count && 
                    perception_stats.active_neuron_count > motor_stats.active_neuron_count) {
                    neuro_controller->enable_focus_mode("PerceptionNet", 0.6f);
                } else if (planning_stats.active_neuron_count > motor_stats.active_neuron_count) {
                    neuro_controller->enable_focus_mode("PlanningNet", 0.6f);
                } else {
                    neuro_controller->enable_focus_mode("MotorControlNet", 0.6f);
                }
            }
        }
        
        // Enhanced structural plasticity with controller coordination
        if (i % structural_plasticity_interval == 0 && i > 0) {
            std::cout << "🌱 Structural plasticity at " << current_time << "ms" << std::endl;
            
            // Controller decides when and where to promote growth
            float system_performance = neuro_controller->calculate_overall_system_performance();
            
            if (system_performance > 0.6f) {
                // Good performance - enable creative mode for exploration
                neuro_controller->enable_creative_mode(0.3f);
            } else if (system_performance < 0.4f) {
                // Poor performance - enable focus mode on a random module for now
                std::vector<std::string> modules = {"PerceptionNet", "PlanningNet", "MotorControlNet"};
                std::string focus_module = modules[rand() % modules.size()];
                neuro_controller->enable_focus_mode(focus_module, 0.7f);
            }
            
            auto* perception_network = perception_net->get_network();
            auto* planning_network = planning_net->get_network();
            auto* motor_network = motor_control_net->get_network();
            
            if (perception_network) {
                perception_network->grow_synapses();
                perception_network->prune_synapses();
            }
            if (planning_network) {
                planning_network->grow_synapses();
                planning_network->prune_synapses();
            }
            if (motor_network) {
                motor_network->grow_synapses();
                motor_network->prune_synapses();
            }
            
            // Coordinate module activities after structural changes
            neuro_controller->coordinate_module_activities();
        }
        
        // Enhanced monitoring output with controller status
        if (i % 1000 == 0) {
            std::cout << "\n📊 Time: " << current_time << "ms (Pattern " << current_pattern << ")" << std::endl;
            std::cout << "   🧠 Perception: " << perception_stats.active_neuron_count 
                      << " active, " << perception_stats.total_synapses << " synapses" << std::endl;
            std::cout << "   🎯 Planning: " << planning_stats.active_neuron_count 
                      << " active, " << planning_stats.total_synapses << " synapses" << std::endl;
            std::cout << "   🏃 Motor: " << motor_stats.active_neuron_count 
                      << " active, " << motor_stats.total_synapses << " synapses" << std::endl;
            
            // Display neuromodulator status
            std::cout << "   🧬 Dopamine: " << std::fixed << std::setprecision(2) 
                      << neuro_controller->get_concentration(NeuromodulatorType::DOPAMINE) << std::endl;
            std::cout << "   🧬 Serotonin: " << neuro_controller->get_concentration(NeuromodulatorType::SEROTONIN) << std::endl;
            std::cout << "   🧬 Norepinephrine: " << neuro_controller->get_concentration(NeuromodulatorType::NOREPINEPHRINE) << std::endl;
            
            // System performance
            float system_perf = neuro_controller->calculate_overall_system_performance();
            std::cout << "   📈 System Performance: " << std::setprecision(1) << system_perf * 100 << "%" << std::endl;
        }
    }

    // Final controller status report
    std::cout << "\n🎛️ ===== FINAL CONTROLLER STATUS =====" << std::endl;
    std::string status_report = neuro_controller->generate_status_report();
    std::cout << status_report << std::endl;

    std::cout << "\n✅ Basic modular simulation completed!" << std::endl;
}

// ============================================================================
// AUTONOMOUS LEARNING SIMULATION (New Version 0.5.5 Feature)
// ============================================================================

void runNLPTrainingSimulation(bool reset_model) {
    std::cout << "\n🚀 Starting NLP Training Simulation..." << std::endl;
    
    // Initialize NLP agent
    AutonomousLearningAgent agent(config);
    if (!agent.initialize(reset_model)) {
        std::cerr << "❌ Failed to initialize NLP agent!" << std::endl;
        return;
    }
    
    // Load training dataset
    if (!agent.loadDataset("datasets/language_corpus.txt")) {
        std::cerr << "❌ Failed to load language dataset!" << std::endl;
        return;
    }
    
    agent.startAutonomousLearning();
    
    int max_epochs = 100;
    std::cout << "🔄 Training for " << max_epochs << " epochs..." << std::endl;
    
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        // Process entire dataset
        while (agent.hasNextBatch()) {
            float progress = agent.autonomousLearningStep(0.1f);
            agent.update(0.1f);
        }
        
        // Epoch summary
        if (epoch % 10 == 0) {
            std::cout << "\n📈 Epoch " << epoch << " Complete:" << std::endl;
            std::cout << agent.getStatusReport() << std::endl;
            
            // Save checkpoint
            agent.saveAgentState("checkpoints/epoch_" + std::to_string(epoch));
        }
        
        // Reset dataset for next epoch
        agent.resetDataset();
    }
    
    std::cout << "🎉 NLP Training Complete!" << std::endl;
}


// ============================================================================
// INTERACTIVE TRAINING MODE
// ============================================================================

void runInteractiveTraining() {
    std::cout << "\n🎮 Interactive Training Mode - Coming Soon!" << std::endl;
    std::cout << "This mode will allow real-time interaction with the learning agent." << std::endl;
}

// ============================================================================
// BENCHMARK SUITE
// ============================================================================

void runBenchmarkSuite() {
    std::cout << "\n� Benchmark Suite - Coming Soon!" << std::endl;
    std::cout << "This will test performance across standardized learning tasks." << std::endl;
}

// ============================================================================
// MAIN FUNCTION WITH MODE SELECTION
// ============================================================================

int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv + 1, argv + argc);

    bool reset_model = false;
    if (std::find(args.begin(), args.end(), "--reset-model") != args.end()) {
        reset_model = true;
        std::cout << "🔥 --reset-model flag detected. Agent state will be reset." << std::endl;
    }

    std::cout << "🧠 NeuroGen 0.5.5 - Advanced Autonomous Learning Framework" << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    auto agent_config = create_default_config();
    AutonomousLearningAgent agent(agent_config);

    if (!agent.initialize(reset_model)) {
        std::cerr << "❌ Failed to initialize autonomous learning agent!" << std::endl;
        return 1;
    }

    std::cout << "✅ Agent initialized. Waiting for commands..." << std::endl;
    std::cout.flush();

    std::string line;
    while (true) {
        if (std::getline(std::cin, line)) {
            if (line.rfind("COMMAND:", 0) == 0) {
                agent.handleCommand(line.substr(8));
                std::cout.flush(); // Ensure any output is immediately visible
            } else if (line == "EXIT" || line == "QUIT") {
                std::cout << "🛑 Exit command received. Shutting down." << std::endl;
                break;
            } else if (!line.empty()) {
                std::cerr << "Warning: Received malformed input: " << line << std::endl;
            }
        } else {
            // Check if stdin was closed (parent process died)
            if (std::cin.eof()) {
                std::cout << "🛑 Input stream closed. Shutting down." << std::endl;
                break;
            }
            // Sleep briefly to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    std::cout << "🛑 Agent shutting down." << std::endl;
    agent.shutdown();

    return 0;
}
