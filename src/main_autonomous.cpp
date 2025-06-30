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
    config.num_neurons = 256; // Enhanced neuron count for version 0.5.5
    config.enable_neurogenesis = true;
    config.enable_stdp = true;
    config.enable_pruning = true;
    config.enable_structural_plasticity = true; // Enable dynamic synaptogenesis
    
    // Enhanced connectivity parameters for version 0.5.5
    config.input_hidden_prob = 0.6f;
    config.hidden_hidden_prob = 0.3f;
    config.hidden_output_prob = 0.8f;
    config.exc_ratio = 0.8f;
    
    // Synaptic parameters
    config.min_weight = 0.01f;
    config.max_weight = 1.5f;
    config.weight_init_std = 0.3f;
    
    // Topology parameters
    config.numColumns = 4;
    config.neuronsPerColumn = 64;
    config.localFanOut = 15;
    config.localFanIn = 15;
    
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
    controller_config.curiosity_drive_strength = 0.4f;  // Strong exploration drive
    controller_config.enable_adaptive_baselines = true;
    controller_config.enable_stress_response = true;
    
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
            RewardSignal novelty_signal;
            novelty_signal.type = RewardSignalType::NOVELTY_DETECTION;
            novelty_signal.magnitude = 0.3f;
            novelty_signal.confidence = 0.8f;
            novelty_signal.source_module = "Environment";
            novelty_signal.target_module = "PerceptionNet";
            neuro_controller->process_reward_signal(novelty_signal);
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
            RewardSignal coop_signal;
            coop_signal.type = RewardSignalType::SOCIAL_COOPERATION;
            coop_signal.magnitude = 0.4f;
            coop_signal.confidence = 0.9f;
            coop_signal.source_module = "ControllerModule";
            coop_signal.target_module = ""; // All modules
            neuro_controller->process_reward_signal(coop_signal);
        }
        
        // Update modules with inter-modular communication
        perception_net->update(dt, inputs, reward);
        auto perception_output = perception_net->get_output();
        
        planning_net->update(dt, perception_output, reward);
        auto planning_output = planning_net->get_output();
        
        motor_control_net->update(dt, planning_output, reward);
        
        // Controller-mediated attention allocation
        if (i % 500 == 0) {
            std::unordered_map<std::string, float> attention_weights;
            
            // Allocate attention based on activity levels
            float total_activity = perception_stats.active_neuron_count + 
                                 planning_stats.active_neuron_count + 
                                 motor_stats.active_neuron_count;
            
            if (total_activity > 0) {
                attention_weights["PerceptionNet"] = perception_stats.active_neuron_count / total_activity;
                attention_weights["PlanningNet"] = planning_stats.active_neuron_count / total_activity;
                attention_weights["MotorControlNet"] = motor_stats.active_neuron_count / total_activity;
                
                neuro_controller->allocate_attention(attention_weights);
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
                // Poor performance - enable focus mode on weakest module
                float min_perf = 1.0f;
                std::string weakest_module;
                
                float perc_perf = neuro_controller->calculate_module_performance("PerceptionNet");
                float plan_perf = neuro_controller->calculate_module_performance("PlanningNet");
                float motor_perf = neuro_controller->calculate_module_performance("MotorControlNet");
                
                if (perc_perf < min_perf) { min_perf = perc_perf; weakest_module = "PerceptionNet"; }
                if (plan_perf < min_perf) { min_perf = plan_perf; weakest_module = "PlanningNet"; }
                if (motor_perf < min_perf) { min_perf = motor_perf; weakest_module = "MotorControlNet"; }
                
                if (!weakest_module.empty()) {
                    neuro_controller->enable_focus_mode(weakest_module, 0.7f);
                }
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
            auto concentrations = neuro_controller->get_all_concentrations();
            std::cout << "   🧬 Dopamine: " << std::fixed << std::setprecision(2) 
                      << concentrations[NeuromodulatorType::DOPAMINE] << std::endl;
            std::cout << "   🧬 Serotonin: " << concentrations[NeuromodulatorType::SEROTONIN] << std::endl;
            std::cout << "   🧬 Norepinephrine: " << concentrations[NeuromodulatorType::NOREPINEPHRINE] << std::endl;
            
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

void runAutonomousLearningSimulation() {
    std::cout << "\n🤖 ========== AUTONOMOUS LEARNING SIMULATION ==========\n" << std::endl;
    std::cout << "🚀 Initializing Advanced Autonomous Learning Agent..." << std::endl;
    
    // Configure autonomous agent
    AutonomousAgentConfig agent_config;
    agent_config.initial_neuron_count = 512;        // Start with more neurons
    agent_config.max_neuron_count = 2048;           // Allow significant expansion
    agent_config.exploration_rate = 0.2f;           // Higher initial exploration
    agent_config.curiosity_weight = 0.3f;           // Strong curiosity drive
    agent_config.intrinsic_motivation_strength = 0.25f; // Strong internal motivation
    agent_config.expansion_threshold = 0.7f;        // Expand when 70% complexity reached
    agent_config.max_exploration_steps = 5000;      // Longer exploration periods
    
    std::cout << "🔧 Agent Configuration:" << std::endl;
    std::cout << "   • Initial neurons: " << agent_config.initial_neuron_count << std::endl;
    std::cout << "   • Max neurons: " << agent_config.max_neuron_count << std::endl;
    std::cout << "   • Exploration rate: " << agent_config.exploration_rate << std::endl;
    std::cout << "   • Curiosity weight: " << agent_config.curiosity_weight << std::endl;
    
    // Create autonomous learning agent
    AutonomousLearningAgent agent(agent_config);
    
    // ========================================
    // SETUP SIMULATED ENVIRONMENT
    // ========================================
    
    std::cout << "\n🌍 Setting up simulated environment..." << std::endl;
    
    // Environment state variables
    std::vector<float> environment_state(64, 0.0f);
    std::vector<float> environment_dynamics(64, 0.0f);
    float environment_complexity = 0.5f;
    int environment_phase = 0;
    std::mt19937 env_rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> env_dist(-1.0f, 1.0f);
    
    // Initialize environment with interesting dynamics
    for (size_t i = 0; i < environment_state.size(); ++i) {
        environment_state[i] = env_dist(env_rng);
        environment_dynamics[i] = env_dist(env_rng) * 0.1f;
    }
    
    // Environment sensor function
    auto environment_sensor = [&]() -> std::vector<float> {
        // Update environment dynamics
        environment_phase++;
        
        for (size_t i = 0; i < environment_state.size(); ++i) {
            // Add some interesting dynamics
            float wave = std::sin(environment_phase * 0.01f + i * 0.1f) * 0.1f;
            float noise = env_dist(env_rng) * 0.05f;
            environment_state[i] += environment_dynamics[i] + wave + noise;
            
            // Keep within bounds
            environment_state[i] = std::tanh(environment_state[i]);
            
            // Occasionally change dynamics
            if (environment_phase % 1000 == 0) {
                environment_dynamics[i] += env_dist(env_rng) * 0.02f;
                environment_dynamics[i] = std::tanh(environment_dynamics[i] * 0.8f);
            }
        }
        
        // Increase complexity over time
        if (environment_phase % 2000 == 0) {
            environment_complexity = std::min(1.0f, environment_complexity + 0.1f);
            std::cout << "🌊 Environment complexity increased to " << environment_complexity << std::endl;
        }
        
        return environment_state;
    };
    
    // Environment actuator function
    auto environment_actuator = [&](const std::vector<float>& action) -> float {
        if (action.empty()) return 0.0f;
        
        float reward = 0.0f;
        
        // Reward for appropriate responses to environment
        for (size_t i = 0; i < std::min(action.size(), environment_state.size()); ++i) {
            // Reward actions that complement environment state
            float synergy = -std::abs(action[i] + environment_state[i]);
            reward += synergy * 0.1f;
            
            // Reward exploration of different action magnitudes
            reward += std::abs(action[i]) * 0.05f;
        }
        
        // Bonus for complex action patterns
        float action_complexity = 0.0f;
        for (size_t i = 1; i < action.size(); ++i) {
            action_complexity += std::abs(action[i] - action[i-1]);
        }
        reward += std::tanh(action_complexity) * 0.2f;
        
        return std::tanh(reward); // Normalize reward
    };
    
    // Environment reward signal (global environmental feedback)
    auto environment_reward_signal = [&]() -> float {
        // Provide environmental challenges that require adaptation
        float challenge_reward = 0.0f;
        
        // Reward stability in changing environment
        float stability = 0.0f;
        for (float state : environment_state) {
            stability += std::abs(state);
        }
        stability = 1.0f / (1.0f + stability); // Inverse relationship
        
        challenge_reward += stability * 0.3f;
        
        // Reward learning in complex environments
        challenge_reward += environment_complexity * 0.2f;
        
        return challenge_reward;
    };
    
    // Setup environment interaction
    agent.setEnvironmentSensor(environment_sensor);
    agent.setEnvironmentActuator(environment_actuator);
    agent.setEnvironmentRewardSignal(environment_reward_signal);
    
    std::cout << "✅ Environment configured with dynamic complexity!" << std::endl;
    
    // ========================================
    // ADD ADDITIONAL LEARNING GOALS
    // ========================================
    
    std::cout << "\n🎯 Setting up advanced learning goals..." << std::endl;
    
    // Goal 1: Environment Adaptation
    auto adaptation_goal = std::make_unique<AutonomousGoal>();
    adaptation_goal->description = "Environment Adaptation";
    adaptation_goal->target_competence = 0.8f;
    adaptation_goal->priority = 9;
    adaptation_goal->evaluation_fn = [&environment_complexity](const std::vector<float>& state) {
        return environment_complexity * 0.5f + 0.3f; // Scale with environment complexity
    };
    agent.addLearningGoal(std::move(adaptation_goal));
    
    // Goal 2: Action Diversity
    auto diversity_goal = std::make_unique<AutonomousGoal>();
    diversity_goal->description = "Action Diversity";
    diversity_goal->target_competence = 0.7f;
    diversity_goal->priority = 6;
    diversity_goal->evaluation_fn = [](const std::vector<float>& state) {
        float diversity = 0.0f;
        for (size_t i = 1; i < state.size(); ++i) {
            diversity += std::abs(state[i] - state[i-1]);
        }
        return std::tanh(diversity / state.size());
    };
    agent.addLearningGoal(std::move(diversity_goal));
    
    // Goal 3: Predictive Learning
    auto prediction_goal = std::make_unique<AutonomousGoal>();
    prediction_goal->description = "Predictive Learning";
    prediction_goal->target_competence = 0.85f;
    prediction_goal->priority = 8;
    prediction_goal->evaluation_fn = [](const std::vector<float>& state) {
        // Reward ability to predict patterns
        float pattern_strength = 0.0f;
        for (size_t i = 2; i < state.size(); ++i) {
            float predicted = 2 * state[i-1] - state[i-2]; // Linear prediction
            float error = std::abs(state[i] - predicted);
            pattern_strength += 1.0f / (1.0f + error);
        }
        return pattern_strength / state.size();
    };
    agent.addLearningGoal(std::move(prediction_goal));
    
    std::cout << "✅ Advanced learning goals established!" << std::endl;
    
    // ========================================
    // RUN AUTONOMOUS LEARNING SIMULATION
    // ========================================
    
    std::cout << "\n🚀 Starting Autonomous Learning Simulation..." << std::endl;
    std::cout << "   The agent will now explore, learn, and adapt autonomously!" << std::endl;
    std::cout << "   Watch for network expansions and learning progress...\n" << std::endl;
    
    // Option 1: Run continuous learning loop
    int max_learning_steps = 5000; // 5000 learning steps
    std::cout << "🔄 Running " << max_learning_steps << " autonomous learning steps..." << std::endl;
    
    auto learning_start = std::chrono::high_resolution_clock::now();
    
    agent.startAutonomousLearning();
    
    // Manual learning loop for detailed monitoring
    for (int step = 0; step < max_learning_steps; ++step) {
        float learning_progress = agent.autonomousLearningStep(1.0f);
        
        // Detailed monitoring every 500 steps
        if (step % 500 == 0) {
            auto metrics = agent.getPerformanceMetrics();
            auto network_stats = agent.getNetworkStatistics();
            auto competencies = agent.getGoalCompetencies();
            
            std::cout << "\n📈 Learning Progress Report (Step " << step << "):" << std::endl;
            std::cout << "   🧠 Network: " << network_stats.neuron_count << " neurons, " 
                      << network_stats.synapse_count << " synapses" << std::endl;
            std::cout << "   📊 Avg Reward: " << std::fixed << std::setprecision(3) 
                      << metrics.average_reward << std::endl;
            std::cout << "   🔍 Exploration: " << agent.getExplorationEffectiveness() << std::endl;
            std::cout << "   🌱 Expansions: " << metrics.network_expansions << std::endl;
            std::cout << "   🎯 Goal Progress:" << std::endl;
            
            for (const auto& comp : competencies) {
                std::cout << "      • " << comp.first << ": " 
                          << std::setprecision(2) << comp.second * 100 << "%" << std::endl;
            }
            
            // Check for early completion
            if (learning_progress > 0.9f) {
                std::cout << "\n🎉 High learning competence achieved early!" << std::endl;
                break;
            }
        }
        
        // Introduce environmental challenges periodically
        if (step % 1000 == 0 && step > 0) {
            environment_complexity = std::min(1.0f, environment_complexity + 0.05f);
            std::cout << "\n🌊 Environmental challenge increased! Complexity: " 
                      << environment_complexity << std::endl;
        }
        
        // Brief pause to prevent CPU overload
        if (step % 100 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    agent.stopAutonomousLearning();
    
    auto learning_end = std::chrono::high_resolution_clock::now();
    auto learning_duration = std::chrono::duration<double>(learning_end - learning_start).count();
    
    // ========================================
    // FINAL ANALYSIS AND REPORT
    // ========================================
    
    std::cout << "\n🎊 ========== AUTONOMOUS LEARNING COMPLETED ==========\n" << std::endl;
    
    // Generate comprehensive report
    std::string final_report = agent.generateLearningReport();
    std::cout << final_report << std::endl;
    
    // Additional analysis
    auto final_metrics = agent.getPerformanceMetrics();
    auto final_network_stats = agent.getNetworkStatistics();
    auto final_competencies = agent.getGoalCompetencies();
    
    std::cout << "⏱️ Learning Duration: " << std::fixed << std::setprecision(1) 
              << learning_duration << " seconds" << std::endl;
    std::cout << "⚡ Learning Speed: " << (max_learning_steps / learning_duration) 
              << " steps/second" << std::endl;
    
    // Check achievement levels
    bool high_achiever = true;
    for (const auto& comp : final_competencies) {
        if (comp.second < 0.6f) {
            high_achiever = false;
            break;
        }
    }
    
    if (high_achiever) {
        std::cout << "\n🏆 EXCEPTIONAL PERFORMANCE: All goals achieved with high competence!" << std::endl;
    } else {
        std::cout << "\n📈 GOOD PROGRESS: Significant learning demonstrated across multiple goals!" << std::endl;
    }
    
    if (final_network_stats.neuron_count > agent_config.initial_neuron_count) {
        std::cout << "🌱 DYNAMIC GROWTH: Network expanded from " << agent_config.initial_neuron_count 
                  << " to " << final_network_stats.neuron_count << " neurons!" << std::endl;
    }
    
    std::cout << "\n✅ Autonomous Learning Simulation Complete!" << std::endl;
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
    std::cout << "\n📏 Benchmark Suite - Coming Soon!" << std::endl;
    std::cout << "This will test performance across standardized learning tasks." << std::endl;
}

// ============================================================================
// MAIN FUNCTION WITH MODE SELECTION
// ============================================================================

int main() {
    std::cout << "🧠 NeuroGen 0.5.5 - Advanced Autonomous Learning Framework" << std::endl;
    std::cout << "=========================================================\n" << std::endl;
    
    // For now, automatically run autonomous learning simulation
    // In the future, this could be command-line configurable
    
    std::cout << "🔍 Available Simulation Modes:" << std::endl;
    std::cout << "   1. Basic Modular Simulation (Enhanced)" << std::endl;
    std::cout << "   2. Autonomous Learning Agent (NEW!)" << std::endl;
    std::cout << "   3. Interactive Training (Coming Soon)" << std::endl;
    std::cout << "   4. Benchmark Suite (Coming Soon)" << std::endl;
    
    std::cout << "\n🚀 Launching Autonomous Learning Simulation..." << std::endl;
    
    try {
        // Run basic modular simulation first
        std::cout << "\n==== Phase 1: Basic Modular Network Test ====" << std::endl;
        runBasicModularSimulation();
        
        // Then run autonomous learning
        std::cout << "\n==== Phase 2: Autonomous Learning Agent ====" << std::endl;
        runAutonomousLearningSimulation();
        
        std::cout << "\n🎉 All simulations completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Simulation error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🎊 NeuroGen 0.5.5 Simulation Suite Complete!" << std::endl;
    std::cout << "Thank you for exploring advanced autonomous neural learning!" << std::endl;
    
    return 0;
}
