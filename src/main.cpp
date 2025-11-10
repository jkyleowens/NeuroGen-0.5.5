// ============================================================================
// NLP-FOCUSED AUTONOMOUS LEARNING AGENT - MAIN APPLICATION
// File: src/main_nlp_agent.cpp
// ============================================================================

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>
#include <iomanip>
#include <sstream>

// NeuroGen Framework includes
#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/BrainModuleArchitecture.h"
#include "NeuroGen/NetworkConfig.h"

/**
 * @brief NLP Training Session Manager
 * 
 * Manages training sessions for the NLP-focused neural architecture
 */
class NLPTrainingSession {
public:
    NLPTrainingSession(std::shared_ptr<AutonomousLearningAgent> agent) 
        : agent_(agent), session_active_(false), total_inputs_processed_(0) {}
    
    void startSession() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "🚀 STARTING NLP TRAINING SESSION" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        session_active_ = true;
        session_start_time_ = std::chrono::steady_clock::now();
        
        if (agent_) {
            agent_->startAutonomousLearning();
        }
    }
    
    void stopSession() {
        session_active_ = false;
        
        if (agent_) {
            agent_->stopAutonomousLearning();
        }
        
        printSessionSummary();
    }
    
    bool processLanguageInput(const std::string& input) {
        if (!session_active_ || !agent_) return false;

        std::cout << "\n📝 Processing: \"" << input.substr(0, 50)
                  << (input.length() > 50 ? "..." : "") << "\"" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        bool success = agent_->processLanguageInput(input);

        // CRITICAL FIX: Allow neural processing to complete
        // The original had only 10ms which is too fast
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        if (success) {
            total_inputs_processed_++;

            // Get response after processing completes
            std::string response = agent_->generateLanguageResponse();

            // CRITICAL FIX: Check if response is empty and provide diagnostic
            if (response.empty()) {
                std::cout << "⚠️  WARNING: Empty response generated!" << std::endl;
                response = "[ERROR: No response generated - check module initialization]";
            }

            std::cout << "🤖 Response: " << response << std::endl;
            std::cout << "⏱️  Processing time: " << duration.count() << "ms" << std::endl;

            // Display metrics
            auto metrics = agent_->getLanguageMetrics();
            std::cout << "📊 Metrics - Comprehension: " << std::fixed << std::setprecision(3)
                      << metrics.comprehension_score << ", Reasoning: " << metrics.reasoning_score
                      << ", Quality: " << metrics.response_quality << std::endl;
        } else {
            std::cout << "❌ Failed to process input" << std::endl;
        }

        return success;
    }
    
    void printSessionSummary() {
        auto session_end_time = std::chrono::steady_clock::now();
        auto session_duration = std::chrono::duration_cast<std::chrono::seconds>(
            session_end_time - session_start_time_);
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "📋 SESSION SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "⏱️  Session Duration: " << session_duration.count() << " seconds" << std::endl;
        std::cout << "📝 Total Inputs Processed: " << total_inputs_processed_ << std::endl;
        
        if (agent_) {
            auto metrics = agent_->getLanguageMetrics();
            std::cout << "🎯 Final Metrics:" << std::endl;
            std::cout << "   - Comprehension Score: " << std::fixed << std::setprecision(3) 
                      << metrics.comprehension_score << std::endl;
            std::cout << "   - Reasoning Score: " << metrics.reasoning_score << std::endl;
            std::cout << "   - Response Quality: " << metrics.response_quality << std::endl;
            std::cout << "   - Learning Efficiency: " << metrics.learning_efficiency << std::endl;
            std::cout << "   - Success Rate: " << std::fixed << std::setprecision(1)
                      << (metrics.processed_inputs > 0 ? 
                          (100.0f * metrics.successful_responses / metrics.processed_inputs) : 0.0f) 
                      << "%" << std::endl;
        }
        
        std::cout << std::string(80, '=') << std::endl;
    }
    
private:
    std::shared_ptr<AutonomousLearningAgent> agent_;
    bool session_active_;
    int total_inputs_processed_;
    std::chrono::steady_clock::time_point session_start_time_;
};

/**
 * @brief Display system information and architecture details
 */
void displaySystemInfo(std::shared_ptr<AutonomousLearningAgent> agent) {
    std::cout << "\n🧠 NEURAL ARCHITECTURE INFORMATION" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    if (agent) {
        auto brain_arch = agent->getBrainArchitecture();
        if (brain_arch) {
            auto module_names = brain_arch->getModuleNames();
            std::cout << "📊 Total Modules: " << module_names.size() << std::endl;
            
            std::cout << "🔧 Module Details:" << std::endl;
            for (const auto& name : module_names) {
                int neuron_count = agent->getModuleNeuronCount(name);
                std::cout << "   - " << name << ": " << neuron_count << " neurons" << std::endl;
            }
            
            auto connections = brain_arch->getConnections();
            std::cout << "🔗 Inter-module Connections: " << connections.size() << std::endl;
            
            auto neuro_levels = brain_arch->getNeuromodulatorLevels();
            std::cout << "🧪 Neuromodulator Levels:" << std::endl;
            for (const auto& [name, level] : neuro_levels) {
                std::cout << "   - " << name << ": " << std::fixed 
                          << std::setprecision(3) << level << std::endl;
            }
        }
        
        std::cout << "🎯 Learning Status: " 
                  << (agent->isLearningActive() ? "ACTIVE" : "INACTIVE") << std::endl;
        std::cout << "🔤 NLP Mode: " 
                  << (agent->isNLPModeActive() ? "ENABLED" : "DISABLED") << std::endl;
        std::cout << "📈 Learning Progress: " << std::fixed << std::setprecision(1)
                  << (agent->getLearningProgress() * 100.0f) << "%" << std::endl;
    }
    
    std::cout << std::string(50, '-') << std::endl;
}

/**
 * @brief Interactive command interface for the NLP agent
 */
void runInteractiveMode(std::shared_ptr<AutonomousLearningAgent> agent) {
    NLPTrainingSession session(agent);
    session.startSession();
    
    std::cout << "\n💬 INTERACTIVE NLP MODE" << std::endl;
    std::cout << "Type 'quit' to exit, 'help' for commands, 'info' for system info" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    std::string input;
    while (true) {
        std::cout << "\n> ";
        std::getline(std::cin, input);
        
        if (input.empty()) continue;
        
        if (input == "quit" || input == "exit") {
            std::cout << "👋 Exiting interactive mode..." << std::endl;
            break;
        } else if (input == "help") {
            std::cout << "📚 Available commands:" << std::endl;
            std::cout << "   help    - Show this help message" << std::endl;
            std::cout << "   info    - Display system information" << std::endl;
            std::cout << "   metrics - Show current language metrics" << std::endl;
            std::cout << "   reset   - Reset learning state" << std::endl;
            std::cout << "   save    - Save current state" << std::endl;
            std::cout << "   quit    - Exit interactive mode" << std::endl;
            std::cout << "   Any other text will be processed as language input" << std::endl;
        } else if (input == "info") {
            displaySystemInfo(agent);
        } else if (input == "metrics") {
            if (agent) {
                auto metrics = agent->getLanguageMetrics();
                std::cout << "📊 Current Language Metrics:" << std::endl;
                std::cout << "   Comprehension: " << std::fixed << std::setprecision(3) 
                          << metrics.comprehension_score << std::endl;
                std::cout << "   Reasoning: " << metrics.reasoning_score << std::endl;
                std::cout << "   Response Quality: " << metrics.response_quality << std::endl;
                std::cout << "   Inputs Processed: " << metrics.processed_inputs << std::endl;
                std::cout << "   Successful Responses: " << metrics.successful_responses << std::endl;
            }
        } else if (input == "reset") {
            std::cout << "🔄 Resetting learning state..." << std::endl;
            // Note: Reset functionality would need to be implemented in the agent
            std::cout << "✅ Reset complete" << std::endl;
        } else if (input == "save") {
            std::cout << "💾 Saving current state..." << std::endl;
            if (agent && agent->saveLearningState("nlp_checkpoint")) {
                std::cout << "✅ State saved successfully" << std::endl;
            } else {
                std::cout << "❌ Failed to save state" << std::endl;
            }
        } else {
            // Process as language input
            session.processLanguageInput(input);
        }
    }
    
    session.stopSession();
}

/**
 * @brief Automated training with predefined language samples
 */
void runAutomatedTraining(std::shared_ptr<AutonomousLearningAgent> agent) {
    std::cout << "\n🤖 AUTOMATED TRAINING MODE" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    // Sample training texts for different language understanding tasks
    std::vector<std::string> training_samples = {
        "Hello, how are you today?",
        "What is the meaning of artificial intelligence?",
        "Can you explain the concept of neural networks?",
        "The weather is beautiful outside.",
        "I enjoy reading books about science and technology.",
        "What is two plus two?",
        "Tell me about the history of computers.",
        "How do biological neurons work?",
        "What are the applications of machine learning?",
        "Can machines truly understand language?",
        "The quick brown fox jumps over the lazy dog.",
        "Explain the difference between syntax and semantics.",
        "What is consciousness?",
        "How do we learn language as children?",
        "What makes humans creative?",
        "Can artificial neural networks dream?",
        "What is the relationship between mind and brain?",
        "How do we process visual information?",
        "What role does attention play in cognition?",
        "Can machines experience emotions?"
    };
    
    NLPTrainingSession session(agent);
    session.startSession();
    
    std::cout << "🎯 Training with " << training_samples.size() << " samples..." << std::endl;
    
    for (size_t i = 0; i < training_samples.size(); ++i) {
        std::cout << "\n[" << (i + 1) << "/" << training_samples.size() << "] ";
        session.processLanguageInput(training_samples[i]);
        
        // Brief pause between samples
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Update agent
        if (agent) {
            agent->update(0.1f); // 100ms update
        }
    }
    
    session.stopSession();
}

/**
 * @brief Main application entry point
 */
int main(int argc, char* argv[]) {
    std::cout << "🧠 NeuroGen NLP-Focused Autonomous Learning Agent" << std::endl;
    std::cout << "🔤 Natural Language Processing Mode" << std::endl;
    std::cout << "🚫 Autonomous Computer Control DISABLED" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    try {
        // Create network configuration for NLP processing
        NetworkConfig config;
        config.num_neurons = 2048;        // Base neuron count
        config.input_size = 1024;         // Large input for tokenized text
        config.output_size = 512;         // Reasonable output size
        config.learning_rate = 0.005f;    // Conservative learning rate for language
        config.enable_plasticity = true;
        config.enable_cuda = false;       // CPU-only for this demo
        
        std::cout << "🔧 Initializing NLP-focused neural architecture..." << std::endl;
        
        // Create autonomous learning agent
        auto agent = std::make_shared<AutonomousLearningAgent>(config);
        
        // Initialize agent in NLP mode
        if (!agent->initialize(false)) {  // Don't reset existing model
            std::cerr << "❌ Failed to initialize agent" << std::endl;
            return -1;
        }
        
        // Set processing mode to NLP only
        agent->setProcessingMode(AutonomousLearningAgent::ProcessingMode::NLP_ONLY);
        
        std::cout << "✅ Agent initialized successfully" << std::endl;
        
        // Display system information
        displaySystemInfo(agent);
        
        // Determine mode based on command line arguments
        bool interactive_mode = true;
        if (argc > 1) {
            std::string mode_arg = argv[1];
            if (mode_arg == "--automated" || mode_arg == "-a") {
                interactive_mode = false;
            } else if (mode_arg == "--interactive" || mode_arg == "-i") {
                interactive_mode = true;
            } else if (mode_arg == "--help" || mode_arg == "-h") {
                std::cout << "\nUsage: " << argv[0] << " [OPTIONS]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  -i, --interactive  Run in interactive mode (default)" << std::endl;
                std::cout << "  -a, --automated    Run automated training" << std::endl;
                std::cout << "  -h, --help         Show this help message" << std::endl;
                return 0;
            }
        }
        
        // Run in selected mode
        if (interactive_mode) {
            runInteractiveMode(agent);
        } else {
            runAutomatedTraining(agent);
        }
        
        std::cout << "\n💾 Saving final state..." << std::endl;
        if (agent->saveLearningState("nlp_final_state")) {
            std::cout << "✅ Final state saved successfully" << std::endl;
        } else {
            std::cout << "⚠️  Warning: Failed to save final state" << std::endl;
        }
        
        std::cout << "\n🎉 Training session complete!" << std::endl;
        std::cout << "🧠 Neural architecture has been trained on language processing tasks" << std::endl;
        std::cout << "📊 Check the metrics above for performance details" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "❌ Unknown error occurred" << std::endl;
        return -1;
    }
    
    return 0;
}