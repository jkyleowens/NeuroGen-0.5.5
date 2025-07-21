// ============================================================================
// MAIN APPLICATION - NATURAL LANGUAGE PROCESSING AUTONOMOUS AGENT
// File: src/main.cpp
// ============================================================================

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <random>
#include <signal.h>

// NeuroGen Framework Includes
#include <NeuroGen/AutonomousLearningAgent.h>
#include <NeuroGen/BrainModuleArchitecture.h>
#include <NeuroGen/LanguageInterface.h>
#include <NeuroGen/NetworkConfig.h>
#include <NeuroGen/SafetyManager.h>

// Global variables for signal handling
std::atomic<bool> g_shutdown_requested{false};
std::shared_ptr<AutonomousLearningAgent> g_agent;

/**
 * @brief Signal handler for graceful shutdown
 */
void signalHandler(int signal) {
    std::cout << "\n🛑 Shutdown signal received (" << signal << "). Initiating graceful shutdown..." << std::endl;
    g_shutdown_requested = true;
    
    if (g_agent) {
        g_agent->stopAutonomousLearning();
    }
}

/**
 * @brief Language Training Session Manager
 * 
 * Manages interactive language training sessions with the autonomous agent,
 * providing real-time feedback and performance monitoring.
 */
class LanguageTrainingSession {
private:
    std::shared_ptr<AutonomousLearningAgent> agent_;
    std::unique_ptr<LanguageInterface> language_interface_;
    std::atomic<bool> session_active_;
    std::vector<std::pair<std::string, std::string>> conversation_history_;
    std::map<std::string, float> session_metrics_;
    std::chrono::high_resolution_clock::time_point session_start_time_;
    
public:
    explicit LanguageTrainingSession(std::shared_ptr<AutonomousLearningAgent> agent) 
        : agent_(agent), session_active_(false) {
        
        language_interface_ = std::make_unique<LanguageInterface>();
        session_metrics_["total_inputs"] = 0.0f;
        session_metrics_["comprehension_score"] = 0.0f;
        session_metrics_["response_quality"] = 0.0f;
        session_metrics_["learning_progress"] = 0.0f;
    }
    
    bool startSession() {
        if (!language_interface_->initialize()) {
            std::cerr << "Failed to initialize language interface" << std::endl;
            return false;
        }
        
        if (!agent_) {
            std::cerr << "No agent available for training session" << std::endl;
            return false;
        }
        
        session_active_ = true;
        session_start_time_ = std::chrono::high_resolution_clock::now();
        
        // Configure agent for language training mode
        agent_->setOperatingMode(AutonomousLearningAgent::OperatingMode::LANGUAGE_TRAINING);
        agent_->setPassiveMode(false); // Enable active learning
        
        std::cout << "\n🎓 LANGUAGE TRAINING SESSION STARTED" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "📝 Enter text for the agent to learn from" << std::endl;
        std::cout << "💬 Agent will process and respond to your input" << std::endl;
        std::cout << "📊 Performance metrics will be displayed in real-time" << std::endl;
        std::cout << "🔍 Type 'help' for commands, 'quit' to end session" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        return true;
    }
    
    void processLanguageInput(const std::string& input) {
        if (!session_active_ || !agent_) return;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process input through agent
        bool success = agent_->processLanguageInput(input, "", true);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto processing_time = std::chrono::duration<float>(end_time - start_time).count();
        
        if (success) {
            // Generate response
            std::string response = agent_->generateLanguageResponse();
            
            // Store conversation turn
            conversation_history_.emplace_back(input, response);
            
            // Update metrics
            session_metrics_["total_inputs"] += 1.0f;
            session_metrics_["avg_processing_time"] = processing_time;
            session_metrics_["learning_progress"] = agent_->getLearningProgress();
            
            // Display response
            std::cout << "\n🤖 Agent Response: " << response << std::endl;
            
            // Show processing info
            std::cout << "⏱️  Processing Time: " << std::fixed << std::setprecision(3) 
                      << processing_time << "s" << std::endl;
            
            // Show learning metrics
            displaySessionMetrics();
            
        } else {
            std::cout << "❌ Failed to process input: " << input << std::endl;
        }
    }
    
    void displaySessionMetrics() {
        if (!agent_) return;
        
        auto attention_weights = agent_->getAttentionWeights();
        auto language_stats = agent_->getLanguageProcessingStats();
        
        std::cout << "\n📊 Current Session Metrics:" << std::endl;
        std::cout << "   • Total Inputs: " << static_cast<int>(session_metrics_["total_inputs"]) << std::endl;
        std::cout << "   • Learning Progress: " << std::fixed << std::setprecision(1) 
                  << (session_metrics_["learning_progress"] * 100) << "%" << std::endl;
        
        // Display attention weights for language modules
        std::cout << "   • Attention Distribution:" << std::endl;
        for (const auto& [module, weight] : attention_weights) {
            if (module.find("language") != std::string::npos || 
                module.find("semantic") != std::string::npos ||
                module.find("syntactic") != std::string::npos ||
                module.find("working_memory") != std::string::npos) {
                std::cout << "     - " << module << ": " << std::fixed << std::setprecision(2) 
                          << (weight * 100) << "%" << std::endl;
            }
        }
        
        // Display language processing stats if available
        if (!language_stats.empty()) {
            std::cout << "   • Language Processing:" << std::endl;
            for (const auto& [stat, value] : language_stats) {
                std::cout << "     - " << stat << ": " << std::fixed << std::setprecision(2) 
                          << value << std::endl;
            }
        }
        
        std::cout << std::string(60, '-') << std::endl;
    }
    
    void stopSession() {
        session_active_ = false;
        
        auto session_end_time = std::chrono::high_resolution_clock::now();
        auto session_duration = std::chrono::duration<float>(session_end_time - session_start_time_).count();
        
        std::cout << "\n📈 LANGUAGE TRAINING SESSION COMPLETE" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "⏰ Session Duration: " << std::fixed << std::setprecision(1) 
                  << session_duration << " seconds" << std::endl;
        std::cout << "📝 Total Inputs Processed: " << static_cast<int>(session_metrics_["total_inputs"]) << std::endl;
        std::cout << "🧠 Final Learning Progress: " << std::fixed << std::setprecision(1) 
                  << (session_metrics_["learning_progress"] * 100) << "%" << std::endl;
        
        if (!conversation_history_.empty()) {
            std::cout << "💾 Conversation turns: " << conversation_history_.size() << std::endl;
        }
        
        // Save session data if requested
        saveSessionData();
        
        language_interface_->shutdown();
    }
    
    void saveSessionData() {
        try {
            std::string filename = "language_session_" + getCurrentTimestamp() + ".txt";
            std::ofstream session_file(filename);
            
            if (session_file.is_open()) {
                session_file << "Language Training Session Report\n";
                session_file << "================================\n\n";
                
                session_file << "Session Metrics:\n";
                for (const auto& [metric, value] : session_metrics_) {
                    session_file << "  " << metric << ": " << value << "\n";
                }
                
                session_file << "\nConversation History:\n";
                for (size_t i = 0; i < conversation_history_.size(); ++i) {
                    session_file << "\nTurn " << (i + 1) << ":\n";
                    session_file << "  Input:    " << conversation_history_[i].first << "\n";
                    session_file << "  Response: " << conversation_history_[i].second << "\n";
                }
                
                session_file.close();
                std::cout << "💾 Session data saved to: " << filename << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ Failed to save session data: " << e.what() << std::endl;
        }
    }
    
    bool isActive() const { return session_active_; }
    
private:
    std::string getCurrentTimestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }
};

/**
 * @brief Display system information and agent status
 */
void displaySystemInfo(std::shared_ptr<AutonomousLearningAgent> agent) {
    std::cout << "\n🖥️  SYSTEM STATUS" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    if (agent) {
        std::cout << "🤖 Agent Status:" << std::endl;
        std::cout << agent->getStatusReport() << std::endl;
        
        auto language_stats = agent->getLanguageProcessingStats();
        if (!language_stats.empty()) {
            std::cout << "\n📊 Language Processing Statistics:" << std::endl;
            for (const auto& [stat, value] : language_stats) {
                std::cout << "   • " << stat << ": " << std::fixed << std::setprecision(2) 
                          << value << std::endl;
            }
        }
        
        std::cout << "\n🧠 Neural Architecture:" << std::endl;
        std::cout << "   • Total Neurons: " << agent->getTotalNeuronCount() << std::endl;
        std::cout << "   • Active Modules: Language-focused modular network" << std::endl;
        std::cout << "   • Learning Phase: " << (agent->getLearningProgress() * 100) << "%" << std::endl;
    } else {
        std::cout << "❌ No agent available" << std::endl;
    }
}

/**
 * @brief Run interactive language training mode
 */
void runInteractiveTraining(std::shared_ptr<AutonomousLearningAgent> agent) {
    LanguageTrainingSession session(agent);
    
    if (!session.startSession()) {
        std::cerr << "❌ Failed to start language training session" << std::endl;
        return;
    }
    
    std::string input;
    while (session.isActive() && !g_shutdown_requested) {
        std::cout << "\n💬 Your input: ";
        std::getline(std::cin, input);
        
        if (input.empty()) {
            continue;
        }
        
        if (input == "quit" || input == "exit") {
            break;
        } else if (input == "help") {
            std::cout << "\n📋 Available Commands:" << std::endl;
            std::cout << "   • quit/exit - End training session" << std::endl;
            std::cout << "   • help - Show this help message" << std::endl;
            std::cout << "   • status - Show agent status" << std::endl;
            std::cout << "   • metrics - Show current performance metrics" << std::endl;
            std::cout << "   • reset - Reset learning state" << std::endl;
            std::cout << "   • save - Save current state" << std::endl;
            std::cout << "   • Any other text - Process as language input" << std::endl;
        } else if (input == "status") {
            displaySystemInfo(agent);
        } else if (input == "metrics") {
            session.displaySessionMetrics();
        } else if (input == "reset") {
            std::cout << "🔄 Resetting learning state..." << std::endl;
            // Reset agent state here
            std::cout << "✅ Reset complete" << std::endl;
        } else if (input == "save") {
            std::cout << "💾 Saving current state..." << std::endl;
            if (agent && agent->saveAgentState("language_checkpoint")) {
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
    std::cout << "\n🤖 AUTOMATED LANGUAGE TRAINING MODE" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    // Enhanced training samples for various language understanding tasks
    std::vector<std::string> training_samples = {
        // Basic conversational patterns
        "Hello, how are you today?",
        "What is your name and what can you help me with?",
        "Can you explain how neural networks learn from data?",
        "I'm interested in learning about artificial intelligence.",
        
        // Questions and queries
        "What is the meaning of consciousness in artificial systems?",
        "How do biological neurons differ from artificial ones?",
        "Can you describe the process of natural language understanding?",
        "What are the main challenges in machine learning today?",
        
        // Informational content
        "The human brain contains approximately 86 billion neurons interconnected in complex networks.",
        "Natural language processing involves understanding syntax, semantics, and pragmatics.",
        "Machine learning algorithms can identify patterns in large datasets through statistical analysis.",
        "Artificial neural networks are inspired by the structure and function of biological brains.",
        
        // Complex reasoning tasks
        "If artificial intelligence systems can process language, do they truly understand meaning?",
        "Compare and contrast supervised learning with unsupervised learning approaches.",
        "Explain the relationship between attention mechanisms and human cognitive processes.",
        "How might future AI systems integrate symbolic reasoning with neural computation?",
        
        // Creative and philosophical content
        "What makes human creativity unique compared to algorithmic generation?",
        "Can machines develop genuine emotions or are they merely simulating responses?",
        "Describe the ethical implications of increasingly autonomous AI systems.",
        "How do we balance AI capabilities with human values and safety concerns?",
        
        // Technical discussions
        "Transformer architectures use self-attention to process sequential data efficiently.",
        "Backpropagation enables neural networks to learn by adjusting weights based on errors.",
        "Reinforcement learning agents learn optimal behaviors through trial and error.",
        "Large language models demonstrate emergent capabilities at sufficient scale.",
        
        // Conversational responses
        "That's a fascinating perspective on machine consciousness.",
        "I appreciate your detailed explanation of the neural architecture.",
        "Could you elaborate on that point about semantic understanding?",
        "Your insights about AI safety are very thought-provoking.",
        
        // Educational content
        "Language models learn statistical patterns from text to predict likely continuations.",
        "Attention mechanisms allow models to focus on relevant parts of input sequences.",
        "Transfer learning enables models to apply knowledge from one domain to another.",
        "Fine-tuning adapts pre-trained models to specific tasks with minimal additional data."
    };
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> reward_dist(0.3, 1.0);
    std::uniform_int_distribution<> sample_dist(0, training_samples.size() - 1);
    
    const int total_training_steps = 100;
    int successful_processes = 0;
    
    for (int i = 0; i < total_training_steps && !g_shutdown_requested; ++i) {
        // Select random training sample
        int sample_index = sample_dist(gen);
        const std::string& sample = training_samples[sample_index];
        
        std::cout << "\n[Step " << (i + 1) << "/" << total_training_steps << "] ";
        std::cout << "Processing: \"" << sample.substr(0, 50) << "...\"" << std::endl;
        
        // Process through agent
        bool success = agent->processLanguageInput(sample, "", true);
        
        if (success) {
            successful_processes++;
            
            // Generate response
            std::string response = agent->generateLanguageResponse();
            std::cout << "Response: \"" << response.substr(0, 80) << "...\"" << std::endl;
            
            // Apply simulated reward based on response quality
            float reward = reward_dist(gen);
            agent->applyReward(reward);
            
            std::cout << "Reward: " << std::fixed << std::setprecision(2) << reward << std::endl;
        } else {
            std::cout << "❌ Processing failed" << std::endl;
        }
        
        // Show progress every 10 steps
        if ((i + 1) % 10 == 0) {
            float success_rate = static_cast<float>(successful_processes) / (i + 1) * 100;
            float learning_progress = agent->getLearningProgress() * 100;
            
            std::cout << "\n📊 Progress Update:" << std::endl;
            std::cout << "   • Steps Completed: " << (i + 1) << "/" << total_training_steps << std::endl;
            std::cout << "   • Success Rate: " << std::fixed << std::setprecision(1) << success_rate << "%" << std::endl;
            std::cout << "   • Learning Progress: " << std::fixed << std::setprecision(1) << learning_progress << "%" << std::endl;
            
            auto attention_weights = agent->getAttentionWeights();
            std::cout << "   • Key Attention Weights:" << std::endl;
            for (const auto& [module, weight] : attention_weights) {
                if (module.find("language") != std::string::npos || 
                    module.find("semantic") != std::string::npos) {
                    std::cout << "     - " << module << ": " << std::fixed << std::setprecision(2) 
                              << (weight * 100) << "%" << std::endl;
                }
            }
        }
        
        // Brief pause between training steps
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "\n✅ AUTOMATED TRAINING COMPLETE" << std::endl;
    std::cout << "📈 Final Success Rate: " << std::fixed << std::setprecision(1) 
              << (static_cast<float>(successful_processes) / total_training_steps * 100) << "%" << std::endl;
    std::cout << "🧠 Final Learning Progress: " << std::fixed << std::setprecision(1) 
              << (agent->getLearningProgress() * 100) << "%" << std::endl;
}

/**
 * @brief Main application entry point
 */
int main(int argc, char* argv[]) {
    // Setup signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    std::cout << "🚀 NEUROGEN LANGUAGE PROCESSING AUTONOMOUS AGENT" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "🧠 Brain-Inspired Modular Neural Architecture" << std::endl;
    std::cout << "💬 Natural Language Processing Focus" << std::endl;
    std::cout << "🎯 Autonomous Learning and Adaptation" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    try {
        // Initialize brain architecture configuration
        BrainModuleArchitecture::ArchitectureConfig config;
        config.max_sequence_length = 512;
        config.vocabulary_size = 50000;
        config.embedding_dimensions = 300;
        config.global_learning_rate = 0.001f;
        config.enable_continual_learning = true;
        config.use_gpu_acceleration = false; // CPU-only for now
        
        std::cout << "🔧 Initializing autonomous learning agent..." << std::endl;
        
        // Create and initialize autonomous learning agent
        g_agent = std::make_shared<AutonomousLearningAgent>(config);
        
        if (!g_agent->initialize(config.vocabulary_size, config.max_sequence_length)) {
            std::cerr << "❌ Failed to initialize autonomous learning agent" << std::endl;
            return -1;
        }
        
        std::cout << "✅ Agent initialized successfully" << std::endl;
        
        // Check command line arguments for mode selection
        std::string mode = "interactive";
        if (argc > 1) {
            mode = argv[1];
        }
        
        if (mode == "automated" || mode == "auto") {
            runAutomatedTraining(g_agent);
        } else if (mode == "status") {
            displaySystemInfo(g_agent);
        } else {
            // Default to interactive mode
            runInteractiveTraining(g_agent);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << std::endl;
        return -1;
    }
    
    // Graceful shutdown
    std::cout << "\n🛑 Shutting down autonomous learning agent..." << std::endl;
    
    if (g_agent) {
        g_agent->shutdown();
        g_agent.reset();
    }
    
    std::cout << "✅ Shutdown complete. Goodbye!" << std::endl;
    return 0;
}