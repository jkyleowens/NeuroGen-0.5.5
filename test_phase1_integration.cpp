#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/NetworkConfig.h"
#include <chrono>
#include <thread>
#include <iostream>

int main() {
    NetworkConfig config;
    AutonomousLearningAgent agent(config);
    if (!agent.initialize()) {
        std::cerr << "Failed to initialize agent" << std::endl;
        return 1;
    }
    agent.startAutonomousLearning();
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
        agent.autonomousLearningStep(0.1f);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    agent.stopAutonomousLearning();
    std::cout << agent.getStatusReport() << std::endl;
    return 0;
}
