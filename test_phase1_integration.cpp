#include "NeuroGen/AutonomousLearningAgent.h"
#include "NeuroGen/NetworkConfig.h"
#include "NeuroGen/BioVisualProcessor.h"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include <chrono>
#include <thread>
#include <iostream>

int main() {
    NetworkConfig config;
    config.num_neurons = 32;
    BioVisualProcessor processor("test_processor", config, 32);
    processor.initialize();
#ifdef USE_OPENCV
    cv::Mat dummy = cv::Mat::zeros(8, 4, CV_8UC1);
    auto feats = processor.processPixels(dummy);
#else
    std::vector<float> dummy(32, 0.0f);
    auto feats = processor.processPixels(dummy);
#endif
    std::cout << "Processor output size: " << feats.size() << std::endl;
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
    agent.saveAgentState("agent_state");
    agent.loadAgentState("agent_state");
    std::cout << agent.getStatusReport() << std::endl;
    return 0;
}
