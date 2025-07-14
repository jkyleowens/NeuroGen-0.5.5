#include "NeuroGen/BrainModuleArchitecture.h"
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>

// Constructor with corrected initializer list order
BrainModuleArchitecture::BrainModuleArchitecture(std::string id, BrainType type)
    : id_(std::move(id)),
      type_(type),
      // FIXED: Initialization order now matches the header declaration order
      creation_time_(std::chrono::steady_clock::now()),
      last_update_time_(std::chrono::steady_clock::now()) {
    std::cout << "BrainModuleArchitecture created with ID: " << id_ << std::endl;
}

BrainModuleArchitecture::~BrainModuleArchitecture() {
    std::cout << "BrainModuleArchitecture destroyed." << std::endl;
}

void BrainModuleArchitecture::initialize() {
    if (type_ == BrainType::LANGUAGE) {
        // Set higher initial neuromodulator levels for language tasks
        global_dopamine_level_ = 0.2f;      // Higher for language reward
        global_acetylcholine_level_ = 0.3f; // Higher for attention in language
        global_norepinephrine_level_ = 0.15f;
        global_serotonin_level_ = 0.1f;

        // FIXED: Calls to previously undeclared functions
        createNLPModules();
        setupNLPConnections();
        initializeNLPAttentionSystem();
    }

    // Initialize CUDA Network
    // FIXED: Use of previously undeclared member 'cuda_network_'
    if (!cuda_network_) {
        // These would be configured from a file or another system
        auto cuda_cfg = std::make_shared<CUDA::Config>();
        cuda_cfg->device_id = 0;
        cuda_network_ = std::make_shared<NetworkCUDA>(cuda_cfg);
        
        auto net_cfg = std::make_shared<Network::Config>();
        // Populate net_cfg...

        auto [success, err] = cuda_network_->initialize(net_cfg);
        if (!success) {
            std::cerr << "CUDA Network initialization failed: " << err << std::endl;
        } else {
            cuda_network_->setBrainArchitecture(shared_from_this());
            cuda_network_->setLearningStateManager(learning_state_manager_);
        }
    }
}

// Out-of-line definition for a previously undeclared function
void BrainModuleArchitecture::createNLPModules() {
    std::cout << "Creating NLP modules..." << std::endl;
    // Dummy implementation
    modules_["Wernicke"] = std::make_shared<int>(1); // Placeholder
    modules_["Broca"] = std::make_shared<int>(2);    // Placeholder
}

// Out-of-line definition for a previously undeclared function
void BrainModuleArchitecture::setupNLPConnections() {
    std::cout << "Setting up NLP connections..." << std::endl;
    // Dummy implementation
}

// Out-of-line definition for a previously undeclared function
void BrainModuleArchitecture::initializeNLPAttentionSystem() {
    std::cout << "Initializing NLP attention system..." << std::endl;
    attention_weights_["Wernicke"] = 0.5f;
    attention_weights_["Broca"] = 0.5f;
    
    for (const auto& pair : attention_weights_) {
        // FIXED: Use push_back to add the float to the vector instead of direct assignment
        attention_history_[pair.first].push_back(pair.second);
    }
}


void BrainModuleArchitecture::update(float dt, float global_reward) {
    // FIXED: Use of previously undeclared member variables and functions
    global_learning_steps_++;
    global_reward_accumulator_ += global_reward;

    updateNeuromodulatorLevels(global_reward, dt);

    // Some logic to update modules...

    // The compiler suggested 'updateGlobalContext' for 'updateGlobalAttention'.
    // This indicates a likely typo. Assuming 'updateGlobalAttention' was intended,
    // and its definition should match the call.
    updateGlobalAttention(dt);

    last_update_time_ = std::chrono::steady_clock::now();
}

BrainState BrainModuleArchitecture::getBrainState() const {
    BrainState state;
    state.architecture_id = id_;
    state.type = type_;
    // FIXED: Use of previously undeclared member variables
    state.total_learning_steps = global_learning_steps_;
    state.cumulative_reward = global_reward_accumulator_;
    state.last_update = last_update_time_;
    return state;
}

// Out-of-line definition for previously undeclared function
void BrainModuleArchitecture::updateNeuromodulatorLevels(float reward, float dt) {
    // Dummy implementation for neuromodulator updates
    global_dopamine_level_ += (reward - 0.1f) * dt;
    global_dopamine_level_ = std::clamp(global_dopamine_level_, 0.0f, 1.0f);
}

// Out-of-line definition for previously undeclared function
void BrainModuleArchitecture::updateGlobalAttention(float dt) {
    for (auto const& [module_name, weight] : attention_weights_) {
        if (!attention_history_.count(module_name) || attention_history_[module_name].empty()) {
             attention_history_[module_name].push_back(weight);
        } else {
            // FIXED: Correctly update the last element in the history vector
            float& last_attention = attention_history_[module_name].back();
            last_attention = 0.9f * last_attention + 0.1f * weight;
        }
    }
}

void BrainModuleArchitecture::saveState(const std::string& path) {
    std::ofstream state_file(path);
    if (state_file.is_open()) {
        // FIXED: Use of previously undeclared member variables
        state_file << "learning_steps: " << global_learning_steps_ << std::endl;
        state_file << "reward_accumulator: " << global_reward_accumulator_ << std::endl;
        // ... save other state ...
    }
}

void BrainModuleArchitecture::loadState(const std::string& path) {
    std::ifstream state_file(path);
    std::string line;
    if (state_file.is_open()) {
        while (getline(state_file, line)) {
            if (line.find("learning_steps:") != std::string::npos) {
                // FIXED: Use of previously undeclared member variable
                global_learning_steps_ = std::stoull(line.substr(line.find(":") + 1));
            } else if (line.find("reward_accumulator:") != std::string::npos) {
                // FIXED: Use of previously undeclared member variable
                global_reward_accumulator_ = std::stod(line.substr(line.find(":") + 1));
            }
        }
    }
}

// Dummy Implementations for other functions to ensure compilation
void BrainModuleArchitecture::updateGlobalContext(const std::vector<float>& new_context) {}
std::vector<InterModuleConnection> BrainModuleArchitecture::getConnections() const { return connections_; }