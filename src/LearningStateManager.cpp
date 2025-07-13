// ============================================================================
// LEARNING STATE MANAGER IMPLEMENTATION
// File: src/LearningStateManager.cpp
// ============================================================================

LearningStateManager::LearningStateManager(std::shared_ptr<BrainModuleArchitecture> architecture, 
                                           const std::string& base_save_path)
    : architecture_(architecture), base_save_path_(base_save_path) {
    
    // Initialize learning statistics
    learning_stats_["total_steps"] = 0.0f;
    learning_stats_["cumulative_reward"] = 0.0f;
    learning_stats_["average_performance"] = 0.0f;
    learning_stats_["learning_efficiency"] = 0.0f;
    
    last_save_time_ = std::chrono::steady_clock::now();
    
    std::cout << "📊 LearningStateManager created with save path: " << base_save_path_ << std::endl;
}

LearningStateManager::~LearningStateManager() {
    if (initialized_) {
        // Auto-save on destruction
        saveLearningState("auto_save");
    }
}

bool LearningStateManager::initialize() {
    try {
        // Create save directory if it doesn't exist
        std::filesystem::create_directories(base_save_path_);
        
        initialized_ = true;
        std::cout << "✅ LearningStateManager initialized" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to initialize LearningStateManager: " << e.what() << std::endl;
        return false;
    }
}

bool LearningStateManager::saveLearningState(const std::string& checkpoint_name) {
    if (!initialized_) {
        std::cerr << "❌ LearningStateManager not initialized" << std::endl;
        return false;
    }
    
    try {
        std::string checkpoint_path = base_save_path_ + "/" + checkpoint_name + "_learning.state";
        std::ofstream state_file(checkpoint_path);
        
        if (state_file.is_open()) {
            // Save learning statistics
            for (const auto& [stat_name, value] : learning_stats_) {
                state_file << stat_name << ": " << value << std::endl;
            }
            
            // Save timestamp
            auto now = std::chrono::steady_clock::now();
            auto duration = now.time_since_epoch();
            auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
            state_file << "save_timestamp: " << timestamp << std::endl;
            
            state_file.close();
            last_save_time_ = now;
            
            std::cout << "💾 Learning state saved: " << checkpoint_name << std::endl;
            return true;
        } else {
            std::cerr << "❌ Failed to open save file: " << checkpoint_path << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to save learning state: " << e.what() << std::endl;
        return false;
    }
}

bool LearningStateManager::loadLearningState(const std::string& checkpoint_name) {
    if (!initialized_) {
        std::cerr << "❌ LearningStateManager not initialized" << std::endl;
        return false;
    }
    
    try {
        std::string checkpoint_path = base_save_path_ + "/" + checkpoint_name + "_learning.state";
        std::ifstream state_file(checkpoint_path);
        
        if (state_file.is_open()) {
            std::string line;
            while (std::getline(state_file, line)) {
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos) {
                    std::string key = line.substr(0, colon_pos);
                    std::string value_str = line.substr(colon_pos + 1);
                    
                    // Trim whitespace
                    key.erase(0, key.find_first_not_of(" \t"));
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value_str.erase(0, value_str.find_first_not_of(" \t"));
                    value_str.erase(value_str.find_last_not_of(" \t") + 1);
                    
                    // Store learning statistic
                    if (key != "save_timestamp") {
                        learning_stats_[key] = std::stof(value_str);
                    }
                }
            }
            
            state_file.close();
            std::cout << "📂 Learning state loaded: " << checkpoint_name << std::endl;
            return true;
        } else {
            std::cout << "⚠️ Checkpoint not found: " << checkpoint_name << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to load learning state: " << e.what() << std::endl;
        return false;
    }
}

void LearningStateManager::updateLearningStats(float reward, float performance) {
    learning_stats_["total_steps"] += 1.0f;
    learning_stats_["cumulative_reward"] += reward;
    
    // Update average performance with exponential moving average
    float alpha = 0.1f;
    learning_stats_["average_performance"] = 
        (1.0f - alpha) * learning_stats_["average_performance"] + alpha * performance;
    
    // Calculate learning efficiency
    if (learning_stats_["total_steps"] > 0) {
        learning_stats_["learning_efficiency"] = 
            learning_stats_["cumulative_reward"] / learning_stats_["total_steps"];
    }
}

std::map<std::string, float> LearningStateManager::getLearningStats() const {
    return learning_stats_;
}

bool LearningStateManager::checkpointExists(const std::string& checkpoint_name) const {
    std::string checkpoint_path = base_save_path_ + "/" + checkpoint_name + "_learning.state";
    std::ifstream file(checkpoint_path);
    return file.good();
}