// ============================================================================
// REIMAGINED BRAIN MODULE ARCHITECTURE
// File: include/NeuroGen/BrainModuleArchitecture.h
// ============================================================================

#ifndef BRAIN_MODULE_ARCHITECTURE_H
#define BRAIN_MODULE_ARCHITECTURE_H

#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief BrainModuleArchitecture now models each module as a standalone
 * cortical column style neural network.  The architecture is intentionally
 * simple – it focuses on the internal structure of a single module so that
 * additional modules can later be connected sparsely to form a larger brain.
 */
class BrainModuleArchitecture : public std::enable_shared_from_this<BrainModuleArchitecture> {
public:
    struct CorticalColumnLayer {
        size_t input_size = 0;
        size_t output_size = 0;
        std::vector<float> weights;
        std::vector<float> biases;
        float activation_gain = 1.0f;

        std::vector<float> forward(const std::vector<float>& input) const;
    };

    struct BrainModule {
        std::string name;
        size_t input_size = 0;
        size_t output_size = 0;
        std::vector<CorticalColumnLayer> cortical_layers;
        std::vector<float> last_output;

        std::vector<float> process(const std::vector<float>& input);
    };

    struct ModuleConnection {
        std::string source_module;
        std::string target_module;
        float strength = 0.0f;
    };

    BrainModuleArchitecture();
    ~BrainModuleArchitecture();

    bool initializeForNLP();
    bool initialize(int input_width = 0, int input_height = 0);
    void shutdown();

    BrainModule& createBrainModule(const std::string& name,
                                   size_t input_size,
                                   size_t output_size,
                                   size_t column_count,
                                   size_t column_width);

    bool hasModule(const std::string& name) const;
    std::shared_ptr<BrainModule> getModule(const std::string& name) const;

    std::vector<float> stimulateModule(const std::string& module_name,
                                       const std::vector<float>& input);

    std::vector<float> getLastModuleOutput(const std::string& module_name) const;

    std::vector<float> processVisualInput(const std::vector<float>& visual_input);

    std::vector<std::string> getModuleNames() const;
    size_t getModuleCount() const;

    bool createConnection(const std::string& source_module,
                          const std::string& target_module,
                          float strength);
    std::vector<ModuleConnection> getConnections() const;

private:
    CorticalColumnLayer createLayer(size_t input_size, size_t output_size);

    std::unordered_map<std::string, std::shared_ptr<BrainModule>> modules_;
    std::vector<ModuleConnection> connections_;

    bool initialized_ = false;
    std::mt19937 random_engine_;
};

#endif // BRAIN_MODULE_ARCHITECTURE_H
