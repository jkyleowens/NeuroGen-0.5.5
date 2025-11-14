// ============================================================================
// REIMAGINED BRAIN MODULE ARCHITECTURE IMPLEMENTATION
// File: src/BrainModuleArchitecture.cpp
// ============================================================================

#include "NeuroGen/BrainModuleArchitecture.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
float safeRandomRange(size_t fan_in) {
    if (fan_in == 0) {
        return 1.0f;
    }
    return 1.0f / std::sqrt(static_cast<float>(fan_in));
}
} // namespace

// ============================================================================
// CorticalColumnLayer
// ============================================================================

std::vector<float> BrainModuleArchitecture::CorticalColumnLayer::forward(
    const std::vector<float>& input) const {
    if (input.size() != input_size) {
        throw std::runtime_error("CorticalColumnLayer received mismatched input size");
    }

    std::vector<float> output(output_size, 0.0f);
    for (size_t row = 0; row < output_size; ++row) {
        float sum = biases[row];
        const size_t row_offset = row * input_size;
        for (size_t col = 0; col < input_size; ++col) {
            sum += weights[row_offset + col] * input[col];
        }
        output[row] = std::tanh(sum * activation_gain);
    }
    return output;
}

// ============================================================================
// BrainModule
// ============================================================================

std::vector<float> BrainModuleArchitecture::BrainModule::process(
    const std::vector<float>& input) {
    if (cortical_layers.empty()) {
        return {};
    }

    std::vector<float> activation = input;
    for (const auto& layer : cortical_layers) {
        activation = layer.forward(activation);
    }

    last_output = activation;
    return last_output;
}

// ============================================================================
// BrainModuleArchitecture core
// ============================================================================

BrainModuleArchitecture::BrainModuleArchitecture()
    : random_engine_(std::random_device{}()) {}

BrainModuleArchitecture::~BrainModuleArchitecture() {
    shutdown();
}

bool BrainModuleArchitecture::initializeForNLP() {
    return initialize();
}

bool BrainModuleArchitecture::initialize(int input_width, int input_height) {
    static_cast<void>(input_width);
    static_cast<void>(input_height);

    shutdown();
    initialized_ = true;

    // Seed the architecture with a simple module that can be re-used as a
    // template for experiments.  Additional modules can be created later.
    createBrainModule("seed_module", 64, 32, 3, 48);

    std::cout << "🧠 BrainModuleArchitecture initialized with cortical column modules" << std::endl;
    return true;
}

void BrainModuleArchitecture::shutdown() {
    modules_.clear();
    connections_.clear();
    initialized_ = false;
}

BrainModuleArchitecture::BrainModule& BrainModuleArchitecture::createBrainModule(
    const std::string& name,
    size_t input_size,
    size_t output_size,
    size_t column_count,
    size_t column_width) {
    if (name.empty()) {
        throw std::invalid_argument("Module name cannot be empty");
    }
    if (column_count == 0) {
        column_count = 1;
    }

    auto module = std::make_shared<BrainModule>();
    module->name = name;
    module->input_size = input_size;
    module->output_size = output_size;

    size_t prev_size = input_size;
    for (size_t layer_index = 0; layer_index < column_count; ++layer_index) {
        size_t layer_output = (layer_index == column_count - 1) ? output_size : column_width;
        module->cortical_layers.push_back(createLayer(prev_size, layer_output));
        prev_size = layer_output;
    }

    modules_[name] = module;
    return *module;
}

bool BrainModuleArchitecture::hasModule(const std::string& name) const {
    return modules_.count(name) > 0;
}

std::shared_ptr<BrainModuleArchitecture::BrainModule> BrainModuleArchitecture::getModule(
    const std::string& name) const {
    auto it = modules_.find(name);
    if (it != modules_.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<float> BrainModuleArchitecture::stimulateModule(
    const std::string& module_name,
    const std::vector<float>& input) {
    auto module = getModule(module_name);
    if (!module) {
        throw std::runtime_error("Requested module does not exist: " + module_name);
    }
    if (input.size() != module->input_size) {
        throw std::runtime_error("Input size does not match module expectations");
    }
    return module->process(input);
}

std::vector<float> BrainModuleArchitecture::getLastModuleOutput(
    const std::string& module_name) const {
    auto module = getModule(module_name);
    if (!module) {
        return {};
    }
    return module->last_output;
}

std::vector<float> BrainModuleArchitecture::processVisualInput(
    const std::vector<float>& visual_input) {
    auto existing = getModule("visual_cortex");
    if (!existing || existing->input_size != visual_input.size()) {
        size_t reduced = std::max<size_t>(visual_input.size() / 4, 64);
        createBrainModule("visual_cortex", visual_input.size(), reduced, 3, reduced);
    }
    return stimulateModule("visual_cortex", visual_input);
}

std::vector<std::string> BrainModuleArchitecture::getModuleNames() const {
    std::vector<std::string> names;
    names.reserve(modules_.size());
    for (const auto& [name, _] : modules_) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

size_t BrainModuleArchitecture::getModuleCount() const {
    return modules_.size();
}

bool BrainModuleArchitecture::createConnection(const std::string& source_module,
                                               const std::string& target_module,
                                               float strength) {
    if (!hasModule(source_module) || !hasModule(target_module)) {
        return false;
    }
    ModuleConnection connection{source_module, target_module, strength};
    connections_.push_back(connection);
    return true;
}

std::vector<BrainModuleArchitecture::ModuleConnection> BrainModuleArchitecture::getConnections() const {
    return connections_;
}

BrainModuleArchitecture::CorticalColumnLayer BrainModuleArchitecture::createLayer(
    size_t input_size,
    size_t output_size) {
    CorticalColumnLayer layer;
    layer.input_size = input_size;
    layer.output_size = output_size;
    layer.weights.resize(input_size * output_size);
    layer.biases.resize(output_size);

    std::uniform_real_distribution<float> dist(-safeRandomRange(input_size), safeRandomRange(input_size));
    for (auto& weight : layer.weights) {
        weight = dist(random_engine_);
    }
    for (auto& bias : layer.biases) {
        bias = dist(random_engine_);
    }

    return layer;
}
