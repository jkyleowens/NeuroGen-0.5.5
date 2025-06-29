#include <NeuroGen/TaskAutomationModules.h>
#include <NeuroGen/NetworkBuilder.h> 
#include <stdexcept>

SensoryProcessingModule::SensoryProcessingModule(const std::string& name, const NetworkConfig& config)
    : NeuralModule(name, config) {}

void SensoryProcessingModule::initialize() {
    if (!internal_network_) {
        throw std::runtime_error("Internal network is not initialized for SensoryProcessingModule.");
    }

    NetworkBuilder builder(internal_network_.get());

    auto input_neurons = builder.addNeuronPopulation(NeuronType::EXCITATORY, 512, Position3D(0, 0, 10));
    this->addNeuronPopulation("INPUT", input_neurons);

    auto output_neurons = builder.addNeuronPopulation(NeuronType::EXCITATORY, 256, Position3D(0, 0, 50));
    this->addNeuronPopulation("OUTPUT", output_neurons);

    auto hidden_neurons = builder.addNeuronPopulation(NeuronType::EXCITATORY, 1024, Position3D(0, 0, 30));
    auto inhibitory_neurons = builder.addNeuronPopulation(NeuronType::INHIBITORY, 256, Position3D(0, 0, 40));

    builder.connect(input_neurons, hidden_neurons, 0.2, 50.0);
    builder.connect(hidden_neurons, output_neurons, 0.1, 50.0);
    builder.connect(hidden_neurons, hidden_neurons, 0.05, 30.0);
    builder.connect(hidden_neurons, inhibitory_neurons, 0.1, 40.0);
    builder.connect(inhibitory_neurons, hidden_neurons, -0.1, 40.0); 
}

// Added constructor definition for ActionSelectionModule
ActionSelectionModule::ActionSelectionModule(const std::string& name, const NetworkConfig& config)
    : NeuralModule(name, config) {}

void ActionSelectionModule::initialize() {
    if (!internal_network_) {
        throw std::runtime_error("Internal network is not initialized for ActionSelectionModule.");
    }
    
    NetworkBuilder builder(internal_network_.get());

    auto sensory_input_neurons = builder.addNeuronPopulation(NeuronType::EXCITATORY, 256, Position3D(100, 0, 0));
    this->addNeuronPopulation("SENSORY_INPUT", sensory_input_neurons);

    auto action_output_neurons = builder.addNeuronPopulation(NeuronType::EXCITATORY, 64, Position3D(100, 0, 50));
    this->addNeuronPopulation("ACTION_OUTPUT", action_output_neurons);
}