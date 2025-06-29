#ifndef NEURON_MODEL_CONSTANTS_H
#define NEURON_MODEL_CONSTANTS_H

#include <cuda_runtime.h>

#ifdef __CUDACC__
    #define CONST_HOST_DEVICE __host__ __device__
#else
    #define CONST_HOST_DEVICE
#endif

namespace NeuronModelConstants {

    // --- THIS IS THE FIX ---
    // Using 'inline' prevents multiple definition errors when this header is included
    // in multiple source files.
    inline constexpr float SPIKE_THRESHOLD = 30.0f;  // mV

    // Membrane Properties
    inline constexpr float RESTING_POTENTIAL = -65.0f; // mV
    inline constexpr float RESET_POTENTIAL = -65.0f;   // mV
    inline constexpr float MEMBRANE_TIME_CONSTANT = 10.0f; // ms
    inline constexpr float MEMBRANE_RESISTANCE = 10.0f; // MOhms

    // Refractory Period
    inline constexpr float ABSOLUTE_REFRACTORY_PERIOD = 2.0f; // ms

    // Synaptic Properties
    inline constexpr float SYNAPTIC_TAU_1 = 0.5f;  // ms (rise time)
    inline constexpr float SYNAPTIC_TAU_2 = 2.0f;  // ms (decay time)

    // STDP Learning Rule Parameters
    inline constexpr float A_PLUS = 0.01f;     // LTP rate
    inline constexpr float A_MINUS = 0.012f;   // LTD rate (A- > A+ for stability)
    inline constexpr float TAU_PLUS = 20.0f;   // ms (LTP time window)
    inline constexpr float TAU_MINUS = 20.0f;  // ms (LTD time window)
    inline constexpr float MAX_WEIGHT = 1.0f;
    inline constexpr float MIN_WEIGHT = 0.01f;

    // Homeostatic Plasticity
    inline constexpr float TARGET_FIRING_RATE = 5.0f; // Hz
    inline constexpr float HOMEOSTATIC_TIMESCALE = 1000.0f; // ms

} // namespace NeuronModelConstants

#endif // NEURON_MODEL_CONSTANTS_H