#ifndef MULTIPHYSICS_SIMULATION_PRESET_H
#define MULTIPHYSICS_SIMULATION_PRESET_H

#include <array>

struct SimulationPreset {
    int boxSize;
    int collisionGridDim;
    int majorEvery;
    bool enabled;
};

// Ordered ladder: queue construction and runtime geometry share this mapping.
inline constexpr std::array<SimulationPreset, 4> kSimulationPresets{{

    {4, 64, 8, true}, 
    {8, 64, 8, true},
    {16, 128, 16, true}, 
    {32, 128, 16, true}
}};

inline constexpr int simulationPresetIndex(int boxSize) {

    for (int i = 0; i < static_cast<int>(kSimulationPresets.size()); i++)
        if (kSimulationPresets[i].boxSize == boxSize) 
            return i;
    return -1;
}

inline constexpr const SimulationPreset* findSimulationPreset(int boxSize) {
    const int index = simulationPresetIndex(boxSize);
    return index < 0 ? nullptr : &kSimulationPresets[index];
}

inline constexpr bool supportedSimulationPreset(int boxSize) {
    const auto* preset = findSimulationPreset(boxSize);
    return preset && preset->enabled;
}

#endif
