#include "fieldSystem.h"
#include "kernel.h"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <filesystem>
#include <memory>
#include <limits>
#include <stdexcept>

#include <GL/glew.h>

using namespace std;

FieldSystem::FieldSystem(
    const FieldGridParams& grid,  
    bool bUseOpenGL) : 
    m_bUseOpenGL(bUseOpenGL), 
    m_grid(grid) {

    _initialize();
}


FieldSystem::~FieldSystem() {

    _finalize();
}


void FieldSystem::_initialize() {
    assert(!m_bInitialized);

    m_bInitialized = true;
}


void FieldSystem::_finalize() {

    if (!m_bInitialized) return;


    m_bInitialized = false;
}


bool FieldSystem::allocateFieldBuffers() {

    return true;
}


void FieldSystem::freeFieldBuffers() {

    return;
}


void FieldSystem::clear() {


    uploadFields();
}


bool FieldSystem::uploadFields() {
    if (!m_bInitialized) return false;


    return true;
}


bool FieldSystem::downloadFields() {

    if (!m_bInitialized)
        return false;


    const int scalarBytes =
        static_cast<int>(sizeof(float) * m_grid.cellCount);

    const int vectorBytes =
        static_cast<int>(sizeof(float4) *m_grid.cellCount);


    


    return true;
}


float* FieldSystem::getScalarDevicePtr(ScalarField field) {

    switch (field) {

    case CHARGE_DENSITY:
        return m_buffers.chargeDensity;

    case ELECTRIC_POTENTIAL:
        return m_buffers.electricPotential;

    case TEMPERATURE:
        return m_buffers.temperature;

    case ELECTRON_DENSITY:
        return m_buffers.electronDensity;

    case ELECTRON_TEMPERATURE:
        return m_buffers.electronTemperature;

    default:
        return nullptr;
    }
}


float4* FieldSystem::getVectorDevicePtr(VectorField field) {

    switch (field) {

    case ELECTRIC_FIELD:
        return m_buffers.electricField;

    case MAGNETIC_FIELD:
        return m_buffers.magneticField;

    case CURRENT_DENSITY:
        return m_buffers.currentDensity;

    default:
        return nullptr;
    }
}


bool FieldSystem::validGrid(const FieldGridParams& grid) const {

    if (grid.dimensions.x == 0 ||
        grid.dimensions.y == 0 ||
        grid.dimensions.z == 0)
        return false;


    if (grid.cellSize.x <= 0.0f ||
        grid.cellSize.y <= 0.0f ||
        grid.cellSize.z <= 0.0f)
        return false;


    const unsigned long long expected = 
        static_cast<unsigned long long>(grid.dimensions.x) * 
        grid.dimensions.y * grid.dimensions.z;


    return expected == grid.cellCount;
}