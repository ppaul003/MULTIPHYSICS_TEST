#ifndef __FIELDSYSTEM_H__
#define __FIELDSYSTEM_H__

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <vector_types.h>

#include "paramsEM_kernel.cuh"
#include "particleFieldType.h"

// Forward declaration for CUDA/OpenGL interop.
struct cudaGraphicsResource;

class FieldSystem {
public:

    enum ScalarField {
        CHARGE_DENSITY = 0,
        ELECTRIC_POTENTIAL,
        TEMPERATURE,

        // Future plasma channels
        ELECTRON_DENSITY,
        ELECTRON_TEMPERATURE,

        _NUM_SCALAR_FIELDS
    };

    enum VectorField {
        ELECTRIC_FIELD = 0,
        MAGNETIC_FIELD,
        CURRENT_DENSITY,

        _NUM_VECTOR_FIELDS
    };

    FieldSystem(const FieldGridParams &grid, bool bUseOpenGL);

    ~FieldSystem();


    /// --------------------------------------------------------
    /// FIELD GRID
    /// --------------------------------------------------------

    const FieldGridParams& getGrid() const {
        return m_grid;
    }

    uint3 getGridSize() const {
        return m_grid.dimensions;
    }

    uint32_t getCellCount() const {
        return m_grid.cellCount;
    }

    float3 getWorldOrigin() const {
        return m_grid.origin;
    }

    float3 getCellSize() const {
        return m_grid.cellSize;
    }

    bool setFieldGrid(
        const FieldGridParams& grid
    );


    /// --------------------------------------------------------
    /// FIELD SOLVER CONFIGURATION
    /// --------------------------------------------------------

    void setSolverParams(
        const FieldSolverParams& params
    );

    const FieldSolverParams& getSolverParams() const {
        return m_solverParams;
    }

    void setWaveParams(
        const AnalyticWaveParams& wave
    );

    const AnalyticWaveParams& getWaveParams() const {
        return m_waveParams;
    }


    /// --------------------------------------------------------
    /// FIELD ACCESS
    /// --------------------------------------------------------

    float* getScalarDevicePtr(
        ScalarField field
    );

    float4* getVectorDevicePtr(
        VectorField field
    );

    const FieldBuffers& getBuffers() const {
        return m_buffers;
    }


    /// --------------------------------------------------------
    /// CPU REFERENCE / DEBUG DATA
    /// --------------------------------------------------------

    const std::vector<float>& getScalarHost(
        ScalarField field
    ) const;

    const std::vector<float4>& getVectorHost(
        VectorField field
    ) const;

    bool downloadFields();
    bool uploadFields();


    /// --------------------------------------------------------
    /// FIELD OPERATIONS
    /// --------------------------------------------------------

    void clear();

    // Later:
    //
    // void depositParticles(...);
    // void solveElectrostatics();
    // void computeElectricField();
    // void computeCurrentDensity();
    // void applyAnalyticWave(double time);
    // void stepMaxwell(float dt);
    // void computeTemperature(...);
    //
    // float sampleScalar(...);
    // float3 sampleVector(...);


    /// --------------------------------------------------------
    /// VISUALIZATION
    /// --------------------------------------------------------

    unsigned int getGlyphBuffer() const {return m_glyphVbo;}

    unsigned int getGlyphVertexCount() const {return m_glyphVertexCount;}


protected:

    FieldSystem() = default;

    void _initialize();
    void _finalize();

    bool allocateFieldBuffers();
    void freeFieldBuffers();

    bool createGlyphBuffer();
    void destroyGlyphBuffer();

    bool validGrid(
        const FieldGridParams& grid
    ) const;


protected:

    bool m_bInitialized = false;
    bool m_bUseOpenGL = false;

    FieldGridParams m_grid{};
    FieldSolverParams m_solverParams{};
    AnalyticWaveParams m_waveParams{};
    FieldGridParams fieldGrid{};
    /// --------------------------------------------------------
    /// CPU REFERENCE STORAGE
    /// --------------------------------------------------------

    std::vector<float> m_hChargeDensity;
    std::vector<float> m_hElectricPotential;
    std::vector<float> m_hTemperature;

    std::vector<float> m_hElectronDensity;
    std::vector<float> m_hElectronTemperature;

    std::vector<float4> m_hElectricField;
    std::vector<float4> m_hMagneticField;
    std::vector<float4> m_hCurrentDensity;


    /// --------------------------------------------------------
    /// GPU FIELD STORAGE
    /// --------------------------------------------------------

    FieldBuffers m_buffers{};


    /// --------------------------------------------------------
    /// CUDA / OPENGL FIELD VISUALIZATION
    /// --------------------------------------------------------

    unsigned int m_glyphVbo = 0;
    unsigned int m_glyphVertexCapacity = 0;
    unsigned int m_glyphVertexCount = 0;

    cudaGraphicsResource* m_cudaGlyphResource;
    
    FieldGridParams m_fgParams;
    FieldSolverParams m_fsParams;
};

#endif