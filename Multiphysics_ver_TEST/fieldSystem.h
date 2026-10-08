#ifndef __FIELDSYSTEM_H__
#define __FIELDSYSTEM_H__

#include <string>
#include <vector>
#include "paramsEM_kernel.cuh"
#include "particleFieldType.h"

struct cudaGraphicsResource;

// Owns field/render resources. Use on the GL/CUDA context thread; destroy before
// context teardown. CPU mirrors are explicit upload/download snapshots.
class FieldSystem {
public:
    enum ScalarField { 
        CHARGE_DENSITY, 
        ELECTRIC_POTENTIAL, 
        TEMPERATURE,
        ELECTRON_DENSITY, 
        ELECTRON_TEMPERATURE, 
        _NUM_SCALAR_FIELDS 
    };

    enum VectorField { 
        ELECTRIC_FIELD, 
        MAGNETIC_FIELD, 
        CURRENT_DENSITY, 
        _NUM_VECTOR_FIELDS 
    };

    FieldSystem(const FieldGridParams& grid, bool useOpenGL);
    ~FieldSystem();

    FieldSystem(const FieldSystem&) = delete;
    FieldSystem& operator=(const FieldSystem&) = delete;

    bool initialized() const { return m_bInitialized; }
    const std::string& lastError() const { return m_error; }
    static bool validGrid(const FieldGridParams& grid);
    const FieldGridParams& getGrid() const { return m_grid; }
    uint3 getGridSize() const { return m_grid.dimensions; }
    uint32_t getCellCount() const { return m_grid.cellCount; }
    float3 getWorldOrigin() const { return m_grid.origin; }
    float3 getCellSize() const { return m_grid.cellSize; }
    bool setFieldGrid(const FieldGridParams& grid);
    bool setSolverParams(const FieldSolverParams& params);
    const FieldSolverParams& getSolverParams() const { return m_solverParams; }
    bool setWaveParams(const AnalyticWaveParams& wave);
    const AnalyticWaveParams& getWaveParams() const { return m_waveParams; }
    bool waveSpatiallyResolved() const;

    float* getScalarDevicePtr(ScalarField field);
    float4* getVectorDevicePtr(VectorField field);
    const FieldBuffers& getBuffers() const { return m_buffers; }
    std::vector<float>& getScalarHost(ScalarField field);
    const std::vector<float>& getScalarHost(ScalarField field) const;
    std::vector<float4>& getVectorHost(VectorField field);
    const std::vector<float4>& getVectorHost(VectorField field) const;
    bool downloadFields();
    bool uploadFields();
    bool clear();

    // Device operations; download explicitly to inspect results on CPU.
    bool solveElectrostatics();
    bool computeElectricField();
    // Additive source: clear/recompute base E/B before each new frame.
    bool applyAnalyticWave(double timeSeconds);
    bool computeLorentzAcceleration(const float4* positions, const float4* velocities,
        const ParticleFieldMarker* markers, float4* acceleration, unsigned count);

    bool buildRenderBuffers(ScalarField scalar, const FieldRenderParams& settings);
    // Optional diagnostic palette, independent of E (never stored in E.w).
    bool setElectricGlyphColors(const std::vector<float4>& colors);
    unsigned getGlyphBuffer() const { return m_glyphVbo; }
    unsigned getGlyphVertexCount() const { return m_glyphVertexCount; }
    unsigned getScalarBuffer() const { return m_scalarVbo; }
    unsigned getScalarVertexCount() const { return m_scalarVertexCount; }

protected:
    void _initialize();
    void _finalize();
    bool allocateFieldBuffers();
    void freeFieldBuffers();
    bool createGlyphBuffer();
    void destroyGlyphBuffer();
    bool result(bool success, const char* operation);
    void swapResources(FieldSystem& other);

protected:
    bool m_bInitialized = false;
    bool m_bUseOpenGL = false;

    FieldGridParams m_grid{};
    FieldSolverParams m_solverParams{};
    AnalyticWaveParams m_waveParams{};
    FieldBuffers m_buffers{};
    std::vector<float> m_scalars[_NUM_SCALAR_FIELDS];
    std::vector<float4> m_vectors[_NUM_VECTOR_FIELDS];
    unsigned m_glyphVbo = 0, m_scalarVbo = 0;
    unsigned m_glyphVertexCount = 0, m_scalarVertexCount = 0;
    cudaGraphicsResource* m_cudaGlyphResource = nullptr;
    cudaGraphicsResource* m_cudaScalarResource = nullptr;
    float4* m_glyphColors = nullptr;
    bool m_useGlyphColors = false;
    std::string m_error;
};
#endif
