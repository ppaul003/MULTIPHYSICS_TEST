#include <GL/glew.h>
#include "fieldSystem.h"
#include "kernel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

FieldSystem::FieldSystem(const FieldGridParams& grid, bool useOpenGL)
    : m_bUseOpenGL(useOpenGL), m_grid(grid) {
    m_solverParams.boundaryMode = FieldBoundaryMode::GroundedDirichlet;
    m_solverParams.poissonIterations = 128;
    m_solverParams.coulombSofteningM = 0.02f;
    _initialize();
}

FieldSystem::~FieldSystem() { _finalize(); }

bool FieldSystem::result(bool success, const char* operation) {
    m_error = success ? "" : operation;
    return success;
}

bool FieldSystem::validGrid(const FieldGridParams& grid) {
    return validFieldGrid(grid);
}

void FieldSystem::_initialize() {
    if (!validGrid(m_grid)) { m_error = "Invalid field grid"; return; }
    try {
        for (auto& values : m_scalars) values.assign(m_grid.cellCount, 0.0f);
        for (auto& values : m_vectors) values.assign(m_grid.cellCount, make_float4(0,0,0,0));
        if (!allocateFieldBuffers() || !setFieldGridParameters(&m_grid) ||
            !setFieldSolverParameters(&m_solverParams) ||
            (m_bUseOpenGL && !createGlyphBuffer())) {
            _finalize();
            m_error = "Field allocation or CUDA/OpenGL registration failed";
            return;
        }
        m_bInitialized = true;
        if (!clear()) {
            _finalize();
            m_error = "Initial field clear failed";
        }
    } catch (const std::exception&) {
        _finalize();
        m_error = "Field host allocation failed";
    }
}

void FieldSystem::_finalize() {
    // Also cleans partially initialized objects; never early-return on the flag.
    destroyGlyphBuffer();
    freeFieldBuffers();
    for (auto& values : m_scalars) std::vector<float>().swap(values);
    for (auto& values : m_vectors) std::vector<float4>().swap(values);
    m_bInitialized = false;
}

bool FieldSystem::allocateFieldBuffers() {
    return allocateFieldArrays(m_buffers, m_grid.cellCount);
}
void FieldSystem::freeFieldBuffers() { freeFieldArrays(m_buffers); }

float* FieldSystem::getScalarDevicePtr(ScalarField field) {
    switch (field) {
    case CHARGE_DENSITY: return m_buffers.chargeDensity;
    case ELECTRIC_POTENTIAL: return m_buffers.electricPotential;
    case TEMPERATURE: return m_buffers.temperature;
    case ELECTRON_DENSITY: return m_buffers.electronDensity;
    case ELECTRON_TEMPERATURE: return m_buffers.electronTemperature;
    default: return nullptr;
    }
}
float4* FieldSystem::getVectorDevicePtr(VectorField field) {
    switch (field) {
    case ELECTRIC_FIELD: return m_buffers.electricField;
    case MAGNETIC_FIELD: return m_buffers.magneticField;
    case CURRENT_DENSITY: return m_buffers.currentDensity;
    default: return nullptr;
    }
}
std::vector<float>& FieldSystem::getScalarHost(ScalarField field) {
    if (field < 0 || field >= _NUM_SCALAR_FIELDS) throw std::out_of_range("Scalar field");
    return m_scalars[field];
}
const std::vector<float>& FieldSystem::getScalarHost(ScalarField field) const {
    if (field < 0 || field >= _NUM_SCALAR_FIELDS) throw std::out_of_range("Scalar field");
    return m_scalars[field];
}
std::vector<float4>& FieldSystem::getVectorHost(VectorField field) {
    if (field < 0 || field >= _NUM_VECTOR_FIELDS) throw std::out_of_range("Vector field");
    return m_vectors[field];
}
const std::vector<float4>& FieldSystem::getVectorHost(VectorField field) const {
    if (field < 0 || field >= _NUM_VECTOR_FIELDS) throw std::out_of_range("Vector field");
    return m_vectors[field];
}

bool FieldSystem::uploadFields() {
    if (!m_bInitialized) return result(false, "Fields uninitialized");
    // Validate all mirror sizes before copying anything.
    for (const auto& v : m_scalars) if (v.size() != m_grid.cellCount)
        return result(false, "Scalar mirror size mismatch");
    for (const auto& v : m_vectors) if (v.size() != m_grid.cellCount)
        return result(false, "Vector mirror size mismatch");
    for (int i = 0; i < _NUM_SCALAR_FIELDS; ++i)
        if (!copyFieldToDevice(getScalarDevicePtr(static_cast<ScalarField>(i)),
            m_scalars[i].data(), m_grid.cellCount * sizeof(float)))
            return result(false, "Scalar upload failed");
    for (int i = 0; i < _NUM_VECTOR_FIELDS; ++i)
        if (!copyFieldToDevice(getVectorDevicePtr(static_cast<VectorField>(i)),
            m_vectors[i].data(), m_grid.cellCount * sizeof(float4)))
            return result(false, "Vector upload failed");
    return result(true, "");
}
bool FieldSystem::downloadFields() {
    if (!m_bInitialized) return result(false, "Fields uninitialized");
    // Mirrors may be resized by callers; reject rather than write out of bounds.
    for (const auto& v : m_scalars) if (v.size() != m_grid.cellCount)
        return result(false, "Scalar mirror size mismatch");
    for (const auto& v : m_vectors) if (v.size() != m_grid.cellCount)
        return result(false, "Vector mirror size mismatch");
    for (int i = 0; i < _NUM_SCALAR_FIELDS; ++i)
        if (!copyFieldToHost(m_scalars[i].data(), getScalarDevicePtr(static_cast<ScalarField>(i)),
            m_grid.cellCount * sizeof(float))) return result(false, "Scalar download failed");
    for (int i = 0; i < _NUM_VECTOR_FIELDS; ++i)
        if (!copyFieldToHost(m_vectors[i].data(), getVectorDevicePtr(static_cast<VectorField>(i)),
            m_grid.cellCount * sizeof(float4))) return result(false, "Vector download failed");
    return result(true, "");
}
bool FieldSystem::clear() {
    m_glyphVertexCount = m_scalarVertexCount = 0;
    m_useGlyphColors = false;
    if (!m_bInitialized) return result(false, "Fields uninitialized");
    for (int i = 0; i < _NUM_SCALAR_FIELDS; ++i) {
        std::fill(m_scalars[i].begin(), m_scalars[i].end(), 0.0f);
        if (!clearScalarField(getScalarDevicePtr(static_cast<ScalarField>(i)), m_grid.cellCount))
            return result(false, "Scalar clear failed");
    }
    for (int i = 0; i < _NUM_VECTOR_FIELDS; ++i) {
        std::fill(m_vectors[i].begin(), m_vectors[i].end(), make_float4(0,0,0,0));
        if (!clearVectorField(getVectorDevicePtr(static_cast<VectorField>(i)), m_grid.cellCount))
            return result(false, "Vector clear failed");
    }
    return result(clearScalarField(m_buffers.electricPotentialScratch, m_grid.cellCount),
        "Potential scratch clear failed");
}

void FieldSystem::swapResources(FieldSystem& other) {
    using std::swap;
    swap(m_grid, other.m_grid);
    swap(m_glyphColors, other.m_glyphColors);
    swap(m_useGlyphColors, other.m_useGlyphColors);
    swap(m_buffers, other.m_buffers);
    swap(m_bInitialized, other.m_bInitialized);
    for (int i=0; i<_NUM_SCALAR_FIELDS; ++i) m_scalars[i].swap(other.m_scalars[i]);
    for (int i=0; i<_NUM_VECTOR_FIELDS; ++i) m_vectors[i].swap(other.m_vectors[i]);
    swap(m_glyphVbo, other.m_glyphVbo);
    swap(m_scalarVbo, other.m_scalarVbo);
    swap(m_glyphVertexCount, other.m_glyphVertexCount);
    swap(m_scalarVertexCount, other.m_scalarVertexCount);
    swap(m_cudaGlyphResource, other.m_cudaGlyphResource);
    swap(m_cudaScalarResource, other.m_cudaScalarResource);
}
bool FieldSystem::setFieldGrid(const FieldGridParams& grid) {
    if (!validGrid(grid)) return result(false, "Invalid field grid");
    // Construct first: invalid/OOM replacement leaves the live resources intact.
    FieldSystem replacement(grid, m_bUseOpenGL);
    if (!replacement.initialized()) return result(false, "Replacement field allocation failed");
    if (!replacement.setSolverParams(m_solverParams)) return result(false, "Solver parameters rejected");
    swapResources(replacement);
    return result(true, "");
}
bool FieldSystem::setSolverParams(const FieldSolverParams& params) {
    if (!setFieldSolverParameters(&params)) return result(false, "Unsupported/invalid field solver");
    m_solverParams = params;
    return result(true, "");
}
bool FieldSystem::setWaveParams(const AnalyticWaveParams& wave) {
    auto normalized=wave;
    if (!normalizeFieldWave(normalized)) return result(false, "Invalid/transversely degenerate wave");
    m_waveParams=normalized;
    return result(true, "");
}
bool FieldSystem::waveSpatiallyResolved() const {
    if (!m_waveParams.enabled || m_waveParams.frequencyHz == 0) return true;
    const float h = (std::max)(m_grid.cellSize.x, (std::max)(m_grid.cellSize.y, m_grid.cellSize.z));
    return 299792458.0 / m_waveParams.frequencyHz >= 10.0 * h;
}

bool FieldSystem::createGlyphBuffer() {
    if (!glGenBuffers || !glGetString(GL_VERSION)) return false;
    GLint previous = 0;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &previous);
    auto create = [&](unsigned& vbo, cudaGraphicsResource*& resource, size_t vertices) {
        glGenBuffers(1, &vbo);
        if (!vbo) return false;
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices * sizeof(FieldGlyphVertex)),
            nullptr, GL_DYNAMIC_DRAW);
        GLint bytes = 0;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bytes);
        return bytes == static_cast<GLint>(vertices * sizeof(FieldGlyphVertex)) &&
            registerFieldBuffer(vbo, &resource);
    };
    const bool ok = create(m_glyphVbo, m_cudaGlyphResource, size_t(m_grid.cellCount) * 10) &&
        create(m_scalarVbo, m_cudaScalarResource, m_grid.cellCount) &&
        allocateFieldColors(&m_glyphColors,m_grid.cellCount);
    glBindBuffer(GL_ARRAY_BUFFER, previous);
    if (!ok) destroyGlyphBuffer();
    return ok;
}
void FieldSystem::destroyGlyphBuffer() {
    if (m_glyphColors) freeArray(m_glyphColors);
    m_glyphColors=nullptr;
    m_useGlyphColors=false;
    if (m_cudaGlyphResource) unregisterFieldBuffer(m_cudaGlyphResource);
    if (m_cudaScalarResource) unregisterFieldBuffer(m_cudaScalarResource);
    m_cudaGlyphResource = m_cudaScalarResource = nullptr;
    if (m_glyphVbo) glDeleteBuffers(1, &m_glyphVbo);
    if (m_scalarVbo) glDeleteBuffers(1, &m_scalarVbo);
    m_glyphVbo = m_scalarVbo = m_glyphVertexCount = m_scalarVertexCount = 0;
}



bool FieldSystem::solveElectrostatics() {
    return result(m_bInitialized && solvePoissonJacobi(m_buffers.chargeDensity,
        m_buffers.electricPotential, m_buffers.electricPotentialScratch, m_grid, m_solverParams) &&
        computeFieldElectric(m_buffers.electricPotential, m_buffers.electricField, m_grid),
        "Grounded Poisson/E solve failed");
}
bool FieldSystem::computeElectricField() {
    return result(m_bInitialized && computeFieldElectric(m_buffers.electricPotential,
        m_buffers.electricField,m_grid), "Potential gradient failed");
}


bool FieldSystem::applyAnalyticWave(double timeSeconds) {
    return result(m_bInitialized && addAnalyticWave(m_buffers.electricField,
        m_buffers.magneticField,m_grid,m_waveParams,timeSeconds), "Analytic wave failed");
}
bool FieldSystem::computeLorentzAcceleration(const float4* positions, const float4* velocities,
    const ParticleFieldMarker* markers, float4* acceleration, unsigned count) {
    return result(m_bInitialized && computeFieldLorentz(positions,velocities,markers,
        acceleration,count,m_buffers.electricField,m_buffers.magneticField,m_grid),
        "Lorentz acceleration failed");
}



bool FieldSystem::setElectricGlyphColors(const std::vector<float4>& colors) {
    m_useGlyphColors=false;
    if (colors.empty()) return true;
    if (!m_bInitialized || !m_glyphColors || colors.size()!=m_grid.cellCount)
        return result(false,"Invalid glyph palette");
    m_useGlyphColors=copyFieldToDevice(m_glyphColors,colors.data(),colors.size()*sizeof(float4));
    return result(m_useGlyphColors,"Glyph palette upload failed");
}
bool FieldSystem::buildRenderBuffers(ScalarField scalar, const FieldRenderParams& settings) {
    m_glyphVertexCount=m_scalarVertexCount=0;
    if (!m_bInitialized || !m_bUseOpenGL || !getScalarDevicePtr(scalar))
        return result(false,"No field render resources");
    if (!buildFieldRenderBuffers(m_buffers.electricField,getScalarDevicePtr(scalar),
        m_grid,settings,m_cudaGlyphResource,m_cudaScalarResource,m_useGlyphColors?m_glyphColors:nullptr))
        return result(false,"Field glyph generation failed");
    m_glyphVertexCount=m_grid.cellCount*10;
    m_scalarVertexCount=m_grid.cellCount;
    return result(true,"");
}
