#ifndef __VOLUMESYSTEM_H__
#define __VOLUMESYSTEM_H__

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <vector_types.h>


struct cudaGraphicsResource;


struct VolumeGridParams {

    uint3 dimensions;

    float3 origin;
    float3 cellSize;

    uint32_t cellCount;
};


struct VolumeBuffers {

    float* density = nullptr;
    float* temperature = nullptr;

    float* electronDensity = nullptr;

    float* opacity = nullptr;
    float* emissivity = nullptr;

    float* phase = nullptr;
};


class VolumeSystem {
public:

    enum VolumeChannel {

        DENSITY = 0,
        TEMPERATURE,
        ELECTRON_DENSITY,

        OPACITY,
        EMISSIVITY,

        PHASE,

        _NUM_VOLUME_CHANNELS
    };


    VolumeSystem(
        const VolumeGridParams& grid,
        bool bUseOpenGL
    );

    ~VolumeSystem();


    /// --------------------------------------------------------
    /// GRID
    /// --------------------------------------------------------

    const VolumeGridParams& getGrid() const {
        return m_grid;
    }

    uint3 getGridSize() const {
        return m_grid.dimensions;
    }

    uint32_t getCellCount() const {
        return m_grid.cellCount;
    }


    /// --------------------------------------------------------
    /// VOLUME CHANNELS
    /// --------------------------------------------------------

    float* getDevicePtr(
        VolumeChannel channel
    );

    const VolumeBuffers& getBuffers() const {
        return m_buffers;
    }


    const std::vector<float>& getHostChannel(
        VolumeChannel channel
    ) const;


    bool uploadVolumes();
    bool downloadVolumes();

    void clear();


    /// --------------------------------------------------------
    /// FUTURE COUPLING
    /// --------------------------------------------------------

    // Later:
    //
    // void importDensityFromFieldSystem(...);
    // void importTemperatureFromFieldSystem(...);
    //
    // void sampleToParticles(...);
    //
    // void buildOpacityTransferFunction();
    // void updateVolumeTexture();
    //
    // void render(...);


protected:

    VolumeSystem() = default;

    bool _initialize();
    void _finalize();

    bool allocateVolumeBuffers();
    void freeVolumeBuffers();

    bool createVolumeResources();
    void destroyVolumeResources();

    bool validGrid(
        const VolumeGridParams& grid
    ) const;


protected:

    bool m_bInitialized = false;
    bool m_bUseOpenGL = false;


    VolumeGridParams m_grid{};


    /// --------------------------------------------------------
    /// CPU REFERENCE STORAGE
    /// --------------------------------------------------------

    std::vector<float> m_hDensity;
    std::vector<float> m_hTemperature;

    std::vector<float> m_hElectronDensity;

    std::vector<float> m_hOpacity;
    std::vector<float> m_hEmissivity;

    std::vector<float> m_hPhase;


    /// --------------------------------------------------------
    /// GPU STORAGE
    /// --------------------------------------------------------

    VolumeBuffers m_buffers{};


    /// --------------------------------------------------------
    /// FUTURE VOLUME RENDERING
    /// --------------------------------------------------------

    unsigned int m_volumeTexture = 0;
    unsigned int m_transferTexture = 0;

    cudaGraphicsResource* m_cudaVolumeResource;

};

#endif