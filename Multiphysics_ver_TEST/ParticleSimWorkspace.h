#ifndef VITRUGEN_PARTICLE_SIM_WORKSPACE_H
#define VITRUGEN_PARTICLE_SIM_WORKSPACE_H

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "IWorkspaceEM.h"
#include "TextEntry.h"
#include "TheArbiterEM.h"
#include "particleSystem.h"

class ParticleSimWorkspace : public IWorkspace {
public:
	ParticleSimWorkspace();
	~ParticleSimWorkspace() override;

	bool initialize(WorkspaceServices& services) override;

	void enter(WorkspaceServices& services) override;
	void exit(WorkspaceServices& services) override;

	void update(
		const WorkspaceFrameContext& frame,
		WorkspaceServices& services
	) override;

	void render(
		const WorkspaceFrameContext& frame,
		WorkspaceServices& services
	) override;

	bool handleInput(
		const WorkspaceInputEvent& input,
		WorkspaceServices& services
	) override;

	WorkspacePresentation buildPresentation() const override;
	WorkspacePresentation buildLayer1TransitionPresentation() const;
	WorkspaceMenuPresentation buildMenu() const override;
	bool handleMenuCommand(int command, WorkspaceServices& services) override;
	bool handleInputRelease(const WorkspaceInputEvent& input, WorkspaceServices& services) override;
	bool handlePointerInput(const WorkspacePointerEvent& input, WorkspaceServices& services) override;
	void renderOverlay(const WorkspaceFrameContext& frame, WorkspaceServices& services) override;
	void cancelInput(WorkspaceServices& services) override;

private:
	enum class Layer1Row {
		WorkspaceSelection = 0,
		GridLayout,
		ColorMode,
		RadiusMode,
		Configure,
		Count
	};

	enum class Layer3Row {
		DisplaySliders = 0,
		CameraView,
		ShootParticles,
		Count
	};

	enum class Layer3CameraView {
		Orbit = 0,
		Free
	};

	enum class GridLayout {
		None = 0,
		MajorGrid,
		Dynamic,
		Count
	};

	enum class ColorMode {
		Default = 0,
		RGB
	};

	enum class RadiusMode {
		Uniform = 0,
		Random
	};

	enum class ColorChannel {
		Red = 0,
		Green,
		Blue,
		Count
	};

	enum class ResetMode {
		Default = 0,
		Random
	};

	struct DraftConfig {
		GridLayout gridLayout = GridLayout::None;
		ColorMode colorMode = ColorMode::Default;
		RadiusMode radiusMode = RadiusMode::Uniform;

		unsigned int defaultParticleCount = 0;
		unsigned int redCount = 0;
		unsigned int greenCount = 0;
		unsigned int blueCount = 0;
		ColorChannel selectedColorChannel = ColorChannel::Red;

		ResetMode resetMode = ResetMode::Default;
		float uniformRadius = 0.0120f;
		float minimumRadius = 0.0098f;
		float maximumRadius = 0.0156f;
		unsigned int spawnSelectionIndex = 0;
	};

	struct RuntimeConfig {
		unsigned int capacity = 0;
		unsigned int activeCount = 0;
		unsigned int redCount = 0;
		unsigned int greenCount = 0;
		unsigned int blueCount = 0;

		float uniformRadius = 0.0120f;
		float minimumRadius = 0.0098f;
		float maximumRadius = 0.0156f;
		float placementRadius = 0.0120f;

		ColorMode colorMode = ColorMode::Default;
		RadiusMode radiusMode = RadiusMode::Uniform;
		ResetMode resetMode = ResetMode::Default;
		GridLayout gridLayout = GridLayout::None;

		unsigned int selectedSpawnSelectionIndex = 0;
		float selectedSpawnVolumeM3 = 0.0f;
		unsigned int activeMacroParticleCount = 0;
	};

	WorkspacePresentation buildLayer1Presentation() const;
	WorkspacePresentation buildLayer2Presentation() const;
	WorkspacePresentation buildLayer3Presentation() const;
	WorkspaceRuntimeStatus buildParticleRuntimeStatus() const;

	void renderConfiguredGrid(WorkspaceServices& services, GridLayout layout) const;
	void renderSelectedSpawnRegion(WorkspaceServices& services) const;
	void renderActiveParticles(WorkspaceServices& services);

	bool handleLayer1Input(
		const WorkspaceInputEvent& input,
		WorkspaceServices& services
	);
	bool handleLayer2Input(
		const WorkspaceInputEvent& input,
		WorkspaceServices& services
	);
	bool handleLayer3Input(
		const WorkspaceInputEvent& input,
		WorkspaceServices& services
	);
	bool handleTextEntry(const WorkspaceInputEvent& input);

	bool resolveRuntimeConfig(RuntimeConfig& resolved) const;
	bool applyRuntimeConfig();

	void moveLayer1Cursor(int direction);
	void moveLayer2Cursor(int direction);
	void moveLayer3Cursor(int direction);
	void adjustLayer1Value(int direction, WorkspaceServices& services);
	void adjustLayer2Value(int direction);
	void adjustLayer3Value(int direction, WorkspaceServices& services);
	void setLayer3CameraView(Layer3CameraView view, WorkspaceServices& services);
	void toggleDisplaySliders(WorkspaceServices& services);
	bool layer3Active() const;
	bool slidersVisible() const;
	bool setFreeMovementKey(WorkspaceInputAction action, bool pressed);
	void applyLiveParameters();
	void beginParticleAmountEntry();

	int layer2RowCount() const;
	int voxelSpawnRowIndex() const;
	int runSimulationRowIndex() const;
	bool isVoxelSpawnRowSelected() const;

	unsigned int selectedColorCount() const;
	unsigned int otherColorCount() const;
	void setSelectedColorCount(unsigned int value);
	unsigned int requestedParticleCount() const;

	static int radiusPresetIndex(float radius);
	static float radiusPreset(int index);

	static std::string radiusText(float radius);
	static std::string spawnSelectionText(unsigned int selectionIndex);

	const char* gridLayoutName() const;
	const char* colorModeName() const;
	const char* radiusModeName() const;
	const char* colorChannelName() const;
	const char* resetModeName() const;
	const char* layer3CameraViewName() const;

private:
	static constexpr float kSimulationBoxSizeM = 4.0f;
	static constexpr float kMaximumSupportedRadius = 0.0156f;
	static constexpr unsigned int kParticleCapacity = 16384;
	static constexpr unsigned int kGridSize = 64;
	static constexpr unsigned int kMajorGridEvery = 8;
	static constexpr unsigned int kDefaultCountStep = 100;
	static constexpr unsigned int kResetSeed = 1973;

	unsigned int m_capacity = kParticleCapacity;
	unsigned int m_activeCount = 0;
	uint3 m_gridDimensions = make_uint3(kGridSize, kGridSize, kGridSize);

	std::unique_ptr<ParticleSystem> m_particleSystem;
	std::vector<float> m_radii;

	TheArbiter* m_arbiter = nullptr;
	DraftConfig m_draftConfig;
	RuntimeConfig m_runtimeConfig;
	SpatialVoxelGrid3D m_baseVoxelGrid;
	SpawnDensityRegionGrid3D m_spawnDensityGrid;
	TextEntrySession m_textEntry;

	Layer1Row m_layer1Selection = Layer1Row::WorkspaceSelection;
	int m_layer2Selection = 0;
	bool m_subLayerPanelOpen = false;
	Layer3Row m_layer3Selection = Layer3Row::DisplaySliders;
	bool m_displaySliders = false;
	Layer3CameraView m_layer3CameraView = Layer3CameraView::Orbit;
	bool m_freeMovementKeys[4] = { false, false, false, false };
	bool m_freeLookDragging = false;
	int m_sliderDragRow = -1;
	struct RuntimeSliders;
	std::unique_ptr<RuntimeSliders> m_runtimeSliders;

	std::string m_statusLine = "READY: CUDA PARTICLE BASELINE.";
	WorkspaceStatusTone m_statusTone = WorkspaceStatusTone::Ready;

	bool m_initialized = false;
	bool m_active = false;
	bool m_paused = true;
	bool m_runtimeEnabled = false;

	float m_elapsedSimulationTime = 0.0f;
};

#endif
