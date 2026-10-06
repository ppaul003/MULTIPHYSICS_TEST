#ifndef NDMSM_RENDERER_EM_EUCLID_H
#define NDMSM_RENDERER_EM_EUCLID_H

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#include <GL/glew.h>
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>

#include <vector_types.h>

#include "particleSystem.h"

class EuclidRenderer {
public:
    enum DisplayMode {
        PARTICLE_POINTS,
        PARTICLE_SPHERES,
        PARTICLE_NUM_MODES
    };

    enum GridMode {
        GRID_3D = 0,
        GRID_2D
    };

	enum GridPlane {
		PLANE_XY = 0,
		PLANE_XZ,
		PLANE_YZ
	};

	enum VolumeAxisGuideMode {
		VOLUME_AXIS_GUIDE_DEFAULT = 0,

		// Scale-edit guides.
		VOLUME_AXIS_GUIDE_X,
		VOLUME_AXIS_GUIDE_Y,
		VOLUME_AXIS_GUIDE_Z,

		// Rotation-edit guides.
		VOLUME_AXIS_GUIDE_PITCH,
		VOLUME_AXIS_GUIDE_YAW,
		VOLUME_AXIS_GUIDE_ROLL
	};

	enum VolumeOffsetAxis {
		VOLUME_OFFSET_AXIS_X = 0,
		VOLUME_OFFSET_AXIS_Y,
		VOLUME_OFFSET_AXIS_Z
	};

	struct UniformGrid {

		glm::ivec3 dimensions{
			kGridDim,
			kGridDim,
			kGridDim
		};

		glm::vec3 origin{
			-kSimHalfBox,
			-kSimHalfBox,
			-kSimHalfBox
		};

		glm::vec3 cellSize{
			kCellSize,
			kCellSize,
			kCellSize
		};

		int majorEvery = kMajorEvery;
	};

	struct GridDisplay {
		bool boundary = true;
		bool majorGrid = true;
		bool minorGrid = false;
		bool axes = true;
	};

public:
    EuclidRenderer();
    ~EuclidRenderer();

    void setWindowSize(int w, int h) {
        m_windowW = w;
        m_windowH = h;
        m_window_h = h;
    }
    void setFOV(float fov) { m_fov = fov; }

	void setSimBoxSize(int x) { m_simBox = x; }
	int getSimBoxSize() const { return m_simBox; }

	void setGridDimSize(int x) { m_gridDimSize = x; }
	int getGridDimSize() const { return m_gridDimSize; }

	void setGridMajorEvery(int x) { m_gridMajorEvery = x; }
	int getGridMajorEvery() const { return m_gridMajorEvery; }

	void setGridStyle(int majorEvery, bool drawMinor);

	void setWorkspaceGridVisibility(
		bool drawBoundary,
		bool drawMajor,
		bool drawMinor,
		bool drawAxes
	);

	void setGridMode3D();
	void setGridMode2D(GridPlane plane, int sliceOffset);
	void setRadius(float* r, int numParticles);
	void setPointSize(float size) { m_pointSize = size; }
	void setParticleRadius(float r) { m_particleRadius = r; }
	
	void setDisplayMode(DisplayMode mode) { m_displayMode = mode; }
	void setParticleSystem(ParticleSystem* psystem) { m_psystem = psystem; }
	void setPositions(float* pos, int numParticles);
	void setVertexBuffer(unsigned int vbo, int numParticles);
	void setColorBuffer(unsigned int vbo) { m_colorVBO = vbo; }
	void setRadiusBuffer(unsigned int vbo) { m_radVBO = vbo; }

	void setGrid(
		const glm::ivec3& gridDim,
		const glm::vec3& worldOrigin,
		const glm::vec3& cellSize
	);

	DisplayMode getDisplayMode() const { return m_displayMode; }

	// ----------------------------
	// GENERIC DIAGNOSTIC GEOMETRY
	// ----------------------------
	void display(DisplayMode mode = PARTICLE_POINTS);
	void displayGrid();

	void drawGridBoundary(const UniformGrid& grid);
	void drawGridAxes(const UniformGrid& grid);
	void drawAxisGizmo(const glm::vec3& origin, float length) {
		if (length <= 0.0f) return;

		glUseProgram(0);
		glLineWidth(2.0f);
		glBegin(GL_LINES);

		glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
		glVertex3f(origin.x, origin.y, origin.z);
		glVertex3f(origin.x + length, origin.y, origin.z);

		glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
		glVertex3f(origin.x, origin.y, origin.z);
		glVertex3f(origin.x, origin.y + length, origin.z);

		glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
		glVertex3f(origin.x, origin.y, origin.z);
		glVertex3f(origin.x, origin.y, origin.z + length);

		glEnd();
		glLineWidth(1.0f);
	}

    void drawGridPlane(
        const UniformGrid& grid,
        GridPlane plane,
        float planePosition,
        bool drawMinorLines = true);

    void drawUniformGrid(
        const UniformGrid& grid,
        const GridDisplay& display);

    void drawUniformGridZRange(
        const UniformGrid& grid,
        float visibleMinZ,
        float visibleMaxZ,
        const GridDisplay& display);

	void drawWireCube(
		const glm::vec3& center,
		const glm::vec3& halfExtent
	);

	void drawHighlightedVoxel(
		const glm::vec3& center,
		const glm::vec3& halfExtent,
		float lineWidth = 3.0f) {

		const glm::vec3 minimum = center - halfExtent;
		const glm::vec3 maximum = center + halfExtent;

		glUseProgram(0);
		glLineWidth(lineWidth > 0.0f ? lineWidth : 1.0f);
		glColor4f(1.0f, 0.5f, 0.0f, 1.0f);
		glBegin(GL_LINES);

		glVertex3f(minimum.x, minimum.y, minimum.z);
		glVertex3f(maximum.x, minimum.y, minimum.z);
		glVertex3f(minimum.x, maximum.y, minimum.z);
		glVertex3f(maximum.x, maximum.y, minimum.z);
		glVertex3f(minimum.x, minimum.y, maximum.z);
		glVertex3f(maximum.x, minimum.y, maximum.z);
		glVertex3f(minimum.x, maximum.y, maximum.z);
		glVertex3f(maximum.x, maximum.y, maximum.z);

		glVertex3f(minimum.x, minimum.y, minimum.z);
		glVertex3f(minimum.x, maximum.y, minimum.z);
		glVertex3f(maximum.x, minimum.y, minimum.z);
		glVertex3f(maximum.x, maximum.y, minimum.z);
		glVertex3f(minimum.x, minimum.y, maximum.z);
		glVertex3f(minimum.x, maximum.y, maximum.z);
		glVertex3f(maximum.x, minimum.y, maximum.z);
		glVertex3f(maximum.x, maximum.y, maximum.z);

		glVertex3f(minimum.x, minimum.y, minimum.z);
		glVertex3f(minimum.x, minimum.y, maximum.z);
		glVertex3f(maximum.x, minimum.y, minimum.z);
		glVertex3f(maximum.x, minimum.y, maximum.z);
		glVertex3f(minimum.x, maximum.y, minimum.z);
		glVertex3f(minimum.x, maximum.y, maximum.z);
		glVertex3f(maximum.x, maximum.y, minimum.z);
		glVertex3f(maximum.x, maximum.y, maximum.z);

		glEnd();
		glLineWidth(1.0f);
	}

    struct DiagnosticParticleVisual {
        glm::vec3 position{0.0f};
        float radius = 0.015f;
        glm::vec4 color{1.0f};
        float emissiveIntensity = 0.0f;
    };
    void displayDiagnosticParticles(const std::vector<DiagnosticParticleVisual>& particles);

	void displayParticleRange(
		int start, 
		int count, 
		bool emissive,
		float emissiveIntensity = 1.0f
	);

private:
	void _initGL();
	void _initialize();
	void _drawPoints(bool useColorBuffer = true);

	void _drawPointsRange(
		int start, 
		int count, 
		bool useColorBuffer = true
	);

	GLuint _compileProgram(
		const char* vsource,
		const char* fsource,
		const char* secondaryAttributeName,
		const char* programLabel
	);

	void drawAxes();
	void drawWorkspaceBoundary();
	void drawWorkspaceMajorGrid();

	void drawXYWorkplane(
		int slice,
		bool hoverValid,
		float hoverX,
		float hoverY
	);


    void drawGridPlaneLines(
        const UniformGrid& grid,
        GridPlane plane,
        float planePosition,
        bool drawMinorLines);

	void drawParticleSphere(
		GLuint program,
		const float pos[4],
		float radius,
		const float color[4],
		bool emissiveBlend
	);

	bool checkShader(GLuint shader, const char* label);
	

private:
	static constexpr float kSimBoxSize = 4.0f;
	static constexpr float kSimHalfBox = kSimBoxSize * 0.5f;
	static constexpr int kMajorEvery = 8;
	static constexpr int kGridDim = 16;
	static constexpr float kCellSize =
		kSimBoxSize / static_cast<float>(kGridDim);

	bool m_bInitialized;
	bool m_gridEnabled = true;
	bool m_drawBoundaryGrid = true;
	bool m_drawMajorGrid = true;
	bool m_drawMinorGrid = false;
	bool m_drawAxes = true;

	GridMode m_gridMode = GRID_3D;
	GridPlane m_workPlane = PLANE_XY;
	DisplayMode m_displayMode = PARTICLE_SPHERES;

	int m_simBox = 4;
	int m_gridDimSize = 64;
	int m_gridMajorEvery = 8;

    float m_fov = 60.0f;
    int m_windowW = 1920;
    int m_windowH = 1080;
	int m_window_h = 1080;

	int m_sliceOffset = 0;

	int m_radCapacity;
	int m_numParticles;

	float m_pointSize;
	float m_particleRadius;
	bool m_particleHighlighted = false;
	float m_particleHighlightScale = 1.35f;

	float* m_rad;
	float* m_pos;

	GLuint m_program0; // sphere shader
	GLuint m_program1; // sphere light
	GLuint m_meshProgram = 0;

	GLint m_meshColorLocation = -1;
	GLint m_meshLightDirLocation = -1;
	GLint m_meshAmbientLocation = -1;
	GLint m_meshTexcoordAttributeLocation = -1;
	GLint m_meshSamplerLocation = -1;
	GLint m_meshUseTextureLocation = -1;
	GLint m_meshAlphaMaskLocation = -1;
	GLint m_meshAlphaCutoffLocation = -1;
	GLint m_meshEmissiveSamplerLocation = -1;
	GLint m_meshUseEmissiveLocation = -1;
	GLint m_meshEmissiveFactorLocation = -1;
	GLint m_meshEmissiveIntensityLocation = -1;

	GLuint m_vbo;
	GLuint m_radVBO;
	GLuint m_colorVBO;

	GLuint m_pbo = 0;
	GLuint m_tex = 0;

	GLuint m_particlePosVBO = 0;
	GLuint m_particleRadVBO = 0;
	GLuint m_particleColorVBO = 0;

	glm::ivec3 m_gridDim{ 0, 0, 0 };
	glm::vec3 m_gridOrigin{ 0, 0, 0 };
	glm::vec3 m_cellSize{ 1, 1, 1 };

	glm::mat4 m_root = glm::mat4(1.0f);

	ParticleSystem* m_psystem = nullptr;
};

#endif
