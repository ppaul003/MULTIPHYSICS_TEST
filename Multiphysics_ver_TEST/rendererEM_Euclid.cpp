#include <GL/glew.h>

#include <stdio.h>
#include <vector>
#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <filesystem>

#include "rendererEM_Euclid.h"
#include "render_utils.h"

using namespace std;
using namespace glm;

static float wrapGuideAngleDeg(float angleDeg) {
    float wrapped = fmodf(angleDeg, 360.0f);

    if (wrapped > 180.0f) wrapped -= 360.0f;
    if (wrapped < -180.0f) wrapped += 360.0f;

    return wrapped;
}

static int getOffsetGridMajorEvery(float normalizedIncrement) {
    if (normalizedIncrement <= 0.0f) return 1;

    const float epsilon = 0.000001f;

    const bool baseTwelveFamily =
        fabsf(normalizedIncrement - 0.012f) < epsilon ||
        fabsf(normalizedIncrement - 0.12f) < epsilon;

    float normalizedMajorStep = 1.0f;

    if (baseTwelveFamily) {

        // 0.012 -> major every 0.12
        // 0.12  -> major every 1.20
        normalizedMajorStep = normalizedIncrement * 10.0f;
    }
    else if (normalizedIncrement < 0.1f) {

        // Decimal-hundredth family:
        //
        // 0.01, 0.02, 0.025, 0.05
        //
        // major lines occur every 0.10.
        normalizedMajorStep = 0.1f;
    }
    else {

        // Decimal-tenth family:
        //
        // 0.1, 0.2, 0.25, 0.5
        //
        // major lines occur every 1.0.
        normalizedMajorStep = 1.0f;
    }

    const int majorEvery =
        static_cast<int>(lroundf(normalizedMajorStep /
            normalizedIncrement));

    return std::max(1, majorEvery);
}

EuclidRenderer::EuclidRenderer() :
    m_bInitialized(false),
    m_rad(0),
    m_pos(0),
    m_radCapacity(0),
    m_numParticles(0),
    m_pointSize(1.0f),
    m_particleRadius(0.125f * 0.5f),
    m_fov(60.0f),
    m_windowW(0),
    m_windowH(0),
    m_program0(0),
    m_program1(0),
    m_vbo(0),
    m_radVBO(0),
    m_colorVBO(0) {

    _initGL();
    _initialize();
}

EuclidRenderer::~EuclidRenderer() {
    m_pos = 0;

    delete[] m_rad;
    m_rad = nullptr;

    if (m_program0) {
        glDeleteProgram(m_program0);
        m_program0 = 0;
    }

    if (m_program1) {
        glDeleteProgram(m_program1);
        m_program1 = 0;
    }

    if (m_particlePosVBO) {
        glDeleteBuffers(
            1,
            &m_particlePosVBO
        );

        m_particlePosVBO = 0;
    }

    if (m_particleRadVBO) {
        glDeleteBuffers(
            1,
            &m_particleRadVBO
        );

        m_particleRadVBO = 0;
    }

    if (m_particleColorVBO) {
        glDeleteBuffers(
            1,
            &m_particleColorVBO
        );

        m_particleColorVBO = 0;
    }
}

void EuclidRenderer::drawGridBoundary(const UniformGrid& grid) {

    const vec3 min = grid.origin;

    const vec3 max = grid.origin +
        vec3(grid.dimensions) * grid.cellSize;

    glUseProgram(0);
    glDisable(GL_TEXTURE_2D);

    glEnable(GL_BLEND);

    glBlendFunc(
        GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA
    );

    glLineWidth(2.0f);
    glColor4f(1.0f, 1.0f, 1.0f, 0.35f);

    glBegin(GL_LINES);

    // Bottom rectangle
    glVertex3f(min.x, min.y, min.z);
    glVertex3f(max.x, min.y, min.z);

    glVertex3f(max.x, min.y, min.z);
    glVertex3f(max.x, min.y, max.z);

    glVertex3f(max.x, min.y, max.z);
    glVertex3f(min.x, min.y, max.z);

    glVertex3f(min.x, min.y, max.z);
    glVertex3f(min.x, min.y, min.z);

    // Top rectangle
    glVertex3f(min.x, max.y, min.z);
    glVertex3f(max.x, max.y, min.z);

    glVertex3f(max.x, max.y, min.z);
    glVertex3f(max.x, max.y, max.z);

    glVertex3f(max.x, max.y, max.z);
    glVertex3f(min.x, max.y, max.z);

    glVertex3f(min.x, max.y, max.z);
    glVertex3f(min.x, max.y, min.z);

    // Verticle edges
    glVertex3f(min.x, min.y, min.z);
    glVertex3f(min.x, max.y, min.z);

    glVertex3f(max.x, min.y, min.z);
    glVertex3f(max.x, max.y, min.z);

    glVertex3f(max.x, min.y, max.z);
    glVertex3f(max.x, max.y, max.z);

    glVertex3f(min.x, min.y, max.z);
    glVertex3f(min.x, max.y, max.z);

    glEnd();
    glLineWidth(1.0f);
    glDisable(GL_BLEND);
}

void EuclidRenderer::drawGridAxes(const UniformGrid& grid) {
    const vec3 extent = vec3(grid.dimensions) * grid.cellSize;
    const float axisLength = 0.20f * std::max({ extent.x, extent.y, extent.z });

    glUseProgram(0);
    glLineWidth(2.0f);
    glBegin(GL_LINES);

    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(axisLength, 0.0f, 0.0f);

    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(0.0f, axisLength, 0.0f);

    glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(0.0f, 0.0f, axisLength);

    glEnd();
    glLineWidth(1.0f);
}

void EuclidRenderer::drawGridPlane(
    const UniformGrid& grid,
    GridPlane plane,
    float planePosition,
    bool drawMinorLines) {

    drawGridPlaneLines(
        grid,
        plane,
        planePosition,
        drawMinorLines
    );
}

void EuclidRenderer::drawGridPlaneLines(
    const UniformGrid& grid,
    GridPlane plane,
    float planePosition,
    bool drawMinorLines) {

    const vec3 min = grid.origin;

    const vec3 max = grid.origin +
        vec3(grid.dimensions) * grid.cellSize;

    const int majorEvery =
        std::max(1, grid.majorEvery);

    glUseProgram(0);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);

    glBlendFunc(
        GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA
    );

    // =========================================================
    // GOLD DIAGNOSTIC SLICE OUTER BORDER
    // =========================================================
    glLineWidth(3.0f);
    glColor4f(1.0f, 1.0f, 1.0f, 0.90f);

    glBegin(GL_LINE_LOOP);

    switch (plane) {
        // --------------------------------
        // XY — constant Z
        // --------------------------------
    case GridPlane::PLANE_XY:

        glVertex3f(min.x, min.y, planePosition);
        glVertex3f(max.x, min.y, planePosition);
        glVertex3f(max.x, max.y, planePosition);
        glVertex3f(min.x, max.y, planePosition);

        break;

        // --------------------------------
        // XZ — constant Y
        // --------------------------------
    case GridPlane::PLANE_XZ:

        glVertex3f(min.x, planePosition, min.z);
        glVertex3f(max.x, planePosition, min.z);
        glVertex3f(max.x, planePosition, max.z);
        glVertex3f(min.x, planePosition, max.z);

        break;

        // --------------------------------
        // YZ — constant X
        // --------------------------------
    case GridPlane::PLANE_YZ:

        glVertex3f(planePosition, min.y, min.z);
        glVertex3f(planePosition, max.y, min.z);
        glVertex3f(planePosition, max.y, max.z);
        glVertex3f(planePosition, min.y, max.z);

        break;

    }

    glEnd();

    // =========================================================
    // GOLD DIAGNOSTIC SLICE MAJOR GRID
    //
    // Always visible.
    // =========================================================
    glLineWidth(2.0f);
    glColor4f(1.0f, 1.0f, 1.0f, 0.35f);

    glBegin(GL_LINES);
    switch (plane) {

        // ---------------------------------------------------------
        // XY — constant Z
        // ---------------------------------------------------------
    case GridPlane::PLANE_XY:

        // Y-parallel lines at major X positions.
        for (int xIdx = 0; xIdx <= grid.dimensions.x; xIdx++) {
            if ((xIdx % majorEvery) != 0)
                continue;

            const float x = min.x +
                static_cast<float>(xIdx) * grid.cellSize.x;

            glVertex3f(x, min.y, planePosition);
            glVertex3f(x, max.y, planePosition);
        }

        // X-parallel lines at major Y positions.
        for (int yIdx = 0; yIdx <= grid.dimensions.y; yIdx++) {
            if ((yIdx % majorEvery) != 0)
                continue;

            const float y = min.y +
                static_cast<float>(yIdx) * grid.cellSize.y;

            glVertex3f(min.x, y, planePosition);
            glVertex3f(max.x, y, planePosition);
        }

        break;

        // ---------------------------------------------------------
        // XZ — constant Y
        // ---------------------------------------------------------
    case GridPlane::PLANE_XZ:

        // Z-parallel lines at major X positions.
        for (int xIdx = 0; xIdx <= grid.dimensions.x; xIdx++) {
            if ((xIdx % majorEvery) != 0)
                continue;

            const float x = min.x +
                static_cast<float>(xIdx) * grid.cellSize.x;

            glVertex3f(x, planePosition, min.z);
            glVertex3f(x, planePosition, max.z);
        }


        // X-parallel lines at major Z positions.
        for (int zIdx = 0; zIdx <= grid.dimensions.z; zIdx++) {
            if ((zIdx % majorEvery) != 0)
                continue;

            const float z = min.z +
                static_cast<float>(zIdx) * grid.cellSize.z;

            glVertex3f(min.x, planePosition, z);
            glVertex3f(max.x, planePosition, z);
        }

        break;

        // ---------------------------------------------------------
        // YZ — constant X
        // ---------------------------------------------------------
    case GridPlane::PLANE_YZ:

        // Z-parallel lines at major Y positions.
        for (int yIdx = 0; yIdx <= grid.dimensions.y; yIdx++) {
            if ((yIdx % majorEvery) != 0)
                continue;

            const float y = min.y +
                static_cast<float>(yIdx) * grid.cellSize.y;

            glVertex3f(planePosition, y, min.z);
            glVertex3f(planePosition, y, max.z);
        }


        // Y-parallel lines at major Z positions.
        for (int zIdx = 0; zIdx <= grid.dimensions.z; zIdx++) {
            if ((zIdx % majorEvery) != 0)
                continue;

            const float z = min.z +
                static_cast<float>(zIdx) * grid.cellSize.z;

            glVertex3f(planePosition, min.y, z);
            glVertex3f(planePosition, max.y, z);
        }

        break;
    }

    glEnd();

    // =========================================================
    // OPTIONAL MINOR GRID
    //
    // GOLD:
    //     width = 1.0
    //     alpha = 0.08
    //
    // Layer-0 currently passes false, so these will normally
    // remain disabled during the idle diagnostic.
    // =========================================================
    if (drawMinorLines) {

        glLineWidth(1.0f);
        glColor4f(1.0f, 1.0f, 1.0f, 0.08f);
        glBegin(GL_LINES);

        switch (plane) {

        case GridPlane::PLANE_XY:

            for (int xIdx = 0; xIdx <= grid.dimensions.x; xIdx++) {
                if ((xIdx % majorEvery) == 0)
                    continue;

                const float x = min.x +
                    static_cast<float>(xIdx) * grid.cellSize.x;

                glVertex3f(x, min.y, planePosition);
                glVertex3f(x, max.y, planePosition);
            }

            for (int yIdx = 0; yIdx <= grid.dimensions.y; yIdx++) {
                if ((yIdx % majorEvery) == 0)
                    continue;


                const float y = min.y +
                    static_cast<float>(yIdx) * grid.cellSize.y;


                glVertex3f(min.x, y, planePosition);
                glVertex3f(max.x, y, planePosition);
            }

            break;


        case GridPlane::PLANE_XZ:

            for (int xIdx = 0; xIdx <= grid.dimensions.x; xIdx++) {
                if ((xIdx % majorEvery) == 0)
                    continue;


                const float x = min.x +
                    static_cast<float>(xIdx) * grid.cellSize.x;


                glVertex3f(x, planePosition, min.z);
                glVertex3f(x, planePosition, max.z);
            }


            for (int zIdx = 0; zIdx <= grid.dimensions.z; zIdx++) {
                if ((zIdx % majorEvery) == 0)
                    continue;


                const float z = min.z +
                    static_cast<float>(zIdx) * grid.cellSize.z;


                glVertex3f(
                    min.x,
                    planePosition,
                    z
                );

                glVertex3f(
                    max.x,
                    planePosition,
                    z
                );
            }

            break;


        case GridPlane::PLANE_YZ:

            for (int yIdx = 0; yIdx <= grid.dimensions.y; yIdx++) {
                if ((yIdx % majorEvery) == 0)
                    continue;

                const float y = min.y +
                    static_cast<float>(yIdx) * grid.cellSize.y;

                glVertex3f(planePosition, y, min.z);
                glVertex3f(planePosition, y, max.z);
            }


            for (int zIdx = 0; zIdx <= grid.dimensions.z; zIdx++) {
                if ((zIdx % majorEvery) == 0)
                    continue;

                const float z = min.z +
                    static_cast<float>(zIdx) * grid.cellSize.z;

                glVertex3f(planePosition, min.y, z);
                glVertex3f(planePosition, max.y, z);
            }

            break;
        }

        glEnd();
    }

    glLineWidth(1.0f);
    glDisable(GL_BLEND);
}

void EuclidRenderer::drawUniformGrid(
    const UniformGrid& grid,
    const GridDisplay& display) {

    // ---------------------------------------------------------
    // Validate grid.
    // ---------------------------------------------------------
    if (grid.dimensions.x <= 0 ||
        grid.dimensions.y <= 0 ||
        grid.dimensions.z <= 0) return;

    const vec3 min = grid.origin;

    const vec3 max = grid.origin +
        vec3(grid.dimensions) * grid.cellSize;

    const int majorEvery =
        std::max(1, grid.majorEvery);

    // ---------------------------------------------------------
    // Outer workspace boundary.
    // ---------------------------------------------------------
    if (display.boundary)
        drawGridBoundary(grid);

    // ---------------------------------------------------------
    // World-reference axes.
    // ---------------------------------------------------------
    if (display.axes)
        drawGridAxes(grid);

    glUseProgram(0);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);

    glBlendFunc(
        GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA
    );

    // =========================================================
    // MAJOR 3D LATTICE
    //
    // at every major-grid intersection.
    //
    // This creates a true sparse 3D volume instead of three
    // intersecting center planes.
    // =========================================================
    if (display.majorGrid) {

        glLineWidth(1.0f);
        glColor4f(1.0f, 1.0f, 1.0f, 0.12f);

        glBegin(GL_LINES);

        // -----------------------------------------------------
        // X-parallel lines.
        //
        // Sweep major Y and Z intersections.
        // -----------------------------------------------------
        for (int yIdx = 0; yIdx <= grid.dimensions.y; yIdx++) {
            if ((yIdx % majorEvery) != 0)
                continue;

            const float y = min.y +
                static_cast<float>(yIdx) * grid.cellSize.y;

            for (int zIdx = 0; zIdx <= grid.dimensions.z; zIdx++) {
                if ((zIdx % majorEvery) != 0)
                    continue;

                const float z = min.z +
                    static_cast<float>(zIdx) * grid.cellSize.z;

                glVertex3f(min.x, y, z);
                glVertex3f(max.x, y, z);

            }
        }

        // -----------------------------------------------------
        // Y-parallel lines.
        //
        // Sweep major X and Z intersections.
        // -----------------------------------------------------
        for (int xIdx = 0; xIdx <= grid.dimensions.x; xIdx++) {
            if ((xIdx % majorEvery) != 0)
                continue;

            const float x = min.x +
                static_cast<float>(xIdx) * grid.cellSize.x;

            for (int zIdx = 0; zIdx <= grid.dimensions.z; zIdx++) {
                if ((zIdx % majorEvery) != 0)
                    continue;

                const float z = min.z +
                    static_cast<float>(zIdx) * grid.cellSize.z;

                glVertex3f(x, min.y, z);
                glVertex3f(x, max.y, z);
            }
        }

        // -----------------------------------------------------
        // Z-parallel lines.
        //
        // Sweep major X and Y intersections.
        // -----------------------------------------------------
        for (int xIdx = 0; xIdx <= grid.dimensions.x; xIdx++) {
            if ((xIdx % majorEvery) != 0)
                continue;

            const float x = min.x +
                static_cast<float>(xIdx) * grid.cellSize.x;

            for (int yIdx = 0; yIdx <= grid.dimensions.y; yIdx++) {
                if ((yIdx % majorEvery) != 0)
                    continue;

                const float y = min.y +
                    static_cast<float>(yIdx) * grid.cellSize.y;

                glVertex3f(x, y, min.z);
                glVertex3f(x, y, max.z);
            }
        }

        glEnd();
    }

    // =========================================================
    // OPTIONAL MINOR 3D LATTICE
    //
    // Normally disabled in Layer-0 idle mode.
    //
    // Kept generic because future workspaces may request the
    // complete collision-cell lattice.
    // =========================================================
    if (display.minorGrid) {

        glLineWidth(1.0f);
        glColor4f(1.0f, 1.0f, 1.0f, 0.04f);

        glBegin(GL_LINES);

        // X-parallel dense lines.
        for (int yIdx = 0; yIdx <= grid.dimensions.y; yIdx++) {

            const float y = min.y +
                static_cast<float>(yIdx) * grid.cellSize.y;

            for (int zIdx = 0; zIdx <= grid.dimensions.z; zIdx++) {

                const float z = min.z +
                    static_cast<float>(zIdx) * grid.cellSize.z;

                glVertex3f(min.x, y, z);
                glVertex3f(max.x, y, z);
            }
        }

        // Y-parallel dense lines.
        for (int xIdx = 0; xIdx <= grid.dimensions.x; xIdx++) {

            const float x = min.x +
                static_cast<float>(xIdx) * grid.cellSize.x;

            for (int zIdx = 0; zIdx <= grid.dimensions.z; zIdx++) {

                const float z = min.z +
                    static_cast<float>(zIdx) * grid.cellSize.z;

                glVertex3f(x, min.y, z);
                glVertex3f(x, max.y, z);
            }
        }

        // Z-parallel dense lines.
        for (int xIdx = 0; xIdx <= grid.dimensions.x; xIdx++) {

            const float x = min.x +
                static_cast<float>(xIdx) * grid.cellSize.x;

            for (int yIdx = 0; yIdx <= grid.dimensions.y; yIdx++) {

                const float y = min.y +
                    static_cast<float>(yIdx) * grid.cellSize.y;

                glVertex3f(x, y, min.z);
                glVertex3f(x, y, max.z);
            }
        }

        glEnd();
    }

    glLineWidth(1.0f);
    glDisable(GL_BLEND);
}

void EuclidRenderer::drawWireCube(
    const vec3& center,
    const vec3& halfExtent) {

    const vec3 min =
        center - halfExtent;

    const vec3 max =
        center + halfExtent;

    glUseProgram(0);
    glEnable(GL_BLEND);

    glBlendFunc(
        GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA
    );

    glLineWidth(2.0f);
    glColor4f(0.85f, 0.95f, 1.0f, 0.90f);

    glBegin(GL_LINES);

    // Bottom
    glVertex3f(min.x, min.y, min.z);
    glVertex3f(max.x, min.y, min.z);

    glVertex3f(max.x, min.y, min.z);
    glVertex3f(max.x, min.y, max.z);

    glVertex3f(max.x, min.y, max.z);
    glVertex3f(min.x, min.y, max.z);

    glVertex3f(min.x, min.y, max.z);
    glVertex3f(min.x, min.y, min.z);

    // Top
    glVertex3f(min.x, max.y, min.z);
    glVertex3f(max.x, max.y, min.z);

    glVertex3f(max.x, max.y, min.z);
    glVertex3f(max.x, max.y, max.z);

    glVertex3f(max.x, max.y, max.z);
    glVertex3f(min.x, max.y, max.z);

    glVertex3f(min.x, max.y, max.z);
    glVertex3f(min.x, max.y, min.z);

    // Vertical
    glVertex3f(min.x, min.y, min.z);
    glVertex3f(min.x, max.y, min.z);

    glVertex3f(max.x, min.y, min.z);
    glVertex3f(max.x, max.y, min.z);

    glVertex3f(max.x, min.y, max.z);
    glVertex3f(max.x, max.y, max.z);

    glVertex3f(min.x, min.y, max.z);
    glVertex3f(min.x, max.y, max.z);

    glEnd();

    glLineWidth(1.0f);
    glDisable(GL_BLEND);
}

void EuclidRenderer::drawUniformGridZRange(
    const UniformGrid& grid,
    float visibleMinZ,
    float visibleMaxZ,
    const GridDisplay& display) {

    if (grid.dimensions.x <= 0 || grid.dimensions.y <= 0 ||
        grid.dimensions.z <= 0) return;

    const vec3 min = grid.origin;
    const vec3 max = grid.origin + vec3(grid.dimensions) * grid.cellSize;
    visibleMinZ = std::max(min.z, std::min(max.z, visibleMinZ));
    visibleMaxZ = std::max(min.z, std::min(max.z, visibleMaxZ));
    if (visibleMaxZ - visibleMinZ <= 0.00001f) return;

    // Preserve the exact established full-grid path at the endpoint.
    if (visibleMinZ <= min.z + 0.00001f &&
        visibleMaxZ >= max.z - 0.00001f) {
        drawUniformGrid(grid, display);
        return;
    }

    if (display.boundary) {
        const vec3 clippedCenter(
            0.5f * (min.x + max.x),
            0.5f * (min.y + max.y),
            0.5f * (visibleMinZ + visibleMaxZ));
        const vec3 clippedHalfExtent(
            0.5f * (max.x - min.x),
            0.5f * (max.y - min.y),
            0.5f * (visibleMaxZ - visibleMinZ));
        drawWireCube(clippedCenter, clippedHalfExtent);
    }

    // The positive axes are meaningful only while their origin remains
    // inside the clipped interval.
    if (display.axes && visibleMinZ <= 0.0f && visibleMaxZ >= 0.0f)
        drawGridAxes(grid);

    const int majorEvery = std::max(1, grid.majorEvery);
    glUseProgram(0);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const auto drawLattice = [&](bool majorOnly) {
        for (int yIdx = 0; yIdx <= grid.dimensions.y; ++yIdx) {
            if (majorOnly && (yIdx % majorEvery) != 0) continue;
            const float y = min.y + static_cast<float>(yIdx) * grid.cellSize.y;
            for (int zIdx = 0; zIdx <= grid.dimensions.z; ++zIdx) {
                if (majorOnly && (zIdx % majorEvery) != 0) continue;
                const float z = min.z + static_cast<float>(zIdx) * grid.cellSize.z;
                if (z < visibleMinZ || z > visibleMaxZ) continue;
                glVertex3f(min.x, y, z); glVertex3f(max.x, y, z);
            }
        }
        for (int xIdx = 0; xIdx <= grid.dimensions.x; ++xIdx) {
            if (majorOnly && (xIdx % majorEvery) != 0) continue;
            const float x = min.x + static_cast<float>(xIdx) * grid.cellSize.x;
            for (int zIdx = 0; zIdx <= grid.dimensions.z; ++zIdx) {
                if (majorOnly && (zIdx % majorEvery) != 0) continue;
                const float z = min.z + static_cast<float>(zIdx) * grid.cellSize.z;
                if (z < visibleMinZ || z > visibleMaxZ) continue;
                glVertex3f(x, min.y, z); glVertex3f(x, max.y, z);
            }
        }
        for (int xIdx = 0; xIdx <= grid.dimensions.x; ++xIdx) {
            if (majorOnly && (xIdx % majorEvery) != 0) continue;
            const float x = min.x + static_cast<float>(xIdx) * grid.cellSize.x;
            for (int yIdx = 0; yIdx <= grid.dimensions.y; ++yIdx) {
                if (majorOnly && (yIdx % majorEvery) != 0) continue;
                const float y = min.y + static_cast<float>(yIdx) * grid.cellSize.y;
                glVertex3f(x, y, visibleMinZ);
                glVertex3f(x, y, visibleMaxZ);
            }
        }
    };

    if (display.majorGrid) {
        glLineWidth(1.0f);
        glColor4f(1.0f, 1.0f, 1.0f, 0.12f);
        glBegin(GL_LINES); drawLattice(true); glEnd();
    }
    if (display.minorGrid) {
        glLineWidth(1.0f);
        glColor4f(1.0f, 1.0f, 1.0f, 0.04f);
        glBegin(GL_LINES); drawLattice(false); glEnd();
    }

    glLineWidth(1.0f);
    glDisable(GL_BLEND);
}

void EuclidRenderer::_initGL() {

    // ---------------------------------------------------------
    // Particle sphere shader.
    // ---------------------------------------------------------
    m_program0 =
        _compileProgram(
            vertexShader,
            spherePixelShader,
            "radius",
            "particle sphere"
        );

    // ---------------------------------------------------------
    // Particle selection/highlight shader.
    // ---------------------------------------------------------
    m_program1 =
        _compileProgram(
            lightVertShader,
            lightFragShader,
            "radius",
            "particle highlight"
        );

    // ---------------------------------------------------------
    // OBJ triangle mesh shader.
    // ---------------------------------------------------------
    m_meshProgram =
        _compileProgram(
            modelVertexShader,
            modelFragmentShader,
            "normal",
            "OBJ mesh"
        );

    m_meshColorLocation = -1;
    m_meshLightDirLocation = -1;
    m_meshAmbientLocation = -1;
    m_meshTexcoordAttributeLocation = -1;
    m_meshSamplerLocation = -1;
    m_meshUseTextureLocation = -1;
    m_meshAlphaMaskLocation = -1;
    m_meshAlphaCutoffLocation = -1;
    m_meshEmissiveSamplerLocation = -1;
    m_meshUseEmissiveLocation = -1;
    m_meshEmissiveFactorLocation = -1;
    m_meshEmissiveIntensityLocation = -1;

    if (m_meshProgram) {

        m_meshColorLocation =
            glGetUniformLocation(
                m_meshProgram,
                "uColor"
            );

        m_meshLightDirLocation =
            glGetUniformLocation(
                m_meshProgram,
                "uLightDir"
            );

        m_meshAmbientLocation =
            glGetUniformLocation(
                m_meshProgram,
                "uAmbient"
            );

        m_meshTexcoordAttributeLocation =
            glGetAttribLocation(m_meshProgram, "texcoord");
        m_meshSamplerLocation =
            glGetUniformLocation(m_meshProgram, "uBaseColorTexture");
        m_meshUseTextureLocation =
            glGetUniformLocation(m_meshProgram, "uUseBaseColorTexture");
        m_meshAlphaMaskLocation =
            glGetUniformLocation(m_meshProgram, "uAlphaMask");
        m_meshAlphaCutoffLocation =
            glGetUniformLocation(m_meshProgram, "uAlphaCutoff");
        m_meshEmissiveSamplerLocation =
            glGetUniformLocation(m_meshProgram, "uEmissiveTexture");
        m_meshUseEmissiveLocation =
            glGetUniformLocation(m_meshProgram, "uUseEmissiveTexture");
        m_meshEmissiveFactorLocation =
            glGetUniformLocation(m_meshProgram, "uEmissiveFactor");
        m_meshEmissiveIntensityLocation =
            glGetUniformLocation(m_meshProgram, "uEmissiveIntensity");

        if (m_meshColorLocation < 0) {
            printf(
                "[EuclidRenderer] WARNING: OBJ mesh shader "
                "uniform uColor was not found.\n"
            );
        }

        if (m_meshLightDirLocation < 0) {
            printf(
                "[EuclidRenderer] WARNING: OBJ mesh shader "
                "uniform uLightDir was not found.\n"
            );
        }

        if (m_meshAmbientLocation < 0) {
            printf(
                "[EuclidRenderer] WARNING: OBJ mesh shader "
                "uniform uAmbient was not found.\n"
            );
        }

        // Establish safe defaults. These values remain stored
        // in the program until the future mesh draw path changes them.
        glUseProgram(m_meshProgram);

        if (m_meshColorLocation >= 0) {
            glUniform4f(
                m_meshColorLocation,
                1.0f,
                0.0f,
                0.0f,
                1.0f
            );
        }

        if (m_meshLightDirLocation >= 0) {
            glUniform3f(
                m_meshLightDirLocation,
                0.577f,
                0.577f,
                0.577f
            );
        }

        if (m_meshAmbientLocation >= 0) {
            glUniform1f(
                m_meshAmbientLocation,
                0.15f
            );
        }
        if (m_meshSamplerLocation >= 0) glUniform1i(m_meshSamplerLocation, 0);
        if (m_meshUseTextureLocation >= 0) glUniform1i(m_meshUseTextureLocation, 0);
        if (m_meshAlphaMaskLocation >= 0) glUniform1i(m_meshAlphaMaskLocation, 0);
        if (m_meshAlphaCutoffLocation >= 0) glUniform1f(m_meshAlphaCutoffLocation, 0.5f);

        glUseProgram(0);

        printf(
            "[EuclidRenderer] OBJ mesh shader uniforms: "
            "color=%d lightDir=%d ambient=%d texcoord=%d sampler=%d\n",
            m_meshColorLocation,
            m_meshLightDirLocation,
            m_meshAmbientLocation,
            m_meshTexcoordAttributeLocation,
            m_meshSamplerLocation
        );
    }

    glClampColorARB(
        GL_CLAMP_VERTEX_COLOR_ARB,
        GL_FALSE
    );

    glClampColorARB(
        GL_CLAMP_FRAGMENT_COLOR_ARB,
        GL_FALSE
    );
}

void EuclidRenderer::_initialize() {
    assert(!m_bInitialized);

    if (!m_radVBO) {
        glGenBuffers(1, &m_radVBO);
    }

    if (!m_particlePosVBO) {
        glGenBuffers(1, &m_particlePosVBO);
    }

    if (!m_particleRadVBO) {
        glGenBuffers(1, &m_particleRadVBO);
    }

    if (!m_particleColorVBO) {
        glGenBuffers(1, &m_particleColorVBO);
    }

    m_bInitialized = true;
}

void EuclidRenderer::_drawPoints(bool useColorBuffer) {
    
    _drawPointsRange(
        0,
        m_numParticles,
        useColorBuffer
    );
}


void EuclidRenderer::_drawPointsRange(
    int start, 
    int count, 
    bool useColorBuffer) {

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo);

    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_radVBO);

    glEnableVertexAttribArrayARB(1);
    glVertexAttribPointerARB(1, 1, GL_FLOAT, GL_FALSE, 0, 0);

    const bool useColors = useColorBuffer && (m_colorVBO != 0);

    if (useColors) {
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVBO);

        glColorPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_COLOR_ARRAY);
    }

    glDrawArrays(GL_POINTS, start, count);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableVertexAttribArrayARB(1);

    if (useColors) {
        glDisableClientState(GL_COLOR_ARRAY);
    }

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
}

void EuclidRenderer::display(DisplayMode mode) {
    switch (mode) {

    case PARTICLE_POINTS: {
        glColor3f(1, 1, 1);
        glPointSize(m_pointSize);
        _drawPoints();
        break;
    }
    default:
    case PARTICLE_SPHERES: {
        glEnable(GL_POINT_SPRITE_ARB);
        glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);

        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
        glDepthMask(GL_TRUE);

        glEnable(GL_DEPTH_TEST);

        glUseProgram(m_program0);
        glUniform1f(glGetUniformLocation(m_program0, "pointScale"),
            m_window_h / tanf(m_fov * 0.5f * (float)M_PI / 180.0f));

        vector<float> radii(m_numParticles);
        for (int i = 0; i < m_numParticles; i++) {

            radii[i] = m_rad[i];
        }

        glBindBuffer(GL_ARRAY_BUFFER, m_radVBO);

        glBufferData(GL_ARRAY_BUFFER,
            m_numParticles * sizeof(float),
            radii.data(),
            GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glColor3f(1, 1, 1);
        _drawPoints();

        if (m_particleHighlighted && m_program1) {
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);

            glUseProgram(m_program1);
            glUniform1f(glGetUniformLocation(m_program1, "emissiveIntensity"), 1.0f);
            glUniform1f(
                glGetUniformLocation(m_program1, "pointScale"),
                m_window_h / tanf(m_fov * 0.5f * (float)M_PI / 180.0f)
            );

            vector<float> highlightRadii(m_numParticles);
            for (int i = 0; i < m_numParticles; i++) {
                highlightRadii[i] = m_rad[i] * m_particleHighlightScale;
            }

            glBindBuffer(GL_ARRAY_BUFFER, m_radVBO);
            glBufferData(
                GL_ARRAY_BUFFER,
                m_numParticles * sizeof(float),
                highlightRadii.data(),
                GL_STATIC_DRAW
            );
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glColor4f(1.0f, 0.35f, 0.05f, 0.85f);
            _drawPoints(false);

            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }

        glUseProgram(0);
        glDisable(GL_POINT_SPRITE_ARB);

        break;
    }

    }
}

void EuclidRenderer::setGridStyle(int majorEvery, bool drawMinor) {
    m_gridMajorEvery = std::max(1, majorEvery);
    m_drawMinorGrid = drawMinor;
}

// --- <TESSERACT OBJECT> ---
void EuclidRenderer::displayGrid() {
    if (!m_gridEnabled) return;

    if (m_gridDim.x <= 0 ||
        m_gridDim.y <= 0 ||
        m_gridDim.z <= 0) return;

    glUseProgram(0);
    glDisable(GL_POINT_SPRITE_ARB);
    glDisable(GL_TEXTURE_2D);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const vec3 mn = m_gridOrigin;
    const vec3 mx = m_gridOrigin + vec3(
        m_cellSize.x * (float)m_gridDim.x,
        m_cellSize.y * (float)m_gridDim.y,
        m_cellSize.z * (float)m_gridDim.z
    );

    // Axes at origin
    if (m_drawAxes)
        drawAxes();

    const int major =
        std::max(1, m_gridMajorEvery);

    if (m_gridMode == GRID_3D) {
        // bounding box
        if (m_drawBoundaryGrid) {
            glLineWidth(2.0f);
            glColor4f(1, 1, 1, 0.35f);
            RenderUtils::draw_aabb_wire(mn, mx);
        }

        // major lattice (sparse)
        if (m_drawMajorGrid) {
            glLineWidth(1.0f);
            glColor4f(1, 1, 1, 0.12f);

            glBegin(GL_LINES);

            // X-parallel
            for (int yi = 0; yi <= m_gridDim.y; yi++) {
                if (!RenderUtils::is_major(yi, major))
                    continue;

                float y = mn.y +
                    yi * m_cellSize.y;

                for (int zi = 0; zi <= m_gridDim.z; zi++) {
                    if (!RenderUtils::is_major(zi, major))
                        continue;

                    float z = mn.z +
                        zi * m_cellSize.z;

                    glVertex3f(mn.x, y, z);
                    glVertex3f(mx.x, y, z);
                }
            }

            // Y-parallel
            for (int xi = 0; xi <= m_gridDim.x; xi++) {
                if (!RenderUtils::is_major(xi, major))
                    continue;

                float x = mn.x +
                    xi * m_cellSize.x;

                for (int zi = 0; zi <= m_gridDim.z; zi++) {
                    if (!RenderUtils::is_major(zi, major))
                        continue;

                    float z = mn.z +
                        zi * m_cellSize.z;

                    glVertex3f(x, mn.y, z);
                    glVertex3f(x, mx.y, z);
                }
            }

            // Z-parallel
            for (int xi = 0; xi <= m_gridDim.x; xi++) {
                if (!RenderUtils::is_major(xi, major))
                    continue;

                float x = mn.x +
                    xi * m_cellSize.x;

                for (int yi = 0; yi <= m_gridDim.y; yi++) {
                    if (!RenderUtils::is_major(yi, major))
                        continue;

                    float y = mn.y +
                        yi * m_cellSize.y;

                    glVertex3f(x, y, mn.z);
                    glVertex3f(x, y, mx.z);
                }
            }

            glEnd();
        }

        // optional minor grid (dense)
        if (m_drawMinorGrid) {

            glLineWidth(1.0f);
            glColor4f(1, 1, 1, 0.04f);

            glBegin(GL_LINES);

            for (int yi = 0; yi <= m_gridDim.y; yi++) {

                float y = mn.y +
                    yi * m_cellSize.y;

                for (int zi = 0; zi <= m_gridDim.z; zi++) {

                    float z = mn.z +
                        zi * m_cellSize.z;

                    glVertex3f(mn.x, y, z);
                    glVertex3f(mx.x, y, z);
                }
            }

            for (int xi = 0; xi <= m_gridDim.x; xi++) {

                float x = mn.x +
                    xi * m_cellSize.x;

                for (int zi = 0; zi <= m_gridDim.z; zi++) {

                    float z = mn.z +
                        zi * m_cellSize.z;

                    glVertex3f(x, mn.y, z);
                    glVertex3f(x, mx.y, z);
                }
            }

            for (int xi = 0; xi <= m_gridDim.x; xi++) {

                float x = mn.x +
                    xi * m_cellSize.x;

                for (int yi = 0; yi <= m_gridDim.y; yi++) {

                    float y = mn.y +
                        yi * m_cellSize.y;

                    glVertex3f(x, y, mn.z);
                    glVertex3f(x, y, mx.z);
                }
            }

            glEnd();
        }
    }
    else {
        // =========================================================
        // 2D DIAGNOSTIC SLICE
        //
        // Boundary and major lines are always visible.
        // Minor collision-cell lines are optional.
        // =========================================================
        if (m_workPlane == PLANE_XY) {

            const int mid = m_gridDim.z / 2;

            const int sliceIndex =
                RenderUtils::clampi(
                    mid + m_sliceOffset,
                    0,
                    m_gridDim.z
                );

            const float z =
                mn.z + sliceIndex * m_cellSize.z;

            // -----------------------------------------------------
            // Plane boundary — always visible.
            // -----------------------------------------------------
            glLineWidth(3.0f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.90f);

            RenderUtils::draw_rect_wire_xy(
                mn.x,
                mn.y,
                mx.x,
                mx.y,
                z
            );

            // -----------------------------------------------------
            // Major grid lines — always visible.
            // -----------------------------------------------------
            glLineWidth(2.0f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.35f);

            glBegin(GL_LINES);

            for (int xi = 0; xi <= m_gridDim.x; xi++) {
                if (!RenderUtils::is_major(xi, major))
                    continue;

                const float x =
                    mn.x + xi * m_cellSize.x;

                glVertex3f(x, mn.y, z);
                glVertex3f(x, mx.y, z);
            }

            for (int yi = 0; yi <= m_gridDim.y; yi++) {
                if (!RenderUtils::is_major(yi, major))
                    continue;

                const float y =
                    mn.y + yi * m_cellSize.y;

                glVertex3f(mn.x, y, z);
                glVertex3f(mx.x, y, z);
            }

            glEnd();

            if (m_drawMinorGrid) {

                glLineWidth(1.0f);
                glColor4f(1, 1, 1, 0.08f);

                glBegin(GL_LINES);

                for (int xi = 0; xi <= m_gridDim.x; xi++) {
                    // Major lines where already rendered above
                    if (RenderUtils::is_major(xi, major))
                        continue;

                    const float x =
                        mn.x + xi * m_cellSize.x;

                    glVertex3f(x, mn.y, z);
                    glVertex3f(x, mx.y, z);
                }

                for (int yi = 0; yi <= m_gridDim.y; yi++) {
                    if (RenderUtils::is_major(yi, major))
                        continue;

                    const float y =
                        mn.y + yi * m_cellSize.y;

                    glVertex3f(mn.x, y, z);
                    glVertex3f(mx.x, y, z);
                }

                glEnd();
            }
        }
        else if (m_workPlane == PLANE_XZ) {

            const int mid = m_gridDim.y / 2;

            const int sliceIndex =
                RenderUtils::clampi(
                    mid + m_sliceOffset,
                    0,
                    m_gridDim.y
                );

            const float y = mn.y +
                sliceIndex * m_cellSize.y;

            // Plane boundary — always visible.
            glLineWidth(3.0f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.90f);

            RenderUtils::draw_rect_wire_xz(
                mn.x,
                mn.z,
                mx.x,
                mx.z,
                y
            );

            // Major grid lines — always visible.
            glLineWidth(2.0f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.35f);

            glBegin(GL_LINES);

            for (int xi = 0; xi <= m_gridDim.x; ++xi) {
                if (!RenderUtils::is_major(xi, major))
                    continue;


                const float x =
                    mn.x + xi * m_cellSize.x;

                glVertex3f(x, y, mn.z);
                glVertex3f(x, y, mx.z);
            }

            for (int zi = 0; zi <= m_gridDim.z; ++zi) {
                if (!RenderUtils::is_major(zi, major))
                    continue;


                const float z =
                    mn.z + zi * m_cellSize.z;

                glVertex3f(mn.x, y, z);
                glVertex3f(mx.x, y, z);
            }

            glEnd();

            if (m_drawMinorGrid) {

                glLineWidth(1.0f);
                glColor4f(1, 1, 1, 0.08f);

                glBegin(GL_LINES);

                for (int xi = 0; xi <= m_gridDim.x; xi++) {
                    if (RenderUtils::is_major(xi, major))
                        continue;

                    const float x =
                        mn.x + xi * m_cellSize.x;

                    glVertex3f(x, y, mn.z);
                    glVertex3f(x, y, mx.z);
                }

                for (int zi = 0; zi <= m_gridDim.z; zi++) {
                    if (RenderUtils::is_major(zi, major))
                        continue;

                    const float z =
                        mn.z + zi * m_cellSize.z;

                    glVertex3f(mn.x, y, z);
                    glVertex3f(mx.x, y, z);
                }

                glEnd();
            }
        }
        else if (m_workPlane == PLANE_YZ) {

            const int mid = m_gridDim.x / 2;

            const int sliceIndex =
                RenderUtils::clampi(mid + m_sliceOffset, 0, m_gridDim.x);

            const float x = mn.x +
                sliceIndex * m_cellSize.x;

            // Plane boundary — always visible.
            glLineWidth(3.0f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.90f);

            RenderUtils::draw_rect_wire_yz(
                mn.y,
                mn.z,
                mx.y,
                mx.z,
                x
            );

            // Major grid lines — always visible.
            glLineWidth(2.0f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.35f);

            glBegin(GL_LINES);

            for (int yi = 0; yi <= m_gridDim.y; yi++) {
                if (!RenderUtils::is_major(yi, major))
                    continue;


                const float y =
                    mn.y + yi * m_cellSize.y;

                glVertex3f(x, y, mn.z);
                glVertex3f(x, y, mx.z);
            }

            for (int zi = 0; zi <= m_gridDim.z; zi++) {
                if (!RenderUtils::is_major(zi, major))
                    continue;


                const float z =
                    mn.z + zi * m_cellSize.z;

                glVertex3f(x, mn.y, z);
                glVertex3f(x, mx.y, z);
            }

            glEnd();

            if (m_drawMinorGrid) {
                glLineWidth(1.0f);
                glColor4f(1, 1, 1, 0.18f);

                glBegin(GL_LINES);

                for (int yi = 0; yi <= m_gridDim.y; yi++) {
                    if (!RenderUtils::is_major(yi, major))
                        continue;

                    const float y =
                        mn.y + yi * m_cellSize.y;

                    glVertex3f(x, y, mn.z);
                    glVertex3f(x, y, mx.z);
                }

                for (int zi = 0; zi <= m_gridDim.z; zi++) {
                    if (!RenderUtils::is_major(zi, major))
                        continue;

                    const float z =
                        mn.z + zi * m_cellSize.z;

                    glVertex3f(x, mn.y, z);
                    glVertex3f(x, mx.y, z);
                }

                glEnd();
            }
        }
    }

    glDisable(GL_BLEND);
    glLineWidth(1.0f);
}
// --- <\TESSERACT OBJECT> ---

void EuclidRenderer::setWorkspaceGridVisibility(
    bool drawBoundary,
    bool drawMajor,
    bool drawMinor,
    bool drawAxes) {

    m_drawBoundaryGrid = drawBoundary;
    m_drawMajorGrid = drawMajor;
    m_drawMinorGrid = drawMinor;
    m_drawAxes = drawAxes;
}

void EuclidRenderer::setGridMode3D() {
    m_gridMode = GRID_3D;
}

void EuclidRenderer::setGridMode2D(GridPlane plane, int sliceOffset) {
    m_gridMode = GRID_2D;
    m_workPlane = plane;
    m_sliceOffset = sliceOffset;
}

void EuclidRenderer::setRadius(float* radiusData, int numParticles) {
    assert(m_bInitialized);

    if (!radiusData || numParticles <= 0) {
        m_numParticles = 0;
        return;
    }

    if (m_radCapacity < numParticles) {

        delete[] m_rad;

        m_rad = new float[numParticles];
        m_radCapacity = numParticles;
    }

    copy_n(radiusData, numParticles, m_rad);
    m_numParticles = numParticles;

    // Keep the GPU radius attribute synchronized here
    // full-population and range rendering path can use it
    glBindBuffer(GL_ARRAY_BUFFER, m_radVBO);

    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(numParticles * sizeof(float)), 
        m_rad,
        GL_DYNAMIC_DRAW
    );

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void EuclidRenderer::setPositions(float* pos, int numParticles) {
    m_pos = pos;
    m_numParticles = numParticles;
}

void EuclidRenderer::setVertexBuffer(unsigned int vbo, int numParticles) {
    m_vbo = vbo;
    m_numParticles = numParticles;
}

void EuclidRenderer::setGrid(
    const ivec3& gridDim,
    const vec3& worldOrigin,
    const vec3& cellSize) {

    m_gridDim = gridDim;
    m_gridOrigin = worldOrigin;
    m_cellSize = cellSize;
}

void EuclidRenderer::displayParticleRange(
    int start,
    int count,
    bool emissive,
    float emissiveIntensity) {

    if (start < 0 ||
        count <= 0 ||
        start >= m_numParticles || 
        start + count > m_numParticles) return;

    const GLuint program =
        emissive
        ? m_program1
        : m_program0;

    if (program == 0) return;
    if (m_vbo == 0 || m_radVBO == 0) return;

    // Preserve the caller's blend and depth state across each population pass.
    glPushAttrib(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_ENABLE_BIT);
    glEnable(GL_POINT_SPRITE_ARB);
    glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);

    glEnable(GL_DEPTH_TEST);
    // Overlay positions equal the opaque spheres: accept equal depth, but
    // retain opaque depth so emission cannot shine through nearer particles.
    glDepthFunc(emissive ? GL_LEQUAL : GL_LESS);
    glDepthMask(emissive ? GL_FALSE : GL_TRUE);
    glDisable(GL_BLEND);

    glUseProgram(program);
    if (emissive) {
        glUniform1f(glGetUniformLocation(program, "emissiveIntensity"),
            (std::max)(0.0f, emissiveIntensity));
    }

    const GLint pointScaleLocation =
        glGetUniformLocation(program, "pointScale");

    if (pointScaleLocation >= 0) {
        glUniform1f(
            pointScaleLocation,
            m_window_h / tanf(m_fov * 0.5f * static_cast<float>(M_PI) / 180.0f)
        );
    }

    if (emissive) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    }

    _drawPointsRange(start, count, true);
    
    if (emissive) {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_BLEND);
    }

    glUseProgram(0);
    glDisable(GL_POINT_SPRITE_ARB);
    glPopAttrib();
}

// =============================================================================
// SHADER PROGRAM COMPILATION
// =============================================================================
GLuint EuclidRenderer::_compileProgram(
    const char* vsource,
    const char* fsource,
    const char* secondaryAttributeName,
    const char* programLabel) {

    const char* safeLabel =
        (programLabel && programLabel[0] != '\0')
        ? programLabel
        : "unnamed program";

    // ---------------------------------------------------------
    // Validate shader source.
    // ---------------------------------------------------------
    if (!vsource || !fsource) {

        printf(
            "[EuclidRenderer] Cannot compile %s: "
            "shader source is null.\n",
            safeLabel
        );

        return 0;
    }

    // ---------------------------------------------------------
    // Create shader objects.
    // ---------------------------------------------------------
    GLuint vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

    if (!vertexShaderObject || !fragmentShaderObject) {

        printf(
            "[EuclidRenderer] Cannot compile %s: "
            "glCreateShader failed.\n",
            safeLabel
        );

        if (vertexShaderObject) {

            glDeleteShader(vertexShaderObject);
        }

        if (fragmentShaderObject) {

            glDeleteShader(fragmentShaderObject);
        }

        return 0;
    }


    // ---------------------------------------------------------
    // Compile vertex shader.
    // ---------------------------------------------------------
    glShaderSource(
        vertexShaderObject,
        1,
        &vsource,
        nullptr
    );

    glCompileShader(vertexShaderObject);
    GLint vertexCompiled = GL_FALSE;

    glGetShaderiv(
        vertexShaderObject,
        GL_COMPILE_STATUS,
        &vertexCompiled
    );

    if (vertexCompiled != GL_TRUE) {

        GLint logLength = 0;

        glGetShaderiv(
            vertexShaderObject,
            GL_INFO_LOG_LENGTH,
            &logLength
        );


        vector<char> log((std::max)(1, logLength));

        glGetShaderInfoLog(
            vertexShaderObject,
            static_cast<GLsizei>(log.size()),
            nullptr,
            log.data()
        );

        printf(
            "[EuclidRenderer] Failed to compile "
            "%s vertex shader:\n%s\n",
            safeLabel,
            log.data()
        );

        glDeleteShader(vertexShaderObject);
        glDeleteShader(fragmentShaderObject);

        return 0;
    }

    // ---------------------------------------------------------
    // Compile fragment shader.
    // ---------------------------------------------------------
    glShaderSource(
        fragmentShaderObject,
        1,
        &fsource,
        nullptr
    );

    glCompileShader(fragmentShaderObject);

    GLint fragmentCompiled = GL_FALSE;

    glGetShaderiv(
        fragmentShaderObject,
        GL_COMPILE_STATUS,
        &fragmentCompiled
    );

    if (fragmentCompiled != GL_TRUE) {

        GLint logLength = 0;

        glGetShaderiv(
            fragmentShaderObject,
            GL_INFO_LOG_LENGTH,
            &logLength
        );

        vector<char> log((std::max)(1, logLength));

        glGetShaderInfoLog(
            fragmentShaderObject,
            static_cast<GLsizei>(log.size()),
            nullptr,
            log.data()
        );

        printf(
            "[EuclidRenderer] Failed to compile "
            "%s fragment shader:\n%s\n",
            safeLabel,
            log.data()
        );

        glDeleteShader(vertexShaderObject);
        glDeleteShader(fragmentShaderObject);

        return 0;
    }

    // ---------------------------------------------------------
    // Create and link program.
    // ---------------------------------------------------------
    GLuint program = glCreateProgram();

    if (!program) {

        printf(
            "[EuclidRenderer] Cannot link %s: "
            "glCreateProgram failed.\n",
            safeLabel
        );

        glDeleteShader(vertexShaderObject);
        glDeleteShader(fragmentShaderObject);

        return 0;
    }

    glAttachShader(program, vertexShaderObject);
    glAttachShader(program, fragmentShaderObject);

    // ---------------------------------------------------------
    // Preserve Ver004 attribute layout.
    //
    // Attribute 0:
    //     vertex position
    //
    // Attribute 1:
    //     radius for particle programs
    //     normal for OBJ mesh program
    // ---------------------------------------------------------
    glBindAttribLocation(program, 0, "position");

    if (secondaryAttributeName && secondaryAttributeName[0] != '\0') {

        glBindAttribLocation(
            program,
            1,
            secondaryAttributeName
        );
    }

    glLinkProgram(program);

    // ---------------------------------------------------------
    // Validate program link.
    // ---------------------------------------------------------
    GLint linkSuccess = GL_FALSE;

    glGetProgramiv(
        program,
        GL_LINK_STATUS,
        &linkSuccess
    );

    if (linkSuccess != GL_TRUE) {

        GLint logLength = 0;

        glGetProgramiv(
            program,
            GL_INFO_LOG_LENGTH,
            &logLength
        );

        vector<char> log((std::max)(1, logLength));

        glGetProgramInfoLog(
            program,
            static_cast<GLsizei>(log.size()),
            nullptr,
            log.data()
        );

        printf(
            "[EuclidRenderer] Failed to link %s:\n%s\n",
            safeLabel,
            log.data()
        );

        glDetachShader(program, vertexShaderObject);
        glDetachShader(program, fragmentShaderObject);

        glDeleteShader(vertexShaderObject);
        glDeleteShader(fragmentShaderObject);

        glDeleteProgram(program);

        return 0;
    }

    // ---------------------------------------------------------
    // Shader objects are no longer needed after linking.
    // ---------------------------------------------------------
    glDetachShader(program, vertexShaderObject);
    glDetachShader(program, fragmentShaderObject);

    glDeleteShader(vertexShaderObject);
    glDeleteShader(fragmentShaderObject);

    printf(
        "[EuclidRenderer] Shader program ready: %s "
        "(program=%u, attribute1=%s)\n",
        safeLabel,
        static_cast<unsigned int>(program),
        secondaryAttributeName
        ? secondaryAttributeName
        : "none"
    );

    return program;
}

void EuclidRenderer::drawParticleSphere(
    GLuint program,
    const float pos[4],
    float radius,
    const float color[4],
    bool emissiveBlend) {

    if (!program) return;

    glUseProgram(program);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POINT_SPRITE_ARB);
    glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);

    if (emissiveBlend) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDepthMask(GL_FALSE);
        glDepthFunc(GL_LEQUAL);
    }
    else {
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
    }

    const float pointScale =
        m_window_h / tanf(
            m_fov * 0.5f *
            static_cast<float>(M_PI) / 180.0f
        );

    GLint pointScaleLoc =
        glGetUniformLocation(program, "pointScale");

    if (pointScaleLoc >= 0) {
        glUniform1f(pointScaleLoc, pointScale);
    }

    // Position VBO.
    glBindBuffer(GL_ARRAY_BUFFER, m_particlePosVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(float) * 4,
        pos,
        GL_DYNAMIC_DRAW
    );

    glVertexPointer(4, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_VERTEX_ARRAY);

    // Radius VBO.
    glBindBuffer(GL_ARRAY_BUFFER, m_particleRadVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(float),
        &radius,
        GL_DYNAMIC_DRAW
    );

    glEnableVertexAttribArrayARB(1);
    glVertexAttribPointerARB(
        1,
        1,
        GL_FLOAT,
        GL_FALSE,
        0,
        nullptr
    );

    // Color VBO.
    glBindBuffer(GL_ARRAY_BUFFER, m_particleColorVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(float) * 4,
        color,
        GL_DYNAMIC_DRAW
    );

    glColorPointer(4, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_COLOR_ARRAY);

    glDrawArrays(GL_POINTS, 0, 1);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableVertexAttribArrayARB(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glDisable(GL_POINT_SPRITE_ARB);

    glUseProgram(0);
}

void EuclidRenderer::drawAxes() {
    const float axisLen = 0.35f;

    glLineWidth(2.0f);
    glBegin(GL_LINES);

    glColor4f(1, 0, 0, 1);
    glVertex3f(0, 0, 0);
    glVertex3f(axisLen, 0, 0);

    glColor4f(0, 1, 0, 1);
    glVertex3f(0, 0, 0);
    glVertex3f(0, axisLen, 0);

    glColor4f(0, 0, 1, 1);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, axisLen);

    glEnd();
    glLineWidth(1.0f);
}

void EuclidRenderer::drawWorkspaceBoundary() {
    const float s = kSimHalfBox;

    glUseProgram(0);
    glDisable(GL_TEXTURE_2D);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Outer bounding cube.
    glLineWidth(1.5f);
    glColor4f(1.0f, 1.0f, 1.0f, 0.55f);

    glBegin(GL_LINES);

    // bottom square
    glVertex3f(-s, -s, -s); glVertex3f(s, -s, -s);
    glVertex3f(s, -s, -s); glVertex3f(s, -s, s);
    glVertex3f(s, -s, s); glVertex3f(-s, -s, s);
    glVertex3f(-s, -s, s); glVertex3f(-s, -s, -s);

    // top square
    glVertex3f(-s, s, -s); glVertex3f(s, s, -s);
    glVertex3f(s, s, -s); glVertex3f(s, s, s);
    glVertex3f(s, s, s); glVertex3f(-s, s, s);
    glVertex3f(-s, s, s); glVertex3f(-s, s, -s);

    // verticals
    glVertex3f(-s, -s, -s); glVertex3f(-s, s, -s);
    glVertex3f(s, -s, -s); glVertex3f(s, s, -s);
    glVertex3f(s, -s, s); glVertex3f(s, s, s);
    glVertex3f(-s, -s, s); glVertex3f(-s, s, s);

    glEnd();

    glLineWidth(1.0f);
    glDisable(GL_BLEND);
}

void EuclidRenderer::drawWorkspaceMajorGrid() {
    const float s = kSimHalfBox;

    glUseProgram(0);
    glDisable(GL_TEXTURE_2D);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Sparse internal major grid, similar to EucliGen_ver002.
    glLineWidth(1.0f);
    glColor4f(1.0f, 1.0f, 1.0f, 0.12f);

    glBegin(GL_LINES);

    const int majorEvery = 4;

    for (int i = 0; i <= kGridDim; i++) {
        if ((i % majorEvery) != 0) continue;

        const float v = -s + i * kCellSize;

        // XY plane lines at z = 0
        glVertex3f(-s, v, 0.0f); glVertex3f(s, v, 0.0f);
        glVertex3f(v, -s, 0.0f); glVertex3f(v, s, 0.0f);

        // XZ plane lines at y = 0
        glVertex3f(-s, 0.0f, v); glVertex3f(s, 0.0f, v);
        glVertex3f(v, 0.0f, -s); glVertex3f(v, 0.0f, s);

        // YZ plane lines at x = 0
        glVertex3f(0.0f, -s, v); glVertex3f(0.0f, s, v);
        glVertex3f(0.0f, v, -s); glVertex3f(0.0f, v, s);
    }

    glEnd();

    glLineWidth(1.0f);
    glDisable(GL_BLEND);
}

void EuclidRenderer::drawXYWorkplane(
    int slice,
    bool hoverValid,
    float hoverX,
    float hoverY) {
    const float s = kSimHalfBox;      // should be 2.0f
    const int sliceRange = 64;

    // slice = -64 -> z = -2
    // slice =   0 -> z =  0
    // slice =  64 -> z = +2
    const float z =
        (static_cast<float>(slice) / static_cast<float>(sliceRange)) * s;

    glUseProgram(0);
    glDisable(GL_TEXTURE_2D);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // The scanner plane should not write into the depth buffer.
    // Otherwise it can accidentally occlude the particle/axes.
    glDepthMask(GL_FALSE);

    // Workplane outer border.
    glLineWidth(2.0f);
    glColor4f(0.82f, 0.90f, 1.0f, 0.75f);

    glBegin(GL_LINE_LOOP);
    glVertex3f(-s, -s, z);
    glVertex3f(s, -s, z);
    glVertex3f(s, s, z);
    glVertex3f(-s, s, z);
    glEnd();

    // Dense XY grid lines.
    const int divisions = 16;

    glLineWidth(1.0f);
    glBegin(GL_LINES);

    for (int i = -divisions; i <= divisions; i++) {
        const float t =
            static_cast<float>(i) / static_cast<float>(divisions);

        const float x = t * s;
        const float y = t * s;

        // Major lines every 4 divisions.
        if ((i % 4) == 0) {
            glColor4f(0.75f, 0.85f, 1.0f, 0.45f);
        }
        else {
            glColor4f(0.55f, 0.65f, 0.90f, 0.20f);
        }

        // Lines parallel to Y.
        glVertex3f(x, -s, z);
        glVertex3f(x, s, z);

        // Lines parallel to X.
        glVertex3f(-s, y, z);
        glVertex3f(s, y, z);
    }

    glEnd();

    // Center cross on the active plane.
    glLineWidth(1.5f);
    glColor4f(0.90f, 0.95f, 1.0f, 0.55f);

    glBegin(GL_LINES);
    glVertex3f(-s, 0.0f, z);
    glVertex3f(s, 0.0f, z);

    glVertex3f(0.0f, -s, z);
    glVertex3f(0.0f, s, z);
    glEnd();

    // Mouse hover crosshair on the workplane.
    if (hoverValid) {
        const float h = 0.08f;

        glLineWidth(2.0f);
        glColor4f(1.0f, 1.0f, 0.15f, 0.95f);

        glBegin(GL_LINES);
        glVertex3f(hoverX - h, hoverY, z + 0.01f);
        glVertex3f(hoverX + h, hoverY, z + 0.01f);

        glVertex3f(hoverX, hoverY - h, z + 0.01f);
        glVertex3f(hoverX, hoverY + h, z + 0.01f);
        glEnd();
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glLineWidth(1.0f);
}

bool EuclidRenderer::checkShader(
    GLuint shader,
    const char* label) {
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        printf("%s shader compile failed:\n%s\n", label, log);
        return false;
    }

    return true;
}

void EuclidRenderer::displayDiagnosticParticles(
    const std::vector<DiagnosticParticleVisual>& particles) {
    if (particles.empty() || !m_program0 || !m_particlePosVBO ||
        !m_particleRadVBO || !m_particleColorVBO) return;

    GLint previousProgram = 0, previousBuffer = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &previousProgram);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &previousBuffer);
    GLint enabled, size, type, normalized, stride, buffer;
    void* pointer = nullptr;
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_SIZE, &size);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_TYPE, &type);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED, &normalized);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_STRIDE, &stride);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &buffer);
    glGetVertexAttribPointerv(1, GL_VERTEX_ATTRIB_ARRAY_POINTER, &pointer);
    glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_COLOR_BUFFER_BIT |
        GL_DEPTH_BUFFER_BIT | GL_TEXTURE_BIT | GL_POINT_BIT);
    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);

    const GLint intensityLocation = m_program1 ?
        glGetUniformLocation(m_program1, "emissiveIntensity") : -1;
    GLfloat previousIntensity = 1.0f;
    if (intensityLocation >= 0) glGetUniformfv(m_program1, intensityLocation, &previousIntensity);

    // All opaque spheres precede all additive passes, retaining mutual occlusion.
    // Dedicated one-sphere VBOs avoid touching CUDA or the base population buffers.
    for (const auto& particle : particles) {
        const float position[4]{particle.position.x, particle.position.y, particle.position.z, 1.0f};
        drawParticleSphere(m_program0, position, particle.radius, &particle.color.x, false);
    }
    if (m_program1) {
        for (const auto& particle : particles) {
            if (particle.emissiveIntensity <= 0.0f) continue;
            const float position[4]{particle.position.x, particle.position.y, particle.position.z, 1.0f};
            glUseProgram(m_program1);
            if (intensityLocation >= 0) glUniform1f(intensityLocation, particle.emissiveIntensity);
            drawParticleSphere(m_program1, position, particle.radius, &particle.color.x, true);
        }
        glUseProgram(m_program1);
        if (intensityLocation >= 0) glUniform1f(intensityLocation, previousIntensity);
    }
    glPopClientAttrib();
    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(buffer));
    glVertexAttribPointer(1, size, static_cast<GLenum>(type), static_cast<GLboolean>(normalized), stride, pointer);
    if (enabled) glEnableVertexAttribArray(1); else glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(previousBuffer));
    glPopAttrib();
    glUseProgram(static_cast<GLuint>(previousProgram));
}
