#ifndef NDMSM_ARBITER_EM_H
#define NDMSM_ARBITER_EM_H

#include "InteractionsEM.h"
#include "WorkspaceInputEM.h"

class TheArbiter {
public:
    enum class ApplicationLayer {
        GLOBAL_SHELL = 0,
        DOMAIN_SELECTION,
        WORKSPACE_CONFIGURATION,
        ACTIVE_WORKSPACE
    };

    enum class UnitMeasurement {
        METRIC = 0,
        IMPERIAL
    };

    enum class WorkspaceDomain {
        NONE = 0,
        GRID_2D,
        GRID_3D,
        MULPHY_SIM
    };

    enum class WorkspaceId {
        NONE = 0,

        DIAGNOSTIC,

        GRAPH_2D,
        HEAT_2D,

        GRAPH_3D,
        ANN_DESIGN,

        PARTICLE_SIMULATION,
        MULTIPHYSICS_SIM
    };

    enum class NavigationRequestType {
        NONE = 0,
        ENTER_DOMAIN,
        RETURN_GLOBAL_SHELL
    };

    enum class ArbiterCommand {
        CMD_NONE = 0,
        CMD_EXIT
    };

    struct NavigationRequest {
        NavigationRequestType type = NavigationRequestType::NONE;
        WorkspaceDomain domain = WorkspaceDomain::NONE;
    };

    struct ArbiterResult {
        ArbiterCommand arbiterCommand = ArbiterCommand::CMD_NONE;
        bool handled = false;
        bool hasWorkspaceInput = false;
        WorkspaceInputEvent workspaceInput;
    };

    struct NavigationState {
        ApplicationLayer layer = ApplicationLayer::GLOBAL_SHELL;
        WorkspaceDomain domain = WorkspaceDomain::NONE;
        WorkspaceId workspace = WorkspaceId::DIAGNOSTIC;
    };

public:
    TheArbiter();

    ArbiterResult routeKeyboard(const KeyboardInput::KeyEvent& event) const;
    WorkspacePointerEvent translateMouseButton(int button, int state, int x, int y) const;
    WorkspacePointerEvent translateMouseMotion(int x, int y, int dx, int dy) const;

    void setApplicationLayer(ApplicationLayer layer) { m_navigation.layer = layer; }
    ApplicationLayer getApplicationLayer() const { return m_navigation.layer; }

    void setWorkspaceDomain(WorkspaceDomain domain) { m_navigation.domain = domain; }
    WorkspaceDomain getWorkspaceDomain() const { return m_navigation.domain; }

    void setActiveWorkspace(WorkspaceId workspace) { m_navigation.workspace = workspace; }
    WorkspaceId getActiveWorkspace() const { return m_navigation.workspace; }

    void requestEnterDomain(WorkspaceDomain domain);
    void requestReturnToGlobalShell(WorkspaceDomain domain);

    bool hasNavigationRequest() const { return m_navigationRequest.type != NavigationRequestType::NONE; }

    NavigationRequest takeNavigationRequest();

    bool isGlobalShell() const;
    bool isDomainSelection() const;
    bool isWorkspaceLayer() const;

    void setUnitMeasurement(UnitMeasurement unit) { m_unitMeasurement = unit; }
    UnitMeasurement getUnitMeasurement() const { return m_unitMeasurement; }

private:
    NavigationState m_navigation;
    NavigationRequest m_navigationRequest;

    UnitMeasurement m_unitMeasurement =
        UnitMeasurement::METRIC;
};

#endif
