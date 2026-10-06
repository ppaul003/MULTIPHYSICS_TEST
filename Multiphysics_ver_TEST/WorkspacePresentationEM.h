#ifndef NDMSM_WORKSPACE_PRESENTATION_EM_H
#define NDMSM_WORKSPACE_PRESENTATION_EM_H

#include <string>
#include <vector>

enum class WorkspaceStatusTone {
    Neutral = 0,
    Ready,
    Warning,
    Transition
};

enum class WorkspacePanelLayout {
    Main = 0,
    SubLayer
};

struct WorkspacePanelRow {
    std::string label;
    std::string value;
    bool selectable = false;
    bool selected = false;
};

struct WorkspacePanelSection {
    std::string heading;
    std::vector<WorkspacePanelRow> rows;
};

struct WorkspaceRuntimeStatus {
    bool visible = false;

    std::string titleLine;
    std::string contextLine;
    std::string objectLine;
    std::string helpLine;
    std::vector<std::string> detailLines;

    WorkspaceStatusTone objectTone = WorkspaceStatusTone::Neutral;
};

struct WorkspaceCapabilityLine {
    std::string text;
    WorkspaceStatusTone tone = WorkspaceStatusTone::Neutral;
};

struct WorkspacePresentation {
    bool panelVisible = false;
    bool statusBlink = false;
    bool frameBlink = false;

    WorkspaceStatusTone frameTone = WorkspaceStatusTone::Neutral;
    WorkspaceStatusTone statusTone = WorkspaceStatusTone::Neutral;
    WorkspacePanelLayout panelLayout = WorkspacePanelLayout::Main;

    std::string workspaceName;
    std::string layerLabel;
    std::string subLayerLabel;
    std::string statusLine;

    std::vector<WorkspacePanelSection> sections;
    std::vector<std::string> postStatusLines;
    std::vector<WorkspaceCapabilityLine> capabilityLines;

    std::string footerLine1;
    std::string footerLine2;

    WorkspaceRuntimeStatus runtimeStatus;
};

#endif
