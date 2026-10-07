#ifndef NDMSM_RUNTIME_SUB_LAYER_TRAVERSAL_H
#define NDMSM_RUNTIME_SUB_LAYER_TRAVERSAL_H

#include "WorkspaceInputEM.h"
#include "WorkspacePresentationEM.h"

enum class RuntimeSubLayer { SubLayer0 = 0, SubLayer1, SubLayer2, SubLayer3, Count };

// Workspace-local navigation only: no simulation, camera, or Arbiter ownership.
class RuntimeSubLayerTraversal {
public:
    enum class ActivationResult { None = 0, ChangedLayer, ExitToRuntime };

    void reset() { m_activeLayer = RuntimeSubLayer::SubLayer0; m_selectedRow = 0; hidePanel(); }
    void togglePanel() { m_panelOpen = !m_panelOpen; }
    void showPanel() { m_panelOpen = true; }
    void hidePanel() { m_panelOpen = false; }
    bool panelOpen() const { return m_panelOpen; }
    RuntimeSubLayer activeLayer() const { return m_activeLayer; }
    int selectedRow() const { return m_selectedRow; }
    int rowCount() const { return m_activeLayer == RuntimeSubLayer::SubLayer0 ? 4 : 5; }

    void moveCursor(int direction) {
        if (direction == 0) return;
        m_selectedRow = (m_selectedRow + (direction < 0 ? -1 : 1) + rowCount()) % rowCount();
    }

    ActivationResult activateSelected() {
        if (m_selectedRow < 3) return ActivationResult::None;
        if (m_selectedRow == 3 && m_activeLayer == RuntimeSubLayer::SubLayer3) {
            reset();
            return ActivationResult::ExitToRuntime;
        }
        m_activeLayer = static_cast<RuntimeSubLayer>(
            static_cast<int>(m_activeLayer) + (m_selectedRow == 3 ? 1 : -1));
        m_selectedRow = 0;
        showPanel();
        return ActivationResult::ChangedLayer;
    }

    // Call after TAB handling, before workspace camera/runtime feature input.
    bool handlePanelInput(const WorkspaceInputEvent& input) {
        if (!panelOpen()) return false;
        switch (input.action) {
        case WorkspaceInputAction::Previous:
            if (!input.repeated) moveCursor(-1);
            return true;
        case WorkspaceInputAction::Next:
            if (!input.repeated) moveCursor(1);
            return true;
        case WorkspaceInputAction::Activate:
            if (!input.repeated) activateSelected();
            return true;
        case WorkspaceInputAction::Decrease:
        case WorkspaceInputAction::Increase:
            return true; // Placeholder values do not modify runtime state.
        default:
            return false; // Space and Q retain their workspace-level meaning.
        }
    }

    std::string layerLabel() const {
        const auto n = std::to_string(static_cast<int>(m_activeLayer));
        return "SUB-LAYER_" + n + " -> SETUP_" + n;
    }
    std::string context(const std::string& workspace) const {
        return workspace + ": " + (panelOpen() ? layerLabel() : "LAYER 3 RUNTIME");
    }
    std::string help() const {
        return panelOpen() ? "TAB: HIDE SUB_LAYER PANEL    W/S: SELECT    E: ACTIVATE"
            : "TAB: SUB_LAYER PANEL DISPLAY";
    }

    WorkspacePresentation buildPresentation(const std::string& heading) const {
        WorkspacePresentation p;
        p.panelLayout = WorkspacePanelLayout::SubLayer;
        p.panelVisible = panelOpen();
        p.workspaceName = heading;
        p.subLayerLabel = layerLabel();
        const int layer = static_cast<int>(m_activeLayer);
        WorkspacePanelSection options;
        options.heading = "SL" + std::to_string(layer) + "_SELECT:";
        for (int row = 0; row < 3; ++row) {
            const auto n = std::to_string(row + 1);
            options.rows.push_back({ "[" + n + "]: OPTION TYPE " + n, "", true, row == m_selectedRow });
        }
        p.sections.push_back(options);
        WorkspacePanelSection navigation;
        navigation.heading = layer == 0 ? "Next Sub-Layer:"
            : layer == 3 ? "EXIT/Previous Sub-Layer(s):" : "Next/Prev Sub-Layer:";
        navigation.rows.push_back({ layer == 3 ? "[4]: EXIT_SUB_LAYERS"
            : "[4]: SUB-LAYER_" + std::to_string(layer + 1), "", true, m_selectedRow == 3 });
        if (layer > 0)
            navigation.rows.push_back({ "[5]: SUB-LAYER_" + std::to_string(layer - 1),
                "", true, m_selectedRow == 4 });
        p.sections.push_back(navigation);
        return p;
    }

private:
    RuntimeSubLayer m_activeLayer = RuntimeSubLayer::SubLayer0;
    int m_selectedRow = 0;
    bool m_panelOpen = false;
};

#endif
