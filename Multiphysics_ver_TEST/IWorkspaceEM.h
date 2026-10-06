#ifndef NDMSM_IWORKSPACE_EM_H
#define NDMSM_IWORKSPACE_EM_H

#include "WorkspaceContextEM.h"
#include "WorkspaceInputEM.h"
#include "WorkspacePresentationEM.h"
#include "WorkspaceMenuEM.h"

class IWorkspace {
public:
    virtual ~IWorkspace() = default;

    virtual bool initialize(WorkspaceServices& services) = 0;
    virtual void enter(WorkspaceServices& services) = 0;
    virtual void exit(WorkspaceServices& services) = 0;

    virtual void update(
        const WorkspaceFrameContext& frame,
        WorkspaceServices& services) = 0;

    virtual void render(
        const WorkspaceFrameContext& frame,
        WorkspaceServices& services) = 0;

    virtual bool handleInput(
        const WorkspaceInputEvent& input,
        WorkspaceServices& services) = 0;

    virtual WorkspacePresentation buildPresentation() const = 0;

    virtual WorkspaceMenuPresentation buildMenu() const {
        return WorkspaceMenuPresentation{};
    }

    virtual bool handleMenuCommand(int command, WorkspaceServices& services) {
        (void)command;
        (void)services;
        return false;
    }

    virtual bool handleInputRelease(
        const WorkspaceInputEvent& input, WorkspaceServices& services) {
        (void)input;
        (void)services;
        return false;
    }

    virtual bool handlePointerInput(
        const WorkspacePointerEvent& input, WorkspaceServices& services) {
        (void)input;
        (void)services;
        return false;
    }

    virtual void cancelInput(WorkspaceServices& services) {
        (void)services;
    }

    virtual void renderOverlay(
        const WorkspaceFrameContext& frame, WorkspaceServices& services) {
        (void)frame;
        (void)services;
    }
};

#endif
