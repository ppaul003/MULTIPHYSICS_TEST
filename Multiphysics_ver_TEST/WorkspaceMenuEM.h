#ifndef WORKSPACE_MENU_H
#define WORKSPACE_MENU_H

#include <string>
#include <vector>

struct WorkspaceMenuItem {
	std::string label;
	int command = 0;
	bool enabled = true;
};

struct WorkspaceMenuPresentation {
	std::vector<WorkspaceMenuItem> items;
};

#endif
