#include "EuclidEngineEM.h"

int main(int argc, char** argv) {
	EuclidEngine engine;
	if (!engine.init(argc, argv)) return 1;

	engine.run();
	return 0;
}