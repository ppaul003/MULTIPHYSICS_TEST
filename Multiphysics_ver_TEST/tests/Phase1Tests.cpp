#include "VoxelField3D.h"
#include "DebugElectrodynamics.h"
#include "CameraEM.h"
#include "TheArbiterEM.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
    unsigned checks = 0;
    void check(bool condition, const char* message) {
        ++checks;
        if (!condition) throw std::runtime_error(message);
    }
    bool closeValue(double a, double b, double absolute = 1e-6) { return std::abs(a-b) <= absolute; }
    bool nearVec(const glm::vec3& a, const glm::vec3& b, double absolute = 1e-5) {
        return vectorMagnitude(a-b) <= absolute;
    }
    void fieldsAndCurl() {
        SpatialVoxelGrid3D grid;
        ScalarField3D scalar(grid);
        VectorField3D field(grid);
        check(scalar.size() == 512 && field.size() == 512, "512 field samples");
        for (unsigned id=0; id<512; ++id) {
            SpatialVoxelRegion cell;
            check(grid.region(id, cell), "valid region");
            scalar.set(cell.index, static_cast<double>(id));
            check(scalar.get(id) == id, "index/flat roundtrip");
            field.set(id, glm::vec3(-cell.center.y, cell.center.x, 0));
        }
        const auto result = curl(field);
        for (unsigned id=0; id<512; ++id)
            check(nearVec(result.get(id), glm::vec3(0,0,2)), "analytic curl including one-sided boundaries");
        computeCurl(field, field);
        check(nearVec(field.get(0), glm::vec3(0,0,2)), "in-place curl safe");
        field.clear(glm::vec3(1e-26f,0,0));
        check(field.maxMagnitude()>0 && closeValue(field.maxMagnitude()/1e-26,1,1e-5), "tiny SI B magnitude does not underflow");
        scalar.clear(-3.0);
        check(scalar.maxMagnitude()==3.0, "signed scalar magnitude");
        bool threw=false;
        try { scalar.get(glm::ivec3(-1,0,0)); } catch (const std::out_of_range&) { threw=true; }
        check(threw, "invalid grid index rejected");
        grid.dimensions=glm::ivec3(1);
        field.initialize(grid); field.clear(glm::vec3(1));
        check(nearVec(curl(field).get(0),glm::vec3(0)), "single cell derivative zero");
        grid.dimensions=glm::ivec3(4,3,2); grid.voxelEdgeM=.25f;
        field.initialize(grid);
        for(unsigned id=0; id<grid.voxelCount(); ++id) {
            SpatialVoxelRegion cell; grid.region(id,cell);
            field.set(cell.index,glm::vec3(cell.center.z,cell.center.x,cell.center.y));
        }
        const auto allComponents=curl(field);
        for(unsigned id=0; id<grid.voxelCount(); ++id)
            check(nearVec(allComponents.get(id),glm::vec3(1)), "curl axes and physical spacing");
    }
    struct Fields {
        VectorField3D E,B,J;
        ScalarField3D rho;
        explicit Fields(const SpatialVoxelGrid3D& grid): E(grid),B(grid),J(grid),rho(grid) {}
    };
    void projectilesAndSigns() {
        SpatialVoxelGrid3D grid;
        DebugElectrodynamics electron,ion,neutral;
        Fields e(grid),i(grid),n(grid);
        const glm::vec3 origin(.1f,.1f,0), direction(0,0,1);
        check(electron.fire(DebugProjectileSpecies::Electron,origin,direction,grid)==DebugElectrodynamics::FireResult::Fired,"fire electron");
        ion.fire(DebugProjectileSpecies::ArgonIon,origin,direction,grid);
        neutral.fire(DebugProjectileSpecies::ArgonNeutral,origin,direction,grid);
        electron.populateFields(grid,e.E,e.B,e.J,e.rho);
        ion.populateFields(grid,i.E,i.B,i.J,i.rho);
        neutral.populateFields(grid,n.E,n.B,n.J,n.rho);
        double totalCharge=0;
        for(unsigned id=0;id<512;++id) {
            check(nearVec(e.E.get(id),-i.E.get(id),1e-18),"E opposite charge signs");
            check(nearVec(e.B.get(id),-i.B.get(id),1e-34),"B opposite charge signs");
            check(closeValue(e.rho.get(id),-i.rho.get(id),1e-30),"rho opposite signs");
            totalCharge+=e.rho.get(id)*.125;
            if(e.rho.get(id)!=0) check(e.rho.get(id)<0 && e.J.get(id).z<0,"electron J conventional current opposes velocity");
            SpatialVoxelRegion cell; grid.region(id,cell);
            check(glm::dot(glm::dvec3(e.E.get(id)),glm::dvec3(cell.center-origin)) < 0,"negative E points inward");
        }
        check(closeValue(totalCharge,-1.602176634e-19,1e-30),"NGP charge sum conserved");
        check(n.E.maxMagnitude()==0 && n.B.maxMagnitude()==0 && n.J.maxMagnitude()==0 && n.rho.maxMagnitude()==0,"neutral fields zero");
        const unsigned rightCell=4+8*(4+8*4);
        check(i.B.get(rightCell).y>0 && e.B.get(rightCell).y<0,"B cross product orientation for +Z velocity");
        electron.fire(DebugProjectileSpecies::ArgonIon,origin,direction,grid);
        electron.populateFields(grid,e.E,e.B,e.J,e.rho);
        check(e.E.maxMagnitude()==0 && e.B.maxMagnitude()==0 && e.J.maxMagnitude()==0 && e.rho.maxMagnitude()==0,"coincident opposite charges cancel");
        auto start=neutral.projectiles().front().position;
        neutral.update(.5,grid);
        check(nearVec(neutral.projectiles().front().position,start+direction*.375f),"neutral ballistic motion");
        neutral.update(100,grid);
        check(neutral.projectiles().empty(),"domain exit removed compactly");
        neutral.populateFields(grid,n.E,n.B,n.J,n.rho);
        check(n.E.maxMagnitude()==0,"exit clears fields");
    }
    void raysAndCapacity() {
        SpatialVoxelGrid3D grid;
        DebugElectrodynamics sim;
        using Result=DebugElectrodynamics::FireResult;
        check(sim.fire(DebugProjectileSpecies::Electron,glm::vec3(0,0,5),glm::vec3(0,0,-2),grid)==Result::Fired,"outside center ray hits");
        check(sim.projectiles().back().position.z<2 && sim.projectiles().back().position.z>1.99f,"spawn inside entry face");
        check(sim.fire(DebugProjectileSpecies::Electron,glm::vec3(3,0,5),glm::vec3(0,0,-1),grid)==Result::MissedDomain,"parallel outside miss");
        check(sim.fire(DebugProjectileSpecies::Electron,glm::vec3(0,0,5),glm::vec3(0,0,1),grid)==Result::MissedDomain,"behind ray miss");
        check(sim.fire(DebugProjectileSpecies::Electron,glm::vec3(0),glm::vec3(0),grid)==Result::InvalidRay,"zero ray rejected");
        sim.clear();
        for(unsigned j=0;j<140;++j) sim.fire(DebugProjectileSpecies::ArgonIon,glm::vec3(0),glm::vec3(1,0,0),grid);
        check(sim.projectiles().size()==128 && sim.firedCount()==140,"capacity bounded with recycling");
        check(sim.projectiles().front().eventId==13,"oldest recycled");
        sim.clear(); check(sim.firedCount()==0 && sim.projectiles().empty(),"clear resets diagnostic session");
    }
    void cameraAndRouting() {
        CameraProcessor camera;
        camera.setBehaviorMode(CameraProcessor::CAM_STANDARD_3D);
        camera.beginTransitionToStandard3D(0);
        glm::vec3 origin,dir;
        check(camera.getCenterViewRay(origin,dir),"camera ray valid without GL context");
        check(nearVec(origin,glm::vec3(0,0,5)) && nearVec(dir,glm::vec3(0,0,-1)),"standard orbit ray");
        camera.orbit(450,150);
        camera.getCenterViewRay(origin,dir);
        check(nearVec(dir,glm::vec3(0,0,-1)),"ray uses displayed lag pose, not target");
        camera.updateLag(); // 9 degrees yaw, 3 degrees pitch
        camera.getCenterViewRay(origin,dir);
        glm::mat4 view=glm::translate(glm::mat4(1),glm::vec3(0,0,-5));
        view=glm::rotate(view,glm::radians(3.0f),glm::vec3(1,0,0));
        view=glm::rotate(view,glm::radians(9.0f),glm::vec3(0,1,0));
        check(nearVec(glm::vec3(view*glm::vec4(origin,1)),glm::vec3(0)),"orbit inverse origin");
        check(nearVec(glm::vec3(view*glm::vec4(dir,0)),glm::vec3(0,0,-1)),"orbit inverse direction");
        const auto orbitOrigin=origin,orbitDir=dir;
        camera.beginFreeView(); camera.getCenterViewRay(origin,dir);
        check(nearVec(origin,orbitOrigin)&&nearVec(dir,orbitDir),"free-view entry continuous");
        camera.moveFree(1,0,.5f); camera.getCenterViewRay(origin,dir);
        check(nearVec(origin,orbitOrigin+orbitDir),"free-view ray follows movement");
        const auto eye=origin;
        camera.lookFree(100,35); camera.getCenterViewRay(origin,dir);
        check(nearVec(origin,eye)&&closeValue(vectorMagnitude(dir),1,1e-5)&&!nearVec(dir,orbitDir),"free look rotates normalized ray with fixed eye");
        camera.endFreeView(); camera.getCenterViewRay(origin,dir);
        check(nearVec(origin,orbitOrigin)&&nearVec(dir,orbitDir),"orbit restored after free view");
        TheArbiter arbiter; KeyboardInput keyboard;
        arbiter.setApplicationLayer(TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE);
        arbiter.setActiveWorkspace(TheArbiter::WorkspaceId::MULTIPHYSICS_SIM);
        for(unsigned char key : {'f','F','v','b','c','g','1','2','3'})
            check(arbiter.routeKeyboard(keyboard.onKey(key,0,0)).hasWorkspaceInput,"debug key routed");
        check(arbiter.routeKeyboard(keyboard.onKey(' ',0,0)).workspaceInput.action==WorkspaceInputAction::TogglePause,"space pause retained");
        check(arbiter.routeKeyboard(keyboard.onKey('q',0,0)).workspaceInput.action==WorkspaceInputAction::Back,"Q back retained");
        check(arbiter.routeKeyboard(keyboard.onKey(27,0,0)).arbiterCommand==TheArbiter::ArbiterCommand::CMD_EXIT,"escape retained");
        arbiter.setActiveWorkspace(TheArbiter::WorkspaceId::PARTICLE_SIMULATION);
        check(!arbiter.routeKeyboard(keyboard.onKey('f',0,0)).hasWorkspaceInput,"other cartridge unchanged");
    }
}
int main() {
    try {
        fieldsAndCurl(); projectilesAndSigns(); raysAndCapacity(); cameraAndRouting();
        std::cout << "PASS: " << checks << " field, curl, electrodynamics, ray, capacity, camera and input checks\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
