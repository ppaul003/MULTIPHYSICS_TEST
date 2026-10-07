#include "VoxelField3D.h"
#include "RuntimeSubLayerTraversal.h"
#include "DebugElectrodynamics.h"
#include "FieldDebugRenderer.h"
#include "SimulationPreset.h"
#include "CameraEM.h"
#include "TheArbiterEM.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <iostream>
#include <limits>
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
    void independentFieldGrid() {
        for (int box : {4,8,16,32}) {
            const auto* preset=findSimulationPreset(box);
            const int fieldDim=box<16 ? 8 : 16;
            check(preset && preset->fieldGridDim==fieldDim,"independent preset field resolution");
            check(preset->collisionGridDim==(box<16 ? 64 : 128),"collision mapping preserved");
            SpatialVoxelGrid3D spawnGrid,fieldGrid;
            spawnGrid.origin=fieldGrid.origin=glm::vec3(-box*.5f);
            spawnGrid.voxelEdgeM=box/8.0f;
            fieldGrid.dimensions=glm::ivec3(preset->fieldGridDim);
            fieldGrid.voxelEdgeM=static_cast<float>(box)/preset->fieldGridDim;
            check(spawnGrid.voxelCount()==512,"spawn base stays 8 cubed");
            check(fieldGrid.voxelCount()==(box<16 ? 512u : 4096u),"field grid sample count");
            Fields sampled(fieldGrid);
            check(sampled.E.size()==fieldGrid.voxelCount(),"field allocation follows field geometry");
            SpawnDensityRegionGrid3D selection;
            check(selection.regionCount(spawnGrid)==64 && selection.selectionCount(spawnGrid)==65,
                "spawn-density selection count independent of field resolution");
            SpawnDensityRegion3D first,last;
            check(selection.region(spawnGrid,0,first) && selection.region(spawnGrid,63,last),"spawn region endpoints");
            check(nearVec(first.minimum,glm::vec3(-box*.5f)) && nearVec(last.maximum,glm::vec3(box*.5f)),
                "spawn regions still span whole domain");
            check(nearVec(first.center,glm::vec3(-box*.375f)),"voxel-center spawn geometry preserved");
        }
    }
    void diagnosticSampling() {
        SpatialVoxelGrid3D grid;
        grid.dimensions=glm::ivec3(9,5,5);
        grid.voxelEdgeM=1.0f;
        grid.origin=glm::vec3(-4.5f,-2.5f,-2.5f);
        auto id=[](int x,int y,int z) { return static_cast<unsigned>((x+4)+9*((y+2)+5*(z+2))); };
        auto fireAt=[&](DebugElectrodynamics& sim,DebugProjectileSpecies species,
            const glm::vec3& position,const glm::vec3& direction=glm::vec3(0,0,1)) {
            // fire() offsets an interior origin slightly along the ray.
            const auto origin=position-direction*(grid.voxelEdgeM*1.0e-4f);
            check(sim.fire(species,origin,direction,grid)==DebugElectrodynamics::FireResult::Fired,
                "diagnostic sampling source fired");
        };
        DebugElectrodynamics positive,negative,pair;
        positive.setSpeedMps(0); negative.setSpeedMps(0); pair.setSpeedMps(0);
        fireAt(positive,DebugProjectileSpecies::ArgonIon,glm::vec3(-1,0,0));
        fireAt(negative,DebugProjectileSpecies::Electron,glm::vec3(1,0,0));
        fireAt(pair,DebugProjectileSpecies::ArgonIon,glm::vec3(-1,0,0));
        fireAt(pair,DebugProjectileSpecies::Electron,glm::vec3(1,0,0));
        Fields p(grid),n(grid),combined(grid),uniform(grid);
        std::vector<DiagnosticFieldSample> pmeta,nmeta,metadata;
        positive.populateFields(grid,p.E,p.B,p.J,p.rho,&pmeta);
        negative.populateFields(grid,n.E,n.B,n.J,n.rho,&nmeta);
        pair.populateFields(grid,combined.E,combined.B,combined.J,combined.rho,&metadata);
        pair.populateFields(grid,uniform.E,uniform.B,uniform.J,uniform.rho);
        check(metadata.size()==grid.voxelCount(),"one visualization sample per field voxel");
        check(p.B.maxMagnitude()==0 && n.B.maxMagnitude()==0,"stationary charges have no B");
        check(vectorMagnitude(positive.projectiles().front().velocity)==0 &&
            vectorMagnitude(negative.projectiles().front().velocity)==0,"stationary source velocity stays zero");
        for(unsigned cell=0;cell<grid.voxelCount();++cell) {
            SpatialVoxelRegion voxel; grid.region(cell,voxel);
            check(pmeta[cell].electricClass==ElectricGlyphClass::PositiveSource,"positive E source metadata");
            if (vectorMagnitude(n.E.get(cell)) > 0.0)
                check(nmeta[cell].electricClass==ElectricGlyphClass::NegativeSource,"negative E source metadata");
            check(vectorMagnitude(pmeta[cell].electricNegative)==0 && vectorMagnitude(nmeta[cell].electricPositive)==0,
                "single-source polarity channels separated");
            check(glm::dot(glm::dvec3(p.E.get(cell)),glm::dvec3(voxel.center-positive.projectiles()[0].position))>=0,
                "stationary positive E points outward");
            check(glm::dot(glm::dvec3(n.E.get(cell)),glm::dvec3(voxel.center-negative.projectiles()[0].position))<=0,
                "stationary negative E points inward");
            check(nearVec(combined.E.get(cell),p.E.get(cell)+n.E.get(cell),1e-15),"dipole remains ordinary superposition");
            check(nearVec(combined.E.get(cell),metadata[cell].electricPositive+metadata[cell].electricNegative,1e-15),
                "polarity contributions sum to physical E");
            check(nearVec(combined.E.get(cell),uniform.E.get(cell),0),"metadata does not alter physical E");
            check(nearVec(combined.B.get(cell),uniform.B.get(cell),0) &&
                nearVec(combined.J.get(cell),uniform.J.get(cell),0) && combined.rho.get(cell)==uniform.rho.get(cell),
                "metadata does not alter B or NGP deposition");
        }
        check(metadata[id(0,0,0)].electricClass==ElectricGlyphClass::DipoleBridge,"near opposite pair midpoint is bridge");
        check(combined.E.get(id(0,0,0)).x>0 && vectorMagnitude(combined.E.get(id(0,0,0)))>p.E.maxMagnitude()*.1,
            "dipole midpoint reinforces toward negative sink");
        check(metadata[id(-2,0,0)].electricClass==ElectricGlyphClass::PositiveSource &&
            metadata[id(2,0,0)].electricClass==ElectricGlyphClass::NegativeSource,"outside bridge uses local dominant source");
        check(metadata[id(0,1,0)].electricClass!=ElectricGlyphClass::DipoleBridge,"off-axis sample outside narrow corridor");
        check(metadata[id(-1,0,0)].electricClass!=ElectricGlyphClass::DipoleBridge &&
            metadata[id(1,0,0)].electricClass!=ElectricGlyphClass::DipoleBridge,"source endpoints are not a bridge");

        DebugElectrodynamics farPair;
        farPair.setSpeedMps(0);
        fireAt(farPair,DebugProjectileSpecies::ArgonIon,glm::vec3(-3,0,0));
        fireAt(farPair,DebugProjectileSpecies::Electron,glm::vec3(3,0,0));
        farPair.populateFields(grid,combined.E,combined.B,combined.J,combined.rho,&metadata);
        check(metadata[id(0,0,0)].electricClass!=ElectricGlyphClass::DipoleBridge && combined.E.get(id(0,0,0)).x>0,
            "distant pair stays physical but does not create bridge classification");

        DebugElectrodynamics unequalContributions;
        unequalContributions.setSpeedMps(0);
        fireAt(unequalContributions,DebugProjectileSpecies::ArgonIon,glm::vec3(-1.3f,0,0));
        fireAt(unequalContributions,DebugProjectileSpecies::Electron,glm::vec3(1,0,0));
        unequalContributions.populateFields(grid,combined.E,combined.B,combined.J,combined.rho,&metadata);
        check(metadata[id(-1,0,0)].electricClass==ElectricGlyphClass::PositiveSource,
            "geometric bridge rejected when negative contribution is negligible");

        DebugElectrodynamics reversedNet;
        reversedNet.setSpeedMps(0);
        fireAt(reversedNet,DebugProjectileSpecies::ArgonIon,glm::vec3(-1,0,0));
        fireAt(reversedNet,DebugProjectileSpecies::Electron,glm::vec3(1,0,0));
        fireAt(reversedNet,DebugProjectileSpecies::ArgonIon,glm::vec3(.25f,0,0));
        reversedNet.populateFields(grid,combined.E,combined.B,combined.J,combined.rho,&metadata);
        check(combined.E.get(id(0,0,0)).x<0 && metadata[id(0,0,0)].electricClass!=ElectricGlyphClass::DipoleBridge,
            "third source reversing net direction prevents fake purple bridge");

        DebugElectrodynamics movingPositive,movingNegative;
        movingPositive.setSpeedMps(.5f); movingNegative.setSpeedMps(.5f);
        fireAt(movingPositive,DebugProjectileSpecies::ArgonIon,glm::vec3(-1,0,0),glm::vec3(1,0,0));
        fireAt(movingNegative,DebugProjectileSpecies::Electron,glm::vec3(-1,0,0),glm::vec3(1,0,0));
        movingPositive.populateFields(grid,p.E,p.B,p.J,p.rho,&pmeta);
        movingNegative.populateFields(grid,n.E,n.B,n.J,n.rho,&nmeta);
        check(p.B.get(id(0,1,0)).z>0 && n.B.get(id(0,1,0)).z<0,"moving charge B circulation follows sign");
        check(nearVec(pmeta[id(0,1,0)].magneticPositive,p.B.get(id(0,1,0)),0) &&
            vectorMagnitude(pmeta[id(0,1,0)].magneticNegative)==0,"positive B provenance retained");
        check(nearVec(nmeta[id(0,1,0)].magneticNegative,n.B.get(id(0,1,0)),0) &&
            vectorMagnitude(nmeta[id(0,1,0)].magneticPositive)==0,"negative B provenance retained");
        const double before=vectorMagnitude(p.E.get(id(0,0,0)));
        const auto directionBefore=glm::normalize(p.E.get(id(0,1,0)));
        const auto beforePosition=movingPositive.projectiles()[0].position;
        const auto velocity=movingPositive.projectiles()[0].velocity;
        movingPositive.update(1,grid);
        movingPositive.populateFields(grid,p.E,p.B,p.J,p.rho,&pmeta);
        check(nearVec(movingPositive.projectiles()[0].position,beforePosition+velocity) &&
            nearVec(movingPositive.projectiles()[0].velocity,velocity,0),"field sampling leaves ballistic trajectory unchanged");
        check(vectorMagnitude(p.E.get(id(0,0,0)))>before,"approaching charge increases sampled magnitude");
        check(!nearVec(glm::normalize(p.E.get(id(0,1,0))),directionBefore),"moving source changes fixed-voxel field direction");
        movingPositive.update(1,grid); // source crosses the sampled voxel center
        movingPositive.populateFields(grid,p.E,p.B,p.J,p.rho,&pmeta);
        for(unsigned cell=0;cell<grid.voxelCount();++cell)
            check(std::isfinite(vectorMagnitude(p.E.get(cell))) && std::isfinite(vectorMagnitude(p.B.get(cell))),
                "softened near-source fields stay finite");

        DebugElectrodynamics movingPair;
        movingPair.setSpeedMps(1);
        fireAt(movingPair,DebugProjectileSpecies::ArgonIon,glm::vec3(-1,0,0),glm::vec3(0,1,0));
        fireAt(movingPair,DebugProjectileSpecies::Electron,glm::vec3(1,0,0),glm::vec3(0,1,0));
        movingPair.populateFields(grid,combined.E,combined.B,combined.J,combined.rho,&metadata);
        check(metadata[id(0,0,0)].electricClass==ElectricGlyphClass::DipoleBridge,"moving pair initially bridges center sample");
        movingPair.update(1,grid);
        movingPair.populateFields(grid,combined.E,combined.B,combined.J,combined.rho,&metadata);
        check(metadata[id(0,0,0)].electricClass!=ElectricGlyphClass::DipoleBridge &&
            metadata[id(0,1,0)].electricClass==ElectricGlyphClass::DipoleBridge,"bridge follows moving sources between fixed samples");
        for(unsigned cell=0;cell<grid.voxelCount();++cell)
            check(nearVec(combined.B.get(cell),metadata[cell].magneticPositive+metadata[cell].magneticNegative,1e-32),
                "mixed-sign B channels preserve physical superposition");
        movingPair.clear();
        movingPair.populateFields(grid,combined.E,combined.B,combined.J,combined.rho,&metadata);
        check(combined.E.maxMagnitude()==0 && combined.B.maxMagnitude()==0,"clearing sources clears total fields");
        for(const auto& sample:metadata)
            check(vectorMagnitude(sample.electricPositive)==0 && vectorMagnitude(sample.electricNegative)==0 &&
                vectorMagnitude(sample.magneticPositive)==0 && vectorMagnitude(sample.magneticNegative)==0 &&
                sample.electricClass!=ElectricGlyphClass::DipoleBridge,"clearing sources clears display metadata");

        grid.voxelEdgeM=2.0f; grid.origin*=2.0f;
        DebugElectrodynamics scaledPair;
        scaledPair.setSpeedMps(0);
        fireAt(scaledPair,DebugProjectileSpecies::ArgonIon,glm::vec3(-3,0,0));
        fireAt(scaledPair,DebugProjectileSpecies::Electron,glm::vec3(3,0,0));
        scaledPair.populateFields(grid,combined.E,combined.B,combined.J,combined.rho,&metadata);
        check(metadata[id(0,0,0)].electricClass==ElectricGlyphClass::DipoleBridge,
            "pair threshold scales with field voxel edge instead of fixed world distance");
    }
    void glyphScaling() {
        VectorFieldRenderSettings settings;
        settings.scale=VectorGlyphScale::LogMagnitude;
        settings.referenceMagnitude=1.44e-8;
        const auto length=[&](double magnitude,double frameMaximum=1e-8) {
            return FieldGlyphDisplay::vectorLengthFraction(magnitude,frameMaximum,settings);
        };
        check(length(1e-10)<length(1e-9) && length(1e-9)<length(1e-8),"log E glyph grows with SI magnitude");
        check(closeValue(length(1e-9,1e-8),length(1e-9,1e-4),0),"stable reference avoids frame normalization");
        check(length(1e10)<=1 && length(0)==0 && length(-1)==0 &&
            length((std::numeric_limits<double>::infinity)())==0 &&
            length((std::numeric_limits<double>::quiet_NaN)())==0,"E glyph scale bounds and finite handling");
        using FieldGlyphDisplay::velocityLengthInVoxels;
        check(velocityLengthInVoxels(0)==0 && velocityLengthInVoxels(-1)==0,"zero velocity has no glyph");
        check(velocityLengthInVoxels(.1)<velocityLengthInVoxels(.75) &&
            velocityLengthInVoxels(.75)<velocityLengthInVoxels(2),"velocity glyph grows with speed");
        check(velocityLengthInVoxels(1e20)<=FieldGlyphDisplay::kMaximumVelocityLengthInVoxels &&
            velocityLengthInVoxels((std::numeric_limits<double>::infinity)())==0 &&
            velocityLengthInVoxels((std::numeric_limits<double>::quiet_NaN)())==0,"velocity glyph bounds and finite handling");
    }
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
    void runtimeSubLayers() {
        RuntimeSubLayerTraversal traversal;
        using Result = RuntimeSubLayerTraversal::ActivationResult;
        check(!traversal.panelOpen() && traversal.selectedRow() == 0 &&
            traversal.activeLayer() == RuntimeSubLayer::SubLayer0, "sub-layer defaults");
        WorkspaceInputEvent input;
        input.action = WorkspaceInputAction::Next;
        check(!traversal.handlePanelInput(input), "hidden panel does not consume navigation");
        traversal.showPanel();
        for (int layer = 0; layer < 4; ++layer) {
            check(static_cast<int>(traversal.activeLayer()) == layer, "forward traversal");
            const int count = layer == 0 ? 4 : 5;
            check(traversal.rowCount() == count, "per-layer row count");
            traversal.moveCursor(-1);
            check(traversal.selectedRow() == count - 1, "up wraps");
            traversal.moveCursor(1);
            for (int row = 0; row < 3; ++row) {
                check(traversal.activateSelected() == Result::None, "placeholder activation no-op");
                traversal.moveCursor(1);
            }
            traversal.togglePanel(); traversal.togglePanel();
            check(traversal.selectedRow() == 3 && static_cast<int>(traversal.activeLayer()) == layer,
                "TAB retains layer and cursor");
            input.action = WorkspaceInputAction::Activate; input.repeated = true;
            check(traversal.handlePanelInput(input) && static_cast<int>(traversal.activeLayer()) == layer,
                "held E does not retrigger transitions");
            input.repeated = false;
            if (layer > 0) {
                traversal.moveCursor(1);
                check(traversal.activateSelected() == Result::ChangedLayer &&
                    static_cast<int>(traversal.activeLayer()) == layer - 1 && traversal.selectedRow() == 0,
                    "previous transition resets cursor");
                for (int step = 0; step < 3; ++step) traversal.moveCursor(1);
                traversal.activateSelected();
                for (int step = 0; step < 3; ++step) traversal.moveCursor(1);
            }
            check(traversal.activateSelected() == (layer == 3 ? Result::ExitToRuntime : Result::ChangedLayer),
                "activation result distinguishes exit");
        }
        check(!traversal.panelOpen() && traversal.activeLayer() == RuntimeSubLayer::SubLayer0 &&
            traversal.selectedRow() == 0, "exit clears traversal only");
        traversal.showPanel();
        for (auto action : {WorkspaceInputAction::TogglePause, WorkspaceInputAction::Back}) {
            input.action = action;
            check(!traversal.handlePanelInput(input), "Space and Q belong to workspace");
        }
        for (auto action : {WorkspaceInputAction::Decrease, WorkspaceInputAction::Increase}) {
            input.action = action;
            check(traversal.handlePanelInput(input) && traversal.selectedRow() == 0, "A/D consumed as no-op");
        }
        traversal.moveCursor(1); traversal.reset();
        check(!traversal.panelOpen() && traversal.selectedRow() == 0, "new runtime reset");
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
        camera.beginRelativeDistanceScale(2.0f, 1.0f);
        camera.updatePoseTransition(0.5f);
        camera.getCenterViewRay(origin,dir);
        check(nearVec(origin,orbitOrigin*1.5f)&&nearVec(dir,orbitDir),"relative distance scales smoothly without rotating");
        camera.updatePoseTransition(0.5f);
        camera.getCenterViewRay(origin,dir);
        check(nearVec(origin,orbitOrigin*2.0f)&&nearVec(dir,orbitDir),"relative distance expansion preserves view angle");
        camera.beginRelativeDistanceScale(0.5f, 0.8f);
        camera.updatePoseTransition(0.8f);
        camera.getCenterViewRay(origin,dir);
        check(nearVec(origin,orbitOrigin)&&nearVec(dir,orbitDir),"relative distance shrink reverses expansion");
        CameraProcessor menuCamera;
        glm::vec3 menuOrigin,menuDir;
        menuCamera.getCenterViewRay(menuOrigin,menuDir);
        menuCamera.beginRelativeDistanceScale(2.0f,0.8f);
        menuCamera.updatePoseTransition(0.8f);
        menuCamera.getCenterViewRay(origin,dir);
        check(nearVec(origin,menuOrigin*2.0f)&&nearVec(dir,menuDir),"off-axis menu translation scales on every axis");
        TheArbiter arbiter; KeyboardInput keyboard;
        arbiter.setApplicationLayer(TheArbiter::ApplicationLayer::ACTIVE_WORKSPACE);
        arbiter.setActiveWorkspace(TheArbiter::WorkspaceId::ATOMIC_PARTICLES);
        for(unsigned char key : {'f','F','v','b','c','g','1','2','3'})
            check(arbiter.routeKeyboard(keyboard.onKey(key,0,0)).workspaceInput.action==WorkspaceInputAction::RawKey,"Atomic debug key routed");
        check(arbiter.routeKeyboard(keyboard.onKey(' ',0,0)).workspaceInput.action==WorkspaceInputAction::TogglePause,"space pause retained");
        check(arbiter.routeKeyboard(keyboard.onKey('q',0,0)).workspaceInput.action==WorkspaceInputAction::Back,"Q back retained");
        check(arbiter.routeKeyboard(keyboard.onKey(27,0,0)).arbiterCommand==TheArbiter::ArbiterCommand::CMD_EXIT,"escape retained");
        arbiter.setActiveWorkspace(TheArbiter::WorkspaceId::PARTICLE_SIM);
        check(!arbiter.routeKeyboard(keyboard.onKey('f',0,0)).hasWorkspaceInput,"other cartridge unchanged");
    }
}
int main() {
    try {
        fieldsAndCurl(); independentFieldGrid(); diagnosticSampling(); glyphScaling();
        projectilesAndSigns(); raysAndCapacity(); cameraAndRouting(); runtimeSubLayers();
        std::cout << "PASS: " << checks << " field, curl, electrodynamics, ray, capacity, camera, input and sub-layer checks\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
