#include "AtomicInitialConditions.h"
#include "TextEntry.h"
#include "VoxelField3D.h"
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

    void initialConditions() {
        using namespace AtomicInitialization;
        double density = 0;
        check(numberDensity(1,14,density) && density==1e14,"scientific notation");
        check(numberDensity(1.5,14,density) && density==1.5e14,"decimal notation");
        check(numberDensity(2,-3,density) && closeValue(density,.002,1e-16),"negative exponent");
        check(!numberDensity(-1,0,density) && !numberDensity(1,309,density),"density bounds");
        check(!numberDensity(std::numeric_limits<double>::infinity(),0,density),"infinite mantissa");
        check(!numberDensity(std::numeric_limits<double>::quiet_NaN(),0,density),"NaN mantissa");
        check(!numberDensity(10,308,density),"density overflow");
        check(closeValue(physicalVolumeM3(32),3.2768e-14,1e-28),"32 micrometer cube");
        Population p; std::string error;
        check(resolve(DensityType::Electron,1,14,.1,32,p,error),"reference population valid");
        check(closeValue(p.expectedHeavyCount,32.768,1e-12) && closeValue(p.expectedNeutralCount,29.4912,1e-12) &&
            closeValue(p.expectedIonCount,3.2768,1e-12) && p.expectedIonCount==p.expectedElectronCount,"reference expected populations");
        const auto reference=p;
        check(resolve(DensityType::Ion,1,14,.1,32,p,error) && p.heavyCount==reference.heavyCount && p.ionCount==reference.ionCount,"ion/electron density equivalence");
        check(resolve(DensityType::Neutral,9,14,.1,32,p,error) && closeValue(p.expectedHeavyCount,32.768,1e-12),"neutral density derivation");
        for (int box : {4,8,16,32}) for (int percent=0;percent<=100;++percent) {
            const double alpha=percent/100.0;
            const auto type=percent==0 ? DensityType::Neutral : DensityType::Electron;
            check(resolve(type,1,14,alpha,box,p,error),"preset/fraction resolves");
            check(p.neutralCount+p.ionCount==p.heavyCount && p.electronCount==p.ionCount,"population identities");
            Population again;
            check(resolve(type,1,14,alpha,box,again,error) && again.heavyCount==p.heavyCount && again.ionCount==p.ionCount,"preview deterministic");
        }
        check(resolve(DensityType::Electron,21000/physicalVolumeM3(32),0,1,32,p,error) && p.heavyCount==21000 &&
            p.ionCount==21000 && p.electronCount==21000 && p.neutralCount==0 && p.activeMarkerCount==42000 &&
            kParticleCapacity-p.activeMarkerCount==7152,"fully ionized reserve");
        check(!resolve(DensityType::Neutral,22000/physicalVolumeM3(32),0,0,32,p,error) && error.find("HEAVY")!=std::string::npos,"heavy limit rejects");
        check(!resolve(DensityType::Ion,30000/physicalVolumeM3(32),0,1,32,p,error) && error.find("CAPACITY")!=std::string::npos,"marker limit rejects");
        for(auto type:{DensityType::Electron,DensityType::Ion})
            check(!resolve(type,1,14,0,32,p,error),"nonzero charge alpha zero rejected");
        check(!resolve(DensityType::Neutral,1,14,1,32,p,error),"nonzero neutral alpha one rejected");
        check(resolve(DensityType::Neutral,0,0,1,32,p,error)&&p.heavyCount==0,"zero neutral endpoint empty");
        check(resolve(DensityType::Electron,0,0,0,32,p,error)&&p.heavyCount==0,"zero charged endpoint empty");
        check(!resolve(DensityType::Electron,1,308,1e-10,32,p,error),"derived overflow rejects before integer conversion");
        std::mt19937 rng(kPopulationRoundingSeed), same(kPopulationRoundingSeed);
        double sum=0;
        for (int i=0;i<20000;++i) {
            double x=stochasticRound(32.768,rng);
            check(x==stochasticRound(32.768,same) && (x==32 || x==33),"fixed seed round reproducible");
            sum+=x;
        }
        check(std::abs(sum/20000-32.768)<.015,"stochastic mean matches expectation");
    }
    void numericTextEntry() {
        TextEntrySession entry;
        auto type=[&](const std::string& text) {for(unsigned char c:text) check(entry.handleRawKey(c)==TextEntryAction::Changed,"numeric character accepted");};
        double real=0; int integer=0; unsigned count=0;
        check(entry.beginNonnegativeReal("mantissa",0),"real editor starts");
        type("1.0");
        check(entry.getBuffer()=="1.0","real text preserved during editing");
        check(entry.handleRawKey(13)==TextEntryAction::Committed && entry.tryGetCommittedReal(real) && real==1 && entry.getNormalizedText()=="1","real normalized on commit");
        check(entry.beginNonnegativeReal("mantissa",1),"real reopen"); type("1.5");
        check(entry.handleRawKey('.')==TextEntryAction::Rejected && entry.handleRawKey('-')==TextEntryAction::Rejected,"invalid mantissa characters");
        check(entry.handleRawKey(13)==TextEntryAction::Committed && entry.tryGetCommittedReal(real) && real==1.5,"decimal commit");
        entry.beginNonnegativeReal("mantissa",1); type("3.2768"); entry.handleRawKey(8);
        check(entry.getBuffer()=="3.276","backspace");
        check(entry.handleRawKey(27)==TextEntryAction::Cancelled && !entry.tryGetCommittedReal(real),"cancel clears result");
        entry.beginNonnegativeReal("mantissa",0); type(std::string(310,'9'));
        check(entry.handleRawKey(13)==TextEntryAction::Rejected && entry.isActive(),"real overflow rejected on commit");
        check(!entry.beginNonnegativeReal("mantissa",std::numeric_limits<double>::infinity()),"nonfinite default rejected");
        for(const auto* text:{"14","-3","0"}) {
            entry.beginSignedInteger("exponent",-308,308,0); type(text);
            check(entry.handleRawKey(13)==TextEntryAction::Committed && entry.tryGetCommittedSigned(integer) && integer==std::stoi(text),"signed exponent commit");
        }
        entry.beginSignedInteger("exponent",-308,308,0); type("309");
        check(entry.handleRawKey(13)==TextEntryAction::Rejected,"exponent bounds");
        entry.beginSignedInteger("exponent",-308,308,0); type("-");
        check(entry.handleRawKey(13)==TextEntryAction::Rejected,"incomplete sign rejected");
        entry.beginUnsignedInteger("count",0,100); type("12");
        check(entry.handleRawKey(13)==TextEntryAction::Committed && entry.tryGetCommittedUnsigned(count) && count==12,"unsigned mode retained");
        entry.beginAssetName("asset","",32); type(" My Asset ");
        check(entry.handleRawKey(13)==TextEntryAction::Committed && entry.getNormalizedText()=="My_Asset","asset mode retained");
        entry.beginGeneralText("text","",32); type("a.b-12");
        check(entry.handleRawKey(13)==TextEntryAction::Committed && entry.getCommittedText()=="a.b-12","general mode retained");
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
            check(selection.regionCount(spawnGrid)==64 && selection.selectionCount(spawnGrid)==66,
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
        for(unsigned char key : {'f','F','v','b','c','g','h','H','1','2','3'})
            check(arbiter.routeKeyboard(keyboard.onKey(key,0,0)).workspaceInput.action==WorkspaceInputAction::RawKey,"Atomic debug key routed");
        check(arbiter.routeKeyboard(keyboard.onKey(' ',0,0)).workspaceInput.action==WorkspaceInputAction::TogglePause,"space pause retained");
        check(arbiter.routeKeyboard(keyboard.onKey('q',0,0)).workspaceInput.action==WorkspaceInputAction::Back,"Q back retained");
        check(arbiter.routeKeyboard(keyboard.onKey(27,0,0)).arbiterCommand==TheArbiter::ArbiterCommand::CMD_EXIT,"escape retained");
        arbiter.setActiveWorkspace(TheArbiter::WorkspaceId::PARTICLE_SIM);
        check(!arbiter.routeKeyboard(keyboard.onKey('f',0,0)).hasWorkspaceInput,"other cartridge unchanged");
    }
}

void micrometerCpuUnits() {
    SpatialVoxelGrid3D g;
    g.dimensions=glm::ivec3(8);g.origin=glm::vec3(-16);g.voxelEdgeM=4;g.metersPerWorldUnit=1e-6;
    SpatialVoxelRegion cell;check(g.region(0,cell),"Metric CPU voxel");
    check(std::abs(double(cell.volumeM3)-64e-18)<1e-23,"CPU voxel volume is SI");
    SpawnDensityRegionGrid3D regions;SpawnDensityRegion3D whole;
    check(regions.wholeDomainRegion(g,whole)&&std::abs(double(whole.volumeM3)-3.2768e-14)<1e-20,"CPU whole-domain SI volume");
    VectorField3D field(g),result(g);
    for(unsigned i=0;i<g.voxelCount();++i) {g.region(i,cell);field.set(i,glm::vec3(-cell.center.y*1e-6f,cell.center.x*1e-6f,0));}
    computeCurl(field,result);
    for(unsigned i=0;i<g.voxelCount();++i) check(std::abs(result.get(i).z-2)<1e-5,"CPU curl uses meter spacing");
    DebugElectrodynamics beam;beam.setSpeedMps(1);
    check(beam.fire(DebugProjectileSpecies::ArgonIon,glm::vec3(0),glm::vec3(0,0,1),g)==DebugElectrodynamics::FireResult::Fired,"Metric CPU beam");
    const float before=beam.projectiles()[0].position.z;beam.update(1e-6,g);
    check(std::abs((beam.projectiles()[0].position.z-before)-1)<1e-6,"SI beam drift converts back to world geometry");
    VectorField3D electric(g),magnetic(g),current(g);ScalarField3D rho(g);
    beam.populateFields(g,electric,magnetic,current,rho);
    double q=0;for(unsigned i=0;i<g.voxelCount();++i) q+=rho.get(i)*double(cell.volumeM3);
    check(std::abs(q-1.602176634e-19)<1e-25,"CPU diagnostic deposition uses SI volume");
}
int main() {
    try {
        micrometerCpuUnits();
        numericTextEntry(); initialConditions(); fieldsAndCurl(); independentFieldGrid(); diagnosticSampling(); glyphScaling();
        projectilesAndSigns(); raysAndCapacity(); cameraAndRouting();
        std::cout << "PASS: " << checks << " field, curl, electrodynamics, ray, capacity, camera and input checks\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
