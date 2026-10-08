#include <GL/glew.h>
#include <GL/freeglut.h>
#include "fieldSystem.h"
#include "kernel.h"
#include <helper_math.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
    unsigned checks = 0;
    void check(bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(message);
    }
    FieldGridParams grid(unsigned n = 7) {
        return {make_uint3(n,n,n), make_float3(-1,-2,-3),
            make_float3(0.25f,0.5f,0.75f), n*n*n};
    }
    void lifecycle() {
        auto g = grid();
        check(FieldSystem::validGrid(g), "Valid nonuniform grid");
        auto bad = g; bad.dimensions.y = 0;
        check(!FieldSystem::validGrid(bad), "Zero dimension rejected");
        FieldSystem invalid(bad, false);
        check(!invalid.initialized() && !invalid.getBuffers().electricField, "Invalid object owns no GPU storage");
        bad = g; bad.cellSize.x = 0;
        check(!FieldSystem::validGrid(bad), "Zero spacing rejected");
        bad = g; bad.cellSize.x = std::numeric_limits<float>::quiet_NaN();
        check(!FieldSystem::validGrid(bad), "NaN spacing rejected");
        bad = g; bad.cellCount--;
        check(!FieldSystem::validGrid(bad), "Mismatched count rejected");
        FieldSystem fields(g, false);
        check(fields.initialized(), "CUDA field allocation");
        for (int k=0; k<FieldSystem::_NUM_SCALAR_FIELDS; ++k) {
            auto& v = fields.getScalarHost(static_cast<FieldSystem::ScalarField>(k));
            for (unsigned i=0; i<g.cellCount; ++i) v[i] = float(i+17*k);
        }
        for (int k=0; k<FieldSystem::_NUM_VECTOR_FIELDS; ++k) {
            auto& v = fields.getVectorHost(static_cast<FieldSystem::VectorField>(k));
            for (unsigned i=0; i<g.cellCount; ++i) v[i] = make_float4(float(i), float(k), -float(i), 0.5f);
        }
        check(fields.uploadFields(), "Upload all channels");
        fields.getScalarHost(FieldSystem::CHARGE_DENSITY).assign(g.cellCount, -99);
        fields.getVectorHost(FieldSystem::ELECTRIC_FIELD).assign(g.cellCount, make_float4(9,9,9,9));
        check(fields.downloadFields(), "Download all channels");
        for (unsigned z=0; z<g.dimensions.z; ++z)
            for (unsigned y=0; y<g.dimensions.y; ++y)
                for (unsigned x=0; x<g.dimensions.x; ++x) {
                    const unsigned i=x+y*g.dimensions.x+z*g.dimensions.x*g.dimensions.y;
                    check(fields.getScalarHost(FieldSystem::CHARGE_DENSITY)[i] == float(i), "X-fastest scalar round trip");
                    const auto v=fields.getVectorHost(FieldSystem::ELECTRIC_FIELD)[i];
                    check(v.x==float(i) && v.y==0 && v.z==-float(i) && v.w==0.5f, "float4 round trip");
                }
        const auto live = fields.getBuffers().electricField;
        check(!fields.setFieldGrid(bad) && fields.getBuffers().electricField==live, "Invalid resize preserves live allocation");
        check(fields.clear() && fields.downloadFields(), "Clear kernels and readback");
        for (int k=0; k<FieldSystem::_NUM_SCALAR_FIELDS; ++k)
            for (float v: fields.getScalarHost(static_cast<FieldSystem::ScalarField>(k))) check(v==0, "Scalar zero");
        for (int k=0; k<FieldSystem::_NUM_VECTOR_FIELDS; ++k)
            for (auto v: fields.getVectorHost(static_cast<FieldSystem::VectorField>(k)))
                check(v.x==0 && v.y==0 && v.z==0 && v.w==0, "Vector zero including w");
        check(fields.setFieldGrid(grid(9)) && fields.getCellCount()==729 &&
            fields.getScalarHost(FieldSystem::TEMPERATURE).size()==729, "Resize rebuilds device and host storage");
        auto unsupported=fields.getSolverParams(); unsupported.boundaryMode=FieldBoundaryMode::Periodic;
        check(!fields.setSolverParams(unsupported), "Periodic explicitly rejected");
        unsupported.boundaryMode=FieldBoundaryMode::Neumann;
        check(!fields.setSolverParams(unsupported), "Neumann explicitly rejected");
        check(cudaDeviceSynchronize()==cudaSuccess, "Allocation/clear CUDA execution");
    }
    void poisson() {
        auto g=grid();
        FieldSystem fields(g,false);
        auto& phi=fields.getScalarHost(FieldSystem::ELECTRIC_POTENTIAL);
        for (unsigned i=0; i<g.cellCount; ++i) phi[i]=g.origin.x+(float(i%7)+0.5f)*g.cellSize.x;
        check(fields.uploadFields() && fields.computeElectricField() && fields.downloadFields(), "Linear potential gradient");
        for (auto e: fields.getVectorHost(FieldSystem::ELECTRIC_FIELD))
            check(std::abs(e.x+1)<1e-5 && e.y==0 && e.z==0 && e.w==0, "E=-grad(x) including one-sided faces");
        check(fields.clear(), "Clear before Poisson");
        auto& rho=fields.getScalarHost(FieldSystem::CHARGE_DENSITY);
        const unsigned center=3+7*(3+7*3);
        rho[center]=1e-12f;
        auto solver=fields.getSolverParams();
        for (unsigned iterations: {31u,32u}) {
            solver.poissonIterations=iterations;
            check(fields.setSolverParams(solver) && fields.uploadFields() &&
                fields.solveElectrostatics() && fields.downloadFields(), "Poisson odd/even ping-pong");
            std::vector<double> cpu(g.cellCount,0), next(g.cellCount,0);
            const double ax=1.0/(g.cellSize.x*g.cellSize.x), ay=1.0/(g.cellSize.y*g.cellSize.y),
                az=1.0/(g.cellSize.z*g.cellSize.z);
            for (unsigned t=0; t<iterations; ++t) {
                for (unsigned z=0; z<7; ++z) for (unsigned y=0; y<7; ++y) for (unsigned x=0; x<7; ++x) {
                    const unsigned i=x+7*(y+7*z);
                    const unsigned coordinate[3]={x,y,z},stride[3]={1,7,49};
                    const double weights[3]={ax,ay,az};
                    double numerator=rho[i]/8.8541878128e-12,denominator=0;
                    for(unsigned axis=0;axis<3;++axis) {
                        if(coordinate[axis]>0) {numerator+=weights[axis]*cpu[i-stride[axis]];denominator+=weights[axis];}
                        else denominator+=2*weights[axis];
                        if(coordinate[axis]<6) {numerator+=weights[axis]*cpu[i+stride[axis]];denominator+=weights[axis];}
                        else denominator+=2*weights[axis];
                    }
                    next[i]=numerator/denominator;
                }
                cpu.swap(next);
            }
            for (unsigned i=0; i<g.cellCount; ++i)
                check(std::abs(phi[i]-cpu[i])<1e-8, "CUDA/CPU anisotropic Jacobi agreement");
            const auto& e=fields.getVectorHost(FieldSystem::ELECTRIC_FIELD);
            check(e[center+1].x>0 && e[center-1].x<0 &&
                e[center+7].y>0 && e[center-7].y<0 &&
                e[center+49].z>0 && e[center-49].z<0, "Positive rho produces outward E");
        }
        const auto positive=fields.getVectorHost(FieldSystem::ELECTRIC_FIELD);
        rho[center]=-rho[center];
        check(fields.uploadFields() && fields.solveElectrostatics() && fields.downloadFields(), "Negative rho solve");
        const auto& negative=fields.getVectorHost(FieldSystem::ELECTRIC_FIELD);
        for (unsigned i=0; i<g.cellCount; ++i)
            check(std::abs(negative[i].x+positive[i].x)<1e-9 &&
                std::abs(negative[i].y+positive[i].y)<1e-9 &&
                std::abs(negative[i].z+positive[i].z)<1e-9, "Negative charge reverses E");
        check(cudaDeviceSynchronize()==cudaSuccess, "Poisson CUDA execution");
    }
    void waveAndLorentz() {
        FieldSystem fields(grid(),false);
        AnalyticWaveParams wave;
        wave.enabled=true; wave.amplitudeVm=2; wave.frequencyHz=1e6f;
        wave.propagationDirection=make_float3(0,0,5);
        wave.polarization=make_float3(3,0,2); // Remove longitudinal part.
        check(fields.setWaveParams(wave) && fields.waveSpatiallyResolved(), "Normalize transverse analytic wave");
        check(fields.applyAnalyticWave(0) && fields.downloadFields(), "Wave at initial time");
        const auto before=fields.getVectorHost(FieldSystem::ELECTRIC_FIELD);
        const auto magnetic=fields.getVectorHost(FieldSystem::MAGNETIC_FIELD);
        for (unsigned i=0; i<before.size(); ++i) {
            check(std::isfinite(before[i].x) && std::abs(before[i].y)<1e-6 && std::abs(before[i].z)<1e-6,
                "Wave finite and transverse");
            check(std::abs(double(magnetic[i].y)*299792458.0-before[i].x)<1e-6 &&
                magnetic[i].x==0 && magnetic[i].z==0, "B=(k cross E)/c");
        }
        check(fields.clear() && fields.applyAnalyticWave(0.5e-6) && fields.downloadFields(), "Wave half period");
        for (unsigned i=0; i<before.size(); ++i)
            check(std::abs(before[i].x+fields.getVectorHost(FieldSystem::ELECTRIC_FIELD)[i].x)<1e-5,
                "Wave time changes phase");
        wave.frequencyHz=1e12f;
        check(fields.setWaveParams(wave) && !fields.waveSpatiallyResolved(), "Under-resolved wave flagged");
        wave.propagationDirection=make_float3(0,0,0);
        check(!fields.setWaveParams(wave), "Zero direction rejected");
        wave.propagationDirection=wave.polarization;
        check(!fields.setWaveParams(wave), "Parallel polarization rejected");
        check(fields.clear(), "Clear wave before Lorentz");
        std::fill(fields.getVectorHost(FieldSystem::ELECTRIC_FIELD).begin(),
            fields.getVectorHost(FieldSystem::ELECTRIC_FIELD).end(),make_float4(2,0,0,0));
        std::fill(fields.getVectorHost(FieldSystem::MAGNETIC_FIELD).begin(),
            fields.getVectorHost(FieldSystem::MAGNETIC_FIELD).end(),make_float4(0,0,3,0));
        check(fields.uploadFields(), "Stored uniform E/B");
        float4 pos[3]={make_float4(0,0,0,1),make_float4(0,0,0,1),make_float4(0,0,0,1)};
        float4 vel[3]={make_float4(0,4,0,0),make_float4(0,4,0,0),make_float4(0,4,0,0)};
        ParticleFieldMarker markers[3]{};
        for (auto& m:markers) m.kind=ParticleKind::Atomic;
        markers[0].chargeC=1; markers[1].chargeC=-1;
        markers[0].chargeToMass=2; markers[1].chargeToMass=-2;
        float4 *dp=nullptr,*dv=nullptr,*da=nullptr;
        ParticleFieldMarker* dm=nullptr;
        check(cudaMalloc(&dp,sizeof(pos))==cudaSuccess && cudaMalloc(&dv,sizeof(vel))==cudaSuccess &&
            cudaMalloc(&da,sizeof(pos))==cudaSuccess && cudaMalloc(&dm,sizeof(markers))==cudaSuccess, "Coupling test allocation");
        check(copyFieldToDevice(dp,pos,sizeof(pos)) && copyFieldToDevice(dv,vel,sizeof(vel)) &&
            copyFieldToDevice(dm,markers,sizeof(markers)), "Coupling test upload");
        float4 seed[3]={make_float4(0,7,0,0),make_float4(0,7,0,0),make_float4(0,7,0,0)};
        check(copyFieldToDevice(da,seed,sizeof(seed)), "Seed non-EM acceleration");
        check(fields.computeLorentzAcceleration(dp,dv,dm,da,3), "Lorentz launch");
        float4 acc[3]{};
        check(copyFieldToHost(acc,da,sizeof(acc)), "Lorentz readback");
        check(acc[0].x==28 && acc[1].x==-28 && acc[0].y==7 && acc[0].z==0,
            "q/m*(E+v cross B), charge sign");
        check(acc[2].x==0 && acc[2].y==7 && acc[2].z==0 && acc[2].w==0, "Neutral preserves non-EM acceleration");
        cudaFree(dp); cudaFree(dv); cudaFree(da); cudaFree(dm);
    }
    void micrometerCoupling() {
        FieldGridParams g{make_uint3(8,8,8),make_float3(-16),make_float3(4),512,1e-6};
        check(std::abs(std::pow(32*g.metersPerWorldUnit,3)-3.2768e-14)<1e-28,"32 um cube SI volume");
        FieldSystem fields(g,false);
        check(fields.initialized(),"Micrometer field grid");
        float4 pos[3]={make_float4(-1,0,0,9),make_float4(1,0,0,8),make_float4(0,0,0,7)};
        float4 vel[3]={make_float4(0,4e6f,0,0.0063f),make_float4(0),make_float4(0)};
        ParticleFieldMarker marker[3]{};
        for(auto& m:marker) m.kind=ParticleKind::Atomic;
        marker[0].chargeC=1.602176634e-19f; marker[0].chargeToMass=2;
        marker[1].chargeC=marker[0].chargeC; marker[1].chargeToMass=2;
        float4 *dp=nullptr,*dv=nullptr,*da=nullptr; ParticleFieldMarker* dm=nullptr;
        check(cudaMalloc(&dp,sizeof(pos))==cudaSuccess && cudaMalloc(&dv,sizeof(vel))==cudaSuccess &&
            cudaMalloc(&da,sizeof(pos))==cudaSuccess && cudaMalloc(&dm,sizeof(marker))==cudaSuccess,"SI particle allocation");
        auto upload=[&]() {check(copyFieldToDevice(dp,pos,sizeof(pos))&&copyFieldToDevice(dv,vel,sizeof(vel))&&
            copyFieldToDevice(dm,marker,sizeof(marker)),"SI particle upload");};
        upload();
        ParticleSimParams params{};
        params.gridSize=make_uint3(8,8,8); params.numCells=512;params.cellSize=make_float3(4);
        params.worldOrigin=make_float3(-16);params.boundary=16;params.boundaryDamping=-0.5f;
        params.globalDamping=1;params.gravity=make_float3(0,7,0);params.metersPerWorldUnit=1e-6;
        setParameters(&params);
        check(initializeParticleAcceleration(da,dm,3),"Gravity owns initialization");
        UniformEMField uniform;uniform.enabled=true;uniform.electricVm=make_float3(2,0,0);uniform.magneticT=make_float3(0,0,3);
        fields.setUniformField(uniform);
        check(fields.updateParticleFields(dp,dv,dm,3,ElectrostaticMode::Off,0)&&fields.downloadFields(),"CIC plus external-only fields");
        double totalCharge=0,totalJy=0;
        const double volume=std::pow(4e-6,3);
        for(float r:fields.getScalarHost(FieldSystem::CHARGE_DENSITY)) totalCharge+=r*volume;
        for(auto j:fields.getVectorHost(FieldSystem::CURRENT_DENSITY)) totalJy+=j.y*volume;
        check(std::abs(totalCharge-2*double(marker[0].chargeC))<1e-25,"CIC integrated charge uses cubic SI scale");
        check(std::abs(totalJy-double(marker[0].chargeC)*4)<1e-25,"J uses velocity converted to m/s");
        check(fields.computeLorentzAcceleration(dp,dv,dm,da,3),"SI Lorentz coupling");
        float4 a[3];check(copyFieldToHost(a,da,sizeof(a)),"SI acceleration readback");
        check(std::abs(a[0].x-28)<1e-5 && a[0].y==7 && a[2].x==0 && a[2].y==7,"Lorentz unit conversion and neutral gravity");
        check(fields.updateParticleFields(dp,dv,dm,3,ElectrostaticMode::Off,0)&&fields.downloadFields(),"Rebuild prescribed fields");
        check(fields.getVectorHost(FieldSystem::ELECTRIC_FIELD)[0].x==2,"External fields do not accumulate across steps");
        check(!computeDirectCoulomb(dp,dm,da,3,ElectrostaticMode::GridField,1e-8),"Grid/direct mutual exclusion");
        check(!computeDirectCoulomb(dp,dm,da,2049,ElectrostaticMode::DirectCoulombDebug,1e-8),"Small-N direct guard");
        check(!initializeParticleAcceleration(da,dm,49153),"Capacity guard before memory access");
        check(initializeParticleAcceleration(da,dm,3)&&computeDirectCoulomb(dp,dm,da,3,ElectrostaticMode::DirectCoulombDebug,1e-8)&&
            copyFieldToHost(a,da,sizeof(a)),"Direct Coulomb SI launch");
        const double r2=4e-12+1e-16;
        const double expected=8.9875517923e9*2*double(marker[1].chargeC)*2e-6/(r2*std::sqrt(r2));
        check(a[0].x<0 && a[1].x>0 && std::abs(a[0].x+expected)<expected*2e-6,"Like charges repel with physical 2 um separation");
        marker[1].chargeC=-marker[1].chargeC;marker[1].chargeToMass=-2;upload();
        check(initializeParticleAcceleration(da,dm,3)&&computeDirectCoulomb(dp,dm,da,3,ElectrostaticMode::DirectCoulombDebug,1e-8)&&
            copyFieldToHost(a,da,sizeof(a)),"Opposite Coulomb launch");
        check(a[0].x>0 && a[1].x<0 && a[2].x==0,"Opposite charges attract, neutral ignores EM");
        // Known SI kick, independent of electrostatic solver and contact parameters.
        params.gravity=make_float3(1,0,0);setParameters(&params);
        pos[0]=make_float4(0,0,0,9);vel[0]=make_float4(0,0,0,0.0063f);
        marker[2].kind=ParticleKind::Inactive;pos[2]=make_float4(3,2,1,7);vel[2]=make_float4(5,0,0,0.01f);upload();
        check(initializeParticleAcceleration(da,dm,3),"Initialize SI integration");
        integrateSystem(reinterpret_cast<float*>(dp),reinterpret_cast<float*>(dv),reinterpret_cast<float*>(da),1e-6f,3,dm);
        check(copyFieldToHost(pos,dp,sizeof(pos))&&copyFieldToHost(vel,dv,sizeof(vel)),"SI push readback");
        check(std::abs(vel[0].x-1)<1e-6 && std::abs(pos[0].x-1e-6)<1e-12 && pos[0].w==9 && vel[0].w==0.0063f,
            "One SI kick and drift, metadata and radius preserved");
        check(pos[2].x==3 && vel[2].x==5,"Inactive slot is not pushed");
        check(initializeParticleAcceleration(nullptr,nullptr,0)&&
            fields.updateParticleFields(nullptr,nullptr,nullptr,0,ElectrostaticMode::GridField,0)&&fields.downloadFields(),"Empty population clears fields safely");
        for(float rho:fields.getScalarHost(FieldSystem::CHARGE_DENSITY)) check(rho==0,"No stale charge at zero particles");
        // Owner-specific scale: alternating a meter grid must not contaminate the um owner.
        FieldSystem meterFields(grid(),false);
        check(meterFields.computeElectricField()&&fields.computeElectricField(),"Alternating meter/micrometer owners");
        auto bad=g;bad.metersPerWorldUnit=0;check(!validFieldGrid(bad),"Reject zero physical scale");
        bad=g;bad.metersPerWorldUnit=std::numeric_limits<double>::quiet_NaN();check(!validFieldGrid(bad),"Reject NaN physical scale");
        // Equal rho on geometrically similar grids: phi scales as L^2, E as L.
        auto meterGrid=g;meterGrid.metersPerWorldUnit=1;
        FieldSystem meterReference(meterGrid,false);
        check(fields.clear(),"Clear unit comparison fields");
        fields.getScalarHost(FieldSystem::CHARGE_DENSITY)[292]=1e-12f;
        meterReference.getScalarHost(FieldSystem::CHARGE_DENSITY)[292]=1e-12f;
        check(fields.uploadFields()&&meterReference.uploadFields()&&meterReference.solveElectrostatics()&&
            fields.solveElectrostatics()&&fields.downloadFields()&&meterReference.downloadFields(),"Alternating SI Poisson owners");
        for(unsigned i=0;i<g.cellCount;++i) {
            const double expectedPhi=meterReference.getScalarHost(FieldSystem::ELECTRIC_POTENTIAL)[i]*1e-12;
            check(std::abs(fields.getScalarHost(FieldSystem::ELECTRIC_POTENTIAL)[i]-expectedPhi)<1e-17,"Poisson metric squared scaling");
            const auto me=meterReference.getVectorHost(FieldSystem::ELECTRIC_FIELD)[i];
            const auto ue=fields.getVectorHost(FieldSystem::ELECTRIC_FIELD)[i];
            check(std::abs(ue.x-double(me.x)*1e-6)<1e-11,"Gradient metric scaling");
        }
        AnalyticWaveParams wave;wave.enabled=true;wave.amplitudeVm=2;wave.frequencyHz=float(299792458.0/64e-6);
        check(fields.setWaveParams(wave)&&fields.waveSpatiallyResolved()&&fields.clear()&&fields.applyAnalyticWave(0)&&fields.downloadFields(),"Resolved micrometer wave");
        const double phase=6.2831853071795864769*double(wave.frequencyHz)*(-14e-6)/299792458.0;
        check(std::abs(fields.getVectorHost(FieldSystem::ELECTRIC_FIELD)[0].x-2*std::cos(phase))<1e-5,"Wave phase uses meters");
        // Contact is additive SI acceleration; neutral pairs still interact.
        params.gravity=make_float3(0,0,4);params.spring=100;params.damping=10;params.shear=5;setParameters(&params);
        pos[0]=make_float4(-0.005f,0,0,1);pos[1]=make_float4(0.005f,0,0,1);
        vel[0]=make_float4(0,1e6f,0,0.0063f);vel[1]=make_float4(0,0,0,0.0063f);
        marker[0]={};marker[1]={};marker[0].kind=marker[1].kind=ParticleKind::Atomic;upload();
        unsigned *indices=nullptr,*start=nullptr,*end=nullptr;
        check(cudaMalloc(&indices,2*sizeof(unsigned))==cudaSuccess&&cudaMalloc(&start,512*sizeof(unsigned))==cudaSuccess&&
            cudaMalloc(&end,512*sizeof(unsigned))==cudaSuccess,"Contact test allocation");
        const unsigned order[2]={0,1};std::vector<unsigned> starts(512,0xffffffffu),ends(512,0);
        starts[291]=0;ends[291]=1;starts[292]=1;ends[292]=2;
        check(copyFieldToDevice(indices,order,sizeof(order))&&copyFieldToDevice(start,starts.data(),512*sizeof(unsigned))&&
            copyFieldToDevice(end,ends.data(),512*sizeof(unsigned)),"Contact neighbor tables");
        check(initializeParticleAcceleration(da,dm,2)&&computeContactAcceleration(da,dp,dv,indices,start,end,dm,2)&&
            copyFieldToHost(a,da,sizeof(a)),"Additive contact launch");
        check(std::abs(a[0].x+2.6e-7)<1e-12&&std::abs(a[1].x-2.6e-7)<1e-12&&
            a[0].y==-15&&a[1].y==15&&a[0].z==4,"Contact spring/damping/shear SI units preserve gravity");
        pos[1]=pos[0];upload();starts[291]=0;ends[291]=2;starts[292]=0xffffffffu;
        check(copyFieldToDevice(start,starts.data(),512*sizeof(unsigned))&&copyFieldToDevice(end,ends.data(),512*sizeof(unsigned))&&
            initializeParticleAcceleration(da,dm,2)&&computeContactAcceleration(da,dp,dv,indices,start,end,dm,2)&&
            copyFieldToHost(a,da,sizeof(a)),"Coincident contact launch");
        check(std::isfinite(a[0].x)&&a[0].x==-a[1].x,"Coincident normal is finite and antisymmetric");
        cudaFree(indices);cudaFree(start);cudaFree(end);
        check(cudaDeviceSynchronize()==cudaSuccess,"SI kernels execute successfully");
        cudaFree(dp);cudaFree(dv);cudaFree(da);cudaFree(dm);
    }
    std::vector<FieldGlyphVertex> readVbo(unsigned vbo, unsigned count) {
        std::vector<FieldGlyphVertex> result(count);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glGetBufferSubData(GL_ARRAY_BUFFER,0,count*sizeof(FieldGlyphVertex),result.data());
        glBindBuffer(GL_ARRAY_BUFFER,0);
        return result;
    }
    void renderBuffers() {
        FieldSystem fields(grid(),true);
        check(fields.initialized(), "Persistent CUDA/GL field buffers allocated");
        const auto lines=fields.getGlyphBuffer(), points=fields.getScalarBuffer();
        auto& rho=fields.getScalarHost(FieldSystem::CHARGE_DENSITY);
        auto& electric=fields.getVectorHost(FieldSystem::ELECTRIC_FIELD);
        for (unsigned i=0; i<fields.getCellCount(); ++i) {
            rho[i]=i%2 ? -1e-19f : 1e-19f;
            electric[i]=make_float4(0,0,1e-26f,0);
        }
        check(fields.uploadFields(), "Render input upload");
        FieldRenderParams settings;
        settings.vectorScale=0;
        settings.scalarThreshold=1e-18; settings.vectorThreshold=1e-25;
        check(fields.buildRenderBuffers(FieldSystem::CHARGE_DENSITY,settings), "Filtered GPU render build");
        const auto invisible=readVbo(lines,fields.getGlyphVertexCount());
        for (const auto& v:invisible) check(v.color.w==0,"Filtered vector transparent");
        for (const auto& v:readVbo(points,fields.getScalarVertexCount()))
            check(v.color.w==0,"Filtered scalar transparent");
        settings.scalarThreshold=0; settings.vectorThreshold=0;
        check(fields.buildRenderBuffers(FieldSystem::CHARGE_DENSITY,settings), "Visible GPU render build");
        const auto visible=readVbo(lines,fields.getGlyphVertexCount());
        check(visible.size()==fields.getCellCount()*6, "Three GL_LINES segments per field cell");
        check(visible[2].position.z==visible[1].position.z && visible[4].position.z==visible[1].position.z &&
            visible[3].position.z<visible[1].position.z && visible[5].position.z<visible[1].position.z &&
            visible[3].position.x<visible[1].position.x && visible[5].position.x>visible[1].position.x,
            "Planar two-sided arrowhead connects to shaft tip");
        check(visible[0].color.w>0.4f && visible[0].color.w<=0.8f, "Translucent vector alpha");
        check(visible[1].position.z>visible[0].position.z && visible[0].color.w>0,
            "Tiny vector survives magnitude calculation and arrow points +z");
        const float length=visible[1].position.z-visible[0].position.z;
        check(std::abs(length-0.65f*fields.getCellSize().x)<1e-6,
            "Glyph length visual cell fraction");
        const auto scalars=readVbo(points,fields.getScalarVertexCount());
        check(scalars[0].color.x>scalars[0].color.z && scalars[1].color.z>scalars[1].color.x,
            "Charge density warm positive / cool negative");
        check(scalars[0].color.w>=0.4f && scalars[0].color.w<=0.8f, "Magnitude controls scalar alpha");
        settings.stride=2;
        check(fields.buildRenderBuffers(FieldSystem::CHARGE_DENSITY,settings), "Per-axis stride");
        const auto strided=readVbo(points,fields.getScalarVertexCount());
        for (unsigned i=0; i<strided.size(); ++i) {
            const bool sampled=(i%7)%2==0 && ((i/7)%7)%2==0 && (i/49)%2==0;
            check((strided[i].color.w>0)==sampled,"Stride samples each coordinate");
        }
        for (int frame=0; frame<10; ++frame)
            check(fields.buildRenderBuffers(FieldSystem::CHARGE_DENSITY,settings) &&
                fields.getGlyphBuffer()==lines && fields.getScalarBuffer()==points, "No per-frame VBO reallocation");
        check(fields.downloadFields(), "Render physics readback");
        for (unsigned i=0; i<fields.getCellCount(); ++i)
            check(rho[i]==(i%2?-1e-19f:1e-19f) && electric[i].z==1e-26f, "Threshold/stride do not change physics");
        check(fields.setFieldGrid(grid(9)), "GL resource resize");
        check(fields.buildRenderBuffers(FieldSystem::CHARGE_DENSITY,settings) &&
            fields.getGlyphVertexCount()==729*6 && fields.getScalarVertexCount()==729, "New grid render storage size");
        check(glGetError()==GL_NO_ERROR && cudaDeviceSynchronize()==cudaSuccess, "GL/CUDA render buffer execution");
    }
}
int main(int argc, char** argv) {
    try {
        lifecycle();
        poisson();
        waveAndLorentz();
        micrometerCoupling();
        glutInit(&argc,argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
        glutInitWindowSize(64,64);
        const int window=glutCreateWindow("FieldSystem GPU tests");
        glutHideWindow();
        check(glewInit()==GLEW_OK,"GLEW context");
        while (glGetError()!=GL_NO_ERROR) {}
        renderBuffers();
        glutDestroyWindow(window);
        std::printf("PASS: %u FieldSystem CUDA checks\n", checks);
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
