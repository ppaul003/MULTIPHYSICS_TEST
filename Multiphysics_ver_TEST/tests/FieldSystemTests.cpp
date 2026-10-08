#include <GL/glew.h>
#include <GL/freeglut.h>
#include "fieldSystem.h"
#include "kernel.h"
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
                for (unsigned z=1; z<6; ++z) for (unsigned y=1; y<6; ++y) for (unsigned x=1; x<6; ++x) {
                    const unsigned i=x+7*(y+7*z);
                    next[i]=(ax*(cpu[i-1]+cpu[i+1])+ay*(cpu[i-7]+cpu[i+7])+
                        az*(cpu[i-49]+cpu[i+49])+rho[i]/8.8541878128e-12)/(2*(ax+ay+az));
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
        markers[0].chargeToMass=2; markers[1].chargeToMass=-2;
        float4 *dp=nullptr,*dv=nullptr,*da=nullptr;
        ParticleFieldMarker* dm=nullptr;
        check(cudaMalloc(&dp,sizeof(pos))==cudaSuccess && cudaMalloc(&dv,sizeof(vel))==cudaSuccess &&
            cudaMalloc(&da,sizeof(pos))==cudaSuccess && cudaMalloc(&dm,sizeof(markers))==cudaSuccess, "Coupling test allocation");
        check(copyFieldToDevice(dp,pos,sizeof(pos)) && copyFieldToDevice(dv,vel,sizeof(vel)) &&
            copyFieldToDevice(dm,markers,sizeof(markers)), "Coupling test upload");
        check(fields.computeLorentzAcceleration(dp,dv,dm,da,3), "Lorentz launch");
        float4 acc[3]{};
        check(copyFieldToHost(acc,da,sizeof(acc)), "Lorentz readback");
        check(acc[0].x==28 && acc[1].x==-28 && acc[0].y==0 && acc[0].z==0,
            "q/m*(E+v cross B), charge sign");
        check(acc[2].x==0 && acc[2].y==0 && acc[2].z==0 && acc[2].w==0, "Neutral has zero EM acceleration");
        cudaFree(dp); cudaFree(dv); cudaFree(da); cudaFree(dm);
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
