@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
pushd build
cl /nologo /std:c++17 /EHsc /W4 /D NOMINMAX /I "..\.." /I "C:\vcpkg\installed\x64-windows\include" /I "C:\Users\richa\cuda-samples-master\Common" ..\Phase1Tests.cpp ..\..\VoxelField3D.cpp ..\..\DebugElectrodynamics.cpp ..\..\CameraEM.cpp ..\..\TheArbiterEM.cpp /Fe:Phase1Tests.exe /link opengl32.lib /LIBPATH:"C:\Users\richa\Anaheim Systems Dynamics - Software Product Dev\PlasmaPhySim_R0\PlasmaPhySim"
if errorlevel 1 (popd & exit /b 1)
Phase1Tests.exe
set TEST_RESULT=%errorlevel%
popd
exit /b %TEST_RESULT%
