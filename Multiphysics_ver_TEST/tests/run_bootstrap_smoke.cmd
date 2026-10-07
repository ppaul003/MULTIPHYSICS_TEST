@echo off
setlocal
rem Build Debug x64 solution first; reuse its real object files (excluding app main).
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
for %%I in ("%~dp0..") do set "testSource=%%~fI"
set "testObj=%testSource%\x64\Debug"
set "testOut=%testSource%\..\x64\Debug"
if not defined CudaSamplesCommon set "CudaSamplesCommon=%USERPROFILE%\cuda-samples-master\Common"
if not defined GlmIncludeDir set "GlmIncludeDir=C:\vcpkg\installed\x64-windows\include"
cl /nologo /std:c++17 /EHsc /MDd /D_USE_MATH_DEFINES /I"%testSource%" /I"%CudaSamplesCommon%" /I"%CUDA_PATH%\include" /I"%GlmIncludeDir%" "%~dp0bootstrap_smoke.cpp" /Fo"%testOut%\bootstrap_smoke.obj" /Fe"%testOut%\bootstrap_smoke.exe" "%testObj%\EuclidEngineEM.obj" "%testObj%\CameraEM.obj" "%testObj%\DebugElectrodynamics.obj" "%testObj%\DiagnosticIdleEM.obj" "%testObj%\FieldDebugRenderer.obj" "%testObj%\MultiPhysicsFieldDebug.obj" "%testObj%\multiPhysicsSimWorkspace.obj" "%testObj%\atomicParticlesSimWorkspace.obj" "%testObj%\ParticleSimWorkspace.obj" "%testObj%\particleSystem.obj" "%testObj%\rendererEM_Euclid.obj" "%testObj%\TextEntry.obj" "%testObj%\TheArbiterEM.obj" "%testObj%\TheTesseractEM.obj" "%testObj%\ViewPortEM.obj" "%testObj%\VoxelField3D.obj" "%testObj%\kernel.cu.obj" /link /DEBUG /OPT:NOICF /LIBPATH:"%testSource%" /LIBPATH:"%CUDA_PATH%\lib\x64" cudart_static.lib freeglut.lib glew64.lib user32.lib gdi32.lib opengl32.lib
if errorlevel 1 exit /b 1
"%testOut%\bootstrap_smoke.exe"
if errorlevel 1 exit /b 1
pushd "%testOut%"
bootstrap_smoke.exe --engine-lifecycle
set "testResult=%errorlevel%"
popd
exit /b %testResult%
