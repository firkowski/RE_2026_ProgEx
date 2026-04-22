@echo off
REM Recursive Estimation 2026: Windows launcher.
REM
REM Usage (from inside the exercise\ folder):
REM     run.bat
REM
REM Requires:
REM   - Docker Desktop : https://docs.docker.com/get-docker/

setlocal EnableDelayedExpansion
set IMAGE=re-student
set VENV=.venv

cd /d "%~dp0"

REM
where docker >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker not found. Install Docker Desktop:
    echo     https://docs.docker.com/get-docker/
    exit /b 1
)

REM
docker image inspect %IMAGE% >nul 2>&1
if errorlevel 1 (
    echo Building Docker image (one-time^)...
    docker build -t %IMAGE% .
    if errorlevel 1 exit /b 1
)

set COMMON=--rm --cpus=4 --memory=2g --memory-swap=2g -v "%cd%":/work -w /work

REM
if /I "%~1"=="headless" (
    echo Running headless...
    docker run %COMMON% %IMAGE% python run.py --headless
    goto :end
)

REM
echo Starting simulation environment (4 CPU cores, 2 GB RAM^)...
docker run %COMMON% %IMAGE%
if errorlevel 1 (
    echo ERROR: Simulation failed.
    exit /b 1
)

REM
if not exist "%VENV%\Scripts\python.exe" (
    echo Setting up Python environment (one-time^)...
    py -m venv %VENV% >nul 2>&1
    if errorlevel 1 python -m venv %VENV% >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Could not create Python environment.
        echo        Install Python 3 from https://www.python.org/downloads/
        exit /b 1
    )
)

REM
echo Installing dependencies...
%VENV%\Scripts\pip install --quiet --upgrade matplotlib numpy

REM
echo Opening plots...
%VENV%\Scripts\python "%~dp0plot_results.py"

:end
endlocal
