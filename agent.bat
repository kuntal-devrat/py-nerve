@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: virtualenv not found at .venv\Scripts\python.exe
    echo Run:  python -m venv .venv ^&^& .venv\Scripts\python -m pip install -e .
    exit /b 1
)
.venv\Scripts\python.exe scripts\desktop_agent_cli.py %*
