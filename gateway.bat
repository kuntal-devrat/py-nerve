@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\llm_gateway.py %*
