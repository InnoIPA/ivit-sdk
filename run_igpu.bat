@echo off
call "%INTEL_OPENVINO_DIR%\setupvars.bat" >nul
venv\Scripts\python.exe -u igpu_probe.py
