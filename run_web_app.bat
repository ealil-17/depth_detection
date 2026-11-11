@echo off
title Arducam Vision Studio - Web UI
echo ====================================================================
echo  Starting Arducam Real-Time Vision Studio Web Server
echo ====================================================================

cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo Launching web application on http://localhost:5000 ...
python web_app.py --device cuda:0

pause

