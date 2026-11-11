@echo off
title Arducam Real-Time Depth Detection - Native App
echo ====================================================================
echo  Starting Arducam Real-Time Multi-Object Detection & Distance Estimation
echo ====================================================================

cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo Controls in video window:
echo   'c' : Switch Camera (Cam 0 / Cam 1)
echo   'd' : Toggle Depth Colormap
echo   's' : Save Screenshot
echo   'q' : Quit
echo.

python main.py --device cuda:0

pause

