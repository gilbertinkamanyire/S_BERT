@echo off
title VeriText + VERITAS Unified Academic Integrity Platform
echo.
echo  =======================================================
echo   VeriText + VERITAS — AI Content ^& Forensic Platform
echo   Modeled after VeriText UI · SBERT 384-D · 100%% Local
echo  =======================================================
echo.
echo  Opening browser at http://127.0.0.1:5050 ...
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5050"
echo  Starting Flask forensic engine on port 5050...
echo  Press Ctrl+C to stop.
echo.

if exist "%~dp0..\.venv314\Scripts\python.exe" (
    "%~dp0..\.venv314\Scripts\python.exe" "%~dp0..\app.py"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
    "%~dp0..\.venv\Scripts\python.exe" "%~dp0..\app.py"
) else if exist "C:\Users\TinkaTech\anaconda3\python.exe" (
    "C:\Users\TinkaTech\anaconda3\python.exe" "%~dp0..\app.py"
) else (
    python "%~dp0..\app.py"
)

pause
