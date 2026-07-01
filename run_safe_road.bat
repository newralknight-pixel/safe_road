@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -m venv .venv
  if errorlevel 1 (
    echo Python was not found. Install Python 3 first, then run this file again.
    pause
    exit /b 1
  )
)

echo Installing requirements...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements.
  pause
  exit /b 1
)

echo Starting Safe Road...
echo Open http://127.0.0.1:8000 in Chrome or Edge.
echo For a phone on the same Wi-Fi, use the Phone URL printed by server.py.
".venv\Scripts\python.exe" server.py
pause
