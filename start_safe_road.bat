@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Run: py -m venv .venv
  echo Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo Starting Safe Road Detector...
echo Open http://127.0.0.1:8000 in your browser.
echo For a phone on the same Wi-Fi, use the Phone URL printed by server.py.
".venv\Scripts\python.exe" server.py
