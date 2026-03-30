@echo off
REM Windows batch script to run scriptnew.py with the correct venv interpreter
REM Simply double-click this file or run: run.bat

pushd "%~dp0"
.venv\Scripts\python.exe scriptnew.py
popd
pause
