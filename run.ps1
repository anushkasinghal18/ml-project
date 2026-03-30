# PowerShell script to run scriptnew.py with the correct venv interpreter
# Run this from terminal: .\run.ps1

$VenvPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$ScriptPath = Join-Path $PSScriptRoot "scriptnew.py"

Write-Host "Running Speaker Identification with venv interpreter..." -ForegroundColor Green
& $VenvPath $ScriptPath
