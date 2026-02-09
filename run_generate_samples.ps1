$ErrorActionPreference = 'Stop'

$repoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$freecadCmd = 'C:\Program Files\FreeCAD 1.0\bin\freecadcmd.exe'
$scriptPath = Join-Path $repoDir 'scripts\generate_training_data.py'
$scriptPath = $scriptPath -replace '\\', '/'

Write-Host "Starting training data generation..."
Write-Host "FreeCAD: $freecadCmd"
Write-Host "Script: $scriptPath"

$pythonCode = "import sys; sys.argv = [r'$scriptPath']; exec(open(r'$scriptPath', encoding='utf-8').read())"
& $freecadCmd -c $pythonCode
