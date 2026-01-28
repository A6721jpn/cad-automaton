$ErrorActionPreference = 'Stop'

$repoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$freecadCmd = 'C:\Program Files\FreeCAD 1.0\bin\freecadcmd.exe'
$scriptPath = Join-Path $repoDir 'src\proto3-codex\run_proto3.py'

& $freecadCmd $scriptPath
