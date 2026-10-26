param([string]$Python = "", [int]$Port = 8780)
$ErrorActionPreference = 'Stop'
if (-not $Python) { $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Python environment missing. Run .\setup.ps1 -Web -SkipUpstream, or pass -Python <existing environment python.exe>.'
}
$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = (Join-Path $PSScriptRoot 'src') + [IO.Path]::PathSeparator + $PreviousPythonPath
Push-Location $PSScriptRoot
try {
    & $Python -m facade_mcp.web_app --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Web server stopped with an error. Check dependencies and .env configuration.' }
} finally {
    Pop-Location
    $env:PYTHONPATH = $PreviousPythonPath
}
