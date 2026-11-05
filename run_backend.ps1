param(
    [ValidateSet('Start','Stop','Status')][string]$Action = 'Start',
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$stateDirectory = Join-Path $PSScriptRoot '.task_tmp\local-backend'
$stateFile = Join-Path $stateDirectory 'state.json'
function Read-BackendState {
    if (Test-Path -LiteralPath $stateFile) { return Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json }
    return $null
}
$backendState = Read-BackendState
$nowSeconds = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$isActive = $backendState -and $backendState.status -in @('starting','running') -and ($nowSeconds - $backendState.heartbeat -lt 15)
if ($Action -eq 'Stop') {
    if (-not $isActive) { Write-Output 'No active managed backend.'; exit 0 }
    Set-Content -LiteralPath (Join-Path $stateDirectory 'stop.request') -Value $backendState.session -Encoding utf8
    for ($attempt = 0; $attempt -lt 25; $attempt++) {
        Start-Sleep -Seconds 1
        $backendState = Read-BackendState
        if ($backendState.status -eq 'stopped') { Write-Output 'Backend stopped.'; exit 0 }
        if ($backendState.status -eq 'failed') { throw "Backend shutdown failed: $($backendState.error)" }
    }
    throw 'Shutdown still in progress; inspect .task_tmp/local-backend/state.json.'
}
if ($Action -eq 'Status') {
    if ($backendState) { $backendState | ConvertTo-Json -Depth 6 } else { Write-Output 'Backend has not been started.' }
    if (-not $isActive) { Write-Output 'No current supervisor heartbeat.' }
    exit 0
}
if ($isActive) { Write-Output 'Backend already managed. Open http://127.0.0.1:8780/'; exit 0 }
if (-not $Python) {
    $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = Join-Path (Split-Path $PSScriptRoot -Parent) 'facade_mcp_langchain\.venv\Scripts\python.exe'
    }
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Run .\setup.ps1 -Web or pass -Python <MCP runtime python.exe>.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.env'))) { throw 'Configure .env first; see docs/local-backend.md.' }
New-Item -ItemType Directory -Force -Path $stateDirectory | Out-Null
$runner = Join-Path $PSScriptRoot 'scripts\local_backend.py'
$process = Start-Process -FilePath $Python -ArgumentList @('"' + $runner + '"') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $stateDirectory 'supervisor.out.log') -RedirectStandardError (Join-Path $stateDirectory 'supervisor.err.log') -PassThru
for ($attempt = 0; $attempt -lt 45; $attempt++) {
    Start-Sleep -Seconds 1
    $backendState = Read-BackendState
    if ($backendState -and $backendState.started -ge $nowSeconds -and $backendState.status -eq 'running') {
        Write-Output 'Backend ready: http://127.0.0.1:8780/'
        $backendState.services | ConvertTo-Json -Depth 4
        exit 0
    }
    if ($process.HasExited) { throw 'Supervisor exited. Inspect .task_tmp/local-backend/supervisor.err.log.' }
    if ($backendState -and $backendState.started -ge $nowSeconds -and $backendState.status -eq 'failed') { throw $backendState.error }
}
Write-Output 'Backend still starting. Use .\run_backend.ps1 Status to inspect progress.'
