param(
    [string]$SourceRepo = (Join-Path (Split-Path $PSScriptRoot -Parent) '项目代码\FacadeFixer-main'),
    [string]$SourcePython = '',
    [string]$Device = 'cpu',
    [string]$Config = (Join-Path $PSScriptRoot 'examples\source_models.workflow.json')
)
$ErrorActionPreference = 'Stop'
if (-not $SourcePython) { $SourcePython = Join-Path $SourceRepo '.venv\Scripts\python.exe' }
$env:FACADE_SOURCE_REPO = (Resolve-Path -LiteralPath $SourceRepo).Path
$env:FACADE_SOURCE_PYTHON = (Resolve-Path -LiteralPath $SourcePython).Path
$env:FACADE_SOURCE_DEVICE = $Device
$resolvedConfig = (Resolve-Path -LiteralPath $Config).Path
Push-Location $PSScriptRoot
try {
    & '.\.venv\Scripts\python.exe' -m facade_mcp.cli run --config $resolvedConfig
    if ($LASTEXITCODE -ne 0) { throw 'Source model MCP workflow failed; inspect the job error and tool_calls.jsonl.' }
} finally { Pop-Location }
