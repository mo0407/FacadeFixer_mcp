param([string]$MaskRepo = "")
$ErrorActionPreference = 'Stop'
if (-not $MaskRepo) {
    $MaskRepo = Join-Path $PSScriptRoot 'vendor\FacadeFixer_mask_api'
    if (-not (Test-Path -LiteralPath $MaskRepo)) {
        $MaskRepo = Join-Path (Split-Path $PSScriptRoot -Parent) '.task_tmp\mask-mcp-feasibility'
    }
}
if (-not (Test-Path -LiteralPath $MaskRepo)) { throw 'Pass -MaskRepo <FacadeFixer_mask_api checkout> or run setup.ps1 first.' }
Push-Location $PSScriptRoot
try {
    & '.\.venv\Scripts\python.exe' -m facade_mcp.cli --mask-repo $MaskRepo demo
    if ($LASTEXITCODE -ne 0) { throw 'Demo failed' }
} finally { Pop-Location }
