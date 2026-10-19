param(
    [string]$DetectorPython = '',
    [string]$Weights = '',
    [string]$MaskRepo = ''
)
$ErrorActionPreference = 'Stop'
$SourceRepo = if ($env:FACADE_SOURCE_REPO) { $env:FACADE_SOURCE_REPO } else {
    Join-Path (Split-Path $PSScriptRoot -Parent) '项目代码\FacadeFixer-main'
}
if (-not $DetectorPython) {
    $DetectorPython = if ($env:FACADE_DETECTOR_PYTHON) { $env:FACADE_DETECTOR_PYTHON } else {
        Join-Path $SourceRepo '.venv\Scripts\python.exe'
    }
}
if (-not $Weights) {
    $Weights = if ($env:FACADE_DETECTOR_WEIGHTS) { $env:FACADE_DETECTOR_WEIGHTS } else {
        Join-Path $SourceRepo 'model_weights\detection\yolo12m_building.pt'
    }
}
if (-not $MaskRepo) {
    $MaskRepo = Join-Path $PSScriptRoot 'vendor\FacadeFixer_mask_api'
    if (-not (Test-Path -LiteralPath $MaskRepo)) {
        $MaskRepo = Join-Path (Split-Path $PSScriptRoot -Parent) '.task_tmp\mask-mcp-feasibility'
    }
}
$env:FACADE_DETECTOR_PYTHON = $DetectorPython
$env:FACADE_DETECTOR_WEIGHTS = $Weights
$env:FACADE_DETECTOR_DEVICE = 'cpu'
Push-Location $PSScriptRoot
try {
    & '.\.venv\Scripts\python.exe' -m facade_mcp.cli --mask-repo $MaskRepo run --config '.\examples\real_photo_sample\workflow.json'
    if ($LASTEXITCODE -ne 0) { throw 'Local YOLO sample failed' }
} finally { Pop-Location }
