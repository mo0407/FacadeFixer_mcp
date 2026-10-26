param([string]$Python = "", [switch]$SkipUpstream, [switch]$Web)
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$TaskCache = Join-Path $ProjectRoot '.uv-cache'
if (-not $Python) {
    $BundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $Python = if (Test-Path -LiteralPath $BundledPython) { $BundledPython } else { '3.12' }
}
$TaskPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $TaskPython)) {
    & uv --cache-dir $TaskCache venv --python $Python (Join-Path $ProjectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment' }
}
& uv --cache-dir $TaskCache pip sync --python $TaskPython (Join-Path $ProjectRoot 'requirements.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install dependencies' }
& uv --cache-dir $TaskCache pip install --python $TaskPython -e $ProjectRoot --no-deps
if ($LASTEXITCODE -ne 0) { throw 'Failed to install project' }
if ($Web) {
    & uv --cache-dir $TaskCache pip install --python $TaskPython 'starlette>=0.40' 'uvicorn>=0.30' 'pypdf>=5,<7'
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install web dependencies' }
}
if (-not $SkipUpstream) {
    $VendorRoot = Join-Path $ProjectRoot 'vendor'
    New-Item -ItemType Directory -Path $VendorRoot -Force | Out-Null
    $Repositories = @(
        @{Name='FacadeFixer_mask_api'; Revision='6b5611f209aa6f0d82d530f35ab7ae264c7abf93'},
        @{Name='FacadeFixer_report'; Revision='0c679ad29980a7bed7998058b68f946d27e2532b'}
    )
    foreach ($Repository in $Repositories) {
        $Checkout = Join-Path $VendorRoot $Repository.Name
        if (-not (Test-Path -LiteralPath $Checkout)) {
            & git -c http.sslBackend=openssl clone "https://github.com/mo0407/$($Repository.Name).git" $Checkout
            if ($LASTEXITCODE -ne 0) { throw "Failed to clone $($Repository.Name)" }
            & git -C $Checkout checkout $Repository.Revision
            if ($LASTEXITCODE -ne 0) { throw "Failed to select verified revision for $($Repository.Name)" }
        }
    }
}
Write-Host 'Setup complete. Set FACADE_MASK_REPO to your upstream checkout, then run the demo.'
