param(
    [string]$ProjectRoot = "D:\deep-person-reid",
    [int]$SetupProcessId = 42960,
    [int]$CpuMasterProcessId = 38124,
    [int[]]$CpuPythonProcessIds = @(38908, 40660)
)

$ErrorActionPreference = "Stop"
$gpuPython = Join-Path $ProjectRoot ".venv\siglip2-gpu\Scripts\python.exe"
$artifactDir = Join-Path $ProjectRoot "artifacts"
$candidates = Join-Path $artifactDir "market_to_msmt17_relative_candidates.json"
$cache = Join-Path $artifactDir "market_to_msmt17_siglip2_scheme2_multiview.json"
$smokeCache = Join-Path $artifactDir "market_to_msmt17_siglip2_gpu_smoke_160.json"
$smokeInspection = Join-Path $artifactDir "market_to_msmt17_siglip2_gpu_smoke_160.csv"
$config = Join-Path $ProjectRoot "configs\scheme2_market_to_duke.json"
$artifact = Join-Path $artifactDir "market_to_msmt17_osnet_x1_0.npz"

Write-Host "Waiting for GPU environment setup process $SetupProcessId"
while (Get-Process -Id $SetupProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

& $gpuPython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    Write-Host "GPU environment verification failed; leaving CPU extraction running."
    exit 2
}

$timer = [System.Diagnostics.Stopwatch]::StartNew()
& $gpuPython -u (Join-Path $ProjectRoot "scripts\extract_siglip2_scheme2_attributes.py") `
    --candidates $candidates --cache $smokeCache --inspection-csv $smokeInspection `
    --hf-cache (Join-Path $artifactDir "siglip2-hf-cache") `
    --device cuda --precision float16 --batch-size 1 --torch-threads 4 `
    --max-images 160 --save-every 160 --sample-size 20 --local-files-only
$smokeExit = $LASTEXITCODE
$timer.Stop()
$seconds = $timer.Elapsed.TotalSeconds
Write-Host ("GPU smoke result: exit={0}, seconds={1:N1}, images_per_second={2:N2}" -f `
    $smokeExit, $seconds, (160.0 / [Math]::Max($seconds, 0.001)))


if ($smokeExit -ne 0 -or $seconds -ge 145.0) {
    Write-Host "GPU is not safely faster than CPU; leaving CPU extraction running."
    exit 3
}

Write-Host "GPU validation passed; stopping the exact CPU extraction processes."
foreach ($processId in @($CpuMasterProcessId) + $CpuPythonProcessIds) {
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) { Stop-Process -Id $processId -Force }
}

if (Test-Path -LiteralPath $cache) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $archive = Join-Path $artifactDir "market_to_msmt17_siglip2_scheme2_multiview.cpu_partial_$stamp.json"
    Move-Item -LiteralPath $cache -Destination $archive
    Write-Host "Archived the internally consistent CPU partial cache to $archive"
}

& $gpuPython -u (Join-Path $ProjectRoot "scripts\extract_siglip2_scheme2_attributes.py") `
    --candidates $candidates --cache $cache `
    --inspection-csv (Join-Path $artifactDir "market_to_msmt17_siglip2_scheme2_inspection.csv") `
    --hf-cache (Join-Path $artifactDir "siglip2-hf-cache") `
    --device cuda --precision float16 --batch-size 1 --torch-threads 4 `
    --save-every 160 --sample-size 20 --local-files-only
if ($LASTEXITCODE -ne 0) { throw "Full MSMT17 GPU extraction failed." }

& $gpuPython -u (Join-Path $ProjectRoot "scripts\run_fixed_scheme2_d.py") `
    --artifact $artifact --scheme2-cache $cache --config $config `
    --ranking-name "OSNet cosine" --target msmt17 `
    --output (Join-Path $artifactDir "market_to_msmt17_scheme2_D.json") `
    --transitions-csv (Join-Path $artifactDir "market_to_msmt17_scheme2_D_transitions.csv") `
    --overwrite
if ($LASTEXITCODE -ne 0) { throw "MSMT17 fixed-D GPU evaluation failed." }
