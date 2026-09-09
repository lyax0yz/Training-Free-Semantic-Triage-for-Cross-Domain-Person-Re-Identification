param([string]$ProjectRoot = "D:\deep-person-reid")

$ErrorActionPreference = "Stop"
$gpuPython = Join-Path $ProjectRoot ".venv\siglip2-gpu\Scripts\python.exe"
$downloadDir = Join-Path $ProjectRoot "artifacts\gpu-wheels"
$torchWheel = Join-Path $downloadDir "torch-2.13.0+cu130-cp314-cp314-win_amd64.whl"
$visionWheel = Join-Path $downloadDir "torchvision-0.28.0+cu130-cp314-cp314-win_amd64.whl"
$torchUrl = "https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp314-cp314-win_amd64.whl"
$visionUrl = "https://download-r2.pytorch.org/whl/cu130/torchvision-0.28.0%2Bcu130-cp314-cp314-win_amd64.whl"

if (-not (Test-Path -LiteralPath $gpuPython)) {
    throw "GPU virtual environment is missing: $gpuPython"
}
New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null

function Download-Wheel([string]$Url, [string]$Destination) {
    Write-Host "Downloading with persistent resume: $Destination"
    $attempt = 0
    while ($true) {
        $attempt += 1
        $existingBytes = if (Test-Path -LiteralPath $Destination) {
            (Get-Item -LiteralPath $Destination).Length
        } else { 0 }
        Write-Host "Download attempt $attempt; resuming from $existingBytes bytes"
        
        & curl.exe -L --fail --connect-timeout 30 -C - `
            --output $Destination $Url
        if ($LASTEXITCODE -eq 0) { break }
        if ($attempt -ge 500) { throw "curl failed 500 times for $Url" }
        Start-Sleep -Seconds 2
    }
}

Download-Wheel $torchUrl $torchWheel
Download-Wheel $visionUrl $visionWheel

& $gpuPython -m pip install $torchWheel $visionWheel --timeout 120 --retries 20
if ($LASTEXITCODE -ne 0) { throw "Local CUDA PyTorch installation failed." }

& $gpuPython -m pip install transformers==5.15.1 pillow==12.3.0 `
    --timeout 120 --retries 20
if ($LASTEXITCODE -ne 0) { throw "SigLIP2 dependency installation failed." }

& $gpuPython -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
if ($LASTEXITCODE -ne 0) { throw "CUDA verification failed." }
