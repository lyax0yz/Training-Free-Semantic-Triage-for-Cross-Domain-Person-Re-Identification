param(
    [string]$ProjectRoot = "D:\deep-person-reid",
    [int]$BatchSize = 16
)

$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\siglip2\Scripts\python.exe"
$candidateFile = Join-Path $ProjectRoot "artifacts\market_to_duke_relative_candidates.json"
$schemeCache = Join-Path $ProjectRoot "artifacts\market_to_duke_siglip2_scheme2_multiview.json"
$inspectionCsv = Join-Path $ProjectRoot "artifacts\market_to_duke_siglip2_scheme2_inspection.csv"
$outputDir = Join-Path $ProjectRoot "artifacts\market_to_duke_scheme2"

Push-Location $ProjectRoot
try {
    & $python -u scripts\extract_siglip2_scheme2_attributes.py `
        --candidates $candidateFile `
        --cache $schemeCache `
        --inspection-csv $inspectionCsv `
        --hf-cache artifacts\siglip2-hf-cache `
            --batch-size $BatchSize `
            --torch-threads 8 `
        --save-every 160 `
        --sample-size 20 `
        --local-files-only `
        --resume
    if ($LASTEXITCODE -ne 0) { throw "Scheme 2 SigLIP2 extraction failed." }

    & $python -u scripts\run_scheme2_ablation.py `
        --artifact artifacts\market_to_duke_osnet_x1_0.npz `
        --candidates $candidateFile `
        --legacy-siglip2-cache artifacts\market_to_duke_siglip2_pedestrian_attributes.json `
        --scheme2-cache $schemeCache `
        --config configs\scheme2_market_to_duke.json `
        --output-dir $outputDir `
        --overwrite
    if ($LASTEXITCODE -ne 0) { throw "Scheme 2 ablation failed." }

    & $python -u scripts\render_scheme2_case_diagnostics.py `
        --diagnostic-dir (Join-Path $outputDir "case_diagnostics") `
        --output-dir (Join-Path $outputDir "case_visuals")
    if ($LASTEXITCODE -ne 0) { throw "Scheme 2 diagnostic rendering failed." }
}
finally {
    Pop-Location
}
