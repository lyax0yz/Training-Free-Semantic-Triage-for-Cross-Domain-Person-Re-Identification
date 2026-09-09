param([string]$ProjectRoot = "D:\deep-person-reid")

$ErrorActionPreference = "Stop"
$semanticPython = Join-Path $ProjectRoot ".venv\siglip2\Scripts\python.exe"
$reidPython = Join-Path $ProjectRoot ".venv\torchreid\python.exe"
$artifactDir = Join-Path $ProjectRoot "artifacts"
$config = Join-Path $ProjectRoot "configs\scheme2_market_to_duke.json"

$dukeArtifact = Join-Path $artifactDir "market_to_duke_osnet_x1_0.npz"
$dukeDistance = Join-Path $artifactDir "market_to_duke_k_reciprocal_distmat.npy"
$dukeDistancePartial = Join-Path $artifactDir "market_to_duke_k_reciprocal_distmat.partial.npy"
$dukeDistanceReport = Join-Path $artifactDir "market_to_duke_k_reciprocal_distmat_report.json"
$dukeCandidates = Join-Path $artifactDir "market_to_duke_k_reciprocal_scheme2_candidates.json"
$dukeOriginalCache = Join-Path $artifactDir "market_to_duke_siglip2_scheme2_multiview.json"
$dukeCombinedCache = Join-Path $artifactDir "market_to_duke_kreciprocal_siglip2_scheme2_multiview.json"

$msmtArtifact = Join-Path $artifactDir "market_to_msmt17_osnet_x1_0.npz"
$msmtCandidates = Join-Path $artifactDir "market_to_msmt17_relative_candidates.json"
$msmtCache = Join-Path $artifactDir "market_to_msmt17_siglip2_scheme2_multiview.json"

Push-Location $ProjectRoot
try {
    # Scheme 1 + Scheme 2: retain the exact published/default parameters used
    # by the existing Duke k-reciprocal baseline.
    if (-not (Test-Path -LiteralPath $dukeDistance)) {
        & $reidPython -u scripts\export_k_reciprocal_distance_matrix.py `
            --artifact $dukeArtifact `
            --output $dukeDistancePartial `
            --report $dukeDistanceReport `
            --k1 20 --k2 6 --lambda-value 0.3 `
            --distance-block-size 128 --overwrite
        if ($LASTEXITCODE -ne 0) { throw "k-reciprocal distance export failed." }
        Move-Item -LiteralPath $dukeDistancePartial -Destination $dukeDistance -Force
    }

    & $semanticPython -u scripts\build_scheme2_candidates_from_distance.py `
        --artifact $dukeArtifact --distance-matrix $dukeDistance `
        --output $dukeCandidates --topk 20 --delta 0.02 `
        --ranking-name "k-reciprocal(k1=20,k2=6,lambda=0.3)"
    if ($LASTEXITCODE -ne 0) { throw "k-reciprocal semantic candidate collection failed." }

    if (-not (Test-Path -LiteralPath $dukeCombinedCache)) {
        Copy-Item -LiteralPath $dukeOriginalCache -Destination $dukeCombinedCache
    }
    & $semanticPython -u scripts\extract_siglip2_scheme2_attributes.py `
        --candidates $dukeCandidates --cache $dukeCombinedCache `
        --inspection-csv (Join-Path $artifactDir "market_to_duke_kreciprocal_scheme2_inspection.csv") `
        --hf-cache artifacts\siglip2-hf-cache --batch-size 16 --torch-threads 8 `
        --save-every 160 --sample-size 20 --local-files-only --resume
    if ($LASTEXITCODE -ne 0) { throw "Duke k-reciprocal Scheme 2 cache completion failed." }

    & $semanticPython -u scripts\run_fixed_scheme2_d.py `
        --artifact $dukeArtifact --scheme2-cache $dukeCombinedCache --config $config `
        --distance-matrix $dukeDistance `
        --ranking-name "k-reciprocal(k1=20,k2=6,lambda=0.3), score=1-distance" `
        --target dukemtmcreid `
        --output (Join-Path $artifactDir "market_to_duke_kreciprocal_scheme2_D.json") `
        --transitions-csv (Join-Path $artifactDir "market_to_duke_kreciprocal_scheme2_D_transitions.csv") `
        --overwrite
    if ($LASTEXITCODE -ne 0) { throw "Duke k-reciprocal + Scheme 2 D evaluation failed." }

    # Frozen D external target validation.  Reuse OSNet features and process
    # each candidate image only once in a resumable cache.
    & $semanticPython -u scripts\extract_siglip2_scheme2_attributes.py `
        --candidates $msmtCandidates --cache $msmtCache `
        --inspection-csv (Join-Path $artifactDir "market_to_msmt17_siglip2_scheme2_inspection.csv") `
        --hf-cache artifacts\siglip2-hf-cache --batch-size 16 --torch-threads 8 `
        --save-every 160 --sample-size 20 --local-files-only --resume
    if ($LASTEXITCODE -ne 0) { throw "MSMT17 fixed-D Scheme 2 extraction failed." }

    & $semanticPython -u scripts\run_fixed_scheme2_d.py `
        --artifact $msmtArtifact --scheme2-cache $msmtCache --config $config `
        --ranking-name "OSNet cosine" --target msmt17 `
        --output (Join-Path $artifactDir "market_to_msmt17_scheme2_D.json") `
        --transitions-csv (Join-Path $artifactDir "market_to_msmt17_scheme2_D_transitions.csv") `
        --overwrite
    if ($LASTEXITCODE -ne 0) { throw "MSMT17 fixed-D evaluation failed." }
}
finally {
    Pop-Location
}
