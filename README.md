# Dissertation experiment package

This directory contains the code, configurations, compact result files, and
selected qualitative examples actually used in the dissertation.

## Contents

- `code/scripts`: experiment entry points and analysis scripts.
- `code/torchreid/semantic`: the final pedestrian semantic arbitration code.
- `code/torchreid/core`: the relevant OSNet, evaluation, dataset, and
  k-reciprocal source files from Torchreid.
- `configs`: the Market-1501 OSNet training configuration and fixed Scheme 2
  configuration.
- `tests`: semantic and k-reciprocal unit tests.
- `results/scheme2`: Duke A-F ablation, frozen MSMT17 validation, and
  k-reciprocal plus configuration D results.
- `results/diagnostics`: failure funnel, reachability, candidate coverage,
  oracle, extractor comparison, and case-level diagnostic tables.
- `results/retrieval_postprocessing`: k-reciprocal, AQE, and DBA reports.
- `figures`: the colour-only and structured-attribute examples selected for
  Chapter 6.
- `environment`: software requirements and recorded environment information.

## Main execution order

1. `export_eval_artifacts.py` exports reusable OSNet features and metadata.
2. `candidate_selection.py` builds the Top-20, delta=0.02 ambiguity sets.
3. `extract_siglip2_scheme2_attributes.py` creates the cached multi-view
   SigLIP2 attributes.
4. `run_scheme2_ablation.py` evaluates configurations A-F on Duke.
5. `run_fixed_scheme2_d.py` evaluates the frozen configuration D on MSMT17 or
   after k-reciprocal reranking.

