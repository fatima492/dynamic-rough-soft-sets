# Dynamic Rough Soft Sets (DRSS)

A unified framework for temporal uncertainty via soft rough approximations.

Analysis code for the manuscript:

> **Dynamic Rough Soft Sets for Modelling Temporal Uncertainty in Systems with
> Time-Varying Parameters**

DRSS equips the Dynamic Soft Set framework with lower and upper rough
approximation operators at every time instant and four cross-temporal operators
(persistent positive region, cumulative upper approximation, strict possibility
region, optimistic positive region), recovering Pawlak rough sets, static soft
rough sets, dynamic soft sets and multi-granulation rough sets as special cases.

## The benchmark, and why it is the one it is

Results in the paper are computed on `SyntheticDRSSBenchmarkLatent`
(`latent_benchmark.py`). Granules are drawn from a latent clustering of the
universe and never reference the target set; the target is a union of some
clusters plus label noise. A granule therefore falls inside the target because
the data has that structure, not because the sampler placed it there.

Two earlier generators remain in `drss_analysis.py` and are **not** used for any
reported result. `SyntheticDRSSBenchmark` samples granules so that their
*intersection* with the target has a given density, and never makes a granule a
*subset* of it, so the lower approximation is identically empty and the
framework is never exercised. `SyntheticDRSSBenchmarkFixed` removes the
emptiness by drawing certifying granules from inside the target, which makes
membership of a lower approximation a guarantee of membership of the target and
leaks the label in the opposite direction. Both are retained so the difference
between them and the generator in use is visible rather than buried in a commit.

`diagnose_synthetic_benchmark()` reports which regime a benchmark is in: the
number of granules contained in the target, the mean and maximum
lower-approximation frequency, and the AUROC of the boundary frequency alone. It
defaults to the generator the paper uses and accepts any other via
`generator=`. On the default it reports 62 certifying granules and a boundary
AUROC of 0.492 — at chance, so no method can score well by that shortcut.

## Contents

| Path | Description |
|------|-------------|
| `drss_analysis.py` | Single-file implementation of the framework, algorithms, experiments and figures |
| `latent_benchmark.py` | Alternative synthetic generator; see the section above |
| `results/` | Console logs from the runs described below |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT License |

## Running

```bash
python drss_analysis.py              # Parts 1-9: core pipeline
python drss_analysis.py --extended   # Part 10: extended analyses
python drss_analysis.py --all        # both
python drss_analysis.py --extended --full   # full replicate counts (slow)
```

Without `--full`, `main_extended()` uses reduced replicate counts so the suite
finishes in a few minutes. The counts reported in the paper require `--full`.

`RANDOM_SEED = 42` is fixed throughout.

## Extended analyses

`RESULT_TABLE_MAP` in the source maps each function to the table it produces:

| Function | Produces |
|---|---|
| `verify_theory()` | Sec. 4.3, 6.3, 7 — 13 assertions, no table |
| `diagnose_synthetic_benchmark()` | Validity check for Tables 10-12 |
| `run_overlap_entropy_study()` | Table 7 |
| `run_calibration_sweep_6x6()` | Table 10 |
| `run_timing_protocol()` | Table 16 |
| `run_diagnostic_regression()` | Table 18 |
| `run_clinical_with_new_baselines()` | clinical harness (not used in the paper) |
| `run_horizon_sweep()` | Sec. 15 |

`run_clinical_with_new_baselines()` is not called by `main_extended()` and does
not correspond to any table in the paper: the clinical case study was withdrawn
because it could not be reproduced to the standard the rest of the work is held
to. The function is retained as a complete matched-tuning harness for a future
evaluation on credentialed data — see the MIMIC-IV section below.

`baseline_B5_gbdt()` uses `xgboost` when it is importable and scikit-learn's
`GradientBoostingClassifier` otherwise, and reports which one produced the
scores rather than assuming either.

Supporting functions: `tune_threshold_f1()` (matched threshold tuning),
`baseline_B6_gru()`, `baseline_B7_temporal_frs()`, `delong_roc_test()`,
`stratified_bootstrap_ci()`, `overlap_index()`.

## Figures

Filenames differ between the plotting defaults and the paper:

| Function | Default output | Paper figure |
|---|---|---|
| `fig_temporal_evolution()` | `fig_temporal.png` | `fig_crosstemp.png` |
| `fig_synthetic_results()` | `fig_synthetic.png` | `fig_synth_bars.png` |
| `fig_sensitivity_studies()` | `fig_sensitivity.png` | `fig_sensitivity.png` |
| `fig_calibration()` | `fig_calibration.png` | `fig_calibration.png` |
| `fig_boundary_evolution()` | `fig_boundary.png` | `fig_boundary_time.png` |

Pass an explicit `savepath` to write the paper filename. Figures 1-3 of the
paper are TikZ with source embedded in the LaTeX and are not produced here.
`fig_abstract_bars.png`, `fig_healthcare.png` and `fig_mimic.png` are not
produced by any function in this file.

## Platform notes

The script runs on Linux, macOS and Windows.

- Peak memory for the timing protocol is read through `peak_rss_mb()`, which
  uses `resource.getrusage` on Unix and `GetProcessMemoryInfo` via `ctypes` on
  Windows. If neither is available the field reports `nan` and nothing else in
  the run is affected.
- Console output contains set-theoretic symbols, Greek letters and box-drawing
  characters. `stdout` and `stderr` are reconfigured to UTF-8 at import, because
  the Windows console default (cp1252) cannot encode them.

## Requirements

Python 3.9+, plus `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy` and
`scikit-learn`:

```bash
pip install -r requirements.txt
```

`torch` is optional. `baseline_B6_gru()` uses PyTorch when it is importable and
otherwise falls back to an MLP over the flattened sequence, reporting the
substitution in its returned dict rather than labelling the fallback "GRU".

## MIMIC-IV data

The MIMIC-IV ICU dataset is not redistributed here. It is available via
PhysioNet credentialed access: <https://physionet.org/content/mimiciv/>

The clinical experiment runs on a **synthetic clinical proxy** that mirrors the
cohort structure (sensor missingness, regime patterns, prevalence) so the
pipeline is runnable without credentialed data. Numbers produced by the proxy
are not MIMIC-IV results and must not be reported as such on the proxy every
method, including every baseline, attains perfect discrimination, which is a
signature of label leakage rather than of a working predictor. To evaluate on
real data, point the cohort loader at a PhysioNet-credentialed extract.

## Citation

If you use this framework or code, please cite the manuscript. A BibTeX entry
will be added once publication details are finalized.

## License

Released under the MIT License see [LICENSE](LICENSE).
