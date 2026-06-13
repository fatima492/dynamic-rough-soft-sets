# Dynamic Rough Soft Sets (DRSS)

A unified framework for temporal uncertainty via soft rough approximations.

This repository contains the full manuscript and the complete, self-contained
analysis code for the paper:

> **Dynamic Rough Soft Sets: A Unified Framework for Temporal Uncertainty via
> Soft Rough Approximations**

DRSS extends rough soft set theory to time-varying parameter systems. It equips
the Dynamic Soft Set framework with lower/upper rough approximation operators at
every time instant and four cross-temporal operators (persistent positive region,
cumulative upper approximation, strict possibility region, optimistic positive
region), recovering Pawlak rough sets, static soft rough sets, and dynamic soft
sets as special cases.

## Contents

| File | Description |
|------|-------------|
| `drss_analysis.py` | Single-file implementation reproducing **all** analyses, tables, and figures in the manuscript |
| `README.md` | This file |
| `LICENSE` | MIT License |

## What the code reproduces

`drss_analysis.py` is organized into nine parts that map directly onto the paper:

1. **Core framework** — dynamic soft approximation space, pointwise approximations, definability classes, the cross-temporal operators, the operator algebra, and the generalization theorems.
2. **Entropy & granularity** — dynamic soft entropy, monotonicity under refinement, temporal granularity drift, and the Liang–Shi comparison.
3. **Worked examples** — the abstract example, the ICU monitoring example, and the boundary non-monotonicity counterexample.
4. **Algorithms** — the base algorithm (Algorithm 1) and the incremental witness-counter variant (Algorithm 2), with a wall-clock comparison demonstrating the ~9× streaming speedup.
5. **Synthetic benchmark** — generative model, baselines B1–B4, the proposed DRSS, the main results table, sensitivity sweeps, calibration grid, and the cross-temporal ablation.
6. **MIMIC-IV experiment** — baselines B1–B5, calibration analysis (ECE, Brier, reliability diagram), per-slice boundary evolution, sliding-window study, and bootstrap confidence intervals.
7. **Failure-mode analysis** — the three failure modes (FM1–FM3) and the diagnostic regression.
8. **DTRS + DRSS integration** — temporally aware three-way decision regions.
9. **Figures** — publication-ready figures saved as PNG.

The script also numerically verifies the generalization theorems (Pawlak RS, DSS)
and the duality results.

## Requirements

- Python 3.9+
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`

Install everything with:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

## Usage

Run the complete analysis pipeline:

```bash
python drss_analysis.py
```

This prints all result tables to the console and writes the following figures to
the working directory:

- `fig4_temporal_evolution.png`
- `fig6_synthetic_results.png`
- `fig7_sensitivity.png`
- `fig9_calibration.png`
- `fig10_boundary_evolution.png`
- `fig_sliding_window.png`

The random seed is fixed (`RANDOM_SEED = 42`) for reproducibility. For faster
runs, the number of synthetic runs is set to 30 in `main()`; set it to 100 to
reproduce the full-scale results reported in the paper.

## A note on the MIMIC-IV data

The MIMIC-IV ICU dataset is **not** redistributed in this repository. It is
available via PhysioNet credentialed access:
<https://physionet.org/content/mimiciv/>

The clinical experiment in `drss_analysis.py` runs on a **synthetic clinical
proxy** that mirrors the cohort structure (sensor missingness, regime patterns,
prevalence) so the pipeline is fully runnable without credentialed data. To
reproduce the exact manuscript numbers, point the cohort loader at your own
PhysioNet-credentialed extract.

## Citation

If you use this framework or code, please cite the manuscript. A BibTeX entry
will be added here once publication details are finalized.

## License

Released under the MIT License — see [LICENSE](LICENSE).
