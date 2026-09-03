# Dynamic Rough Soft Sets (DRSS)

Reproducibility bundle for the paper

> **Dynamic Rough Soft Sets for Temporal Uncertainty Modelling via Soft Rough
> Approximations**

DRSS extends rough soft set theory to time-varying parameter systems. It equips the
Dynamic Soft Set framework with lower and upper rough approximation operators at every
time instant, together with four cross-temporal operators (persistent positive region,
cumulative upper approximation, strict possibility region, optimistic positive
region), recovering Pawlak rough sets, static soft rough sets, and dynamic soft sets
as special cases.

Every synthetic-benchmark number quoted in the paper is reproducible from this bundle
(see the table below). Section, table and figure numbers refer to the revised
manuscript.

---

## Contents

| File | Description |
|------|-------------|
| `drss_analysis.py` | Core framework: approximations, cross-temporal operators, operator algebra, entropy, Algorithms 1 and 2, examples, DTRS integration, MIMIC-IV proxy pipeline |
| `drss_generative_model.py` | Class-pure synthetic generative model (`LatentRiskBenchmark`) and the baseline decision rules B1–B3, B4 features, DRSS score |
| `drss_extended_analyses.py` | Produces every synthetic-benchmark table and figure under one protocol; writes `extended_results.json` and the two manuscript figures |
| `extended_results.json` | Output of `drss_extended_analyses.py` as shipped (the numbers in the paper) |
| `test_regression.py` | Tests pinning the adopted score weights and the structural properties on which the results depend |
| `fig_synth_bars.png`, `fig_sensitivity.png` | Figures 6 and 7 as produced by the script |
| `requirements.txt`, `LICENSE`, `gitignore` | Environment and licensing |

## Usage

Requires Python 3.9 or later. Source files are UTF-8; on Windows the scripts set the
console to replace unprintable characters, so no code-page setting is needed.

```bash
pip install -r requirements.txt

python test_regression.py           # test suite; or: python -m pytest test_regression.py -q
python drss_extended_analyses.py    # all synthetic-benchmark tables/figures -> extended_results.json
python drss_analysis.py             # core framework, examples, algorithms, MIMIC-IV proxy, failure modes, DTRS
```

Random seeds are fixed (`RANDOM_SEED = 42` plus per-experiment seeds). The extended
analyses take a few minutes on one core. Wall-clock timings depend on hardware and
will differ in absolute value from run to run; the speedup ratios reported in the
paper were obtained on one core under Python 3.12 / NumPy 2.4.

## Score weights

The element-wise score of Section 11.3 is `s(u) = f^L + alpha~ f^U - beta~ f^B` with
`(alpha~, beta~) = (alpha|U|, beta|U|)`. The adopted default is **(0.5, 0.25)**
(Table 8, Table 9) and is the default in every code path (`ALPHA_MULT_DEFAULT`,
`BETA_MULT_DEFAULT` in `drss_analysis.py`). The superseded default (1, 2) lies in the
penalised region and is kept only as `ALPHA_MULT_LEGACY`, `BETA_MULT_LEGACY` so that
the comparison rows of Tables 8 and 15 can be regenerated. The test
`test_default_weights_match_manuscript` asserts this.

## Which function produces which number

All in `drss_extended_analyses.py` unless stated. Every synthetic result uses the
class-pure `LatentRiskBenchmark`, threshold tuning on a training split for every
method alike, and DRSS at the adopted weights.

| Manuscript item | Entry point | JSON key |
|---|---|---|
| Proposition 4.3 (containment without fullness) | `monotone_containment_check` | `containment` |
| Definition 4.6 (relative definability, ICU example) | `partial_coverage_definability` | `partial_coverage` |
| Proposition 9.4, Section 9.2, Table 5 (entropy) | `entropy_normalisation_study`, `entropy_overlap_study` | `entropy_normalisation`, `entropy_overlap` |
| Section 10.1 entropy values | `abstract_example_entropy` | `abstract_entropy` |
| Section 11.2 work ratio 0.38 | `incremental_cost_accounting` | `incremental_cost` |
| Table 8, Figure 7 (left) | `alpha_beta_grid` | `alpha_beta` |
| Table 11, Figure 6 | `synthetic_benchmark_main` | `synthetic_main` |
| Table 12, Figure 7 (right) | `rho_sensitivity` | `rho_sensitivity` |
| Section 12.3 further sensitivities | `mean_uptime_sweep`, `active_set_sweep`, `crosstemporal_ablation` | `mean_uptime_sweep`, `active_set_sweep`, `crosstemporal_ablation` |
| Section 12.4 wall-clock and scaling; Section 15 | `wall_clock_protocol`, `wall_clock_scaling` | `wall_clock`, `wall_clock_scaling` |
| Table 15 | `threshold_tuned_baselines` | `threshold_tuned` |
| Section 13.1 FM1 ablations | `no_regime_ablation`, `static_granule_ablation` | `fm1_no_regime`, `fm1_static_granules` |
| Table 17 | `diagnostic_regression` | `diagnostic_regression` |
| Figures 6 and 7 | `make_manuscript_figures` | — |
| Section 10 examples, Table 18, Algorithms 1–2 | `drss_analysis.py` (`run_abstract_example`, `run_icu_example`, `run_dtrs_example`, `DRSSAlgorithm`) | — |

## Notes on the implementation

**Granule purity drives the lower approximation.** A granule enters a soft lower
approximation only if it is entirely contained in the target set. A generative model
in which every granule mixes members and non-members of the target yields an
identically empty lower approximation at every slice, and the framework degenerates.
`LatentRiskBenchmark` draws class-pure granules with probability `rho`, which is what
makes `rho` a fidelity parameter. The degenerate mixture generator is retained in
`drss_analysis.DegenerateMixtureBenchmark` purely as a documented counter-example
(`test_mixture_generator_is_degenerate`) and is not used for any manuscript number.

**Baselines must not read the target.** The dynamic soft set baseline B3 scores
objects by granule incidence alone; `test_b3_has_no_target_leakage` asserts that the
score is invariant to the choice of target.

**Two structural properties are pinned:** Algorithms 1 and 2 produce identical
approximations at every slice (`test_incremental_matches_base`), and the union-static
baseline sits at chance because pooling across time destroys granule purity
(`test_union_static_baseline_collapses`).

## A note on the MIMIC-IV data

MIMIC-IV is **not** redistributed in this bundle. It is available via PhysioNet
credentialed access: <https://physionet.org/content/mimiciv/>

The clinical experiment in `drss_analysis.py` runs on a **synthetic clinical proxy**
mirroring the cohort structure (sensor missingness, regime patterns, prevalence), so
the pipeline is runnable without credentialed data. Numbers produced by the proxy are
**not** the manuscript's MIMIC-IV numbers (Table 14, Table 16, Figures 8–10) and
should not be read as such; reproducing those requires pointing the cohort loader at
a credentialed extract. The discretisation thresholds, temporal binning, and
missing-value policy are given in full in the paper.

## Citation
If you use this framework or code, please cite the manuscript. A BibTeX entry will be added once publication details are finalized.

## License

MIT — see [LICENSE](LICENSE).
