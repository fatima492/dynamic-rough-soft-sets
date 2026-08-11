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

Every number quoted in the paper is reproducible from this bundle.

---

## Contents

| File | Description |
|------|-------------|
| `drss_analysis.py` | Core framework: approximations, cross-temporal operators, operator algebra, entropy, algorithms, figures |
| `drss_generative_model.py` | Synthetic generative model and the baseline decision rules |
| `drss_extended_analyses.py` | Supporting analyses for the theoretical and empirical claims |
| `test_regression.py` | Tests pinning the structural properties on which the results depend |
| `requirements.txt`, `LICENSE`, `gitignore` | Environment and licensing |

## Usage

```bash
pip install -r requirements.txt

python drss_analysis.py            # core analyses, tables and figures
python drss_extended_analyses.py   # supporting analyses; writes extended_results.json
python test_regression.py          # test suite (pytest optional)
```

The random seed is fixed (`RANDOM_SEED = 42`).

## Supporting analyses

`drss_extended_analyses.py` is organised into seven parts, each supporting a specific
claim in the paper:

| Part | Claim supported | Entry point |
|------|-----------------|-------------|
| A | The lower approximation is contained in the upper, and both in the active universe, with no structural hypothesis (Proposition 4.4) | `monotone_containment_check` |
| B | The mixed and Shannon-normalised entropies agree on partitions and diverge under overlap (Proposition 9.3, Section 9.4) | `entropy_normalisation_study`, `shannon_entropy` |
| C | Definability relativised to the active universe when fullness fails (Definition 4.6, Section 10.2) | `partial_coverage_definability`, `relative_definability_class` |
| D | Sensitivity of the element-wise score to its weights (Table 3) | `alpha_beta_grid` |
| E | Like-for-like baseline comparison with every decision threshold tuned (Section 11.5) | `threshold_tuned_baselines` |
| F | Per-update cost of the incremental algorithm, including changes to the mappings `F_t` (Section 10.2) | `incremental_cost_accounting` |
| G | Wall-clock protocol: stream size, repetitions, variability, peak memory (Section 11.4) | `wall_clock_protocol` |
| — | Granularity measures against overlap density (Table 5) | `entropy_overlap_study` |

## Notes on the implementation

Two points are worth stating explicitly, because both affect how the results should be
read and both are asserted by tests.

**Granule purity drives the lower approximation.** A granule enters a soft lower
approximation only if it is entirely contained in the target set. A generative model
in which every granule mixes members and non-members of the target therefore yields an
identically empty lower approximation at every slice, and the framework degenerates.
`LatentRiskBenchmark` in `drss_generative_model.py` accordingly draws class-pure
granules with probability `rho`, which is what makes `rho` a fidelity parameter. The
degenerate behaviour of mixture sampling is retained in `SyntheticDRSSBenchmark` and
asserted by `test_mixture_generator_is_degenerate`, so the distinction cannot be lost
by accident.

**Baselines must not read the target.** The dynamic soft set baseline scores objects by
granule incidence alone. Scoring by membership in `granule & X_star` would read the
labels directly and make the baseline an oracle; `test_b3_has_no_target_leakage`
asserts that the score is invariant to the choice of target.

Two quantitative properties reported in the paper are also pinned here: the incremental
and base algorithms produce identical approximations at every slice
(`test_incremental_matches_base`), and the union-static baseline sits at chance because
pooling across time destroys granule purity (`test_union_static_baseline_collapses`).

## A note on the MIMIC-IV data

MIMIC-IV is **not** redistributed in this bundle. It is available via PhysioNet
credentialed access: <https://physionet.org/content/mimiciv/>

The clinical experiment runs on a **synthetic clinical proxy** mirroring the cohort
structure (sensor missingness, regime patterns, prevalence), so the pipeline is
runnable without credentialed data. Numbers produced by the proxy are **not** the
manuscript's MIMIC-IV numbers and should not be read as such; reproducing the MIMIC-IV
table requires pointing the cohort loader at a credentialed extract. The
discretisation thresholds, temporal binning, and missing-value policy are given in
full in the paper.

## Citation
If you use this framework or code, please cite the manuscript. A BibTeX entry will be added once publication details are finalized.

## License

MIT — see [LICENSE](LICENSE).

