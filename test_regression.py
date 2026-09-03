"""
Regression tests for the DRSS reproducibility bundle.

Run with:  python -m pytest test_regression.py -q      (or)  python test_regression.py

These tests pin the structural properties on which the reported results depend, so
that a change to the framework or to the generative model cannot silently invalidate
them.
"""
import math
import numpy as np

from drss_analysis import (ALPHA_MULT_DEFAULT, BETA_MULT_DEFAULT, DRSS,
                           DRSSAlgorithm, DegenerateMixtureBenchmark,
                           DynamicSoftSet, build_abstract_example,
                           default_alpha_beta, dynamic_soft_entropy,
                           liang_shi_entropy)
from drss_generative_model import (LatentRiskBenchmark, b1_scores, b3_scores,
                                   drss_scores)
from drss_extended_analyses import (active_universe, shannon_entropy,
                                    build_icu_example,
                                    relative_definability_class)


def test_lower_subset_upper_without_fullness():
    """Proposition 4.3: containment holds with no fullness hypothesis."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(3, 10))
        U = list(range(n))
        mapping = {f"e{a}": set(rng.choice(n, size=int(rng.integers(0, n + 1)),
                                           replace=False).tolist())
                   for a in range(int(rng.integers(1, 6)))}
        d = DRSS(DynamicSoftSet(U, {"t": mapping}))
        X = set(rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False).tolist())
        L, Up = d.lower("t", X), d.upper("t", X)
        assert L <= Up
        assert Up <= active_universe(d.dss, "t")


def test_entropy_matches_manuscript_values():
    """The entropy values quoted in Section 10.1 must agree with the code."""
    dss, _, _ = build_abstract_example()
    assert abs(dynamic_soft_entropy(dss, "t1") - 1.459) < 5e-3
    assert abs(dynamic_soft_entropy(dss, "t2") - 1.491) < 5e-3
    assert abs(dynamic_soft_entropy(dss, "t3") - 1.396) < 5e-3
    assert abs(liang_shi_entropy(dss) - 0.331) < 5e-3


def test_entropy_normalisations_agree_on_partitions():
    """Proposition 9.4: H_t = H^S_t whenever the cover is a partition."""
    rng = np.random.default_rng(1)
    U = list(range(24))
    for _ in range(300):
        assign = rng.integers(0, int(rng.integers(2, 8)), size=len(U))
        blocks = {}
        for u, b in zip(U, assign):
            blocks.setdefault(f"b{b}", set()).add(u)
        dss = DynamicSoftSet(U, {"t": blocks})
        assert abs(dynamic_soft_entropy(dss, "t") - shannon_entropy(dss, "t")) < 1e-9


def test_icu_example_fails_fullness_at_t1():
    """The ICU example of Section 10.2 violates the fullness hypothesis at t1."""
    dss, drss, X = build_icu_example()
    assert not dss.is_full_at("t1")
    assert active_universe(dss, "t1") == frozenset({"N", "AR"})
    assert relative_definability_class(drss, "t1", X) == "totally_indefinable_rel"


def test_b3_has_no_target_leakage():
    """The corrected B3 must not depend on the target set."""
    bench = LatentRiskBenchmark(seed=3)
    dss, X_star, _ = bench.generate()
    other = set(list(dss.U)[: len(X_star)])
    assert np.allclose(b3_scores(dss, X_star), b3_scores(dss, other))


def test_mixture_generator_is_degenerate():
    """
    The mixture-sampled generator is degenerate: because every granule contains
    both members and non-members of X*, no granule is ever a subset of X* and the
    lower approximation is identically empty.  Asserted so that the property is
    documented rather than rediscovered.
    """
    dss, X_star, _ = DegenerateMixtureBenchmark(rho=0.75, seed=1).generate()
    pure = sum(1 for t in dss.T for g in dss.granules_at(t) if g <= X_star)
    assert pure == 0
    assert len(DRSS(dss).lower(dss.T[0], X_star)) == 0


def test_latent_risk_generator_produces_pure_granules():
    """
    Target-pure granules occur, so the lower approximation carries information, so the lower
    approximation carries information.  Purity is stochastic: individual seeds
    can yield none, which is failure mode FM2 rather than a defect, so the
    property is asserted across seeds rather than for a single cohort.
    """
    total_pure, slices_with_lower, total_slices = 0, 0, 0
    for seed in range(10):
        dss, X_star, _ = LatentRiskBenchmark(rho=0.75, seed=seed).generate()
        d = DRSS(dss)
        total_pure += sum(1 for t in dss.T for g in dss.granules_at(t) if g <= X_star)
        slices_with_lower += sum(1 for t in dss.T if len(d.lower(t, X_star)) > 0)
        total_slices += len(dss.T)
    assert total_pure > 0
    assert slices_with_lower / total_slices > 0.10


def test_union_static_baseline_collapses():
    """The paper's structural claim: pooling across time destroys certification."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for s in range(10):
        dss, X_star, y = LatentRiskBenchmark(seed=s).generate()
        aucs.append(roc_auc_score(y, b1_scores(dss, X_star)))
    assert abs(float(np.mean(aucs)) - 0.5) < 0.05


def test_default_weights_match_manuscript():
    """
    Every code path must default to the adopted score weights of Section 11.3,
    (alpha~, beta~) = (0.5, 0.25) (Tables 8 and 9).  The superseded (1, 2) is
    available only through the explicit *_LEGACY constants.
    """
    import inspect
    assert (ALPHA_MULT_DEFAULT, BETA_MULT_DEFAULT) == (0.5, 0.25)
    assert default_alpha_beta(200) == (0.5 / 200, 0.25 / 200)
    sig = inspect.signature(drss_scores)
    assert sig.parameters["alpha_mult"].default == 0.5
    assert sig.parameters["beta_mult"].default == 0.25
    dss, X_star, _ = LatentRiskBenchmark(n_T=10, seed=11).generate()
    alg = DRSSAlgorithm(dss, X_star)
    assert abs(alg.alpha * len(dss.U) - 0.5) < 1e-12
    assert abs(alg.beta * len(dss.U) - 0.25) < 1e-12
    # the bundle must contain no remaining hard-coded (1/|U|, 2/|U|) defaults
    import pathlib
    for fn in ["drss_analysis.py", "drss_extended_analyses.py", "drss_generative_model.py"]:
        src = pathlib.Path(__file__).with_name(fn).read_text(encoding="utf-8")
        assert "1.0 / n_U, 2.0 / n_U" not in src, fn
        assert "alpha_mult=1.0, beta_mult=2.0" not in src, fn


def test_incremental_matches_base():
    """Algorithms 1 and 2 must produce identical approximations."""
    dss, X_star, _ = LatentRiskBenchmark(n_T=30, seed=5).generate()
    alg = DRSSAlgorithm(dss, X_star)
    base, _ = alg.run_base()
    incr, _ = alg.run_incremental()
    for b, i in zip(base.to_dict("records"), incr.to_dict("records")):
        assert b["lower"] == i["lower"]
        assert b["upper"] == i["upper"]


if __name__ == "__main__":
    import sys
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS  {name}")
            except Exception as e:  # report any failure, with traceback
                fails += 1; print(f"FAIL  {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print(f"\n{'all tests passed' if not fails else str(fails) + ' FAILURES'}")
    sys.exit(1 if fails else 0)
