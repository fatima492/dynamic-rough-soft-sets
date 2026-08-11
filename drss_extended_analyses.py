"""
=============================================================================
  Dynamic Rough Soft Sets (DRSS) — Extended Analyses
=============================================================================

Supporting analyses for the theoretical and empirical claims of the paper:

  A. Containment of the lower approximation in the upper approximation, and of
     both in the active universe, without structural hypotheses.
  B. Entropy normalisation: agreement of the mixed and Shannon-normalised
     forms on partitions, and their divergence under overlapping granules.
  C. Definability under partial coverage, relativised to the active universe.
  D. Sensitivity of the element-wise score to its weights.
  E. Like-for-like baseline comparison with all decision thresholds tuned.
  F. Per-update cost of the incremental algorithm, including changes to the
     mappings F_t.
  G. Wall-clock protocol: stream size, repetitions, variability, peak memory.

Run with:

    python drss_extended_analyses.py

All results are printed to stdout and written to `extended_results.json`.
The random seed is fixed so that every number quoted in the paper is
reproducible from this file alone.
=============================================================================
"""

from __future__ import annotations

import json
import math
import time
import tracemalloc
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from drss_analysis import (
    DRSS,
    DRSSAlgorithm,
    DynamicSoftSet,
    build_abstract_example,
    dynamic_soft_entropy,
    liang_shi_entropy,
)
from drss_generative_model import (
    LatentRiskBenchmark,
    b1_scores,
    b2_scores,
    b3_scores,
    drss_scores,
    feature_matrix,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



# =============================================================================
# A.  Lower / upper containment
# =============================================================================

def shannon_entropy(dss: DynamicSoftSet, t) -> float:
    """
    Fully Shannon-normalised granule entropy.

        H^S_t = - sum_a (|F_t(a)| / N_t) log2(|F_t(a)| / N_t),
        N_t   = sum_b |F_t(b)|.

    Both the weight and the argument of the logarithm are normalised by the
    same constant, so H^S_t is the Shannon entropy of a genuine probability
    distribution over granules. On a partition N_t = |U| and H^S_t coincides
    with the mixed-normalisation H_t.
    """
    granules = [g for g in dss.granules_at(t) if len(g) > 0]
    if not granules:
        return 0.0
    N_t = sum(len(g) for g in granules)
    if N_t == 0:
        return 0.0
    H = 0.0
    for g in granules:
        p = len(g) / N_t
        if p > 0:
            H -= p * math.log2(p)
    return H


def unnormalised_entropy(dss: DynamicSoftSet, t) -> float:
    """Unnormalised variant of the granule entropy."""
    granules = [g for g in dss.granules_at(t) if len(g) > 0]
    n_U = len(dss.U)
    return -sum(len(g) * math.log2(len(g) / n_U) for g in granules)


def active_universe(dss: DynamicSoftSet, t) -> frozenset:
    """U*_t = union of the active granules at t (the 'active universe')."""
    covered = set()
    for g in dss.granules_at(t):
        covered |= set(g)
    return frozenset(covered)


def monotone_containment_check(n_trials: int = 4000, seed: int = 0) -> Dict:
    """
    Verifies empirically that the lower approximation is always contained in
    the upper approximation, WITHOUT any fullness or partition hypothesis, and
    that both are always contained in the active universe U*_t.

    The proof is one line (a nonempty granule contained in X necessarily meets
    X); this check exercises it across degenerate covers, sampling families
    that are variously partial, overlapping and empty.
    """
    rng = np.random.default_rng(seed)
    violations_lower_upper = 0
    violations_active = 0
    full_cases = 0
    nonfull_cases = 0

    for _ in range(n_trials):
        n_U = int(rng.integers(3, 12))
        U = list(range(n_U))
        n_params = int(rng.integers(1, 7))
        mapping = {}
        for a in range(n_params):
            size = int(rng.integers(0, n_U + 1))  # size 0 allowed: empty granule
            members = set(rng.choice(n_U, size=size, replace=False).tolist())
            mapping[f"e{a}"] = members
        dss = DynamicSoftSet(U, {"t": mapping})
        drss = DRSS(dss)

        X_size = int(rng.integers(0, n_U + 1))
        X = set(rng.choice(n_U, size=X_size, replace=False).tolist())

        L = drss.lower("t", X)
        Up = drss.upper("t", X)
        Ustar = active_universe(dss, "t")

        if not (L <= Up):
            violations_lower_upper += 1
        if not (Up <= Ustar):
            violations_active += 1
        if dss.is_full_at("t"):
            full_cases += 1
        else:
            nonfull_cases += 1

    return {
        "n_trials": n_trials,
        "violations_lower_subset_upper": violations_lower_upper,
        "violations_upper_subset_active_universe": violations_active,
        "full_covers_sampled": full_cases,
        "non_full_covers_sampled": nonfull_cases,
    }


# =============================================================================
# B.  Entropy normalisation
# =============================================================================

def _random_partition(U: List[int], n_blocks: int, rng) -> Dict[str, set]:
    """Random partition of U into at most n_blocks nonempty blocks."""
    assign = rng.integers(0, n_blocks, size=len(U))
    blocks: Dict[str, set] = {}
    for u, b in zip(U, assign):
        blocks.setdefault(f"b{b}", set()).add(u)
    return blocks


def _refine_partition(blocks: Dict[str, set], rng) -> Dict[str, set]:
    """Split every block of size >= 2 into two nonempty parts."""
    refined: Dict[str, set] = {}
    for name, blk in blocks.items():
        members = sorted(blk)
        if len(members) < 2:
            refined[name] = set(members)
            continue
        cut = int(rng.integers(1, len(members)))
        refined[f"{name}_a"] = set(members[:cut])
        refined[f"{name}_b"] = set(members[cut:])
    return refined


def _random_cover(U: List[int], n_granules: int, overlap: float, rng) -> Dict[str, set]:
    """
    Random cover of U whose granules overlap with controlled density.

    `overlap` in [0, 1] scales the expected granule size: overlap = 0 gives
    near-disjoint small granules, overlap = 1 gives heavily redundant granules.
    Coverage of U is enforced so the cover is full.
    """
    n_U = len(U)
    base = max(1, int(round(n_U / max(1, n_granules))))
    size = max(1, min(n_U, int(round(base * (1.0 + 3.0 * overlap)))))
    granules: Dict[str, set] = {}
    for a in range(n_granules):
        members = set(rng.choice(n_U, size=size, replace=False).tolist())
        granules[f"e{a}"] = members
    # guarantee full coverage
    covered = set().union(*granules.values()) if granules else set()
    missing = set(U) - covered
    if missing:
        granules.setdefault("e_fill", set()).update(missing)
    return granules


def entropy_normalisation_study(n_trials: int = 400, n_U: int = 24,
                                seed: int = 1) -> Dict:
    """
    Two questions are answered numerically:

    (a) On partitions, does the manuscript's mixed normalisation H_t agree with
        the fully Shannon-normalised H^S_t?  (Answer: identically, because
        N_t = |U| on a partition.)

    (b) Does monotonicity under refinement survive under each normalisation on
        partitions, and does it fail for overlapping covers?
    """
    rng = np.random.default_rng(seed)
    U = list(range(n_U))

    max_abs_gap_partition = 0.0
    mono_violations_H = 0
    mono_violations_HS = 0

    for _ in range(n_trials):
        n_blocks = int(rng.integers(2, 8))
        blocks = _random_partition(U, n_blocks, rng)
        refined = _refine_partition(blocks, rng)

        dss_c = DynamicSoftSet(U, {"t": blocks})
        dss_f = DynamicSoftSet(U, {"t": refined})

        H_c, H_f = dynamic_soft_entropy(dss_c, "t"), dynamic_soft_entropy(dss_f, "t")
        S_c, S_f = shannon_entropy(dss_c, "t"), shannon_entropy(dss_f, "t")

        max_abs_gap_partition = max(max_abs_gap_partition,
                                    abs(H_c - S_c), abs(H_f - S_f))
        if H_f < H_c - 1e-12:
            mono_violations_H += 1
        if S_f < S_c - 1e-12:
            mono_violations_HS += 1

    # Overlapping covers: refinement no longer guarantees monotonicity for the
    # mixed normalisation, because N_t changes under refinement.
    overlap_violations_H = 0
    overlap_violations_HS = 0
    overlap_trials = 0
    for _ in range(n_trials):
        n_gran = int(rng.integers(3, 8))
        cover = _random_cover(U, n_gran, overlap=float(rng.uniform(0.4, 1.0)), rng=rng)
        # Refine by splitting one granule into two overlapping halves
        refined = dict(cover)
        target = max(cover, key=lambda k: len(cover[k]))
        members = sorted(cover[target])
        if len(members) < 2:
            continue
        cut = max(1, len(members) // 2)
        refined.pop(target)
        refined[f"{target}_a"] = set(members[:cut])
        refined[f"{target}_b"] = set(members[cut:])

        dss_c = DynamicSoftSet(U, {"t": cover})
        dss_f = DynamicSoftSet(U, {"t": refined})
        overlap_trials += 1
        if dynamic_soft_entropy(dss_f, "t") < dynamic_soft_entropy(dss_c, "t") - 1e-12:
            overlap_violations_H += 1
        if shannon_entropy(dss_f, "t") < shannon_entropy(dss_c, "t") - 1e-12:
            overlap_violations_HS += 1

    return {
        "partition_trials": n_trials,
        "max_abs_difference_H_vs_Shannon_on_partitions": max_abs_gap_partition,
        "monotonicity_violations_H_partition": mono_violations_H,
        "monotonicity_violations_Shannon_partition": mono_violations_HS,
        "overlap_trials": overlap_trials,
        "monotonicity_violations_H_overlapping": overlap_violations_H,
        "monotonicity_violations_Shannon_overlapping": overlap_violations_HS,
    }


def entropy_overlap_study(n_configs: int = 200, n_U: int = 24, seed: int = 2) -> Dict:
    """
    Compares the dynamic soft entropy with the Liang-Shi entropy over a
    family of synthetic granulation configurations with varying overlap density.

    Overlap density is measured as

        delta = (sum_a |F(a)|) / |U|  - 1,

    i.e. the mean number of *extra* granules covering each element (delta = 0
    for a partition).  For each configuration we record the dynamic soft
    entropy, its Shannon-normalised counterpart, and the Liang-Shi rough
    entropy, together with their spread across configurations.
    """
    rng = np.random.default_rng(seed)
    U = list(range(n_U))
    rows = []
    for overlap in np.linspace(0.0, 1.0, 6):
        H_vals, S_vals, LS_vals, deltas = [], [], [], []
        for _ in range(n_configs):
            n_gran = int(rng.integers(3, 9))
            cover = _random_cover(U, n_gran, overlap=float(overlap), rng=rng)
            dss = DynamicSoftSet(U, {"t": cover})
            N_t = sum(len(g) for g in dss.granules_at("t"))
            deltas.append(N_t / n_U - 1.0)
            H_vals.append(dynamic_soft_entropy(dss, "t"))
            S_vals.append(shannon_entropy(dss, "t"))
            LS_vals.append(liang_shi_entropy(dss, "t"))
        rows.append({
            "overlap_setting": round(float(overlap), 2),
            "mean_delta": float(np.mean(deltas)),
            "H_mean": float(np.mean(H_vals)),
            "H_std": float(np.std(H_vals)),
            "Shannon_mean": float(np.mean(S_vals)),
            "Shannon_std": float(np.std(S_vals)),
            "LiangShi_mean": float(np.mean(LS_vals)),
            "LiangShi_std": float(np.std(LS_vals)),
        })

    # Sensitivity = spread of each measure across the overlap sweep, expressed
    # relative to its own mean (a scale-free comparison of responsiveness).
    def rel_range(key: str) -> float:
        vals = [r[key] for r in rows]
        m = float(np.mean(vals))
        return float((max(vals) - min(vals)) / m) if m else 0.0

    return {
        "rows": rows,
        "relative_range_H": rel_range("H_mean"),
        "relative_range_Shannon": rel_range("Shannon_mean"),
        "relative_range_LiangShi": rel_range("LiangShi_mean"),
    }


def abstract_example_entropy() -> Dict:
    """Entropy values for the abstract example."""
    dss, drss, X = build_abstract_example()
    per_slice = {}
    for t in dss.T:
        per_slice[t] = {
            "H": dynamic_soft_entropy(dss, t),
            "Shannon": shannon_entropy(dss, t),
            "N_t": sum(len(g) for g in dss.granules_at(t)),
            "is_partition": sum(len(g) for g in dss.granules_at(t)) == len(dss.U)
                            and dss.is_full_at(t),
        }
    return {
        "per_slice": per_slice,
        "liang_shi_static_aggregate": liang_shi_entropy(dss),
        "drift_t1_t2": abs(per_slice["t1"]["H"] - per_slice["t2"]["H"]),
    }


# =============================================================================
# C.  Definability under partial coverage
# =============================================================================

def relative_definability_class(drss: DRSS, t, X: set) -> str:
    """
    Definability class relativised to the active universe U*_t, for use when
    the fullness hypothesis of Definition 4.3 fails.
    """
    L = drss.lower(t, X)
    Up = drss.upper(t, X)
    Ustar = active_universe(drss.dss, t)
    if L and Up != Ustar:
        return "roughly_definable_rel"
    if not L and Up != Ustar:
        return "internally_indefinable_rel"
    if L and Up == Ustar:
        return "externally_indefinable_rel"
    return "totally_indefinable_rel"


def build_icu_example() -> Tuple[DynamicSoftSet, DRSS, set]:
    """ICU monitoring example of Section 10.2 (fails fullness at t1)."""
    U = ["N", "AR", "C"]
    mappings = {
        "t1": {"BP": {"N", "AR"}, "HR": {"N"}, "SpO2": {"N", "AR"}},
        "t2": {"BP": {"AR"}, "HR": {"AR", "C"}, "SpO2": {"N"}, "Temp": {"AR", "C"}},
        "t3": {"BP": {"AR", "C"}, "HR": {"AR", "C"}},
        "t4": {"BP": {"AR"}, "HR": {"AR", "C"}, "Temp": {"AR", "C"}},
    }
    dss = DynamicSoftSet(U, mappings)
    return dss, DRSS(dss), {"AR", "C"}


def partial_coverage_definability() -> Dict:
    """
    Reports, for the ICU example, the active universe at each slice, whether
    fullness holds, and both the absolute and the relativised definability
    class.
    """
    dss, drss, X = build_icu_example()
    rows = []
    for t in dss.T:
        Ustar = active_universe(dss, t)
        rows.append({
            "t": t,
            "active_universe": sorted(Ustar),
            "is_full": dss.is_full_at(t),
            "lower": sorted(drss.lower(t, X)),
            "upper": sorted(drss.upper(t, X)),
            "class_absolute": drss.definability_class(t, X),
            "class_relative": relative_definability_class(drss, t, X),
        })
    return {"target": sorted(X), "rows": rows}


# =============================================================================
# D.  Score-function sensitivity
# =============================================================================

def _trial_fractions(rho: float, seed: int):
    """
    Per-element frequencies of lower/upper/boundary membership for one cohort.
    Returning these once lets the whole (alpha, beta) grid be evaluated without
    recomputing approximations.
    """
    bench = LatentRiskBenchmark(rho=rho, seed=seed)
    dss, X_star, labels = bench.generate()
    drss = DRSS(dss)
    T = dss.T
    n_T = len(T)
    lowers = [drss.lower(t, X_star) for t in T]
    uppers = [drss.upper(t, X_star) for t in T]
    fl = np.array([sum(1 for L in lowers if u in L) / n_T for u in dss.U])
    fu = np.array([sum(1 for Up in uppers if u in Up) / n_T for u in dss.U])
    return fl, fu, fu - fl, labels


def alpha_beta_grid(n_runs: int = 30, rho: float = 0.75, seed: int = 3) -> Dict:
    """
    Sweeps a broad grid of the score-function weights and reports BOTH
    classification error and AUROC.  alpha_mult and beta_mult denote the
    products alpha.|U| and beta.|U|, so the published default
    (alpha, beta) = (1/|U|, 2/|U|) is the cell (1, 2).
    """
    alpha_mults = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    beta_mults = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    trials = [_trial_fractions(rho, seed + r) for r in range(n_runs)]

    rows = []
    for am in alpha_mults:
        for bm in beta_mults:
            errs, aurocs = [], []
            for fl, fu, fb, y in trials:
                score = fl + am * fu - bm * fb
                k = int(y.sum())
                order = np.argsort(-score)
                pred = np.zeros_like(y)
                pred[order[:k]] = 1
                errs.append(float(np.mean(pred != y)) * 100.0)
                try:
                    aurocs.append(float(roc_auc_score(y, score)))
                except ValueError:
                    aurocs.append(0.5)
            rows.append({
                "alpha_mult": am, "beta_mult": bm,
                "error_mean": float(np.mean(errs)),
                "error_std": float(np.std(errs)),
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs)),
            })

    errs = [r["error_mean"] for r in rows]
    aurs = [r["auroc_mean"] for r in rows]
    default = next(r for r in rows if r["alpha_mult"] == 1.0 and r["beta_mult"] == 2.0)
    return {
        "n_runs": n_runs, "rho": rho, "rows": rows,
        "error_range_pp": float(max(errs) - min(errs)),
        "auroc_range": float(max(aurs) - min(aurs)),
        "default_cell": default,
        "best_error_cell": min(rows, key=lambda r: r["error_mean"]),
        "best_auroc_cell": max(rows, key=lambda r: r["auroc_mean"]),
    }


# =============================================================================
# E.  Threshold-tuned baselines
# =============================================================================

def threshold_tuned_baselines(n_runs: int = 30, rho: float = 0.75,
                              seed: int = 4) -> Dict:
    """
    The DRSS decision rule resolves boundary elements with a secondary majority
    vote, which acts as an implicit threshold adjustment that the competing
    baselines were not granted.  Here every method is treated identically: each
    score is thresholded at the value that minimises training-split error, and
    evaluated on a held-out split.  Both the published default (alpha, beta)
    and the revised weights are reported for DRSS.
    """
    rng = np.random.default_rng(seed)
    names = ["B1", "B2", "B3", "B4", "DRSS_default", "DRSS_revised"]
    res = {m: {"auroc": [], "error": []} for m in names}

    for r in range(n_runs):
        bench = LatentRiskBenchmark(rho=rho, seed=seed + r)
        dss, X_star, y = bench.generate()
        n_U = len(dss.U)
        idx = rng.permutation(n_U)
        tr, te = idx[: n_U // 2], idx[n_U // 2:]

        scores = {
            "B1": b1_scores(dss, X_star),
            "B2": b2_scores(dss, X_star),
            "B3": b3_scores(dss, X_star),
            "DRSS_default": drss_scores(dss, X_star, alpha_mult=1.0, beta_mult=2.0),
            "DRSS_revised": drss_scores(dss, X_star, alpha_mult=0.25, beta_mult=0.0),
        }
        feats = feature_matrix(dss)
        try:
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(feats[tr], y[tr])
            scores["B4"] = clf.predict_proba(feats)[:, 1]
        except ValueError:
            scores["B4"] = np.full(n_U, 0.5)

        for m, sc in scores.items():
            # threshold tuned on the training split for every method alike
            best_thr, best_err = 0.5, 1.0
            for thr in np.unique(sc):
                err = float(np.mean((sc[tr] >= thr).astype(int) != y[tr]))
                if err < best_err:
                    best_err, best_thr = err, float(thr)
            res[m]["error"].append(
                float(np.mean((sc[te] >= best_thr).astype(int) != y[te])) * 100.0)
            try:
                res[m]["auroc"].append(float(roc_auc_score(y[te], sc[te])))
            except ValueError:
                res[m]["auroc"].append(0.5)

    out = {m: {"error_mean": float(np.mean(d["error"])),
               "error_std": float(np.std(d["error"])),
               "auroc_mean": float(np.mean(d["auroc"])),
               "auroc_std": float(np.std(d["auroc"]))}
           for m, d in res.items()}

    tests = {}
    for m in ["B1", "B2", "B3", "B4", "DRSS_default"]:
        try:
            _, p = wilcoxon(res["DRSS_revised"]["auroc"], res[m]["auroc"])
            tests[f"DRSS_revised_vs_{m}_auroc_p"] = float(p)
        except ValueError:
            tests[f"DRSS_revised_vs_{m}_auroc_p"] = float("nan")
    return {"n_runs": n_runs, "summary": out, "tests": tests}


# =============================================================================
# F.  Incremental cost accounting and wall-clock protocol
# =============================================================================

def incremental_cost_accounting(n_U: int = 200, n_E: int = 12,
                                n_T: int = 50, seed: int = 5) -> Dict:
    """
    A complexity bound of the form O(|Delta A_t| . |U|) counts only parameter
    activation and deactivation.  Granules whose *mapping* changes while the
    parameter stays active must also be touched.  We therefore measure the
    modified-parameter set

        A^m_t = { a in A_t ∩ A_{t+1} : F_{t+1}(a) != F_t(a) }

    alongside A^+ and A^-, and report the true per-update work as a fraction of
    the work done by the base algorithm.
    """
    rng = np.random.default_rng(seed)
    bench = LatentRiskBenchmark(n_U=n_U, n_E=n_E, n_T=n_T, rho=0.75, seed=seed)
    dss, X_star, _ = bench.generate()

    deltas, mods, touched, base_work = [], [], [], []
    for i in range(len(dss.T) - 1):
        t, tn = dss.T[i], dss.T[i + 1]
        A_t, A_n = dss.active_params(t), dss.active_params(tn)
        A_plus, A_minus = A_n - A_t, A_t - A_n
        A_mod = {p for p in (A_t & A_n)
                 if dss.granule(tn, p) != dss.granule(t, p)}
        deltas.append(len(A_plus) + len(A_minus))
        mods.append(len(A_mod))
        # elements actually visited by the incremental update
        work = 0
        for p in A_minus | A_mod:
            work += len(dss.granule(t, p))
        for p in A_plus | A_mod:
            work += len(dss.granule(tn, p))
        touched.append(work)
        base_work.append(sum(len(g) for g in dss.granules_at(tn)))

    return {
        "n_updates": len(deltas),
        "mean_delta_A": float(np.mean(deltas)),
        "mean_modified_A": float(np.mean(mods)),
        "modified_share_of_touched_params": float(
            np.mean(mods) / max(1e-9, np.mean(deltas) + np.mean(mods))),
        "mean_elements_touched_incremental": float(np.mean(touched)),
        "mean_elements_touched_base": float(np.mean(base_work)),
        "work_ratio_incremental_over_base": float(
            np.mean(touched) / max(1e-9, np.mean(base_work))),
    }


def wall_clock_protocol(n_U: int = 200, n_E: int = 12, n_T: int = 72,
                        n_reps: int = 20, seed: int = 6) -> Dict:
    """
    Full experimental protocol for the speedup measurement: stream size, number of
    repetitions, run-to-run variability and peak memory are all recorded, so
    the reported speedup is verifiable rather than a bare ratio.
    """
    bench = LatentRiskBenchmark(n_U=n_U, n_E=n_E, n_T=n_T, rho=0.75, seed=seed)
    dss, X_star, _ = bench.generate()
    alg = DRSSAlgorithm(dss, X_star)

    base_times, incr_times = [], []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        alg.run_base()
        base_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        alg.run_incremental()
        incr_times.append(time.perf_counter() - t0)

    tracemalloc.start()
    alg.run_base()
    _, peak_base = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    alg.run_incremental()
    _, peak_incr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    base_times = np.array(base_times)
    incr_times = np.array(incr_times)
    return {
        "universe_size": n_U,
        "n_parameters": n_E,
        "stream_length_slices": n_T,
        "n_repetitions": n_reps,
        "base_mean_s": float(base_times.mean()),
        "base_std_s": float(base_times.std()),
        "incremental_mean_s": float(incr_times.mean()),
        "incremental_std_s": float(incr_times.std()),
        "speedup_mean": float(base_times.mean() / incr_times.mean()),
        "speedup_min": float(base_times.min() / incr_times.max()),
        "speedup_max": float(base_times.max() / incr_times.min()),
        "peak_memory_base_kib": peak_base / 1024.0,
        "peak_memory_incremental_kib": peak_incr / 1024.0,
    }


# =============================================================================
# G.  Runner
# =============================================================================

def _fmt(x, nd=3):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> Dict:
    print("=" * 78)
    print("  DRSS — EXTENDED ANALYSES")
    print("=" * 78)

    results: Dict = {}

    print("\n[A] Lower/upper containment without fullness")
    results["containment"] = monotone_containment_check()
    for k, v in results["containment"].items():
        print(f"    {k:<48} {v}")

    print("\n[B1] Abstract-example entropy")
    results["abstract_entropy"] = abstract_example_entropy()
    for t, d in results["abstract_entropy"]["per_slice"].items():
        print(f"    {t}: H={_fmt(d['H'])}  H^S={_fmt(d['Shannon'])}  "
              f"N_t={d['N_t']}  partition={d['is_partition']}")
    print(f"    Liang-Shi (static aggregate) = "
          f"{_fmt(results['abstract_entropy']['liang_shi_static_aggregate'])}")
    print(f"    drift dH(t1,t2) = {_fmt(results['abstract_entropy']['drift_t1_t2'])}")

    print("\n[B2] Entropy normalisation study")
    results["entropy_normalisation"] = entropy_normalisation_study()
    for k, v in results["entropy_normalisation"].items():
        print(f"    {k:<52} {_fmt(v, 6) if isinstance(v, float) else v}")

    print("\n[B3] Entropy vs overlap density")
    results["entropy_overlap"] = entropy_overlap_study()
    print(f"    {'overlap':>8} {'delta':>8} {'H':>18} {'Shannon':>18} {'LiangShi':>18}")
    for r in results["entropy_overlap"]["rows"]:
        print(f"    {r['overlap_setting']:>8} {r['mean_delta']:>8.2f} "
              f"{r['H_mean']:>10.3f}±{r['H_std']:<6.3f} "
              f"{r['Shannon_mean']:>10.3f}±{r['Shannon_std']:<6.3f} "
              f"{r['LiangShi_mean']:>10.3f}±{r['LiangShi_std']:<6.3f}")
    print(f"    relative range  H={_fmt(results['entropy_overlap']['relative_range_H'])}  "
          f"Shannon={_fmt(results['entropy_overlap']['relative_range_Shannon'])}  "
          f"LiangShi={_fmt(results['entropy_overlap']['relative_range_LiangShi'])}")

    print("\n[C] Partial-coverage definability")
    results["partial_coverage"] = partial_coverage_definability()
    for r in results["partial_coverage"]["rows"]:
        print(f"    {r['t']}: U*={r['active_universe']} full={r['is_full']} "
              f"L={r['lower']} Up={r['upper']}")
        print(f"        absolute={r['class_absolute']}  relative={r['class_relative']}")

    print("\n[D] (alpha, beta) sensitivity grid")
    results["alpha_beta"] = alpha_beta_grid()
    ab = results["alpha_beta"]
    print(f"    error range across grid : {_fmt(ab['error_range_pp'])} p.p.")
    print(f"    AUROC range across grid : {_fmt(ab['auroc_range'], 4)}")
    print(f"    default cell (1/|U|, 2/|U|): err={_fmt(ab['default_cell']['error_mean'])} "
          f"auroc={_fmt(ab['default_cell']['auroc_mean'], 4)}")
    print(f"    best error cell : alpha={ab['best_error_cell']['alpha_mult']}/|U| "
          f"beta={ab['best_error_cell']['beta_mult']}/|U| "
          f"err={_fmt(ab['best_error_cell']['error_mean'])}")
    print(f"    best AUROC cell : alpha={ab['best_auroc_cell']['alpha_mult']}/|U| "
          f"beta={ab['best_auroc_cell']['beta_mult']}/|U| "
          f"auroc={_fmt(ab['best_auroc_cell']['auroc_mean'], 4)}")

    print("\n[E] Threshold-tuned baselines")
    results["threshold_tuned"] = threshold_tuned_baselines()
    for m, d in results["threshold_tuned"]["summary"].items():
        print(f"    {m:<20} err={_fmt(d['error_mean'])}±{_fmt(d['error_std'])}  "
              f"auroc={_fmt(d['auroc_mean'], 4)}±{_fmt(d['auroc_std'], 4)}")
    for k, v in results["threshold_tuned"]["tests"].items():
        print(f"    {k:<32} {v:.3e}")

    print("\n[F] Incremental cost accounting")
    results["incremental_cost"] = incremental_cost_accounting()
    for k, v in results["incremental_cost"].items():
        print(f"    {k:<44} {_fmt(v)}")

    print("\n[G] Wall-clock protocol")
    results["wall_clock"] = wall_clock_protocol()
    for k, v in results["wall_clock"].items():
        print(f"    {k:<36} {_fmt(v, 4)}")

    with open("extended_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("\nWritten: extended_results.json")
    return results


if __name__ == "__main__":
    main()
