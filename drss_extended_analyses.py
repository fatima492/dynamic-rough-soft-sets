"""
=============================================================================
  Dynamic Rough Soft Sets (DRSS) — Extended and Empirical Analyses
=============================================================================

This file produces EVERY synthetic-benchmark number in the manuscript, under
one protocol (class-pure LatentRiskBenchmark; every method's decision
threshold tuned on a training split and evaluated on a held-out split; DRSS
at the adopted score weights (alpha~, beta~) = (0.5, 0.25)), together with the
supporting analyses for the theoretical claims.  Section, table and figure
numbers refer to the revised manuscript.

  A. Containment of lower in upper, and of both in the active universe,
     without structural hypotheses                     (Proposition 4.3)
  B. Entropy normalisation and behaviour under overlap (Proposition 9.4,
     Section 9.2, Table 5)
  C. Definability under partial coverage               (Definition 4.6)
  D. Sensitivity of the element-wise score to its weights   (Table 8)
  E. Threshold-tuned like-for-like comparison, (1,2) vs default (Table 15)
  F. Per-update cost accounting of Algorithm 2         (Section 11.2)
  G. Wall-clock protocol and scaling series            (Section 12.4, Section 15)
  H. Main synthetic benchmark, 100 cohorts             (Table 11, Figure 6)
  I. Granule-fidelity sensitivity, 50 cohorts          (Table 12, Figure 7)
  J. Further sensitivities of Section 12.3: mean uptime mu, active-set size
     |A_t|, the cross-temporal ablation, and the FM1 i.i.d.-availability
     ablation of Section 13.1
  K. Manuscript figures fig_synth_bars.png and fig_sensitivity.png

Run with:

    python drss_extended_analyses.py

All results are printed to stdout and written to `extended_results.json`.
The random seeds are fixed so that every number quoted in the paper is
reproducible from this file.  Runtime is a few minutes on one core.
=============================================================================
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):          # Python 3.7+: never crash on non-ASCII console
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

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
    ALPHA_MULT_DEFAULT,
    ALPHA_MULT_LEGACY,
    BETA_MULT_DEFAULT,
    BETA_MULT_LEGACY,
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
    the fullness hypothesis (Definition 4.1, Section 4.1) fails.
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
    Table 8.  Sweeps a broad grid of the score-function weights and reports
    BOTH classification error and AUROC.  alpha_mult and beta_mult denote the
    products alpha.|U| and beta.|U|.  The adopted default is the cell
    (0.5, 0.25); the superseded default (1, 2) is reported as `legacy_cell`.
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
    default = next(r for r in rows if r["alpha_mult"] == ALPHA_MULT_DEFAULT
                   and r["beta_mult"] == BETA_MULT_DEFAULT)
    legacy = next(r for r in rows if r["alpha_mult"] == ALPHA_MULT_LEGACY
                  and r["beta_mult"] == BETA_MULT_LEGACY)
    return {
        "n_runs": n_runs, "rho": rho, "rows": rows,
        "error_range_pp": float(max(errs) - min(errs)),
        "auroc_range": float(max(aurs) - min(aurs)),
        "default_cell": default,
        "legacy_cell": legacy,
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
    evaluated on a held-out split (Table 15).  DRSS is reported both at the
    superseded weights (1, 2) and at the adopted default (0.5, 0.25).
    """
    rng = np.random.default_rng(seed)
    names = ["B1", "B2", "B3", "B4", "DRSS_legacy_1_2", "DRSS_default"]
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
            "DRSS_legacy_1_2": drss_scores(dss, X_star, ALPHA_MULT_LEGACY, BETA_MULT_LEGACY),
            "DRSS_default": drss_scores(dss, X_star, ALPHA_MULT_DEFAULT, BETA_MULT_DEFAULT),
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
    for m in ["B1", "B2", "B3", "B4", "DRSS_legacy_1_2"]:
        try:
            _, p = wilcoxon(res["DRSS_default"]["auroc"], res[m]["auroc"])
            tests[f"DRSS_default_vs_{m}_auroc_p"] = float(p)
        except ValueError:
            tests[f"DRSS_default_vs_{m}_auroc_p"] = float("nan")
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
                        n_reps: int = 25, seed: int = 6) -> Dict:
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


def wall_clock_scaling(configs: Sequence[Tuple[int, int]] = ((200, 72), (1000, 72), (2000, 144)),
                       n_E: int = 12, n_reps: int = 25, seed: int = 6) -> List[Dict]:
    """
    Scaling series of Section 12.4 / Section 15, measured under exactly the
    protocol of `wall_clock_protocol` (same harness, same repetition count),
    so that the |U|=200, |T|=72 entry IS the headline result and the series
    cannot disagree with it.
    """
    return [wall_clock_protocol(n_U=nU, n_E=n_E, n_T=nT, n_reps=n_reps, seed=seed)
            for nU, nT in configs]


# =============================================================================
# H–J.  Synthetic benchmark under one protocol
#       (Tables 11, 12 and the sensitivity items of Section 12.3)
# =============================================================================

METHOD_LABELS = {"B1": "B1 (Union-static SRS)", "B2": "B2 (Per-slice SRS)",
                 "B3": "B3 (DSS)", "B4": "B4 (Logistic reg.)", "DRSS": "DRSS"}


def _tuned_error(sc: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> float:
    """Threshold chosen to minimise training-split error, applied to the held-out split."""
    best_thr, best_err = 0.5, 1.0
    for thr in np.unique(sc):
        err = float(np.mean((sc[tr] >= thr).astype(int) != y[tr]))
        if err < best_err:
            best_err, best_thr = err, float(thr)
    return float(np.mean((sc[te] >= best_thr).astype(int) != y[te])) * 100.0


def _safe_auroc(y, sc) -> float:
    try:
        return float(roc_auc_score(y, sc))
    except ValueError:
        return 0.5


def _single_slice_score(dss: DynamicSoftSet, X_star: set) -> np.ndarray:
    """
    Cross-temporal ablation: the element-wise score evaluated at ONE slice (the
    last slice of the window) instead of as fractions over the whole window, so
    that no aggregation over I takes place.  Uses the adopted weights.
    """
    drss = DRSS(dss)
    t = dss.T[-1]
    L, Up = drss.lower(t, X_star), drss.upper(t, X_star)
    return np.array([
        (1.0 if u in L else 0.0)
        + ALPHA_MULT_DEFAULT * (1.0 if u in Up else 0.0)
        - BETA_MULT_DEFAULT * (1.0 if (u in Up and u not in L) else 0.0)
        for u in dss.U])


def run_cohort(bench_kwargs: Dict, seed: int, rng, methods: Sequence[str],
               with_boundary: bool = False) -> Dict:
    """
    One cohort under the common protocol.  Returns per-method threshold-tuned
    error (%) and held-out AUROC, and optionally mean boundary sizes.
    """
    bench = LatentRiskBenchmark(seed=seed, **bench_kwargs)
    dss, X_star, y = bench.generate()
    n_U = len(dss.U)
    idx = rng.permutation(n_U)
    tr, te = idx[: n_U // 2], idx[n_U // 2:]

    scores = {}
    if "B1" in methods:
        scores["B1"] = b1_scores(dss, X_star)
    if "B2" in methods:
        scores["B2"] = b2_scores(dss, X_star)
    if "B3" in methods:
        scores["B3"] = b3_scores(dss, X_star)
    if "B4" in methods:
        feats = feature_matrix(dss)
        try:
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(feats[tr], y[tr])
            scores["B4"] = clf.predict_proba(feats)[:, 1]
        except ValueError:
            scores["B4"] = np.full(n_U, 0.5)
    if "DRSS" in methods:
        scores["DRSS"] = drss_scores(dss, X_star)      # adopted default weights
    if "ABLATED" in methods:
        scores["ABLATED"] = _single_slice_score(dss, X_star)

    out = {}
    for m, sc in scores.items():
        out[m] = {"error": _tuned_error(sc, y, tr, te),
                  "auroc": _safe_auroc(y[te], sc[te])}

    if with_boundary:
        drss = DRSS(dss)
        per_slice = [len(drss.boundary(t, X_star)) for t in dss.T]
        agg: Dict = {}
        for t in dss.T:
            for p, g in dss.mappings[t].items():
                agg.setdefault(p, set()).update(g)
        static = DRSS(DynamicSoftSet(dss.U, {"t0": agg}))
        out["boundary"] = {"B1": float(len(static.boundary("t0", X_star))),
                           "B2": float(np.mean(per_slice)),
                           "DRSS": float(np.mean(per_slice))}
    return out


def _summarise(runs: List[Dict], methods: Sequence[str], ref: str = "DRSS") -> Dict:
    summ = {}
    for m in methods:
        e = np.array([r[m]["error"] for r in runs])
        a = np.array([r[m]["auroc"] for r in runs])
        summ[m] = {"error_mean": float(e.mean()), "error_std": float(e.std()),
                   "auroc_mean": float(a.mean()), "auroc_std": float(a.std())}
        if "boundary" in runs[0] and m in runs[0]["boundary"]:
            summ[m]["mean_boundary"] = float(np.mean([r["boundary"][m] for r in runs]))
    tests = {}
    if ref in methods:
        a_ref = np.array([r[ref]["auroc"] for r in runs])
        for m in methods:
            if m == ref:
                continue
            try:
                _, p = wilcoxon(a_ref, np.array([r[m]["auroc"] for r in runs]))
                tests[f"{ref}_vs_{m}_auroc_p"] = float(p)
            except ValueError:
                tests[f"{ref}_vs_{m}_auroc_p"] = float("nan")
    return {"summary": summ, "tests": tests}


def synthetic_benchmark_main(n_runs: int = 100, rho: float = 0.75, seed: int = 4) -> Dict:
    """Table 11 and Figure 6: main synthetic benchmark, 100 cohorts, rho = 0.75."""
    rng = np.random.default_rng(seed)
    methods = ["B1", "B2", "B3", "B4", "DRSS"]
    runs = [run_cohort({"rho": rho}, seed + r, rng, methods, with_boundary=True)
            for r in range(n_runs)]
    out = _summarise(runs, methods)
    out.update({"n_runs": n_runs, "rho": rho,
                "alpha_mult": ALPHA_MULT_DEFAULT, "beta_mult": BETA_MULT_DEFAULT})
    return out


def rho_sensitivity(n_runs: int = 50, rhos: Sequence[float] = (0.60, 0.75, 0.90),
                    seed: int = 7) -> Dict:
    """Table 12 and Figure 7 (right): sensitivity to granule fidelity rho, 50 cohorts."""
    methods = ["B1", "B2", "B4", "DRSS"]
    out = {"n_runs": n_runs, "rhos": list(rhos), "by_rho": {}}
    for rho in rhos:
        rng = np.random.default_rng(seed)
        runs = [run_cohort({"rho": rho}, seed + r, rng, methods) for r in range(n_runs)]
        out["by_rho"][str(rho)] = _summarise(runs, methods)
    return out


def mean_uptime_sweep(mus: Sequence[float] = (0.3, 0.5, 0.7, 0.9), n_runs: int = 30,
                      rho: float = 0.75, seed: int = 8) -> Dict:
    """Section 12.3, further sensitivities: mean parameter availability mu."""
    methods = ["B1", "DRSS"]
    out = {"n_runs": n_runs, "mus": list(mus), "by_mu": {}}
    for mu in mus:
        rng = np.random.default_rng(seed)
        runs = [run_cohort({"rho": rho, "mean_uptime": mu}, seed + r, rng, methods)
                for r in range(n_runs)]
        out["by_mu"][str(mu)] = _summarise(runs, methods)
    return out


def active_set_sweep(caps: Sequence[int] = (4, 8, 12), n_runs: int = 30,
                     rho: float = 0.75, seed: int = 9) -> Dict:
    """Section 12.3, further sensitivities: cap on the active-set size |A_t|."""
    methods = ["B1", "DRSS"]
    out = {"n_runs": n_runs, "caps": list(caps), "by_cap": {}}
    for cap in caps:
        rng = np.random.default_rng(seed)
        runs = [run_cohort({"rho": rho, "max_active": cap}, seed + r, rng, methods)
                for r in range(n_runs)]
        res = _summarise(runs, methods)
        e_d = np.array([r["DRSS"]["error"] for r in runs])
        e_b = np.array([r["B1"]["error"] for r in runs])
        try:
            _, p = wilcoxon(e_d, e_b)
        except ValueError:
            p = float("nan")
        res["tests"]["DRSS_vs_B1_error_p"] = float(p)
        out["by_cap"][str(cap)] = res
    return out


def crosstemporal_ablation(n_runs: int = 100, rho: float = 0.75, seed: int = 4) -> Dict:
    """
    Section 12.3, cross-temporal ablation: the DRSS score restricted to a single
    slice (no aggregation over I) against the full cross-temporal score, both at
    the adopted weights and both threshold-tuned.  Same seeds as Table 11.
    """
    rng = np.random.default_rng(seed)
    methods = ["DRSS", "ABLATED"]
    runs = [run_cohort({"rho": rho}, seed + r, rng, methods) for r in range(n_runs)]
    res = _summarise(runs, methods)
    e_d = np.array([r["DRSS"]["error"] for r in runs])
    e_a = np.array([r["ABLATED"]["error"] for r in runs])
    try:
        _, p = wilcoxon(e_d, e_a)
    except ValueError:
        p = float("nan")
    res["tests"]["DRSS_vs_ABLATED_error_p"] = float(p)
    res["n_runs"] = n_runs
    return res


def no_regime_ablation(n_runs: int = 100, rho: float = 0.75, seed: int = 4) -> Dict:
    """
    Section 13.1, FM1: parameter availability made i.i.d. across slices (constant
    activation probability, no autocorrelation), so that cross-temporal
    aggregation has no regime structure to exploit.  Same protocol and seeds as
    Table 11, DRSS at the adopted weights.
    """
    rng = np.random.default_rng(seed)
    methods = ["B1", "DRSS"]
    runs = [run_cohort({"rho": rho, "regime": False, "avail_persistence": 0.0},
                       seed + r, rng, methods) for r in range(n_runs)]
    res = _summarise(runs, methods)
    res["n_runs"] = n_runs
    return res


def static_granule_ablation(n_runs: int = 100, rho: float = 0.75, seed: int = 4) -> Dict:
    """
    Section 13.1, FM1 (operative condition): granules frozen across the window
    (drift = 0, no reinterpretation) with regime-structured availability.  With
    no temporal change in F_t, pooling across time loses nothing and the
    cross-temporal aggregation has no signal beyond the static baselines.
    """
    rng = np.random.default_rng(seed)
    methods = ["B1", "B2", "DRSS"]
    runs = [run_cohort({"rho": rho, "drift": 0.0}, seed + r, rng, methods)
            for r in range(n_runs)]
    res = _summarise(runs, methods)
    res["n_runs"] = n_runs
    return res


def diagnostic_regression(n_runs: int = 500, n_fit: int = 350, rho: float = 0.75,
                          seed: int = 12) -> Dict:
    """
    Table 17 (Section 13.2): do three label-free diagnostics predict the DRSS
    advantage over B1?  Protocol as stated in the manuscript: n_runs cohorts
    from the class-pure generator with its structural parameters drawn at
    random (regime on/off, granule drift, mean availability, availability
    persistence, granule size); response = threshold-tuned error advantage of
    DRSS over B1 in percentage points; predictors = Var_t(H_t), mean |A_t|,
    mean max_a |F_t(a)|/|U|, plus the single pre-specified interaction
    Var_t(H_t) x mean|A_t|; predictors standardised on the fitting sample;
    OLS; R^2 reported on the held-out runs.
    """
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(seed)
    rows = []
    for run in range(n_runs):
        kw = {"rho": rho,
              "regime": bool(rng.random() < 0.5),
              "drift": float(rng.uniform(0.0, 0.3)),
              "mean_uptime": float(rng.uniform(0.2, 0.9)),
              "avail_persistence": float(rng.uniform(0.0, 0.95)),
              "granule_size": int(rng.integers(6, 100))}
        bench = LatentRiskBenchmark(seed=seed + run, **kw)
        dss, X_star, y = bench.generate()
        n_U = len(dss.U)
        idx = rng.permutation(n_U)
        tr, te = idx[: n_U // 2], idx[n_U // 2:]
        e_drss = _tuned_error(drss_scores(dss, X_star), y, tr, te)
        e_b1 = _tuned_error(b1_scores(dss, X_star), y, tr, te)
        H = [dynamic_soft_entropy(dss, t) for t in dss.T]
        At = [len(dss.active_params(t)) for t in dss.T]
        gm = [max((len(g) / n_U for g in dss.granules_at(t)), default=0.0) for t in dss.T]
        rows.append({"advantage": e_b1 - e_drss, "var_H": float(np.var(H)),
                     "mean_At": float(np.mean(At)), "mean_max_g": float(np.mean(gm)), **kw})
    import pandas as pd
    df = pd.DataFrame(rows)
    df["interaction"] = df["var_H"] * df["mean_At"]
    corr = []
    for col, label in [("var_H", "Var_t(H_t)"), ("mean_At", "Mean |A_t|"),
                       ("mean_max_g", "Mean max_a |F_t(a)|/|U|"),
                       ("interaction", "Var_t(H_t) x Mean|A_t|")]:
        r, pval = pearsonr(df[col], df["advantage"])
        corr.append({"diagnostic": label, "pearson_r": float(r), "p_value": float(pval)})
    fit, held = df.iloc[:n_fit], df.iloc[n_fit:]
    main_cols = ["var_H", "mean_At", "mean_max_g"]
    full_cols = main_cols + ["interaction"]
    mu, sd = fit[full_cols].mean(), fit[full_cols].std(ddof=0).replace(0, 1.0)
    Zf, Zh = (fit[full_cols] - mu) / sd, (held[full_cols] - mu) / sd
    r2_main = LinearRegression().fit(Zf[main_cols], fit["advantage"]).score(Zh[main_cols], held["advantage"])
    r2_full = LinearRegression().fit(Zf[full_cols], fit["advantage"]).score(Zh[full_cols], held["advantage"])
    return {"n_runs": n_runs, "n_fit": n_fit, "n_heldout": n_runs - n_fit,
            "correlations": corr, "r2_heldout_main": float(r2_main),
            "r2_heldout_interaction": float(r2_full),
            "advantage_mean_pp": float(df["advantage"].mean()),
            "advantage_std_pp": float(df["advantage"].std())}


# =============================================================================
# K.  Manuscript figures
# =============================================================================

def make_manuscript_figures(main: Dict, rho: Dict, grid: Dict, outdir: str = ".") -> None:
    """Regenerate fig_synth_bars.png (Figure 6) and fig_sensitivity.png (Figure 7)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {"B1": "#5B9BD5", "B2": "#70AD47", "B3": "#FFC000",
               "B4": "#ED7D31", "DRSS": "#C00000"}
    labels = {"B1": "B1\nUnion-static", "B2": "B2\nPer-slice", "B3": "B3\nDSS",
              "B4": "B4\nLogReg", "DRSS": "DRSS\n(ours)"}
    order = ["B1", "B2", "B3", "B4", "DRSS"]
    sm = main["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=200)
    for ax, key, ttl, ylab in [
        (axes[0], "error", "Synthetic benchmark: error (lower is better)", "Classification error (%)"),
        (axes[1], "auroc", "Synthetic benchmark: AUROC (higher is better)", "AUROC")]:
        means = [sm[m][f"{key}_mean"] for m in order]
        stds = [sm[m][f"{key}_std"] for m in order]
        bars = ax.bar(range(len(order)), means, yerr=stds, capsize=4,
                      color=[palette[m] for m in order], edgecolor="black", linewidth=0.8)
        for b, v, sd in zip(bars, means, stds):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + sd + (1.2 if key == "error" else 0.015),
                    f"{v:.1f}" if key == "error" else f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels[m] for m in order])
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(axis="y", alpha=0.3)
        if key == "auroc":
            ax.set_ylim(0, 1.0)
            ax.axhline(0.5, ls="--", color="grey", lw=1)
            ax.text(0.02, 0.515, "chance", color="grey", fontsize=8, transform=ax.get_yaxis_transform())
        else:
            ax.set_ylim(0, max(means) + max(stds) + 12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_synth_bars.png"))
    plt.close(fig)

    # Figure 7: heatmap of AUROC over the (alpha~, beta~) grid + rho sensitivity
    am = sorted({r["alpha_mult"] for r in grid["rows"]})
    bm = sorted({r["beta_mult"] for r in grid["rows"]})
    Z = np.zeros((len(am), len(bm)))
    for r in grid["rows"]:
        Z[am.index(r["alpha_mult"]), bm.index(r["beta_mult"])] = r["auroc_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=200)
    ax = axes[0]
    im = ax.imshow(Z, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(bm))); ax.set_xticklabels([str(b) for b in bm])
    ax.set_yticks(range(len(am))); ax.set_yticklabels([str(a) for a in am])
    ax.set_xlabel(r"$\tilde\beta$"); ax.set_ylabel(r"$\tilde\alpha$")
    ax.set_title("Score-weight sensitivity (AUROC)")
    for i in range(len(am)):
        for j in range(len(bm)):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if Z[i, j] < Z.max() - 0.08 else "black")
    ax.plot(bm.index(BETA_MULT_LEGACY), am.index(ALPHA_MULT_LEGACY), "x", color="red", ms=14, mew=2.5)
    ax.plot(bm.index(BETA_MULT_DEFAULT), am.index(ALPHA_MULT_DEFAULT), "o", mfc="none",
            mec="red", ms=16, mew=2.5)
    fig.colorbar(im, ax=ax)
    ax = axes[1]
    rhos = [float(x) for x in rho["rhos"]]
    for m, lab in [("B1", "B1"), ("B2", "B2"), ("B4", "B4 (LogReg)"), ("DRSS", "DRSS")]:
        ys = [rho["by_rho"][str(x)]["summary"][m]["auroc_mean"] for x in rho["rhos"]]
        ax.plot(rhos, ys, "-o", color=palette[m], label=lab, lw=2.2 if m == "DRSS" else 1.6)
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_xlabel(r"granule fidelity $\rho$"); ax.set_ylabel("AUROC")
    ax.set_title(r"Sensitivity to $\rho$"); ax.set_ylim(0.45, 0.85); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_sensitivity.png"))
    plt.close(fig)


# =============================================================================
# Runner
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
    print(f"    adopted default (0.5, 0.25): err={_fmt(ab['default_cell']['error_mean'])} "
          f"auroc={_fmt(ab['default_cell']['auroc_mean'], 4)}")
    print(f"    legacy cell (1, 2)        : err={_fmt(ab['legacy_cell']['error_mean'])} "
          f"auroc={_fmt(ab['legacy_cell']['auroc_mean'], 4)}")
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

    print("\n[G] Wall-clock protocol and scaling series (Section 12.4)")
    results["wall_clock_scaling"] = wall_clock_scaling()
    results["wall_clock"] = results["wall_clock_scaling"][0]   # |U|=200, |T|=72 headline
    for wc in results["wall_clock_scaling"]:
        print(f"    |U|={wc['universe_size']:<5} |T|={wc['stream_length_slices']:<4} "
              f"base={wc['base_mean_s']*1000:.2f}±{wc['base_std_s']*1000:.2f} ms  "
              f"incr={wc['incremental_mean_s']*1000:.2f}±{wc['incremental_std_s']*1000:.2f} ms  "
              f"speedup={wc['speedup_mean']:.2f}x  range {wc['speedup_min']:.2f}–{wc['speedup_max']:.2f}x  "
              f"reps={wc['n_repetitions']}")

    print("\n[H] Main synthetic benchmark (Table 11)")
    results["synthetic_main"] = synthetic_benchmark_main()
    for m, d in results["synthetic_main"]["summary"].items():
        print(f"    {METHOD_LABELS[m]:<24} B={d.get('mean_boundary', float('nan')):6.1f}  "
              f"err={_fmt(d['error_mean'],1)}±{_fmt(d['error_std'],1)}  "
              f"auroc={_fmt(d['auroc_mean'])}±{_fmt(d['auroc_std'])}")
    for k, v in results["synthetic_main"]["tests"].items():
        print(f"    {k:<28} {v:.2e}")

    print("\n[I] Granule-fidelity sensitivity (Table 12)")
    results["rho_sensitivity"] = rho_sensitivity()
    for rho, blk in results["rho_sensitivity"]["by_rho"].items():
        print(f"    rho={rho}")
        for m, d in blk["summary"].items():
            print(f"      {m:<5} err={_fmt(d['error_mean'],1)}±{_fmt(d['error_std'],1)}  "
                  f"auroc={_fmt(d['auroc_mean'])}")

    print("\n[J1] Mean uptime sweep")
    results["mean_uptime_sweep"] = mean_uptime_sweep()
    for mu, blk in results["mean_uptime_sweep"]["by_mu"].items():
        d, b = blk["summary"]["DRSS"], blk["summary"]["B1"]
        print(f"    mu={mu}: DRSS err={_fmt(d['error_mean'],1)}±{_fmt(d['error_std'],1)}  "
              f"B1 err={_fmt(b['error_mean'],1)}")

    print("\n[J2] Active-set size sweep")
    results["active_set_sweep"] = active_set_sweep()
    for cap, blk in results["active_set_sweep"]["by_cap"].items():
        d, b = blk["summary"]["DRSS"], blk["summary"]["B1"]
        print(f"    |A_t|<={cap}: DRSS err={_fmt(d['error_mean'],1)}±{_fmt(d['error_std'],1)}  "
              f"B1 err={_fmt(b['error_mean'],1)}  p={blk['tests']['DRSS_vs_B1_error_p']:.2e}")

    print("\n[J3] Cross-temporal ablation")
    results["crosstemporal_ablation"] = crosstemporal_ablation()
    for m, d in results["crosstemporal_ablation"]["summary"].items():
        print(f"    {m:<8} err={_fmt(d['error_mean'],1)}±{_fmt(d['error_std'],1)}  "
              f"auroc={_fmt(d['auroc_mean'])}")
    print(f"    p (error, DRSS vs ablated) = "
          f"{results['crosstemporal_ablation']['tests']['DRSS_vs_ABLATED_error_p']:.2e}")

    print("\n[J4] FM1: i.i.d. availability ablation (Section 13.1)")
    results["fm1_no_regime"] = no_regime_ablation()
    for m, d in results["fm1_no_regime"]["summary"].items():
        print(f"    {m:<5} err={_fmt(d['error_mean'],1)}±{_fmt(d['error_std'],1)}  "
              f"auroc={_fmt(d['auroc_mean'])}")

    print("\n[J5] FM1: static-granule ablation, drift = 0 (Section 13.1)")
    results["fm1_static_granules"] = static_granule_ablation()
    for m, d in results["fm1_static_granules"]["summary"].items():
        print(f"    {m:<5} err={_fmt(d['error_mean'],1)}±{_fmt(d['error_std'],1)}  "
              f"auroc={_fmt(d['auroc_mean'])}")

    print("\n[J6] Diagnostic regression (Table 17)")
    results["diagnostic_regression"] = diagnostic_regression()
    for c in results["diagnostic_regression"]["correlations"]:
        print(f"    {c['diagnostic']:<28} r={c['pearson_r']:+.3f}  p={c['p_value']:.2e}")
    print(f"    held-out R^2 main={results['diagnostic_regression']['r2_heldout_main']:.3f}  "
          f"+interaction={results['diagnostic_regression']['r2_heldout_interaction']:.3f}")

    print("\n[K] Manuscript figures")
    make_manuscript_figures(results["synthetic_main"], results["rho_sensitivity"],
                            results["alpha_beta"])
    print("    written fig_synth_bars.png, fig_sensitivity.png")

    with open("extended_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("\nWritten: extended_results.json")
    return results


if __name__ == "__main__":
    main()
