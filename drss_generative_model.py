"""
=============================================================================
  DRSS — Corrected synthetic generative model
=============================================================================

Design requirements
-------------------
A generative model intended to exercise a soft rough framework must satisfy two
conditions that are easy to overlook.

If every granule is sampled as a mixture of X* members and non-members, then no
granule is ever a subset of X*, so the soft lower approximation

    lower_t(X*) = union { F_t(a) : F_t(a) subseteq X* }

is identically empty at every slice and the DRSS score degenerates to
-frac_upper, which is anti-correlated with membership.  The model below
therefore admits target-pure granules.  A counter-example that does not is kept
in ``drss_analysis.DegenerateMixtureBenchmark``.

The model implements the construction the paper describes:

  * rho is the probability that an active parameter contributes *class-pure*
    (certifying) evidence at a slice.  A pure granule lies entirely inside X*
    or entirely inside its complement, so it can enter a lower approximation.
    With probability 1 - rho the parameter contributes a *mixed* granule that
    straddles the target and therefore only enters the upper approximation.

  * Granules persist across time.  Each parameter owns a base granule that is
    re-drawn only with probability `drift` per slice ("reinterpretation" in the
    terminology of the dynamic soft set literature).  Parameter availability
    A_t still follows the three-regime activation pattern.  Persistence is what
    makes an incremental update cheaper than recomputation: without it every
    parameter is modified at every slice and Algorithm 2 has nothing to reuse.

Both properties are needed for the framework to behave as described.
=============================================================================
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from drss_analysis import DRSS, DynamicSoftSet


class LatentRiskBenchmark:
    """
    Non-degenerate synthetic benchmark based on a latent risk variable.

    Each element u of the universe carries a latent risk r_u.  The label is
    drawn as a noisy threshold of that risk, so no granulation is perfectly
    aligned with X* and no method attains a trivial ceiling.

    A parameter e observes the risk through its own noisy channel and, at each
    slice where it is active, contributes the granule

        F_t(e) = { u : observed risk of u through channel e lies in band B_e },

    sampled to a bounded size.  High bands yield granules that are mostly (and
    sometimes entirely) inside X*; such granules are exactly the ones that can
    enter a soft lower approximation.  The parameter `rho` controls channel
    fidelity: a high rho means the observed risk tracks the latent risk closely,
    so certifying granules are more often genuinely pure.

    Granules persist across slices and are re-drawn only with probability
    `drift`, so parameter reinterpretation is occasional rather than constant.
    """

    def __init__(self, n_U: int = 200, n_Xstar: int = 60, n_E: int = 12,
                 n_T: int = 50, rho: float = 0.75, drift: float = 0.15,
                 label_noise: float = 0.08, granule_size: int = 14,
                 regime: bool = True, mean_uptime: float | None = None,
                 avail_persistence: float = 0.9, seed: int | None = None):
        self.n_U = n_U
        self.n_Xstar = n_Xstar
        self.n_E = n_E
        self.n_T = n_T
        self.rho = rho
        self.drift = drift
        self.label_noise = label_noise
        self.granule_size = granule_size
        self.regime = regime
        self.avail_persistence = avail_persistence
        self.mean_uptime = mean_uptime
        self.rng = np.random.default_rng(seed)

    def _pi(self, t: int) -> float:
        if not self.regime:
            base = 0.6
        elif t <= self.n_T * 0.4:
            base = 0.4
        elif t <= self.n_T * 0.7:
            base = 0.9
        else:
            base = 0.5
        if self.mean_uptime is not None:
            nominal = 0.6 if not self.regime else 0.58
            base = float(np.clip(base * self.mean_uptime / nominal, 0.02, 1.0))
        return base

    def _draw_granule(self, risk: np.ndarray, channel_noise: float) -> set:
        """Granule = bounded sample from a random band of the observed risk."""
        observed = risk + self.rng.normal(0.0, channel_noise, size=risk.shape)
        order = np.argsort(-observed)
        # band top position: biased toward the high-risk end
        top = int(self.rng.integers(0, max(1, self.n_U - self.granule_size)))
        top = int(min(top, self.n_U - self.granule_size))
        idx = order[top: top + self.granule_size]
        return set(int(i) for i in idx)

    def generate(self) -> Tuple[DynamicSoftSet, set, np.ndarray]:
        U = list(range(self.n_U))
        risk = self.rng.uniform(0.0, 1.0, size=self.n_U)

        # label: noisy threshold at the prevalence quantile
        thr = np.quantile(risk, 1.0 - self.n_Xstar / self.n_U)
        labels = (risk > thr).astype(int)
        flip = self.rng.random(self.n_U) < self.label_noise
        labels[flip] = 1 - labels[flip]
        X_star = set(int(u) for u in np.where(labels == 1)[0])

        channel_noise = 0.35 * (1.0 - self.rho) + 0.02
        base = {e: self._draw_granule(risk, channel_noise) for e in range(self.n_E)}

        # Sensor availability is autocorrelated: a parameter that is online at
        # slice t tends to remain online at t+1.  Independent per-slice sampling
        # would make every parameter churn at every slice, which is neither
        # clinically realistic nor the regime the incremental algorithm targets.
        online = {e: self.rng.random() < self._pi(1) for e in range(self.n_E)}

        mappings: Dict = {}
        for t in range(1, self.n_T + 1):
            pi_t = self._pi(t)
            params = {}
            for e in range(self.n_E):
                if self.rng.random() < self.drift:
                    base[e] = self._draw_granule(risk, channel_noise)
                if self.rng.random() < self.avail_persistence:
                    pass                      # hold previous availability state
                else:
                    online[e] = self.rng.random() < pi_t
                if online[e]:
                    params[e] = set(base[e])
            mappings[t] = params

        dss = DynamicSoftSet(U, mappings)
        return dss, X_star, labels


# ── decision rules ─────────────────────────────────────────────────────────

def drss_scores(dss: DynamicSoftSet, X_star: set,
                alpha_mult: float = 1.0, beta_mult: float = 2.0) -> np.ndarray:
    """
    DRSS score of Algorithm 1, expressed with the per-slice frequencies

        score(u) = frac_lower(u) + alpha|U| . frac_upper(u)
                                 - beta|U| . frac_boundary(u).

    `alpha_mult` and `beta_mult` are the products alpha.|U| and beta.|U|, so the
    published default (alpha, beta) = (1/|U|, 2/|U|) corresponds to (1, 2).
    """
    drss = DRSS(dss)
    T = dss.T
    n_T = len(T)
    lower_sets = [drss.lower(t, X_star) for t in T]
    upper_sets = [drss.upper(t, X_star) for t in T]
    scores = np.zeros(len(dss.U))
    for i, u in enumerate(dss.U):
        fl = sum(1 for L in lower_sets if u in L) / n_T
        fu = sum(1 for Up in upper_sets if u in Up) / n_T
        fb = fu - fl
        scores[i] = fl + alpha_mult * fu - beta_mult * fb
    return scores


def b1_scores(dss: DynamicSoftSet, X_star: set) -> np.ndarray:
    """
    B1, union-static soft rough set.  Graded score: 1 on the lower
    approximation, 0.5 on the boundary, 0 outside the upper approximation.
    """
    agg: Dict = {}
    for t in dss.T:
        for p, g in dss.mappings[t].items():
            agg.setdefault(p, set()).update(g)
    static = DRSS(DynamicSoftSet(dss.U, {"t0": agg}))
    L = static.lower("t0", X_star)
    Up = static.upper("t0", X_star)
    return np.array([1.0 if u in L else (0.5 if u in Up else 0.0) for u in dss.U])


def b2_scores(dss: DynamicSoftSet, X_star: set) -> np.ndarray:
    """B2, per-slice soft rough set with majority vote over lower memberships."""
    drss = DRSS(dss)
    T = dss.T
    return np.array([
        sum(1 for t in T if u in drss.lower(t, X_star)) / len(T)
        for u in dss.U
    ])


def b3_scores(dss: DynamicSoftSet, X_star: set) -> np.ndarray:
    """
    B3, dynamic soft set with no approximation operators.

    Scoring by membership in `granule & X_star` would read the target set
    directly and make B3 an oracle (AUROC 1.0 by construction).  This baseline
    uses granule incidence only, which is all a dynamic soft set provides in the
    absence of approximation operators.
    """
    counts = {u: 0 for u in dss.U}
    for t in dss.T:
        for g in dss.granules_at(t):
            for u in g:
                counts[u] += 1
    mx = max(counts.values()) or 1
    return np.array([counts[u] / mx for u in dss.U])


def feature_matrix(dss: DynamicSoftSet) -> np.ndarray:
    """Binary parameter-incidence features for the logistic-regression baseline."""
    params = sorted({p for t in dss.T for p in dss.mappings[t]})
    pidx = {p: j for j, p in enumerate(params)}
    feats = np.zeros((len(dss.U), len(params)))
    uidx = {u: i for i, u in enumerate(dss.U)}
    for t in dss.T:
        for p, g in dss.mappings[t].items():
            for u in g:
                feats[uidx[u], pidx[p]] += 1.0
    if len(dss.T):
        feats /= len(dss.T)
    return feats
