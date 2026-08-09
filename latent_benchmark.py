"""A synthetic benchmark that actually exercises the DRSS lower approximation.

The shipped generator builds every granule as `pos_sample + neg_sample` with at
least one negative element, so no granule can ever be a subset of X* and the
lower approximation is empty by construction.  `SyntheticDRSSBenchmarkFixed`
solves that by drawing certifying granules *from* X*, which makes membership of
a lower approximation a guarantee of membership of X* -- the label leaks and
AUROC pins at 1.000.

This generator does neither.  Granules are drawn from a latent clustering of U
and never reference X* at all.  X* is a union of some latent clusters, plus
label noise.  A granule therefore lands inside X* when the data has that
structure, not because the sampler put it there, and it straddles the boundary
when its home cluster bleeds -- which is what the boundary region is meant to
represent.

Difficulty is controlled by two interpretable knobs:
    p_bleed     probability that a granule leaks elements from a neighbouring
                cluster, i.e. how often granules fail to respect the latent
                structure
    label_noise fraction of elements whose X* membership is flipped after the
                clusters are fixed, so that lower-approximation membership is
                strong evidence but never proof
"""
from typing import Dict, Tuple
import numpy as np


class SyntheticDRSSBenchmarkLatent:
    """|U|=200, |X*|~60, |E|=12, |T|=50; three-regime activation as Section 11.3."""

    def __init__(self, n_U=200, n_Xstar=60, n_E=12, n_T=50,
                 n_clusters=20, p_bleed=None, bleed_size=3,
                 p_drop=0.10, label_noise=0.05, rho=None, seed=None):
        self.n_U = n_U
        self.n_Xstar = n_Xstar
        self.n_E = n_E
        self.n_T = n_T
        self.n_clusters = n_clusters
        self.p_bleed = 0.30 if p_bleed is None else float(p_bleed)
        self.bleed_size = bleed_size
        self.p_drop = p_drop
        self.label_noise = label_noise
        # `rho` is accepted for signature compatibility with the shipped
        # generator and, when given, is mapped onto the bleed probability:
        # a higher rho means granules respect the latent structure more often.
        # An explicit p_bleed always wins; rho only maps onto it when the
        # caller did not set the bleed probability directly.
        if rho is not None and p_bleed is None:
            self.p_bleed = float(np.clip(1.0 - rho, 0.02, 0.9))
        self.rng = np.random.default_rng(seed)

    def _pi(self, t: int) -> float:
        if t <= 20:
            return 0.4
        elif t <= 35:
            return 0.9
        return 0.5

    def generate(self) -> Tuple[object, set, np.ndarray]:
        from drss_analysis import DynamicSoftSet

        U = list(range(self.n_U))
        perm = self.rng.permutation(self.n_U)
        clusters = [set(int(v) for v in c)
                    for c in np.array_split(perm, self.n_clusters)]

        # X* is a union of whole clusters -- chosen without reference to any
        # granule -- taken in random order until it is about the target size.
        order = self.rng.permutation(self.n_clusters)
        positive, X_core = set(), set()
        for ci in order:
            if len(X_core) >= self.n_Xstar:
                break
            positive.add(int(ci))
            X_core |= clusters[ci]

        # Label noise: membership is decided per element, so a granule inside a
        # positive cluster is strong evidence for X* but never a proof of it.
        X_star = set(X_core)
        for u in U:
            if self.rng.random() < self.label_noise:
                X_star.symmetric_difference_update({u})

        # Each parameter keeps a home cluster for the whole horizon; this is
        # what gives the stream its temporal structure.
        home = {e: int(self.rng.integers(self.n_clusters)) for e in range(self.n_E)}

        mappings: Dict = {}
        for t in range(1, self.n_T + 1):
            pi_t = self._pi(t)
            params = {}
            for e in range(self.n_E):
                if self.rng.random() >= pi_t:
                    continue
                base = set(clusters[home[e]])
                granule = {u for u in base if self.rng.random() >= self.p_drop}
                if self.rng.random() < self.p_bleed:
                    other = int(self.rng.integers(self.n_clusters))
                    if other != home[e] and clusters[other]:
                        pool = sorted(clusters[other])
                        k = min(len(pool), int(self.rng.integers(1, self.bleed_size + 1)))
                        granule |= set(int(v) for v in
                                       self.rng.choice(pool, size=k, replace=False))
                if granule:
                    params[e] = granule
            mappings[t] = params

        dss = DynamicSoftSet(U, mappings)
        labels = np.array([1 if u in X_star else 0 for u in U])
        return dss, X_star, labels
