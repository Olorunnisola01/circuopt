"""A self-contained NSGA-III for the mixed continuous/categorical ecodesign
problem, plus the reference-point machinery a three-objective front needs.

NSGA-III rather than NSGA-II: with three objectives, crowding distance is a
weak diversity signal, and the reference-point niching of Deb & Jain (2014)
both spreads the front evenly and gives a natural place to hang decision-maker
preferences — each stakeholder profile is simply a different reference
direction on the same normalised hyperplane.

Variation operators are hybrid, because the design vector is hybrid:
  * continuous genes  -> simulated binary crossover + polynomial mutation
  * categorical genes -> uniform crossover + random reset mutation

Constraints are handled by Deb's feasibility rules: a feasible solution always
dominates an infeasible one, and two infeasible solutions are ordered by total
violation.

Dependencies: numpy only.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Reference directions
# --------------------------------------------------------------------------

def das_dennis(n_obj: int, n_partitions: int) -> np.ndarray:
    """Uniformly spaced points on the unit simplex (Das & Dennis, 1998)."""
    def rec(left, total, depth):
        if depth == n_obj - 1:
            return [[left / total]]
        out = []
        for i in range(left + 1):
            for tail in rec(left - i, total, depth + 1):
                out.append([i / total] + tail)
        return out
    return np.array(rec(n_partitions, n_partitions, 0), dtype=float)


# --------------------------------------------------------------------------
# Non-dominated sorting with constraint domination
# --------------------------------------------------------------------------

def dominates(f1, cv1, f2, cv2) -> bool:
    if cv1 <= 0 and cv2 > 0:
        return True
    if cv1 > 0 and cv2 <= 0:
        return False
    if cv1 > 0 and cv2 > 0:
        return cv1 < cv2
    return bool(np.all(f1 <= f2) and np.any(f1 < f2))


def nondominated_mask(F, CV):
    """Boolean mask of the non-dominated, feasible points.

    Equivalent to taking the first front out of ``fast_non_dominated_sort``,
    but incremental rather than pairwise: it maintains a running archive and
    tests each candidate against it, which is O(n |A|) instead of O(n^2).
    That matters for the equal-budget random-search baseline, where n is tens
    of thousands and the pairwise sort is not tractable.
    """
    F = np.asarray(F, dtype=float)
    CV = np.asarray(CV, dtype=float)
    feasible = np.where(CV <= 0)[0]
    if feasible.size == 0:
        return np.zeros(len(F), dtype=bool)

    # Sorting by the objective sum makes early candidates hard to dominate,
    # which keeps the archive small.
    order = feasible[np.argsort(F[feasible].sum(axis=1))]
    arch_idx = []
    arch = np.empty((0, F.shape[1]))
    for i in order:
        f = F[i]
        if arch.shape[0]:
            if np.any(np.all(arch <= f, axis=1) & np.any(arch < f, axis=1)):
                continue
            keep = ~(np.all(f <= arch, axis=1) & np.any(f < arch, axis=1))
            if not keep.all():
                arch = arch[keep]
                arch_idx = [j for j, k in zip(arch_idx, keep) if k]
        arch = np.vstack([arch, f])
        arch_idx.append(int(i))

    mask = np.zeros(len(F), dtype=bool)
    mask[arch_idx] = True
    return mask


def fast_non_dominated_sort(F, CV):
    n = len(F)
    S = [[] for _ in range(n)]
    nd = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], CV[p], F[q], CV[q]):
                S[p].append(q)
            elif dominates(F[q], CV[q], F[p], CV[p]):
                nd[p] += 1
        if nd[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


# --------------------------------------------------------------------------
# Normalisation and niching (Deb & Jain, 2014)
# --------------------------------------------------------------------------

def _achievement(F, w):
    w = np.where(w < 1e-6, 1e-6, w)
    return np.max(F / w, axis=1)


def normalise(F, ideal):
    Fp = F - ideal
    n_obj = F.shape[1]
    extreme = np.zeros(n_obj, dtype=int)
    for j in range(n_obj):
        w = np.full(n_obj, 1e-6)
        w[j] = 1.0
        extreme[j] = int(np.argmin(_achievement(Fp, w)))
    try:
        Z = Fp[extreme]
        b = np.linalg.solve(Z, np.ones(n_obj))
        if np.any(np.abs(b) < 1e-12):
            raise np.linalg.LinAlgError
        intercepts = 1.0 / b
        if np.any(intercepts <= 1e-9) or np.any(~np.isfinite(intercepts)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = np.max(Fp, axis=0)
    intercepts = np.where(intercepts < 1e-9, 1e-9, intercepts)
    return Fp / intercepts


def associate(Fn, refs):
    # perpendicular distance from each point to each reference direction
    norm = np.linalg.norm(refs, axis=1, keepdims=True)
    unit = refs / np.where(norm == 0, 1, norm)
    proj = Fn @ unit.T                      # (n_pop, n_ref)
    d = np.sqrt(np.maximum(
        (Fn ** 2).sum(axis=1, keepdims=True) - proj ** 2, 0.0))
    niche = np.argmin(d, axis=1)
    dist = d[np.arange(len(Fn)), niche]
    return niche, dist


# --------------------------------------------------------------------------
# Variation
# --------------------------------------------------------------------------

def sbx(p1, p2, lo, hi, eta, rng, prob=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > prob:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > 0.5:
            continue
        if abs(p1[i] - p2[i]) < 1e-12:
            continue
        x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        u = rng.random()
        beta = 1.0 + 2.0 * (x1 - lo[i]) / (x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1)
        bq = (u * alpha) ** (1.0 / (eta + 1)) if u <= 1.0 / alpha \
            else (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1))
        c1[i] = 0.5 * ((x1 + x2) - bq * (x2 - x1))
        beta = 1.0 + 2.0 * (hi[i] - x2) / (x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1)
        bq = (u * alpha) ** (1.0 / (eta + 1)) if u <= 1.0 / alpha \
            else (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1))
        c2[i] = 0.5 * ((x1 + x2) + bq * (x2 - x1))
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def polynomial_mutation(x, lo, hi, eta, rng, prob=None):
    x = x.copy()
    prob = prob if prob is not None else 1.0 / max(len(x), 1)
    for i in range(len(x)):
        if rng.random() > prob:
            continue
        y, yl, yu = x[i], lo[i], hi[i]
        if yu - yl < 1e-12:
            continue
        d1, d2 = (y - yl) / (yu - yl), (yu - y) / (yu - yl)
        u = rng.random()
        if u <= 0.5:
            dq = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** (1 / (eta + 1)) - 1
        else:
            dq = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** (1 / (eta + 1))
        x[i] = np.clip(y + dq * (yu - yl), yl, yu)
    return x


def uniform_crossover_int(p1, p2, rng, prob=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > prob:
        return c1, c2
    mask = rng.random(len(p1)) < 0.5
    c1[mask], c2[mask] = p2[mask], p1[mask]
    return c1, c2


def random_reset(x, cardinality, rng, prob=None):
    x = x.copy()
    prob = prob if prob is not None else 1.0 / max(len(x), 1)
    for i in range(len(x)):
        if rng.random() < prob:
            x[i] = rng.integers(0, cardinality[i])
    return x


# --------------------------------------------------------------------------
# The algorithm
# --------------------------------------------------------------------------

class NSGA3:
    def __init__(self, eval_fn, n_cont, cont_bounds, int_cardinality,
                 n_obj=3, partitions=12, pop_size=None, generations=250,
                 eta_c=20.0, eta_m=20.0, seed=0):
        self.eval_fn = eval_fn
        self.n_cont = n_cont
        self.lo = np.array([b[0] for b in cont_bounds], dtype=float)
        self.hi = np.array([b[1] for b in cont_bounds], dtype=float)
        self.card = np.array(int_cardinality, dtype=int)
        self.n_int = len(int_cardinality)
        self.n_obj = n_obj
        self.refs = das_dennis(n_obj, partitions)
        self.pop_size = pop_size or int(np.ceil(len(self.refs) / 4.0) * 4)
        self.generations = generations
        self.eta_c, self.eta_m = eta_c, eta_m
        self.rng = np.random.default_rng(seed)
        self.history = []

    # -- population helpers -------------------------------------------------
    def _init_pop(self):
        n = self.pop_size
        XC = self.rng.random((n, self.n_cont)) * (self.hi - self.lo) + self.lo
        XI = np.column_stack([self.rng.integers(0, c, n) for c in self.card])
        return XC, XI

    def _evaluate(self, XC, XI):
        F = np.zeros((len(XC), self.n_obj))
        CV = np.zeros(len(XC))
        for i in range(len(XC)):
            F[i], CV[i] = self.eval_fn(XC[i], XI[i])
        return F, CV

    def _tournament(self, F, CV, k=2):
        idx = self.rng.integers(0, len(F), k)
        best = idx[0]
        for j in idx[1:]:
            if dominates(F[j], CV[j], F[best], CV[best]):
                best = j
        return best

    def _offspring(self, XC, XI, F, CV):
        oc, oi = [], []
        while len(oc) < self.pop_size:
            a, b = self._tournament(F, CV), self._tournament(F, CV)
            c1, c2 = sbx(XC[a], XC[b], self.lo, self.hi, self.eta_c, self.rng)
            i1, i2 = uniform_crossover_int(XI[a], XI[b], self.rng)
            for c, i in ((c1, i1), (c2, i2)):
                oc.append(polynomial_mutation(c, self.lo, self.hi,
                                              self.eta_m, self.rng))
                oi.append(random_reset(i, self.card, self.rng))
        return np.array(oc[:self.pop_size]), np.array(oi[:self.pop_size])

    # -- environmental selection -------------------------------------------
    def _select(self, XC, XI, F, CV):
        fronts = fast_non_dominated_sort(F, CV)
        chosen, last = [], []
        for fr in fronts:
            if len(chosen) + len(fr) <= self.pop_size:
                chosen.extend(fr)
                if len(chosen) == self.pop_size:
                    last = []
                    break
            else:
                last = fr
                break
        if not last:
            sel = np.array(chosen[:self.pop_size], dtype=int)
            return XC[sel], XI[sel], F[sel], CV[sel]

        pool = np.array(chosen + last, dtype=int)
        ideal = F[pool].min(axis=0)
        Fn = normalise(F[pool], ideal)
        niche, dist = associate(Fn, self.refs)

        n_chosen = len(chosen)
        counts = np.zeros(len(self.refs), dtype=int)
        for i in range(n_chosen):
            counts[niche[i]] += 1

        remaining = list(range(n_chosen, len(pool)))
        picked = list(chosen)
        while len(picked) < self.pop_size:
            avail = [j for j in np.argsort(counts) if any(
                niche[i] == j for i in remaining)]
            if not avail:
                break
            jmin = min(avail, key=lambda j: counts[j])
            members = [i for i in remaining if niche[i] == jmin]
            if counts[jmin] == 0:
                pick = min(members, key=lambda i: dist[i])
            else:
                pick = members[int(self.rng.integers(0, len(members)))]
            picked.append(int(pool[pick]))
            remaining.remove(pick)
            counts[jmin] += 1

        sel = np.array(picked[:self.pop_size], dtype=int)
        return XC[sel], XI[sel], F[sel], CV[sel]

    # -- main loop ----------------------------------------------------------
    def run(self, verbose=False):
        XC, XI = self._init_pop()
        F, CV = self._evaluate(XC, XI)
        for gen in range(self.generations):
            OC, OI = self._offspring(XC, XI, F, CV)
            OF, OCV = self._evaluate(OC, OI)
            XC = np.vstack([XC, OC]); XI = np.vstack([XI, OI])
            F = np.vstack([F, OF]); CV = np.concatenate([CV, OCV])
            XC, XI, F, CV = self._select(XC, XI, F, CV)
            feas = CV <= 0
            self.history.append({
                "gen": gen,
                "feasible": int(feas.sum()),
                "ideal": F[feas].min(axis=0).tolist() if feas.any() else None,
            })
            if verbose and gen % 25 == 0:
                print(f"  gen {gen:4d}  feasible {feas.sum():3d}/{len(F)}  "
                      f"ideal {np.round(F[feas].min(axis=0), 3) if feas.any() else '-'}")
        fronts = fast_non_dominated_sort(F, CV)
        front = np.array([i for i in fronts[0] if CV[i] <= 0], dtype=int)
        if front.size == 0:
            front = np.array(fronts[0], dtype=int)
        return {"XC": XC, "XI": XI, "F": F, "CV": CV, "front": front}


# --------------------------------------------------------------------------
# Quality indicators
# --------------------------------------------------------------------------

def hypervolume(F, ref):
    """Monte-Carlo hypervolume, adequate for 3 objectives and reported with
    its own sampling error."""
    F = np.asarray(F, dtype=float)
    F = F[np.all(F <= ref, axis=1)]
    if len(F) == 0:
        return 0.0, 0.0
    lo = F.min(axis=0)
    rng = np.random.default_rng(12345)
    n = 200_000
    pts = rng.random((n, F.shape[1])) * (ref - lo) + lo
    dominated = np.zeros(n, dtype=bool)
    for f in F:
        dominated |= np.all(f <= pts, axis=1)
    p = dominated.mean()
    vol = float(np.prod(ref - lo))
    return p * vol, float(np.sqrt(p * (1 - p) / n) * vol)


def spacing(F):
    """Schott's spacing metric: lower is a more uniform front."""
    F = np.asarray(F, dtype=float)
    if len(F) < 2:
        return 0.0
    d = np.array([np.min(np.sum(np.abs(F - f), axis=1)[np.arange(len(F)) != i])
                  for i, f in enumerate(F)])
    return float(np.sqrt(np.mean((d.mean() - d) ** 2)))


# --------------------------------------------------------------------------
# NSGA-II, for the comparison that justifies choosing NSGA-III
# --------------------------------------------------------------------------

def crowding_distance(F):
    """Standard NSGA-II crowding distance within one front."""
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(F[:, j])
        d[order[0]] = d[order[-1]] = np.inf
        span = F[order[-1], j] - F[order[0], j]
        if span <= 0:
            continue
        d[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return d


class NSGA2(NSGA3):
    """NSGA-II: identical variation, crowding-distance selection.

    Subclassing NSGA3 keeps the two algorithms sharing one set of operators,
    so the comparison isolates the selection mechanism rather than confounding
    it with different crossover or mutation.
    """

    def _select(self, XC, XI, F, CV):
        fronts = fast_non_dominated_sort(F, CV)
        picked = []
        for fr in fronts:
            if len(picked) + len(fr) <= self.pop_size:
                picked.extend(fr)
            else:
                need = self.pop_size - len(picked)
                d = crowding_distance(F[np.array(fr)])
                order = np.argsort(-d)
                picked.extend([fr[i] for i in order[:need]])
                break
        sel = np.array(picked[:self.pop_size], dtype=int)
        return XC[sel], XI[sel], F[sel], CV[sel]
