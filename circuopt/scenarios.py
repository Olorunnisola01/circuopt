"""Dynamic scenario analysis.

A Pareto front computed on today's factors answers the wrong question.  The
design being specified now will be manufactured against a grid that is
decarbonising, commodity prices that move violently (the USGS series in
data/usgs show 2020-2024 swings of tens of per cent), an electricity tariff
that is not stable, a carbon price that may or may not be internalised, and an
ESPR delegated act that may impose a minimum recycled content.

So the question this module answers is not "which design is best?" but
"which designs stay good across the futures we cannot rule out?".

Three things are produced, and ``run`` documents the difference between them:

  nondominance how often a design stays non-dominated inside the archive.
               Reported for completeness; it saturates, because the archive is
               already a mutually non-dominated set, and so it ranks nothing.

  robustness   how often a design is a top-decile answer across
               future x preference draws.  This is the measure that
               discriminates, and the one the report ranks on.

  profiles     for each decision-maker profile (design team, product manager,
               business-unit leader, regulator-facing), the design that an
               augmented achievement scalarising function selects, per future.
               This is the decision aid: one front, read through four
               different sets of eyes.
"""

from __future__ import annotations

import math

import numpy as np

from .model import Assumptions, Context, Design, evaluate
from .datasets import load_usgs_prices


def empirical_price_sigma(commodity: str, floor: float = 0.05,
                          cap: float = 0.60) -> float:
    """Log-return standard deviation of a commodity's published price series.

    The USGS Mineral Commodity Summaries publish an annual price series per
    commodity.  Using its own realised volatility to set the width of that
    commodity's scenario distribution replaces a guessed sigma with a measured
    one.  It is still a crude estimator -- five annual observations -- so it is
    clamped, and the clamp is reported.
    """
    hist = load_usgs_prices().get(commodity, {}).get("history", {})
    vals = [v for _, v in sorted(hist.items()) if v and v > 0]
    if len(vals) < 3:
        return 0.20        # ASSUMPTION where the series is too short
    rets = [math.log(vals[i + 1] / vals[i]) for i in range(len(vals) - 1)]
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / max(len(rets) - 1, 1)
    return float(min(max(math.sqrt(var), floor), cap))

#: Preference weights over (planet, cost, social), each summing to 1.
#: These encode who is reading the front, not what is true.
PROFILES = {
    "design team":          np.array([0.45, 0.15, 0.40]),
    "product manager":      np.array([0.20, 0.55, 0.25]),
    "business unit leader": np.array([0.30, 0.50, 0.20]),
    "regulator-facing":     np.array([0.55, 0.20, 0.25]),
}


#: Cached so the USGS files are not re-read per scenario.
_SIGMA = {}


def price_sigmas() -> dict:
    """Realised log-return volatility per commodity, from USGS 2020-2024."""
    if not _SIGMA:
        for fam, commodity in (("copper", "Copper"),
                               ("steel", "Iron and Steel Scrap"),
                               ("aluminium", "Aluminum")):
            _SIGMA[fam] = empirical_price_sigma(commodity)
    return dict(_SIGMA)


def sample_assumptions(rng: np.random.Generator, base: Assumptions,
                       sigma_scale: float = 1.0,
                       sample_grid_region: bool = True) -> Assumptions:
    """Draw one plausible future.

    Distribution widths come from three places, and which is which is stated
    rather than blurred:

      MEASURED   commodity price volatility, from the realised log-return
                 standard deviation of the USGS 2020-2024 annual series
      DECLARED   the grid region, drawn uniformly over the six European
                 markets Idemat resolves -- a discrete, data-backed set rather
                 than an invented decarbonisation rate
      JUDGEMENT  everything else.  These are expert-judgement widths with no
                 empirical backing, which is why ``sigma_scale`` exists: E8
                 halves and doubles them and checks whether the robustness
                 ranking survives.

    ``sigma_scale`` multiplies only the JUDGEMENT widths.
    """
    a = Assumptions(**{k: v for k, v in base.__dict__.items()})
    sig = price_sigmas()
    j = float(sigma_scale)

    if sample_grid_region:
        regions = ("Sweden", "EU-27", "France", "Germany", "Netherlands", "Poland")
        a.grid_region = str(rng.choice(regions))

    # UK grid decarbonisation: 0-7 % per year, compounding over the service
    # life.  Expressed as the multiplier on today's average factor for a
    # 12-year horizon, which is the middle of the lifetime design space.
    delta = rng.uniform(0.0, 0.07)
    horizon = 12
    a.use_grid_multiplier = float(
        np.mean([(1.0 - delta) ** t for t in range(horizon)]))

    # Electricity tariff and carbon price
    a.electricity_price_usd_per_kwh = float(base.electricity_price_usd_per_kwh
                                            * rng.lognormal(0.0, 0.30))
    a.carbon_price_usd_per_tonne = float(rng.choice(
        [0.0, 0.0, 40.0, 90.0, 160.0], p=[0.30, 0.20, 0.20, 0.20, 0.10]))

    # Commodity and resin prices; the metal side is perturbed through the
    # secondary-price ratio and the steel multiple, the resin side directly.
    a.plastic_virgin_usd_per_kg = float(base.plastic_virgin_usd_per_kg
                                        * rng.lognormal(0.0, 0.22 * j))
    a.plastic_recyclate_price_ratio = float(np.clip(
        rng.normal(base.plastic_recyclate_price_ratio, 0.18 * j), 0.45, 1.45))
    # MEASURED: the secondary-price ratio and the steel multiple are moved by
    # the realised volatility of the underlying commodity series.
    a.metal_secondary_price_ratio = float(np.clip(
        rng.normal(base.metal_secondary_price_ratio,
                   0.10 * sig["copper"] / 0.20), 0.40, 1.00))
    a.steel_primary_multiple = float(np.clip(
        rng.normal(base.steel_primary_multiple,
                   base.steel_primary_multiple * sig["steel"]), 1.4, 4.2))
    a.electronics_usd_per_kg = float(base.electronics_usd_per_kg
                                     * rng.lognormal(0.0, 0.18 * j))
    a.labour_rate_usd_per_hour = float(base.labour_rate_usd_per_hour
                                       * rng.lognormal(0.0, 0.15 * j))

    # How badly recyclate behaves: the penalty terms that stop recycled
    # content from being a free lunch are themselves uncertain.
    a.recyclate_threshold = float(np.clip(
        rng.normal(base.recyclate_threshold, 0.12 * j), 0.15, 0.75))
    a.recyclate_cost_penalty = float(np.clip(
        rng.normal(base.recyclate_cost_penalty, 0.25 * j), 0.05, 1.40))
    a.recyclate_reject_rate = float(np.clip(
        rng.normal(base.recyclate_reject_rate, 0.04 * j), 0.0, 0.22))

    # Circular-process performance
    a.scrap_recovery_efficiency = float(np.clip(
        rng.normal(base.scrap_recovery_efficiency, 0.08 * j), 0.55, 0.97))
    a.reman_life_extension = float(np.clip(
        rng.normal(base.reman_life_extension, 0.12 * j), 0.25, 0.90))

    # Use intensity: how hard the station is actually worked.  The published
    # duty cycle (20 000 sessions / 15 years) is the central case; a station
    # at a busy site or a quiet one departs from it substantially.
    a.duty_multiplier = float(base.duty_multiplier * rng.lognormal(0.0, 0.35 * j))
    a.kwh_per_charging_session = float(base.kwh_per_charging_session
                                       * rng.lognormal(0.0, 0.15 * j))

    # Proxy uncertainty: the families whose DEFRA factor is a stand-in are
    # perturbed hardest, because that is where the model is weakest.
    # The electronics mapping is the weakest in the model (a white-goods board
    # standing in for a charging-station board), so it is perturbed hardest.
    a.proxy_multiplier = {
        "steel":       float(rng.lognormal(0.0, 0.12 * j)),
        "aluminium":   float(rng.lognormal(0.0, 0.12 * j)),
        "copper":      float(rng.lognormal(0.0, 0.15 * j)),
        "electronics": float(rng.lognormal(0.0, 0.40 * j)),
    }

    # Regulation: an ESPR delegated act imposing minimum recycled content
    a.mandated_recycled_content = float(rng.choice(
        [0.00, 0.00, 0.15, 0.25, 0.35], p=[0.35, 0.20, 0.20, 0.15, 0.10]))

    return a


def evaluate_under(designs, a: Assumptions):
    """Return (F, CV) for a list of designs under one future."""
    ctx = Context.build(a)
    F = np.zeros((len(designs), 3))
    CV = np.zeros(len(designs))
    for i, d in enumerate(designs):
        r = evaluate(d, ctx)
        F[i] = (r["f_planet_kgco2e_per_service_year"],
                r["f_cost_usd_per_service_year"],
                -r["f_social_index"])
        CV[i] = r["constraint_violation"]
    return F, CV


def _non_dominated(F, CV):
    n = len(F)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if CV[i] > 0:
            nd[i] = False
            continue
        for j in range(n):
            if i == j or CV[j] > 0:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                nd[i] = False
                break
    return nd


def asf(Fn, w, rho=1e-4):
    """Augmented achievement scalarising function (Wierzbicki).

    Unlike a weighted sum it can reach non-convex parts of the front, which
    matters here because the cost/social trade-off is visibly non-convex.
    """
    w = np.where(w < 1e-9, 1e-9, w)
    return np.max(Fn / w, axis=1) + rho * np.sum(Fn / w, axis=1)


def run(designs, n_scenarios=400, seed=7, base: Assumptions | None = None,
        progress=None, n_tastes=24, top_quantile=0.10, sigma_scale=1.0):
    """Score every candidate design across ``n_scenarios`` futures.

    Three things are measured, and the difference between them matters.

    ``nondominance``  the share of futures in which the design remains
        non-dominated inside the candidate set.  Reported for completeness,
        but it saturates: the candidates are *already* a mutually
        non-dominated set, so perturbing the factors rarely creates a
        dominance relation between them.  A number near 1.0 for everything is
        the expected, and uninformative, outcome.

    ``robustness``  the share of (future x preference) draws in which the
        design lands in the best ``top_quantile`` of feasible candidates by
        achievement scalarising value.  Preferences are drawn from a
        Dirichlet, so this asks the question that actually discriminates:
        does this design stay a *good answer* when both the world and the
        decision-maker's priorities move?

    ``wins``  per named profile, the share of futures in which that profile's
        weights select this design outright.
    """
    base = base or Assumptions()
    rng = np.random.default_rng(seed)
    n = len(designs)

    nd_count = np.zeros(n)
    feasible_count = np.zeros(n)
    top_count = np.zeros(n)
    draws = 0
    regret = {p: np.zeros(n) for p in PROFILES}
    regret_seen = {p: np.zeros(n) for p in PROFILES}
    wins = {p: np.zeros(n) for p in PROFILES}
    F_stack = []

    for s in range(n_scenarios):
        a = sample_assumptions(rng, base, sigma_scale=sigma_scale)
        F, CV = evaluate_under(designs, a)
        F_stack.append(F)
        feas = CV <= 0
        feasible_count += feas
        nd_count += _non_dominated(F, CV)
        if not feas.any():
            continue

        idx = np.where(feas)[0]
        Ff = F[feas]
        lo, hi = Ff.min(axis=0), Ff.max(axis=0)
        Fn = (Ff - lo) / np.where(hi - lo < 1e-12, 1e-12, hi - lo)

        # named profiles: outright winner and normalised regret
        for name, w in PROFILES.items():
            v = asf(Fn, w)
            wins[name][idx[int(np.argmin(v))]] += 1
            spread = v.max() - v.min()
            regret[name][idx] += (v - v.min()) / (spread if spread > 1e-12 else 1.0)
            regret_seen[name][idx] += 1

        # random tastes: does the design stay in the top decile?
        k = max(int(np.ceil(top_quantile * len(idx))), 1)
        for _ in range(n_tastes):
            w = rng.dirichlet(np.ones(3) * 1.5)
            v = asf(Fn, w)
            top_count[idx[np.argsort(v)[:k]]] += 1
            draws += 1

        if progress and (s + 1) % progress == 0:
            print(f"  scenario {s + 1}/{n_scenarios}", flush=True)

    F_stack = np.array(F_stack)
    # A design that is infeasible in a future simply does not appear in that
    # future's top decile, so the denominator is every draw, not only the
    # draws in which the design was admissible.  Regulatory infeasibility is
    # part of what robustness has to survive.
    total_draws = max(n_scenarios * n_tastes, 1)
    return {
        "n_scenarios": n_scenarios,
        "n_tastes": n_tastes,
        "top_quantile": top_quantile,
        "robustness": top_count / total_draws,
        "nondominance": nd_count / n_scenarios,
        "feasibility": feasible_count / n_scenarios,
        "mean_regret": {p: np.divide(regret[p], np.maximum(regret_seen[p], 1))
                        for p in PROFILES},
        "wins": {k: v / n_scenarios for k, v in wins.items()},
        "F_mean": F_stack.mean(axis=0),
        "F_std": F_stack.std(axis=0),
        "F_p05": np.percentile(F_stack, 5, axis=0),
        "F_p95": np.percentile(F_stack, 95, axis=0),
    }
