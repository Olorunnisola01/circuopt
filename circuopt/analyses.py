"""Sensitivity and verification analyses.

Each function here exists because a specific criticism of the model would
otherwise be unanswerable.  They are grouped by what they defend against:

  indicator_reordering    "you optimised carbon and called it LCA"
  allocation_sensitivity  "your end-of-life credit convention decides the answer"
  parameter_sensitivity   "your headline finding is an artifact of three
                           invented constants"
  weight_sensitivity      "the social objective is whatever you weighted it to be"
  geography_sensitivity   "a Swedish grid flatters every conclusion"
  algorithm_comparison    "random search does nearly as well, so why NSGA-III"
  validation_decomposition "your 18% agreement was luck, not accuracy"
  distributional_sensitivity "the Monte-Carlo widths were guessed"

Nothing here changes the model.  Everything here tries to break its
conclusions and reports how far it got.
"""

from __future__ import annotations

import math

import numpy as np

from . import model as M
from . import scenarios as SC
from .idemat import INDICATOR_LABELS


# --------------------------------------------------------------------------
# Rank statistics, written out rather than pulled from scipy so the report can
# state exactly what was computed.
# --------------------------------------------------------------------------

def _rank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        r = (sums / counts)[inv]
    return r


def spearman(a, b) -> float:
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def kendall_tau(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    conc = disc = 0
    for i in range(n - 1):
        da = a[i + 1:] - a[i]
        db = b[i + 1:] - b[i]
        s = np.sign(da) * np.sign(db)
        conc += int((s > 0).sum())
        disc += int((s < 0).sum())
    tot = conc + disc
    return float((conc - disc) / tot) if tot else float("nan")


def mannwhitney_u(x, y):
    """Two-sided Mann-Whitney U with a normal approximation, plus rank-biserial
    effect size.  Sample sizes here are small (tens of seeds), so the effect
    size matters more than the p-value and both are reported."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    r = _rank(np.concatenate([x, y]))
    rx = r[:nx].sum()
    u1 = rx - nx * (nx + 1) / 2.0
    u2 = nx * ny - u1
    u = min(u1, u2)
    mu = nx * ny / 2.0
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12.0)
    z = (u - mu) / sigma if sigma > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))
    effect = 1.0 - 2.0 * u / (nx * ny)      # rank-biserial correlation
    return {"U": float(u), "z": float(z), "p_two_sided": float(p),
            "rank_biserial": float(abs(effect)),
            "n_x": nx, "n_y": ny}


# --------------------------------------------------------------------------
# 1. Does the front reorder between impact categories?
# --------------------------------------------------------------------------

def indicator_reordering(designs, base: M.Assumptions | None = None) -> dict:
    """Rank every archive design under each impact category and compare.

    If the ranking is essentially identical across categories, then optimising
    carbon was a fair proxy for optimising the environment and saying so is
    honest.  If it is not, then a carbon-only study would have selected the
    wrong designs, and that is the finding.
    """
    base = base or M.Assumptions()
    out = {"indicators": list(base.indicators), "per_indicator": {}}
    values = {}
    for ind in base.indicators:
        a = M.Assumptions(**{**base.__dict__, "impact_indicator": ind})
        ctx = M.Context.build(a)
        v = np.array([M.evaluate(d, ctx)["f_planet"] for d in designs])
        values[ind] = v
        out["per_indicator"][ind] = {
            "label": INDICATOR_LABELS.get(ind, ind),
            "min": float(v.min()), "max": float(v.max()),
            "best_design_index": int(np.argmin(v)),
        }
    ref = base.indicators[0]
    out["reference_indicator"] = ref
    out["rank_correlation_vs_reference"] = {
        ind: {"spearman": spearman(values[ref], values[ind]),
              "kendall_tau": kendall_tau(values[ref], values[ind])}
        for ind in base.indicators if ind != ref}
    out["best_design_agrees_with_reference"] = {
        ind: bool(np.argmin(values[ind]) == np.argmin(values[ref]))
        for ind in base.indicators if ind != ref}
    out["values"] = {k: v.tolist() for k, v in values.items()}
    return out


# --------------------------------------------------------------------------
# 2. Allocation convention (ISO 14044 requires this to be tested)
# --------------------------------------------------------------------------

def allocation_sensitivity(designs, base: M.Assumptions | None = None) -> dict:
    """cut-off vs. substitution vs. the EU PEF Circular Footprint Formula.

    CFF is included as a third convention, not a separate experiment,
    because it answers the same question the other two do (how is recycling
    credited?) with a citable regulatory method rather than an independently
    invented compromise -- see inventory.ALLOCATIONS for the formula and its
    stated scope limit (material term only).
    """
    base = base or M.Assumptions()
    res = {}
    vals = {}
    for alloc in ("cut_off", "substitution", "cff"):
        a = M.Assumptions(**{**base.__dict__, "allocation": alloc})
        ctx = M.Context.build(a)
        rr = [M.evaluate(d, ctx) for d in designs]
        v = np.array([r["f_planet"] for r in rr])
        vals[alloc] = v
        eol = np.array([r["gwp_breakdown"]["end_of_life"] for r in rr])
        res[alloc] = {
            "planet_min": float(v.min()), "planet_mean": float(v.mean()),
            "eol_gwp_mean": float(eol.mean()),
            "eol_share_of_total_mean": float(
                np.mean(np.abs(eol) / np.array([r["gwp_total_kgco2e"] for r in rr]))),
            "best_design_index": int(np.argmin(v)),
        }
    res["spearman_between_conventions"] = spearman(vals["cut_off"],
                                                   vals["substitution"])
    res["kendall_between_conventions"] = kendall_tau(vals["cut_off"],
                                                     vals["substitution"])
    res["same_best_design"] = bool(res["cut_off"]["best_design_index"]
                                   == res["substitution"]["best_design_index"])
    res["mean_shift_pct"] = float(
        100 * (vals["substitution"].mean() / vals["cut_off"].mean() - 1))
    res["cff_vs_cutoff_spearman"] = spearman(vals["cut_off"], vals["cff"])
    res["cff_vs_substitution_spearman"] = spearman(vals["substitution"], vals["cff"])
    res["cff_same_best_as_cutoff"] = bool(res["cut_off"]["best_design_index"]
                                          == res["cff"]["best_design_index"])
    res["cff_same_best_as_substitution"] = bool(
        res["substitution"]["best_design_index"] == res["cff"]["best_design_index"])

    # CFF's own A parameter, swept, since A=0.5 is a cited default, not a
    # verified value for this product's materials
    cff_sweep = {}
    for A in (0.0, 0.5, 1.0):
        a = M.Assumptions(**{**base.__dict__, "allocation": "cff",
                            "cff_allocation_factor": A})
        ctx = M.Context.build(a)
        v = np.array([M.evaluate(d, ctx)["f_planet"] for d in designs])
        cff_sweep[str(A)] = {"planet_mean": float(v.mean()),
                             "best_design_index": int(np.argmin(v))}
    res["cff_allocation_factor_sweep"] = cff_sweep
    # A=1 gives full credit to the recycled-content user on the input side,
    # which is exactly this model's "substitution" treatment on both sides
    # (cff falls back to substitution at end-of-life regardless of A); the
    # two conventions should therefore coincide exactly at A=1, and this
    # equality is a correctness check on the formula's implementation, not
    # an empirical finding.
    res["cff_A1_equals_substitution_exactly"] = math.isclose(
        cff_sweep["1.0"]["planet_mean"], res["substitution"]["planet_mean"],
        rel_tol=1e-6)
    return res


# --------------------------------------------------------------------------
# 3. Is the interior optimum in recycled content real, or an artifact?
# --------------------------------------------------------------------------

def recyclate_parameter_sensitivity(base: M.Assumptions | None = None,
                                    n_grid: int = 81) -> dict:
    """Sweep the three invented recyclate-penalty constants.

    The finding that recycled content has an interior cost optimum -- rather
    than being driven to 1.0 -- is produced by these three constants.  None of
    them has an empirical source.  This function reports, over their plausible
    ranges, how often an interior optimum exists at all and where it sits, so
    the finding can be stated with its true conditionality instead of as a
    fact about recyclate.
    """
    base = base or M.Assumptions()
    rs = np.linspace(0.0, 1.0, n_grid)

    def optimum_for(threshold, penalty, reject):
        a = M.Assumptions(**{**base.__dict__,
                             "recyclate_threshold": threshold,
                             "recyclate_cost_penalty": penalty,
                             "recyclate_reject_rate": reject})
        ctx = M.Context.build(a)
        costs = []
        for r in rs:
            d = M.Design(recycled={f: float(r) for f in M.DESIGN_FAMILIES},
                         joints=(0, 1, 1), lifetime=15, reman=1,
                         eol={f: "Closed-loop" for f in M.EOL_FAMILIES},
                         standby_class=2)
            costs.append(M.evaluate(d, ctx)["f_cost_usd_per_service_year"])
        costs = np.array(costs)
        i = int(np.argmin(costs))
        return float(rs[i]), costs

    nominal_r, nominal_costs = optimum_for(base.recyclate_threshold,
                                           base.recyclate_cost_penalty,
                                           base.recyclate_reject_rate)

    # one-at-a-time sweeps
    oat = {}
    sweeps = {
        "recyclate_threshold": np.linspace(0.15, 0.75, 13),
        "recyclate_cost_penalty": np.linspace(0.0, 1.40, 15),
        "recyclate_reject_rate": np.linspace(0.0, 0.22, 12),
    }
    for name, grid in sweeps.items():
        rows = []
        for val in grid:
            kw = {"recyclate_threshold": base.recyclate_threshold,
                  "recyclate_cost_penalty": base.recyclate_cost_penalty,
                  "recyclate_reject_rate": base.recyclate_reject_rate}
            kw[name] = float(val)
            r_opt, _ = optimum_for(kw["recyclate_threshold"],
                                   kw["recyclate_cost_penalty"],
                                   kw["recyclate_reject_rate"])
            rows.append({"value": float(val), "argmin_recycled_content": r_opt,
                         "interior": bool(0.02 < r_opt < 0.98)})
        oat[name] = rows

    # full-factorial corner check
    corners = []
    for th in (0.15, 0.40, 0.75):
        for pen in (0.05, 0.60, 1.40):
            for rej in (0.0, 0.09, 0.22):
                r_opt, _ = optimum_for(th, pen, rej)
                corners.append({"threshold": th, "penalty": pen, "reject": rej,
                                "argmin_recycled_content": r_opt,
                                "interior": bool(0.02 < r_opt < 0.98)})
    n_int = sum(1 for c in corners if c["interior"])
    return {
        "nominal_argmin_recycled_content": nominal_r,
        "nominal_interior": bool(0.02 < nominal_r < 0.98),
        "cost_curve_recycled_content": rs.tolist(),
        "cost_curve_nominal": nominal_costs.tolist(),
        "one_at_a_time": oat,
        "corners": corners,
        "corner_interior_fraction": n_int / len(corners),
        "verdict": ("The interior optimum is a property of the assumed penalty "
                    "parameters, not an empirical result: it appears in "
                    f"{n_int} of {len(corners)} corner combinations."),
    }


# --------------------------------------------------------------------------
# 4. Is the social ranking an artifact of the chosen weights?
# --------------------------------------------------------------------------

def social_weight_sensitivity(designs, base: M.Assumptions | None = None,
                              n_draws: int = 400, seed: int = 3) -> dict:
    """Redraw the four social sub-weights from a flat Dirichlet.

    The nominal weights (0.35, 0.20, 0.20, 0.25) were chosen, not elicited.
    If the identity of the best-scoring design is stable when the weights are
    resampled, the ranking is a property of the designs; if it is not, the
    ranking is a property of the weights and must be reported as such.
    """
    base = base or M.Assumptions()
    rng = np.random.default_rng(seed)
    ctx0 = M.Context.build(base)
    nominal = np.array([M.evaluate(d, ctx0)["f_social_index"] for d in designs])
    nominal_best = int(np.argmax(nominal))

    # sub-scores are weight-independent, so compute them once
    subs = np.array([[M.evaluate(d, ctx0)["social_breakdown"][k]
                      for k in ("repairability", "modularity",
                                "reman_labour", "supply_risk_relief")]
                     for d in designs])

    wins = np.zeros(len(designs))
    rhos = []
    for _ in range(n_draws):
        w = rng.dirichlet(np.ones(4))
        s = subs @ w
        wins[int(np.argmax(s))] += 1
        rhos.append(spearman(nominal, s))

    return {
        "n_draws": n_draws,
        "nominal_weights": list(base.social_weights),
        "nominal_best_index": nominal_best,
        "nominal_best_win_share": float(wins[nominal_best] / n_draws),
        "distinct_winners": int((wins > 0).sum()),
        "top_winner_index": int(np.argmax(wins)),
        "top_winner_share": float(wins.max() / n_draws),
        "spearman_vs_nominal_mean": float(np.mean(rhos)),
        "spearman_vs_nominal_p05": float(np.percentile(rhos, 5)),
    }


# --------------------------------------------------------------------------
# 5. Geography
# --------------------------------------------------------------------------

def geography_sensitivity(designs, base: M.Assumptions | None = None) -> dict:
    base = base or M.Assumptions()
    inv = M.Context.build(base).inv
    out, vals = {}, {}
    for region in inv.grid_regions:
        a = M.Assumptions(**{**base.__dict__, "grid_region": region})
        ctx = M.Context.build(a)
        v = np.array([M.evaluate(d, ctx)["f_planet"] for d in designs])
        vals[region] = v
        out[region] = {"planet_mean": float(v.mean()),
                       "planet_min": float(v.min()),
                       "best_design_index": int(np.argmin(v)),
                       "grid_kgco2e_per_kwh": float(
                           inv.electricity(region)["gwp_kgco2e"])}
    ref = base.grid_region
    out_rank = {r: {"spearman": spearman(vals[ref], vals[r]),
                    "same_best": bool(np.argmin(vals[r]) == np.argmin(vals[ref]))}
                for r in vals if r != ref}
    means = [out[r]["planet_mean"] for r in out]
    return {"reference_region": ref, "per_region": out,
            "rank_correlation_vs_reference": out_rank,
            "span_ratio": float(max(means) / min(means))}


# --------------------------------------------------------------------------
# 6. Does the optimiser earn its keep?
# --------------------------------------------------------------------------

def algorithm_comparison(hv_by_method: dict) -> dict:
    """Pairwise Mann-Whitney U tests between per-seed hypervolume samples."""
    methods = list(hv_by_method)
    tests = {}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            t = mannwhitney_u(hv_by_method[a], hv_by_method[b])
            t["median_a"] = float(np.median(hv_by_method[a]))
            t["median_b"] = float(np.median(hv_by_method[b]))
            t["median_ratio"] = t["median_a"] / t["median_b"] if t["median_b"] else float("nan")
            tests[f"{a} vs {b}"] = t
    return {"methods": {m: {"median": float(np.median(v)),
                            "mean": float(np.mean(v)),
                            "std": float(np.std(v)),
                            "n": len(v)} for m, v in hv_by_method.items()},
            "tests": tests}


# --------------------------------------------------------------------------
# 7. Where does the validation discrepancy actually come from?
# --------------------------------------------------------------------------

def validation_decomposition(ctx) -> dict:
    """Attribute the model-vs-published gap to individual material families.

    Reporting only the aggregate agreement hides whether it is accuracy or
    cancellation.  This splits the difference by family and reports the sum of
    absolute contributions alongside the net, so the reader can see how much
    of the agreement is offsetting error.
    """
    pub = M.GARO["published_results"]
    pubf = M.GARO["published_emission_factors_gco2e_per_kg"]
    published_total = pub["raw_material_and_component_production_kgco2e"]

    # published-study factor per model family
    fam_pub = {
        "aluminium": pubf["Aluminium"] / 1000.0,
        "steel": pubf["Steel"] / 1000.0,
        "plastic": pubf["General plastic"] / 1000.0,
        "copper": pubf["Copper/plastic"] / 1000.0,
    }

    rows, net, absolute = [], 0.0, 0.0
    model_total = 0.0
    for fam in ("aluminium", "electronics", "plastic", "steel", "copper"):
        mass = sum(c.mass_kg for c in M.BOM if c.family == fam)
        model_f = ctx.inv.material_input(fam, 0.0)["gwp_kgco2e"]
        model_total += mass * model_f
        if fam not in fam_pub:
            rows.append({"family": fam, "mass_kg": mass,
                         "model_factor": model_f, "published_factor": None,
                         "contribution_kgco2e": None,
                         "note": "no comparable factor published"})
            continue
        contrib = mass * (model_f - fam_pub[fam])
        net += contrib
        absolute += abs(contrib)
        rows.append({"family": fam, "mass_kg": mass,
                     "model_factor": model_f,
                     "published_factor": fam_pub[fam],
                     "contribution_kgco2e": contrib,
                     "direction": "model higher" if contrib > 0 else "model lower"})

    cancellation = 1.0 - abs(net) / absolute if absolute > 0 else 0.0
    return {
        "rows": rows,
        "net_difference_kgco2e": net,
        "sum_absolute_difference_kgco2e": absolute,
        "cancellation_fraction": cancellation,
        "model_total_kgco2e": model_total,
        "published_total_kgco2e": published_total,
        "ratio": model_total / published_total,
        "verdict": (f"{100 * cancellation:.0f}% of the family-level disagreement "
                    f"cancels in the total, so the aggregate agreement "
                    f"overstates the model's per-material accuracy."),
    }


# --------------------------------------------------------------------------
# 8. Were the Monte-Carlo widths load-bearing?
# --------------------------------------------------------------------------

def distributional_sensitivity(designs, n_scenarios=120, seed=11,
                               scales=(0.5, 1.0, 2.0),
                               base: M.Assumptions | None = None) -> dict:
    """Rerun the ensemble with the judgement widths halved and doubled.

    The measured widths (commodity volatility) and the declared discrete set
    (grid region) are left alone; only the expert-judgement sigmas move.  If
    the robustness ranking survives a factor of four in those widths, the
    conclusions do not rest on them.
    """
    base = base or M.Assumptions()
    rank_ref = None
    out = {}
    for s in scales:
        res = SC.run(designs, n_scenarios=n_scenarios, seed=seed, base=base,
                     sigma_scale=s)
        rob = np.asarray(res["robustness"])
        if s == 1.0:
            rank_ref = rob
        out[str(s)] = {"robustness": rob.tolist(),
                       "top_index": int(np.argmax(rob)),
                       "robust_core": int((rob >= 2.0 * res["top_quantile"]).sum())}
    for s in scales:
        out[str(s)]["spearman_vs_nominal"] = spearman(
            rank_ref, np.asarray(out[str(s)]["robustness"]))
    out["_scales"] = list(scales)
    out["_n_scenarios"] = n_scenarios
    return out


# --------------------------------------------------------------------------
# 9. How wrong would the electronics proxy have to be to matter?
# --------------------------------------------------------------------------

def electronics_breakeven(designs, base: M.Assumptions | None = None,
                          multipliers=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                                      5.0, 8.0, 12.0)) -> dict:
    """Scale the electronics factor and see what it takes to change a
    conclusion, rather than just widening its error bar and hoping.

    Electronics is the weakest mapping in the inventory (an averaged
    white-goods control board, excluding ICs, standing in for a metering/RCD
    board -- see families.electronics.virgin in the process map). Two
    specific things are checked, because "we perturbed it" is a weaker
    statement than "here is exactly how far wrong it would have to be":

      crossover   the multiplier at which electronics overtakes aluminium as
                  the largest material-stage carbon contributor, for the
                  reference (linear-baseline) design
      rank_stability  whether scaling electronics re-ranks the archive at
                  all.  Electronics mass is not itself a decision variable
                  (no family in DESIGN_FAMILIES touches it), so scaling its
                  factor is expected to shift every design's Planet score by
                  a near-constant amount and barely re-rank them -- a
                  structural insensitivity worth demonstrating, not asserting.
    """
    base = base or M.Assumptions()
    ref = M.baseline_design()

    def shares(mult):
        a = M.Assumptions(**{**base.__dict__,
                             "proxy_multiplier": {"electronics": mult}})
        ctx = M.Context.build(a)
        per_family = {}
        for c in M.BOM:
            f = ctx.inv.material_input(c.family, 0.0)["gwp_kgco2e"]
            m = a.proxy_multiplier.get(c.family, 1.0)
            per_family[c.family] = per_family.get(c.family, 0.0) + c.mass_kg * f * m
        return per_family

    crossover = None
    fine = np.linspace(0.1, 20.0, 400)
    for m in fine:
        sh = shares(float(m))
        if sh.get("electronics", 0.0) >= sh.get("aluminium", 0.0):
            crossover = float(m)
            break

    at_1 = shares(1.0)

    rank_ref = None
    rho_by_mult = {}
    for m in multipliers:
        a = M.Assumptions(**{**base.__dict__,
                             "proxy_multiplier": {"electronics": m}})
        ctx = M.Context.build(a)
        F = np.array([M.evaluate(d, ctx)["f_planet"] for d in designs])
        if m == 1.0:
            rank_ref = F
        rho_by_mult[str(m)] = {"best_design_index": int(np.argmin(F))}
    for m in multipliers:
        a = M.Assumptions(**{**base.__dict__,
                             "proxy_multiplier": {"electronics": m}})
        ctx = M.Context.build(a)
        F = np.array([M.evaluate(d, ctx)["f_planet"] for d in designs])
        rho_by_mult[str(m)]["spearman_vs_nominal"] = spearman(rank_ref, F)

    return {
        "aluminium_share_at_1x": at_1.get("aluminium", 0.0)
        / sum(at_1.values()),
        "electronics_share_at_1x": at_1.get("electronics", 0.0)
        / sum(at_1.values()),
        "crossover_multiplier": crossover,
        "rank_stability_by_multiplier": rho_by_mult,
        "min_spearman_vs_nominal": min(
            v["spearman_vs_nominal"] for v in rho_by_mult.values()),
        "tested_multiplier_range": [float(fine.min()), float(fine.max())],
        "verdict": (
            (f"Electronics would have to be roughly {crossover:.1f}x its "
             f"current factor before it overtakes aluminium as the largest "
             f"material contributor -- a far larger error than the scenario "
             f"ensemble's own perturbation of this factor.")
            if crossover else
            (f"Electronics does not overtake aluminium as the largest "
             f"material contributor anywhere in the tested "
             f"{fine.min():.0f}x-{fine.max():.0f}x range: aluminium's factor "
             f"is now large enough (E12/aluminium reconciliation) that the "
             f"electronics proxy would have to be wrong by more than a "
             f"defensible margin to change which material dominates.")
        ) + (
            f" Across the tested range the archive's ranking barely moves "
            f"regardless (worst-case Spearman "
            f"{min(v['spearman_vs_nominal'] for v in rho_by_mult.values()):+.3f}) "
            f"because electronics mass is not a decision variable -- scaling "
            f"its factor shifts every design's score together."
        ),
    }


# --------------------------------------------------------------------------
# 10. Does the design decision matter more than the modelling decision?
# --------------------------------------------------------------------------

def methodological_vs_design_spread(designs, allocation_result, geography_result,
                                    base: M.Assumptions | None = None) -> dict:
    """Compare the archive's own spread in the Planet objective against the
    spread induced by four purely methodological choices, design space held
    fixed.  A synthesis of E5-E9 and E12, not a new measurement: every
    number here is either already computed (allocation, geography) or a
    single cheap re-evaluation (backend).

    The question this answers: does deciding WHAT to build move the result
    more than deciding HOW to model it?  If methodology dwarfs design choice,
    that is worth knowing on its own, independent of any single design
    recommendation in this report.
    """
    base = base or M.Assumptions()
    ctx0 = M.Context.build(base)
    F0 = np.array([M.evaluate(d, ctx0)["f_planet"] for d in designs])
    design_span = float(F0.max() - F0.min())

    a_defra = M.Assumptions(**{**base.__dict__, "backend": "defra"})
    ctx_defra = M.Context.build(a_defra)
    F_defra = np.array([M.evaluate(d, ctx_defra)["f_planet"] for d in designs])
    backend_span = abs(float(F_defra.mean()) - float(F0.mean()))

    allocation_span = abs(allocation_result["substitution"]["planet_mean"]
                          - allocation_result["cut_off"]["planet_mean"])

    region_means = [v["planet_mean"] for v in geography_result["per_region"].values()]
    geography_span = float(max(region_means) - min(region_means))

    spans = {
        "design choice (whole Pareto archive)": design_span,
        "inventory backend (Idemat vs DEFRA)": backend_span,
        "allocation convention (cut-off vs substitution)": allocation_span,
        "grid region (6 European markets)": geography_span,
    }
    ranked = sorted(spans.items(), key=lambda kv: -kv[1])
    return {
        "design_mean_planet": float(F0.mean()),
        "spans_kgco2e_per_service_year": spans,
        "ranked_largest_first": [k for k, _ in ranked],
        "design_choice_rank": [k for k, _ in ranked].index(
            "design choice (whole Pareto archive)") + 1,
        "verdict": ("Ranked by how much each moves the Planet objective: "
                    + " > ".join(f"{k} ({v:.1f})" for k, v in ranked)),
    }


# --------------------------------------------------------------------------
# 11. Does the cost conclusion depend on which price source is used?
# --------------------------------------------------------------------------

def price_source_sensitivity(designs, base: M.Assumptions | None = None) -> dict:
    """USGS vs. World Bank commodity prices, archive-wide.

    The two sources disagree (World Bank's copper figure runs above USGS's,
    for instance -- see the printed comparison), which is the expected
    behaviour of two independently compiled global price series rather than
    an error in either.  This tests whether that disagreement matters:
    does the archive's cost ranking, or its best design, change with the
    source?  Steel has no comparable World Bank entry (that source publishes
    iron ORE, not scrap -- a different commodity), so this test covers
    aluminium and copper, the two families it can speak to.
    """
    base = base or M.Assumptions()
    vals, prices = {}, {}
    for src in ("usgs", "worldbank"):
        a = M.Assumptions(**{**base.__dict__, "price_source": src})
        ctx = M.Context.build(a)
        F = np.array([M.evaluate(d, ctx)["f_cost_usd_per_service_year"]
                     for d in designs])
        vals[src] = F
        prices[src] = {"aluminium": ctx.price("aluminium"),
                       "copper": ctx.price("copper")}

    return {
        "prices_usd_per_kg": prices,
        "cost_mean": {s: float(v.mean()) for s, v in vals.items()},
        "cost_span_pct": float(
            100 * (vals["worldbank"].mean() / vals["usgs"].mean() - 1)),
        "spearman_between_sources": spearman(vals["usgs"], vals["worldbank"]),
        "same_best_design": bool(np.argmin(vals["usgs"])
                                 == np.argmin(vals["worldbank"])),
        "note": ("World Bank has no directly comparable steel-scrap entry "
                 "(it publishes iron ore, a different commodity), so this "
                 "test covers aluminium and copper only."),
    }
