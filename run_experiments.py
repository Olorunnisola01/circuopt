#!/usr/bin/env python3
"""CircuOpt experimental protocol.

  E0  provenance   what every factor is, and where it came from
  E1  baseline     the linear-economy reference design
  E2  optimisation NSGA-III on today's factors, 5 seeds, vs. an equal-budget
                   random search; hypervolume, spacing, seed stability
  E3  scenarios    Monte-Carlo robustness of the nominal front across futures
  E4  profiles     which design each decision-maker profile selects, and how
                   often that choice survives the scenario ensemble
  E5  indicators   does the front reorder between impact categories, or was
                   optimising carbon a fair proxy for optimising the environment
  E6  allocation   cut-off versus substitution, the ISO 14044 methodological
                   choice that end-of-life results are most sensitive to
  E7  parameters   are the recyclate-penalty findings artifacts of three
                   invented constants
  E8  weights      is the social ranking a property of the designs or of the
                   weights chosen for them
  E9  geography    how much of the answer is the grid the station is plugged into
  E10 algorithm    NSGA-III against NSGA-II and random search, with a
                   Mann-Whitney test and an effect size rather than a claim

Writes results.json and figures/*.png.  Everything downstream (the report)
reads results.json; nothing is transcribed by hand.

    python3 run_experiments.py [--quick]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from circuopt import model as M
from circuopt import scenarios as SC
from circuopt.nsga3 import (NSGA2, NSGA3, hypervolume, spacing,
                            fast_non_dominated_sort, nondominated_mask)
from circuopt import analyses as AN

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

QUICK = "--quick" in sys.argv
GENERATIONS = 60 if QUICK else 300
SEEDS = (1, 2) if QUICK else (1, 2, 3, 4, 5)
N_SCENARIOS = 60 if QUICK else 500
PARTITIONS = 8 if QUICK else 12
ALGO_SEEDS = (1, 2, 3) if QUICK else tuple(range(1, 11))
ALGO_GENERATIONS = 30 if QUICK else 120
DIST_SCENARIOS = 40 if QUICK else 150

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})
C_PLANET, C_COST, C_SOCIAL = "#2b7a4b", "#b3541e", "#33628f"


def design_key(d: M.Design) -> str:
    return json.dumps(d.as_dict(), sort_keys=True)


# --------------------------------------------------------------------------
def e0_provenance(ctx):
    print("\n=== E0  factor provenance ===")
    rows = []
    for fam, (name, tag) in M.FAMILY_DEFRA.items():
        prim = ctx.defra.primary[name]
        closed = ctx.defra.closed_loop.get(name)
        rows.append({
            "family": fam, "defra_material": name, "provenance": tag,
            "primary_kgco2e_per_kg": round(prim, 4),
            "closed_loop_kgco2e_per_kg": round(closed, 4) if closed else None,
            "reduction_pct": round(100 * (1 - closed / prim), 1) if closed else None,
        })
        print(f"  {fam:12s} {tag:10s} {name:48s} "
              f"{prim:7.3f} -> {closed if closed else float('nan'):7.3f}")
    prices = []
    for commodity, rec in sorted(ctx.prices.items()):
        prices.append({"commodity": commodity,
                       "usd_per_kg": round(rec["usd_per_kg"], 4),
                       "reported": f"{rec['reported_value']} {rec['reported_unit']}",
                       "year": rec["year"], "column": rec["column"]})
        print(f"  price {commodity:22s} {rec['usd_per_kg']:9.3f} USD/kg "
              f"({rec['reported_value']} {rec['reported_unit']}, {rec['year']})")
    print(f"  UK grid  {ctx.grid:.5f} kg CO2e/kWh "
          f"(generation {ctx.defra.electricity_generation} + T&D "
          f"{ctx.defra.electricity_td})")
    missing = [f for f in M.FAMILY_USGS
               if M.FAMILY_USGS[f][0] not in ctx.prices]
    if missing:
        wb_covers = [f for f in missing
                    if M.FAMILY_USGS[f][0].rstrip("s") in ctx.worldbank_prices
                    or M.FAMILY_USGS[f][0] in ctx.worldbank_prices]
        print(f"  NOTE: no USGS price file for {missing}; "
              f"{'covered by World Bank data instead' if wb_covers else 'the hardcoded fallback constant is used'} "
              f"(see fix 3 / E14).")
        print(f"  aluminium price now in use: {ctx.price('aluminium'):.3f} USD/kg "
              f"(World Bank, trailing 12-month mean)")
    inv_desc = ctx.inv.describe()
    print(f"  inventory backend  : {inv_desc['backend']}")
    print(f"  allocation         : {inv_desc['allocation']}")
    print(f"  grid region        : {inv_desc['grid_region']} "
          f"({inv_desc.get('grid_process', {}).get('process', '')})")
    print(f"  indicators carried : {', '.join(inv_desc['indicators'])}")
    return {"materials": rows, "prices": prices, "inventory": inv_desc,
            "grid_kgco2e_per_kwh": ctx.grid,
            "usgs_families_missing": missing,
            "criticality_weights": {k: round(v, 4)
                                    for k, v in ctx.criticality.items()}}


# --------------------------------------------------------------------------
def e1_baseline(ctx):
    print("\n=== E1  linear-economy baseline ===")
    print(f"  case product: {M.GARO['product']['name']}")
    print(f"  cited mass  : {M.TOTAL_MASS:.2f} kg product "
          f"({M.GARO['excluded_packaging_pct']:.1f}% packaging excluded from "
          f"the published {M.GARO['product']['bom_mass_incl_packaging_kg']} kg BOM)")
    for c in M.BOM:
        print(f"    {c.name:46s} {c.mass_kg:6.3f} kg  [{c.family}]")
    b = M.baseline_design()
    r = M.evaluate(b, ctx)
    print(f"  service life            {r['service_years']:.1f} years")
    print(f"  cradle-to-grave GWP     {r['gwp_total_kgco2e']:.1f} kg CO2e")
    print(f"  planet objective        {r['f_planet_kgco2e_per_service_year']:.3f} "
          f"kg CO2e / service-year")
    print(f"  cost objective          {r['f_cost_usd_per_service_year']:.2f} "
          f"USD / service-year")
    print(f"  social index            {r['f_social_index']:.3f}")
    print(f"  MCI                     {r['material_circularity_indicator']:.3f}")

    v = M.validate_against_published(ctx)
    print("\n  -- cross-check against the published LCA of the same unit --")
    print(f"  cradle-to-gate materials, this model  {v['model_materials_kgco2e']:7.1f} "
          f"kg CO2e")
    print(f"  cradle-to-gate materials, published   "
          f"{v['published_materials_kgco2e']:7.1f} kg CO2e")
    print(f"  ratio                                 "
          f"{v['ratio_model_over_published']:7.3f}")
    print("  material shares, this model: " + ", ".join(
        f"{k} {100 * s:.0f}%" for k, s in v["model_material_shares"].items()))
    print(f"  published: {v['published_hotspot_note']}")
    print(f"  factor cross-check (kg CO2e/kg), backend = {v['backend']}:")
    for k, d in v["factor_cross_check_kgco2e_per_kg"].items():
        print(f"    {k:24s} model {d['defra_model']:6.3f}  "
              f"published {d['published_study']:6.3f}  ratio {d['ratio']:.3f}")
    return {"design": b.as_dict(), "result": _clean(r), "validation": v,
            "bom": [{"component": c.name, "mass_kg": round(c.mass_kg, 4),
                     "family": c.family, "interface": c.joint,
                     "priority_repair": c.priority_repair} for c in M.BOM],
            "source": M.GARO["source"], "product": M.GARO["product"]}


def _clean(r):
    """Round floats for the JSON payload, at any nesting depth."""
    if isinstance(r, float):
        return round(r, 6)
    if isinstance(r, dict):
        return {k: _clean(v) for k, v in r.items()}
    if isinstance(r, (list, tuple)):
        return [_clean(v) for v in r]
    return r


# --------------------------------------------------------------------------
def e2_optimise(ctx):
    print("\n=== E2  NSGA-III on today's factors ===")
    def ev(xc, xi):
        return M.objectives(xc, xi, ctx)

    bounds = [(0.0, 1.0)] * M.N_CONT
    runs, fronts, times, cpu_times = [], [], [], []
    for seed in SEEDS:
        t0, c0 = time.time(), time.process_time()
        alg = NSGA3(ev, M.N_CONT, bounds, M.INT_CARDINALITY,
                    partitions=PARTITIONS, generations=GENERATIONS, seed=seed)
        out = alg.run(verbose=(seed == SEEDS[0]))
        dt, dc = time.time() - t0, time.process_time() - c0
        times.append(dt); cpu_times.append(dc)
        runs.append((out, alg))
        fronts.append(out["F"][out["front"]])
        print(f"  seed {seed}: |front| {len(out['front']):3d}  "
              f"pop {alg.pop_size}  refs {len(alg.refs)}  "
              f"{dc:.1f}s cpu / {dt:.1f}s wall")

    # a reference point for hypervolume, fixed across seeds and methods
    allF = np.vstack(fronts)
    ref = np.array([allF[:, 0].max() * 1.05,
                    allF[:, 1].max() * 1.05,
                    0.0])
    hvs, hverr, sps = [], [], []
    for F in fronts:
        hv, err = hypervolume(F, ref)
        hvs.append(hv); hverr.append(err); sps.append(spacing(F))
    print(f"  hypervolume  {np.mean(hvs):.2f} +/- {np.std(hvs):.2f} "
          f"(sampling error ~{np.mean(hverr):.2f})")
    print(f"  spacing      {np.mean(sps):.4f} +/- {np.std(sps):.4f}")

    # equal-budget random search, as a floor
    budget = runs[0][1].pop_size * (GENERATIONS + 1)
    rng = np.random.default_rng(99)
    RXC = rng.random((budget, M.N_CONT))
    RXI = np.column_stack([rng.integers(0, c, budget) for c in M.INT_CARDINALITY])
    RF = np.zeros((budget, 3)); RCV = np.zeros(budget)
    for i in range(budget):
        RF[i], RCV[i] = ev(RXC[i], RXI[i])
    rfront = np.where(nondominated_mask(RF, RCV))[0]
    hv_rand, _ = hypervolume(RF[rfront], ref) if rfront.size else (0.0, 0.0)
    print(f"  random search, equal budget ({budget} evals): "
          f"hypervolume {hv_rand:.2f}  ({100 * hv_rand / np.mean(hvs):.1f}% of NSGA-III)")

    # merge the seeds into one archive of unique non-dominated designs
    XC = np.vstack([o["XC"][o["front"]] for o, _ in runs])
    XI = np.vstack([o["XI"][o["front"]] for o, _ in runs])
    F = np.vstack(fronts)
    CV = np.zeros(len(F))
    keep = np.where(nondominated_mask(F, CV))[0]
    seen, uniq = set(), []
    for i in keep:
        d = M.Design.decode(XC[i], XI[i])
        k = design_key(d)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((d, F[i]))
    print(f"  merged archive: {len(uniq)} unique non-dominated designs")

    return {
        "generations": GENERATIONS, "seeds": list(SEEDS),
        "pop_size": runs[0][1].pop_size, "n_refs": len(runs[0][1].refs),
        "runtime_cpu_s_per_seed": [round(c, 1) for c in cpu_times],
        "runtime_cpu_s_mean": float(np.mean(cpu_times)),
        "runtime_wall_s_per_seed": [round(t, 1) for t in times],
        #: Wall-clock is reported only for completeness.  It is not a measure
        #: of the algorithm's cost: a suspended machine inflates it without
        #: consuming a single cycle, which is exactly what happened on one
        #: seed of an earlier run of this protocol.
        "runtime_wall_s_mean": float(np.mean(times)),
        "hv_ref_point": ref.tolist(),
        "hypervolume_mean": float(np.mean(hvs)),
        "hypervolume_std": float(np.std(hvs)),
        "hypervolume_mc_error": float(np.mean(hverr)),
        "spacing_mean": float(np.mean(sps)), "spacing_std": float(np.std(sps)),
        "random_search_budget": int(budget),
        "random_search_hypervolume": float(hv_rand),
        "front_sizes": [int(len(f)) for f in fronts],
        "archive_size": len(uniq),
    }, uniq, runs, ref


# --------------------------------------------------------------------------
def e5_to_e9(uniq, ctx):
    designs = [d for d, _ in uniq]

    print("\n=== E5  do the impact categories agree? ===")
    ir = AN.indicator_reordering(designs)
    print(f"  reference indicator: {ir['reference_indicator']}")
    for ind, rc in ir["rank_correlation_vs_reference"].items():
        agree = ir["best_design_agrees_with_reference"][ind]
        print(f"    {ind:36s} rho={rc['spearman']:+.3f}  tau={rc['kendall_tau']:+.3f}  "
              f"same best design: {agree}")

    print("\n=== E6  allocation convention (ISO 14044 + EU PEF CFF) ===")
    al = AN.allocation_sensitivity(designs)
    for conv in ("cut_off", "substitution", "cff"):
        r = al[conv]
        print(f"  {conv:13s} mean planet {r['planet_mean']:8.3f}  "
              f"end-of-life is {100 * r['eol_share_of_total_mean']:5.2f}% of total GWP")
    print(f"  rank correlation between conventions rho={al['spearman_between_conventions']:+.3f}, "
          f"same best design: {al['same_best_design']}")
    print(f"  mean shift when crediting recovery: {al['mean_shift_pct']:+.1f}%")
    print(f"  CFF vs cut-off rho={al['cff_vs_cutoff_spearman']:+.3f}  "
          f"CFF vs substitution rho={al['cff_vs_substitution_spearman']:+.3f}")
    print(f"  CFF correctness check (A=1 == substitution exactly): "
          f"{al['cff_A1_equals_substitution_exactly']}")
    for A, v in al["cff_allocation_factor_sweep"].items():
        print(f"    CFF A={A:<4s} mean planet {v['planet_mean']:8.3f}")

    print("\n=== E7  are the recyclate findings artifacts? ===")
    ps = AN.recyclate_parameter_sensitivity(n_grid=61)
    print(f"  nominal cost-optimal recycled content: "
          f"{ps['nominal_argmin_recycled_content']:.2f} (interior: {ps['nominal_interior']})")
    print(f"  {ps['verdict']}")
    for name, rows in ps["one_at_a_time"].items():
        lo = min(r["argmin_recycled_content"] for r in rows)
        hi = max(r["argmin_recycled_content"] for r in rows)
        n_int = sum(1 for r in rows if r["interior"])
        print(f"    {name:26s} interior in {n_int:2d}/{len(rows):2d} sweeps, "
              f"optimum ranges {lo:.2f}-{hi:.2f}")

    print("\n=== E8  is the social ranking the designs or the weights? ===")
    ws = AN.social_weight_sensitivity(designs)
    print(f"  nominal best design #{ws['nominal_best_index']} stays best in "
          f"{100 * ws['nominal_best_win_share']:.1f}% of {ws['n_draws']} random weightings")
    print(f"  distinct winners across weightings: {ws['distinct_winners']}")
    print(f"  rank correlation with the nominal weighting: mean rho="
          f"{ws['spearman_vs_nominal_mean']:+.3f}, 5th pct {ws['spearman_vs_nominal_p05']:+.3f}")

    print("\n=== E9  geography ===")
    gs = AN.geography_sensitivity(designs)
    for r, v in gs["per_region"].items():
        print(f"    {r:12s} grid {v['grid_kgco2e_per_kwh']:.4f} kgCO2e/kWh -> "
              f"mean planet {v['planet_mean']:8.2f}")
    print(f"  span across European grids: {gs['span_ratio']:.1f}x")
    worst = max(gs["rank_correlation_vs_reference"].items(),
                key=lambda kv: -kv[1]["spearman"])
    print(f"  weakest rank agreement vs {gs['reference_region']}: "
          f"{worst[0]} rho={worst[1]['spearman']:+.3f}")

    print("\n=== E13  electronics: how wrong would it have to be? ===")
    eb = AN.electronics_breakeven(designs)
    print(f"  material-stage shares at 1x: aluminium "
          f"{100 * eb['aluminium_share_at_1x']:.1f}%, electronics "
          f"{100 * eb['electronics_share_at_1x']:.1f}%")
    print(f"  {eb['verdict']}")

    print("\n=== E14  price source: USGS vs. World Bank ===")
    pr = AN.price_source_sensitivity(designs)
    for src, p in pr["prices_usd_per_kg"].items():
        print(f"    {src:10s} aluminium {p['aluminium']:7.3f}  "
              f"copper {p['copper']:7.3f}  USD/kg")
    print(f"  mean cost shift, World Bank vs USGS: {pr['cost_span_pct']:+.2f}%")
    print(f"  rank correlation between sources: rho={pr['spearman_between_sources']:+.3f}, "
          f"same best design: {pr['same_best_design']}")
    print(f"  {pr['note']}")

    print("\n=== E15  design choice vs. modelling choice ===")
    sp = AN.methodological_vs_design_spread(designs, al, gs)
    for k in sp["ranked_largest_first"]:
        print(f"    {k:52s} {sp['spans_kgco2e_per_service_year'][k]:8.1f}")
    print(f"  design choice ranks #{sp['design_choice_rank']} of "
          f"{len(sp['spans_kgco2e_per_service_year'])} by how much it moves "
          f"the Planet objective")

    return {"E5_indicators": ir, "E6_allocation": al, "E7_parameters": ps,
            "E8_weights": ws, "E9_geography": gs, "E13_electronics": eb,
            "E14_prices": pr, "E15_spread": sp}


def e10_algorithm(ctx, hv_ref):
    """NSGA-III vs NSGA-II vs random search, on equal evaluation budgets."""
    print("\n=== E10  does the optimiser earn its keep? ===")
    def ev(xc, xi):
        return M.objectives(xc, xi, ctx)
    bounds = [(0.0, 1.0)] * M.N_CONT
    seeds = ALGO_SEEDS
    hv = {"NSGA-III": [], "NSGA-II": [], "random search": []}
    budget = None

    for cls, name in ((NSGA3, "NSGA-III"), (NSGA2, "NSGA-II")):
        for seed in seeds:
            alg = cls(ev, M.N_CONT, bounds, M.INT_CARDINALITY,
                      partitions=PARTITIONS, generations=ALGO_GENERATIONS,
                      seed=seed)
            out = alg.run()
            budget = alg.pop_size * (ALGO_GENERATIONS + 1)
            h, _ = hypervolume(out["F"][out["front"]], hv_ref)
            hv[name].append(h)
        print(f"  {name:14s} median HV {np.median(hv[name]):8.2f} "
              f"over {len(seeds)} seeds")

    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        RXC = rng.random((budget, M.N_CONT))
        RXI = np.column_stack([rng.integers(0, c, budget)
                               for c in M.INT_CARDINALITY])
        RF = np.zeros((budget, 3)); RCV = np.zeros(budget)
        for i in range(budget):
            RF[i], RCV[i] = ev(RXC[i], RXI[i])
        mask = nondominated_mask(RF, RCV)
        h, _ = hypervolume(RF[mask], hv_ref)
        hv["random search"].append(h)
    print(f"  {'random search':14s} median HV {np.median(hv['random search']):8.2f} "
          f"over {len(seeds)} seeds, budget {budget} evaluations each")

    stats = AN.algorithm_comparison(hv)
    for pair, t in stats["tests"].items():
        print(f"    {pair:32s} ratio {t['median_ratio']:.4f}  "
              f"U={t['U']:.0f}  p={t['p_two_sided']:.4f}  "
              f"effect r={t['rank_biserial']:.2f}")
    stats["budget_per_run"] = budget
    stats["generations"] = ALGO_GENERATIONS
    stats["seeds"] = list(seeds)
    return stats


def e3_scenarios(uniq):
    print(f"\n=== E3  scenario ensemble ({N_SCENARIOS} futures) ===")
    designs = [d for d, _ in uniq]
    t0 = time.time()
    res = SC.run(designs, n_scenarios=N_SCENARIOS, seed=7,
                 progress=max(N_SCENARIOS // 4, 1))
    print(f"  {time.time() - t0:.1f}s")
    rb = res["robustness"]
    nd = res["nondominance"]
    print(f"  within-front non-dominance: mean {nd.mean() * 100:.1f}%, "
          f"min {nd.min() * 100:.1f}% - saturated, as expected, and therefore "
          f"not used to rank")
    order = np.argsort(-rb)
    print("  most robust designs (share of future x preference draws in which "
          "the design is a top-decile answer):")
    for i in order[:6]:
        d = designs[i].as_dict()
        print(f"    #{i:3d}  {rb[i] * 100:5.1f}%  feasible "
              f"{res['feasibility'][i] * 100:5.1f}%  "
              f"L={d['design_lifetime_years']}y reman={d['remanufacture_cycles']} "
              f"rPlastic={d['recycled_content']['plastic']:.2f} "
              f"joints={d['joints']} eol_plastic={d['end_of_life_route']['plastic']}")
    # A design picked at random would be a top-decile answer in
    # `top_quantile` of draws.  The robust core is the set that clears twice
    # that chance rate, which is scale-free in the archive size.
    thr = 2.0 * res["top_quantile"]
    print(f"  robust core (top-decile answer at >=2x the chance rate, "
          f">={100 * thr:.0f}% of draws): {(rb >= thr).sum()} of {len(rb)}")
    print(f"  never a top-decile answer: {(rb == 0).sum()} of {len(rb)}")
    return res


# --------------------------------------------------------------------------
def e4_profiles(uniq, sc):
    print("\n=== E4  decision-maker profiles ===")
    designs = [d for d, _ in uniq]
    out = {}
    for name, w in SC.PROFILES.items():
        wins = sc["wins"][name]
        i = int(np.argmax(wins))
        d = designs[i].as_dict()
        out[name] = {
            "weights": w.tolist(),
            "modal_choice_index": i,
            "modal_choice_share": float(wins[i]),
            "design": d,
            "robustness": float(sc["robustness"][i]),
            "mean_regret": float(sc["mean_regret"][name][i]),
            "mean_objectives": sc["F_mean"][i].tolist(),
            "p05_objectives": sc["F_p05"][i].tolist(),
            "p95_objectives": sc["F_p95"][i].tolist(),
            "distinct_winners": int((wins > 0).sum()),
        }
        print(f"  {name:22s} picks design #{i:3d} in {wins[i] * 100:5.1f}% of futures "
              f"({(wins > 0).sum()} distinct winners) | L={d['design_lifetime_years']}y "
              f"reman={d['remanufacture_cycles']} standby={d['standby_class']}")
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_front(uniq, base_r):
    F = np.array([f for _, f in uniq])
    fig = plt.figure(figsize=(10, 3.3))

    ax = fig.add_subplot(1, 3, 1)
    s = ax.scatter(F[:, 0], -F[:, 2], c=F[:, 1], cmap="cividis_r", s=20)
    ax.scatter(base_r["f_planet_kgco2e_per_service_year"],
               base_r["f_social_index"],
               marker="X", s=80, color="crimson", zorder=5)
    ax.set_xlabel("Planet  [kg CO$_2$e / service-year]")
    ax.set_ylabel("People index")
    plt.colorbar(s, ax=ax, label="USD / yr", pad=0.02)

    ax = fig.add_subplot(1, 3, 2)
    s = ax.scatter(F[:, 0], F[:, 1], c=-F[:, 2], cmap="viridis", s=20)
    ax.scatter(base_r["f_planet_kgco2e_per_service_year"],
               base_r["f_cost_usd_per_service_year"],
               marker="X", s=80, color="crimson", label="linear baseline", zorder=5)
    ax.set_xlabel("Planet  [kg CO$_2$e / service-year]")
    ax.set_ylabel("Prosperity  [USD / service-year]")
    ax.legend(fontsize=7, loc="upper right")
    plt.colorbar(s, ax=ax, label="People index", pad=0.02)

    ax = fig.add_subplot(1, 3, 3)
    s = ax.scatter(-F[:, 2], F[:, 1], c=F[:, 0], cmap="magma_r", s=20)
    ax.scatter(base_r["f_social_index"], base_r["f_cost_usd_per_service_year"],
               marker="X", s=80, color="crimson", zorder=5)
    ax.set_xlabel("People index")
    ax.set_ylabel("Prosperity  [USD / service-year]")
    plt.colorbar(s, ax=ax, label="kg CO$_2$e / yr", pad=0.02)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_front.png"))
    plt.close(fig)


def fig_convergence(runs):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for (out, alg), seed in zip(runs, SEEDS):
        gens = [h["gen"] for h in alg.history if h["ideal"]]
        y = [h["ideal"][0] for h in alg.history if h["ideal"]]
        ax.plot(gens, y, lw=1.1, label=f"seed {seed}")
    ax.set_xlabel("generation")
    ax.set_ylabel("best Planet objective\n[kg CO$_2$e / service-year]")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Convergence of the ideal point", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_convergence.png"))
    plt.close(fig)


def fig_robustness(uniq, sc):
    F = np.array([f for _, f in uniq])
    rb = sc["robustness"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    s = axes[0].scatter(F[:, 0], F[:, 1], c=rb, cmap="RdYlGn", s=26,
                        vmin=0, vmax=max(rb.max(), 1e-6),
                        edgecolor="k", linewidth=0.2)
    axes[0].set_xlabel("Planet  [kg CO$_2$e / service-year]")
    axes[0].set_ylabel("Prosperity  [USD / service-year]")
    axes[0].set_title("Nominal front, coloured by robustness", fontsize=9)
    plt.colorbar(s, ax=axes[0], label="robustness\n(top-decile share)", pad=0.02)

    err_lo = sc["F_mean"][:, 0] - sc["F_p05"][:, 0]
    err_hi = sc["F_p95"][:, 0] - sc["F_mean"][:, 0]
    o = np.argsort(sc["F_mean"][:, 0])
    axes[1].errorbar(np.arange(len(o)), sc["F_mean"][o, 0],
                     yerr=[err_lo[o], err_hi[o]], fmt="o", ms=2.5,
                     lw=0.7, color=C_PLANET, ecolor="#9ec7ad", capsize=1.5)
    axes[1].set_xlabel("archive design, ranked by mean Planet objective")
    axes[1].set_ylabel("kg CO$_2$e / service-year")
    axes[1].set_title("5th-95th percentile across futures", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_robustness.png"))
    plt.close(fig)


def fig_profiles(uniq, sc, prof):
    names = list(SC.PROFILES)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))

    width = 0.2
    top = np.argsort(-sum(sc["wins"][n] for n in names))[:8]
    x = np.arange(len(top))
    for k, n in enumerate(names):
        axes[0].bar(x + k * width, sc["wins"][n][top] * 100, width, label=n)
    axes[0].set_xticks(x + 1.5 * width)
    axes[0].set_xticklabels([f"#{i}" for i in top], fontsize=7)
    axes[0].set_ylabel("% of futures in which\nthis design is selected")
    axes[0].set_xlabel("archive design")
    axes[0].legend(fontsize=6.5)
    axes[0].set_title("Who chooses what, across futures", fontsize=9)

    keys = ["recycled_content_plastic", "lifetime", "reman", "standby_w",
            "disassembly_s"]
    labels = ["recycled\nplastic", "design life\n[yr]", "reman\ncycles",
              "standby\n[W]", "disassembly\n[s]"]
    vals = []
    for n in names:
        d = prof[n]["design"]
        vals.append([d["recycled_content"]["plastic"],
                     d["design_lifetime_years"] / 15.0,
                     d["remanufacture_cycles"] / 3.0,
                     1.0 - d["standby_w"],
                     0.0])
    vals = np.array(vals)
    xx = np.arange(len(labels) - 1)
    for k, n in enumerate(names):
        axes[1].bar(xx + k * width, vals[k, :4], width, label=n)
    axes[1].set_xticks(xx + 1.5 * width)
    axes[1].set_xticklabels(labels[:4], fontsize=7)
    axes[1].set_ylabel("normalised design lever")
    axes[1].set_title("The levers each profile pulls", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_profiles.png"))
    plt.close(fig)


def fig_breakdown(ctx, base_r, prof):
    names = list(prof)
    designs = {"linear baseline": M.baseline_design()}
    for n in names:
        designs[n] = _design_from_dict(prof[n]["design"])

    labels = list(designs)
    gwp_keys = ["materials", "manufacture", "use", "remanufacture", "end_of_life"]
    cost_keys = ["material", "assembly", "standby_premium", "pv_energy",
                 "pv_remanufacture", "pv_end_of_life", "carbon"]
    G = np.zeros((len(labels), len(gwp_keys)))
    Cc = np.zeros((len(labels), len(cost_keys)))
    S = np.zeros(len(labels))
    for i, l in enumerate(labels):
        r = M.evaluate(designs[l], ctx)
        yrs = r["service_years"]
        G[i] = [r["gwp_breakdown"][k] / yrs for k in gwp_keys]
        ann = (1 - (1 + ctx.a.discount_rate) ** (-yrs)) / ctx.a.discount_rate
        Cc[i] = [r["cost_breakdown"][k] / ann for k in cost_keys]
        S[i] = r["f_social_index"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, Mx, keys, title, unit in (
            (axes[0], G, gwp_keys, "Planet, by life-cycle stage",
             "kg CO$_2$e / service-year"),
            (axes[1], Cc, cost_keys, "Prosperity, by cost element",
             "USD / service-year")):
        bottom_pos = np.zeros(len(labels)); bottom_neg = np.zeros(len(labels))
        for j, k in enumerate(keys):
            v = Mx[:, j]
            base = np.where(v >= 0, bottom_pos, bottom_neg)
            ax.bar(labels, v, bottom=base, label=k.replace("_", " "))
            bottom_pos = np.where(v >= 0, bottom_pos + v, bottom_pos)
            bottom_neg = np.where(v < 0, bottom_neg + v, bottom_neg)
        ax.set_ylabel(unit); ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        for t in ax.get_xticklabels():
            t.set_ha("right")
        ax.legend(fontsize=6, ncol=2)
        ax.axhline(0, color="k", lw=0.6)

    axes[2].bar(labels, S, color=C_SOCIAL)
    axes[2].set_ylabel("People index"); axes[2].set_ylim(0, 1)
    axes[2].set_title("People", fontsize=9)
    axes[2].tick_params(axis="x", rotation=30, labelsize=7)
    for t in axes[2].get_xticklabels():
        t.set_ha("right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_breakdown.png"))
    plt.close(fig)


def _design_from_dict(d) -> M.Design:
    joints = tuple(next(i for i, j in enumerate(M.JOINTS) if j[0] == name)
                   for name in d["joints"])
    sc = next(i for i, c in enumerate(M.STANDBY_CLASSES)
              if c[0] == d["standby_class"])
    return M.Design(recycled=dict(d["recycled_content"]), joints=joints,
                    lifetime=d["design_lifetime_years"],
                    reman=d["remanufacture_cycles"],
                    eol=dict(d["end_of_life_route"]), standby_class=sc)


def fig_datasets(ctx, prov):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
    fams = [r["family"] for r in prov["materials"] if r["closed_loop_kgco2e_per_kg"]]
    prim = [r["primary_kgco2e_per_kg"] for r in prov["materials"]
            if r["closed_loop_kgco2e_per_kg"]]
    clos = [r["closed_loop_kgco2e_per_kg"] for r in prov["materials"]
            if r["closed_loop_kgco2e_per_kg"]]
    x = np.arange(len(fams))
    axes[0].bar(x - 0.2, prim, 0.4, label="primary production", color="#9c9c9c")
    axes[0].bar(x + 0.2, clos, 0.4, label="closed-loop source", color=C_PLANET)
    axes[0].set_xticks(x); axes[0].set_xticklabels(fams, fontsize=7)
    axes[0].set_ylabel("kg CO$_2$e / kg")
    axes[0].set_title("DEFRA 2025: the recycled-content lever", fontsize=9)
    axes[0].legend(fontsize=7)

    for commodity, rec in sorted(ctx.prices.items()):
        h = rec["history"]
        if len(h) < 3:
            continue
        yrs = [int(y) for y in h]
        v = np.array(list(h.values()))
        axes[1].plot(yrs, v / v[0], marker="o", ms=3, lw=1.1, label=commodity)
    axes[1].set_ylabel("price, indexed to first year")
    axes[1].set_xlabel("year")
    axes[1].set_title("USGS MCS 2025: the volatility the model must survive",
                      fontsize=9)
    axes[1].legend(fontsize=6.5, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_datasets.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    ctx = M.Context.build()
    prov = e0_provenance(ctx)
    base = e1_baseline(ctx)
    opt, uniq, runs, ref = e2_optimise(ctx)
    sc = e3_scenarios(uniq)
    prof = e4_profiles(uniq, sc)
    extra = e5_to_e9(uniq, ctx)
    algo = e10_algorithm(ctx, np.array(opt["hv_ref_point"]))

    print("\n=== E11  were the Monte-Carlo widths load-bearing? ===")
    ds = AN.distributional_sensitivity([d for d, _ in uniq],
                                       n_scenarios=DIST_SCENARIOS)
    for sc_ in ds["_scales"]:
        row = ds[str(sc_)]
        print(f"  judgement sigmas x{sc_:<4} top design #{row['top_index']:3d}  "
              f"robust core {row['robust_core']:3d}  "
              f"rho vs nominal {row['spearman_vs_nominal']:+.3f}")

    print("\n=== E12  where does the validation gap come from? ===")
    vd = AN.validation_decomposition(ctx)
    for r in vd["rows"]:
        if r.get("contribution_kgco2e") is None:
            print(f"    {r['family']:12s} {r['mass_kg']:6.2f} kg  "
                  f"model {r['model_factor']:7.3f}  (no published counterpart)")
        else:
            print(f"    {r['family']:12s} {r['mass_kg']:6.2f} kg  "
                  f"model {r['model_factor']:7.3f} vs published "
                  f"{r['published_factor']:7.3f}  "
                  f"contributes {r['contribution_kgco2e']:+8.1f} kg CO2e")
    print(f"  net difference {vd['net_difference_kgco2e']:+.1f}, "
          f"sum of absolute differences {vd['sum_absolute_difference_kgco2e']:.1f}")
    print(f"  {vd['verdict']}")

    print("\n=== figures ===")
    fig_datasets(ctx, prov)
    fig_front(uniq, base["result"])
    fig_convergence(runs)
    fig_robustness(uniq, sc)
    fig_profiles(uniq, sc, prof)
    fig_breakdown(ctx, base["result"], prof)
    for f in sorted(os.listdir(FIGS)):
        print("  figures/" + f)

    # headline comparisons against the baseline
    F = np.array([f for _, f in uniq])
    b = base["result"]
    best_planet = int(np.argmin(F[:, 0]))
    best_cost = int(np.argmin(F[:, 1]))
    both = np.where((F[:, 0] < b["f_planet_kgco2e_per_service_year"]) &
                    (F[:, 1] < b["f_cost_usd_per_service_year"]))[0]
    headline = {
        "n_dominating_baseline_on_planet_and_cost": int(len(both)),
        "archive_size": len(uniq),
        "max_planet_reduction_pct": float(
            100 * (1 - F[:, 0].min() / b["f_planet_kgco2e_per_service_year"])),
        "max_cost_reduction_pct": float(
            100 * (1 - F[:, 1].min() / b["f_cost_usd_per_service_year"])),
    }
    if len(both):
        k = both[int(np.argmin(F[both, 0]))]
        headline["win_win_example"] = {
            "index": int(k),
            "design": uniq[k][0].as_dict(),
            "planet_reduction_pct": float(
                100 * (1 - F[k, 0] / b["f_planet_kgco2e_per_service_year"])),
            "cost_reduction_pct": float(
                100 * (1 - F[k, 1] / b["f_cost_usd_per_service_year"])),
            "social_gain_pct": float(100 * (-F[k, 2] / b["f_social_index"] - 1)),
            "robustness": float(sc["robustness"][k]),
        }
    print("\n=== headline ===")
    print(f"  {headline['n_dominating_baseline_on_planet_and_cost']} of "
          f"{headline['archive_size']} Pareto designs beat the linear baseline "
          f"on carbon AND cost simultaneously")
    print(f"  best carbon reduction vs baseline: "
          f"{headline['max_planet_reduction_pct']:.1f}%")
    print(f"  best cost reduction vs baseline:   "
          f"{headline['max_cost_reduction_pct']:.1f}%")

    payload = {
        "meta": {
            "quick": QUICK, "generations": GENERATIONS, "seeds": list(SEEDS),
            "n_scenarios": N_SCENARIOS, "runtime_s": round(time.time() - t0, 1),
            "total_mass_kg": M.TOTAL_MASS,
        },
        "E0_provenance": prov,
        "E1_baseline": base,
        "E2_optimisation": opt,
        "E3_scenarios": {
            "n_scenarios": int(sc["n_scenarios"]),
            "n_tastes": int(sc["n_tastes"]),
            "top_quantile": float(sc["top_quantile"]),
            "robustness": sc["robustness"].tolist(),
            "nondominance": sc["nondominance"].tolist(),
            "feasibility": sc["feasibility"].tolist(),
            "mean_regret": {k: v.tolist() for k, v in sc["mean_regret"].items()},
            "robust_core": int((sc["robustness"] >= 2.0 * sc["top_quantile"]).sum()),
            "robust_core_threshold": float(2.0 * sc["top_quantile"]),
            "never_top_decile": int((sc["robustness"] == 0).sum()),
            "F_mean": sc["F_mean"].tolist(),
            "F_p05": sc["F_p05"].tolist(),
            "F_p95": sc["F_p95"].tolist(),
        },
        "E4_profiles": prof,
        "E5_indicators": extra["E5_indicators"],
        "E6_allocation": extra["E6_allocation"],
        "E7_parameters": extra["E7_parameters"],
        "E8_weights": extra["E8_weights"],
        "E9_geography": extra["E9_geography"],
        "E10_algorithm": algo,
        "E11_distributional": ds,
        "E12_validation_decomposition": vd,
        "E13_electronics": extra["E13_electronics"],
        "E14_prices": extra["E14_prices"],
        "E15_spread": extra["E15_spread"],
        "archive": [{"index": i, "design": d.as_dict(),
                     "objectives": {"planet": f[0], "cost": f[1],
                                    "social": -f[2]},
                     "robustness": float(sc["robustness"][i])}
                    for i, (d, f) in enumerate(uniq)],
        "headline": headline,
    }
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote results.json   total runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
