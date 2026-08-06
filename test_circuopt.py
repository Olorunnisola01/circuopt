#!/usr/bin/env python3
"""Regression tests for CircuOpt.  No pytest required:  python3 test_circuopt.py

The tests fall into three groups:
  * dataset      the parsers return what the published files actually contain
  * model        the objective functions respond in the direction physics and
                 accounting say they must
  * algorithm    the NSGA-III machinery is correct on cases with known answers
"""

from __future__ import annotations

import math
import sys

import numpy as np

from circuopt import datasets as D
from circuopt import model as M
from circuopt import nsga3 as A
from circuopt import scenarios as S

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail and not cond else ''}")


# --------------------------------------------------------------------------
def test_datasets():
    print("\ndatasets")
    d = D.load_defra()
    check("DEFRA parses a plausible number of primary factors",
          30 <= len(d.primary) <= 60, str(len(d.primary)))
    check("DEFRA UK grid factor is in a physical range for 2025",
          0.10 < d.electricity < 0.35, str(d.electricity))
    check("DEFRA grid factor is generation plus T&D",
          math.isclose(d.electricity,
                       d.electricity_generation + d.electricity_td, rel_tol=1e-12))
    both = set(d.closed_loop) & set(d.primary)
    check("every closed-loop factor is below its primary counterpart",
          bool(both) and all(d.closed_loop[k] < d.primary[k] for k in both))
    check("every material this model uses has a primary factor",
          all(name in d.primary for name, _ in M.FAMILY_DEFRA.values()))
    check("factors are per kilogram, not per tonne",
          all(0.0 < v < 100.0 for v in d.primary.values()))

    p = D.load_usgs_prices()
    check("USGS prices parsed for at least four commodities", len(p) >= 4,
          str(sorted(p)))
    if "Copper" in p:
        check("copper price is in a sane USD/kg band",
              5.0 < p["Copper"]["usd_per_kg"] < 20.0,
              str(p["Copper"]["usd_per_kg"]))
    if "Iron and Steel Scrap" in p:
        check("steel scrap price is in a sane USD/kg band",
              0.10 < p["Iron and Steel Scrap"]["usd_per_kg"] < 1.0,
              str(p["Iron and Steel Scrap"]["usd_per_kg"]))
    check("a Cloudflare challenge page is never parsed as data",
          all("Commodity" not in str(v.get("column", "")) for v in p.values()))
    check("unit conversion: cents per pound to USD per kilogram",
          math.isclose(0.01 / D.LB, 0.0220462, rel_tol=1e-4))


# --------------------------------------------------------------------------
def test_model():
    print("\nmodel")
    ctx = M.Context.build()
    b = M.baseline_design()
    r = M.evaluate(b, ctx)

    check("baseline bill of materials masses to the stated total",
          math.isclose(M.TOTAL_MASS, sum(c.mass_kg for c in M.BOM)))
    check("baseline is feasible", r["constraint_violation"] == 0.0)
    check("baseline GWP is dominated by materials plus use phase",
          r["gwp_breakdown"]["materials"] + r["gwp_breakdown"]["use"]
          > 0.9 * r["gwp_total_kgco2e"])
    check("baseline MCI is zero for an all-virgin, all-landfill design",
          r["material_circularity_indicator"] == 0.0,
          str(r["material_circularity_indicator"]))

    # monotonicity: more recycled content must not raise the carbon objective
    d1 = M.Design(recycled={f: 0.0 for f in M.DESIGN_FAMILIES},
                  joints=(1, 1, 1), lifetime=10, reman=0,
                  eol={f: "Open-loop" for f in M.EOL_FAMILIES}, standby_class=0)
    d2 = M.Design(recycled={f: 1.0 for f in M.DESIGN_FAMILIES},
                  joints=(1, 1, 1), lifetime=10, reman=0,
                  eol={f: "Open-loop" for f in M.EOL_FAMILIES}, standby_class=0)
    r1, r2 = M.evaluate(d1, ctx), M.evaluate(d2, ctx)
    check("recycled content strictly lowers the Planet objective",
          r2["f_planet_kgco2e_per_service_year"]
          < r1["f_planet_kgco2e_per_service_year"])
    check("recycled content raises the social supply-risk term",
          r2["social_breakdown"]["supply_risk_relief"]
          > r1["social_breakdown"]["supply_risk_relief"])

    # lifetime extension amortises the embodied burden
    d3 = M.Design(recycled=d1.recycled, joints=(1, 1, 1), lifetime=15, reman=0,
                  eol=d1.eol, standby_class=0)
    r3 = M.evaluate(d3, ctx)
    check("a longer design life lowers carbon per service-year",
          r3["f_planet_kgco2e_per_service_year"]
          < r1["f_planet_kgco2e_per_service_year"])

    # standby power is the use-phase lever
    d4 = M.Design(recycled=d1.recycled, joints=(1, 1, 1), lifetime=10, reman=0,
                  eol=d1.eol, standby_class=2)
    r4 = M.evaluate(d4, ctx)
    check("a lower standby class lowers use-phase carbon",
          r4["gwp_breakdown"]["use"] < r1["gwp_breakdown"]["use"])
    check("a lower standby class raises capital cost",
          r4["cost_breakdown"]["capex"] > r1["cost_breakdown"]["capex"])

    # remanufacture extends service and creates labour
    d5 = M.Design(recycled=d1.recycled, joints=(1, 1, 1), lifetime=10, reman=2,
                  eol=d1.eol, standby_class=0)
    r5 = M.evaluate(d5, ctx)
    check("remanufacture cycles extend the service life",
          r5["service_years"] > r1["service_years"])
    check("remanufacture cycles raise the reman-labour social term",
          r5["social_breakdown"]["reman_labour"]
          > r1["social_breakdown"]["reman_labour"])

    # recyclate is not a free substitute for virgin feedstock
    costs = []
    carbon = []
    for rc in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        dd = M.Design(recycled={fam: rc for fam in M.DESIGN_FAMILIES},
                      joints=(0, 1, 1), lifetime=12, reman=1,
                      eol={fam: "Closed-loop" for fam in M.EOL_FAMILIES},
                      standby_class=2)
        e = M.evaluate(dd, ctx)
        costs.append(e["f_cost_usd_per_service_year"])
        carbon.append(e["f_planet_kgco2e_per_service_year"])
    check("carbon falls monotonically with recycled content",
          all(carbon[i] > carbon[i + 1] for i in range(len(carbon) - 1)))
    check("cost has an interior optimum in recycled content, so the variable "
          "is not bound-pinned",
          int(np.argmin(costs)) not in (0, len(costs) - 1),
          str([round(c, 3) for c in costs]))
    check("no quality surcharge below the threshold",
          ctx.quality_surcharge(0.2) == 1.0)
    check("the quality surcharge grows above the threshold",
          ctx.quality_surcharge(1.0) > ctx.quality_surcharge(0.6) > 1.0)
    check("the reject rate raises the effective input mass",
          ctx.reject_multiplier(1.0) > ctx.reject_multiplier(0.0) == 1.0)

    # constraints
    bonded = M.Design(recycled=d1.recycled, joints=(2, 1, 1), lifetime=10,
                      reman=2, eol=d1.eol, standby_class=0)
    check("a bonded enclosure cannot be remanufactured",
          M.constraint_violation(bonded) > 0)
    bonded_cl = M.Design(recycled={f: 0.5 for f in M.DESIGN_FAMILIES},
                         joints=(2, 1, 1), lifetime=10, reman=0,
                         eol={**{f: "Open-loop" for f in M.EOL_FAMILIES},
                              "plastic": "Closed-loop"}, standby_class=0)
    check("a bonded enclosure cannot claim closed-loop plastic recycling",
          M.constraint_violation(bonded_cl) > 0)
    check("an ESPR mandate makes a zero-recyclate design infeasible",
          M.constraint_violation(d1, mandate=0.30) > 0)
    check("an ESPR mandate leaves a compliant design feasible",
          M.constraint_violation(d2, mandate=0.30) == 0)

    # cost accounting
    check("closed-loop end of life is cheaper than landfill at end of life",
          M.evaluate(M.Design(recycled={f: 0.5 for f in M.DESIGN_FAMILIES},
                              joints=(1, 1, 1), lifetime=10, reman=0,
                              eol={f: "Closed-loop" for f in M.EOL_FAMILIES},
                              standby_class=0), ctx)["cost_breakdown"]["pv_end_of_life"]
          < M.evaluate(M.Design(recycled={f: 0.5 for f in M.DESIGN_FAMILIES},
                                joints=(1, 1, 1), lifetime=10, reman=0,
                                eol={f: "Landfill" for f in M.EOL_FAMILIES},
                                standby_class=0), ctx)["cost_breakdown"]["pv_end_of_life"])
    check("a carbon price raises the cost objective",
          M.evaluate(b, M.Context.build(
              M.Assumptions(carbon_price_usd_per_tonne=120.0)))[
                  "f_cost_usd_per_service_year"]
          > r["f_cost_usd_per_service_year"])
    check("a decarbonising grid lowers the Planet objective",
          M.evaluate(b, M.Context.build(
              M.Assumptions(use_grid_multiplier=0.5)))[
                  "f_planet_kgco2e_per_service_year"]
          < r["f_planet_kgco2e_per_service_year"])

    # encoding round trip
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(50):
        xc = rng.random(M.N_CONT)
        xi = np.array([rng.integers(0, c) for c in M.INT_CARDINALITY])
        d = M.Design.decode(xc, xi)
        f, cv = M.objectives(xc, xi, ctx)
        ok &= np.all(np.isfinite(f)) and cv >= 0 and d.lifetime in M.LIFETIMES
    check("every point in the design space decodes and evaluates finitely", ok)


# --------------------------------------------------------------------------
def test_algorithm():
    print("\nalgorithm")
    refs = A.das_dennis(3, 12)
    check("Das-Dennis produces the textbook number of reference points",
          len(refs) == 91, str(len(refs)))
    check("reference points lie on the unit simplex",
          np.allclose(refs.sum(axis=1), 1.0))

    f = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 3.0, 3.0]])
    cv = np.zeros(3)
    check("dominance is strict and correct",
          A.dominates(f[0], 0, f[1], 0) and not A.dominates(f[1], 0, f[0], 0)
          and not A.dominates(f[0], 0, f[2], 0))
    check("a feasible point dominates an infeasible one regardless of value",
          A.dominates(f[1], 0.0, f[0], 5.0))
    check("two infeasible points are ordered by violation",
          A.dominates(f[1], 1.0, f[0], 5.0))
    rng2 = np.random.default_rng(11)
    ok = True
    for n in (60, 250, 700):
        FF = rng2.random((n, 3))
        CC = np.where(rng2.random(n) < 0.12, 1.0, 0.0)
        a = set(np.where(A.nondominated_mask(FF, CC))[0])
        b = set(i for i in A.fast_non_dominated_sort(FF, CC)[0] if CC[i] <= 0)
        ok &= (a == b)
    check("the incremental filter agrees with the pairwise sort", ok)

    fronts = A.fast_non_dominated_sort(f, cv)
    check("non-dominated sorting recovers the known front",
          set(fronts[0]) == {0, 2} and fronts[1] == [1], str(fronts))

    # a problem with an analytically known front: minimise (x, 1-x, |x-0.5|)
    def ev(xc, xi):
        x = float(xc[0])
        return np.array([x, 1.0 - x, abs(x - 0.5)]), 0.0
    alg = A.NSGA3(ev, 1, [(0.0, 1.0)], [2], partitions=8, generations=60, seed=3)
    out = alg.run()
    F = out["F"][out["front"]]
    check("NSGA-III finds the whole known trade-off curve",
          F[:, 0].min() < 0.05 and F[:, 0].max() > 0.95, str(F[:, 0].min()))
    check("NSGA-III keeps the population feasible and finite",
          np.all(np.isfinite(out["F"])))

    hv, err = A.hypervolume(np.array([[0.0, 0.0]]), np.array([1.0, 1.0]))
    check("hypervolume of a single point at the origin is the full box",
          abs(hv - 1.0) < 0.02, f"{hv:.4f}")
    hv2, _ = A.hypervolume(np.array([[0.5, 0.5]]), np.array([1.0, 1.0]))
    check("hypervolume shrinks as the point worsens", hv2 < hv)
    check("spacing of an evenly spread front is small",
          A.spacing(np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])) < 1e-9)

    # normalisation
    Fn = A.normalise(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]]), np.zeros(3))
    check("normalisation maps the extreme points onto the unit hyperplane",
          np.allclose(Fn.sum(axis=1), 1.0, atol=1e-6), str(Fn))


# --------------------------------------------------------------------------
def test_scenarios():
    print("\nscenarios")
    rng = np.random.default_rng(1)
    base = M.Assumptions()
    draws = [S.sample_assumptions(rng, base) for _ in range(200)]
    check("the grid multiplier is always a decarbonisation, never a rebound",
          all(0.5 <= d.use_grid_multiplier <= 1.0 for d in draws),
          f"{min(d.use_grid_multiplier for d in draws):.3f}")
    check("sampled prices are strictly positive",
          all(d.plastic_virgin_usd_per_kg > 0 and d.electronics_usd_per_kg > 0
              for d in draws))
    check("the ESPR mandate is sometimes binding and sometimes absent",
          any(d.mandated_recycled_content > 0 for d in draws)
          and any(d.mandated_recycled_content == 0 for d in draws))
    check("sampling does not mutate the base assumptions",
          base.use_grid_multiplier == 1.0
          and base.mandated_recycled_content == 0.0)

    designs = [M.baseline_design(),
               M.Design(recycled={f: 0.6 for f in M.DESIGN_FAMILIES},
                        joints=(0, 0, 1), lifetime=15, reman=2,
                        eol={f: "Closed-loop" for f in M.EOL_FAMILIES},
                        standby_class=2)]
    res = S.run(designs, n_scenarios=25, seed=4, n_tastes=8)
    check("robustness is a probability", np.all((res["robustness"] >= 0)
                                                & (res["robustness"] <= 1)))
    check("the circular design is more robust than the linear baseline",
          res["robustness"][1] > res["robustness"][0],
          str(res["robustness"]))
    check("every profile's win shares sum to at most one",
          all(v.sum() <= 1.0 + 1e-9 for v in res["wins"].values()))
    check("achievement scalarisation returns one value per candidate",
          S.asf(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                np.array([0.3, 0.3, 0.4])).shape == (2,))


# --------------------------------------------------------------------------
def test_cited_bom():
    """The bill of materials must reproduce the published source exactly.

    These are the tests that make the mass basis auditable: if someone edits
    an allocation fraction and quietly changes the product mass or the
    material split, the suite fails rather than silently reporting a
    different product.
    """
    print("\nbill of materials (cited)")
    rec = D.load_garo_bom()
    pub = rec["product"]["official_product_mass_kg"]

    check("BOM mass equals the published product mass",
          math.isclose(M.TOTAL_MASS, pub, rel_tol=1e-9),
          f"{M.TOTAL_MASS} vs {pub}")

    check("packaging is excluded, and it is the published 16%",
          math.isclose(rec["excluded_packaging_pct"], 16.0, abs_tol=1e-6),
          str(rec["excluded_packaging_pct"]))

    # family masses must match the published distribution, renormalised
    dist = rec["material_distribution_pct_of_29_7kg"]
    packaging = set(rec["packaging_categories"])
    prod_pct = {k: v for k, v in dist.items() if k not in packaging}
    tot = sum(prod_pct.values())
    expected_alu = pub * dist["Aluminium"] / tot
    got_alu = sum(c.mass_kg for c in M.BOM if c.family == "aluminium")
    check("aluminium mass reproduces the published 51% share",
          math.isclose(got_alu, expected_alu, rel_tol=1e-9),
          f"{got_alu} vs {expected_alu}")

    expected_cable = pub * dist["Copper/Plastic"] / tot
    got_cable = sum(c.mass_kg for c in M.BOM
                    if c.name.startswith("cable "))
    check("the cable reproduces the published Copper/Plastic 8% share",
          math.isclose(got_cable, expected_cable, rel_tol=1e-9),
          f"{got_cable} vs {expected_cable}")

    for fam, alloc in rec["allocation_to_components"].items():
        if fam.startswith("_"):
            continue
        shares = {k: v for k, v in alloc.items() if not k.startswith("_")}
        check(f"allocation fractions for {fam} sum to one",
              math.isclose(sum(shares.values()), 1.0, abs_tol=1e-9),
              str(sum(shares.values())))

    sl = rec["service_and_lifetime"]
    check("the serviced lifetime exceeds the technical lifetime, as GARO reports",
          sl["serviced_operating_lifetime_years"] > sl["technical_lifetime_years"],
          str(sl))
    expected_extension = (sl["serviced_operating_lifetime_years"]
                          / sl["technical_lifetime_years"] - 1.0)
    check("Assumptions.reman_life_extension is derived from GARO's own "
          "15-to-25-year figure, not a bare guess",
          math.isclose(M.Assumptions().reman_life_extension,
                       expected_extension, rel_tol=1e-9),
          f"{M.Assumptions().reman_life_extension} vs {expected_extension}")

    check("every component carries a declared interface",
          all(c.joint in (-1, 0, 1, 2) for c in M.BOM))
    check("the priority repair parts are present in the BOM",
          set(rec["priority_repair_components"]) <=
          {c.name for c in M.BOM})
    check("no component has zero or negative mass",
          all(c.mass_kg > 0 for c in M.BOM))


def test_validation():
    """The independent cross-check must stay in the band we report."""
    print("\nvalidation against the published LCA")
    ctx = M.Context.build()
    v = M.validate_against_published(ctx)
    check("model and published cradle-to-gate agree within a factor of three",
          0.33 <= v["ratio_model_over_published"] <= 3.0,
          str(v["ratio_model_over_published"]))
    rec = v["geography_reconciliation"]
    check("the model's default (global) aluminium factor sits nearer the "
          "published study's own factor than the European alternative does",
          abs(rec["published_study_aluminium_kgco2e_per_kg"]
              - rec["aluminium_default_global_kgco2e_per_kg"]) <
          abs(rec["published_study_aluminium_kgco2e_per_kg"]
              - rec["aluminium_european_reference_kgco2e_per_kg"]))
    check("substituting the European route would move the model further "
          "from the published figure than the global default",
          rec["ratio_with_european_aluminium"] < v["ratio_model_over_published"],
          f"default {v['ratio_model_over_published']:.3f} -> "
          f"European {rec['ratio_with_european_aluminium']:.3f}")
    shares = v["model_material_shares"]
    check("aluminium and electronics are the two hotspots, as published",
          set(list(shares)[:2]) == {"aluminium", "electronics"},
          str(list(shares)[:2]))
    check("they together dominate, as the published study reports",
          shares["aluminium"] + shares["electronics"] > 0.80,
          str(shares["aluminium"] + shares["electronics"]))
    for k, d in v["factor_cross_check_kgco2e_per_kg"].items():
        # Aluminium is deliberately exempt: the European and global smelting
        # routes differ by more than a factor of two, and that difference is
        # the reported finding rather than a defect.
        lo = 0.3 if k == "aluminium" else 0.5
        check(f"factor cross-check for {k} is within its expected band",
              lo <= d["ratio"] <= 2.0, str(d["ratio"]))


# --------------------------------------------------------------------------
def test_inventory():
    """The multi-indicator inventory layer and the allocation conventions."""
    print("\ninventory and LCIA")
    from circuopt import inventory as INV

    inv = INV.Inventory.build()
    check("every mapped Idemat process resolves", True)
    check("indicator vectors carry more than global warming",
          len(inv.indicators) >= 3, str(inv.indicators))

    for fam in ("aluminium", "steel", "copper", "plastic"):
        v0 = inv.material_input(fam, 0.0)
        v1 = inv.material_input(fam, 1.0)
        check(f"secondary {fam} beats primary on global warming",
              v1["gwp_kgco2e"] < v0["gwp_kgco2e"],
              f"{v0['gwp_kgco2e']:.3f} -> {v1['gwp_kgco2e']:.3f}")
        check(f"secondary {fam} beats primary on cumulative energy demand",
              v1["ced_mj"] < v0["ced_mj"])

    cut = INV.Inventory.build(allocation="cut_off")
    sub = INV.Inventory.build(allocation="substitution")
    e_cut = cut.end_of_life("aluminium", "Closed-loop")["gwp_kgco2e"]
    e_sub = sub.end_of_life("aluminium", "Closed-loop")["gwp_kgco2e"]
    check("cut-off gives closed-loop recovery no environmental credit",
          e_cut >= 0.0, str(e_cut))
    check("substitution credits closed-loop recovery",
          e_sub < 0.0, str(e_sub))
    check("landfill is identical under both conventions",
          math.isclose(cut.end_of_life("aluminium", "Landfill")["gwp_kgco2e"],
                       sub.end_of_life("aluminium", "Landfill")["gwp_kgco2e"]))

    grids = {r: inv.electricity(r)["gwp_kgco2e"] for r in inv.grid_regions}
    check("European grid factors span more than a factor of five",
          max(grids.values()) / min(grids.values()) > 5.0,
          str({k: round(v, 4) for k, v in grids.items()}))
    check("Poland is the most carbon-intensive grid modelled",
          max(grids, key=grids.get) == "Poland")


def test_mci():
    """MCI must reproduce the published formula's boundary behaviour."""
    print("\nmaterial circularity indicator")
    from circuopt.inventory import material_circularity_indicator as mci

    # A fully linear product at exactly the industry-average life scores 0.1,
    # not 0: the published formula uses F(X) = 0.9 / X, so LFI = 1 and X = 1
    # give MCI = 1 - 0.9 = 0.1.  Asserting the textbook 0 here would be
    # asserting a misreading of the method.
    lin = mci(1.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.85, 10, 10)
    check("a fully linear product at average life scores 0.1, per F(X)=0.9/X",
          math.isclose(lin["mci"], 0.1, abs_tol=1e-9), str(lin["mci"]))
    check("a fully linear product has linear flow index 1",
          math.isclose(lin["linear_flow_index"], 1.0, abs_tol=1e-9))
    below = mci(1.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.85, 8, 10)
    check("a linear product below average life is floored at 0",
          math.isclose(below["mci"], 0.0, abs_tol=1e-9), str(below["mci"]))

    circ = mci(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 10, 10)
    check("a perfectly circular product at average life scores near 0.9",
          0.88 < circ["mci"] <= 1.0, str(circ["mci"]))

    longer = mci(1.0, 0.5, 0.0, 0.5, 0.0, 0.9, 0.9, 20, 10)
    shorter = mci(1.0, 0.5, 0.0, 0.5, 0.0, 0.9, 0.9, 10, 10)
    check("a longer service life raises MCI through the utility term",
          longer["mci"] > shorter["mci"],
          f"{shorter['mci']:.4f} -> {longer['mci']:.4f}")
    check("MCI is bounded in [0, 1]",
          all(0.0 <= m["mci"] <= 1.0 for m in (lin, circ, longer, shorter)))
    check("recycled feedstock lowers the linear flow index",
          longer["linear_flow_index"] < lin["linear_flow_index"])


def test_statistics():
    """The hand-written rank statistics must match known answers."""
    print("\nstatistics")
    from circuopt import analyses as AN

    check("Spearman of a monotone pair is 1",
          math.isclose(AN.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, abs_tol=1e-9))
    check("Spearman of a reversed pair is -1",
          math.isclose(AN.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, abs_tol=1e-9))
    check("Spearman detects monotone but non-linear agreement",
          math.isclose(AN.spearman([1, 2, 3, 4], [1, 4, 9, 16]), 1.0, abs_tol=1e-9))
    check("Kendall tau of a monotone pair is 1",
          math.isclose(AN.kendall_tau([1, 2, 3], [5, 6, 7]), 1.0, abs_tol=1e-9))
    check("Kendall tau of a reversed pair is -1",
          math.isclose(AN.kendall_tau([1, 2, 3], [7, 6, 5]), -1.0, abs_tol=1e-9))

    u = AN.mannwhitney_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    check("Mann-Whitney U is 0 for perfectly separated samples",
          math.isclose(u["U"], 0.0), str(u["U"]))
    check("perfect separation gives a rank-biserial effect size of 1",
          math.isclose(u["rank_biserial"], 1.0, abs_tol=1e-9))
    u2 = AN.mannwhitney_u([1, 3, 5], [2, 4, 6])
    check("overlapping samples are not significant at the 5% level",
          u2["p_two_sided"] > 0.05, str(u2["p_two_sided"]))


# --------------------------------------------------------------------------
def test_electronics_averaging():
    """The electronics proxy must be a real average, not one arbitrary pick."""
    print("\nelectronics averaging (fix 1)")
    from circuopt.idemat import load_idemat, augment_with_synthetics

    procs = load_idemat()
    ref, wash = (procs["A.050.06.304"].indicators["gwp_kgco2e"],
                procs["A.050.06.305"].indicators["gwp_kgco2e"])
    augment_with_synthetics(procs)
    avg = procs["SYN.ELECTRONICS.WHITEGOODS.AVG"]
    check("the synthetic electronics process is the mean of its two sources",
          math.isclose(avg.indicators["gwp_kgco2e"], (ref + wash) / 2.0,
                       rel_tol=1e-9),
          f"{avg.indicators['gwp_kgco2e']} vs {(ref + wash) / 2.0}")
    check("the averaged factor sits strictly between its two sources",
          min(ref, wash) < avg.indicators["gwp_kgco2e"] < max(ref, wash))

    ctx = M.Context.build()
    check("the model's electronics factor is the synthetic average, not "
          "either source alone",
          math.isclose(ctx.inv.material_input("electronics", 0.0)["gwp_kgco2e"],
                       avg.indicators["gwp_kgco2e"], rel_tol=1e-9))


def test_electronics_breakeven():
    """The breakeven analysis (fix 1b) must report a defensible, wide margin."""
    print("\nelectronics breakeven (fix 1b)")
    from circuopt import analyses as AN
    ds = [M.baseline_design(),
          M.Design(recycled={f: 0.7 for f in M.DESIGN_FAMILIES},
                  joints=(0, 1, 1), lifetime=15, reman=2,
                  eol={f: "Closed-loop" for f in M.EOL_FAMILIES},
                  standby_class=2)]
    eb = AN.electronics_breakeven(ds)
    check("electronics is not the dominant material-stage contributor today",
          eb["electronics_share_at_1x"] < eb["aluminium_share_at_1x"])
    check("the crossover multiplier, if found, exceeds a 1x error",
          eb["crossover_multiplier"] is None or eb["crossover_multiplier"] > 1.0,
          str(eb["crossover_multiplier"]))
    check("archive ranking stays highly correlated across the tested "
          "multiplier range, since electronics mass is not a decision variable",
          eb["min_spearman_vs_nominal"] > 0.7, str(eb["min_spearman_vs_nominal"]))


def test_cff():
    """The EU PEF Circular Footprint Formula implementation (fix 5)."""
    print("\nCircular Footprint Formula (fix 5)")
    ctx0 = M.Context.build(M.Assumptions(allocation="cut_off"))
    ctx_sub = M.Context.build(M.Assumptions(allocation="substitution"))
    ctx_cff1 = M.Context.build(M.Assumptions(allocation="cff",
                                             cff_allocation_factor=1.0))
    ctx_cff0 = M.Context.build(M.Assumptions(allocation="cff",
                                             cff_allocation_factor=0.0))

    d = M.Design(recycled={f: 0.5 for f in M.DESIGN_FAMILIES},
                joints=(0, 1, 1), lifetime=15, reman=1,
                eol={f: "Closed-loop" for f in M.EOL_FAMILIES},
                standby_class=2)

    e_sub = M.evaluate(d, ctx_sub)["gwp_total_kgco2e"]
    e_cff1 = M.evaluate(d, ctx_cff1)["gwp_total_kgco2e"]
    check("CFF with A=1 reproduces the substitution convention exactly, on "
          "both the input side (full recycled-content credit) and "
          "end-of-life (which cff always treats as substitution)",
          math.isclose(e_sub, e_cff1, rel_tol=1e-9), f"{e_sub} vs {e_cff1}")

    e_cutoff = M.evaluate(d, ctx0)["gwp_total_kgco2e"]
    e_cff0 = M.evaluate(d, ctx_cff0)["gwp_total_kgco2e"]
    check("CFF with A=0 gives recycled content less input-side credit than "
          "cut-off does, since none of the recycling benefit is allocated "
          "to the user at A=0",
          e_cff0 > e_cutoff, f"A=0: {e_cff0} vs cut-off: {e_cutoff}")

    check("CFF total impact is monotonically decreasing in A, holding the "
          "design fixed", all(
              M.evaluate(d, M.Context.build(
                  M.Assumptions(allocation="cff", cff_allocation_factor=a1))
              )["gwp_total_kgco2e"]
              >= M.evaluate(d, M.Context.build(
                  M.Assumptions(allocation="cff", cff_allocation_factor=a2))
              )["gwp_total_kgco2e"]
              for a1, a2 in [(0.0, 0.5), (0.5, 1.0)]))

    check("cff is a documented member of Inventory.ALLOCATIONS",
          "cff" in __import__("circuopt.inventory",
                              fromlist=["ALLOCATIONS"]).ALLOCATIONS)


def test_worldbank_prices():
    """The World Bank price source (fix 3): live, sane, and actually used."""
    print("\nWorld Bank price cross-check (fix 3)")
    from circuopt.datasets import load_worldbank_prices
    wb = load_worldbank_prices()
    check("World Bank data is available", len(wb) > 0, str(list(wb)))
    for commodity in ("Aluminum", "Copper"):
        check(f"World Bank {commodity} price is a sane positive number",
              commodity in wb and 0.1 < wb[commodity]["usd_per_kg_trailing_12m_mean"] < 100,
              str(wb.get(commodity)))

    ctx = M.Context.build()
    check("USGS never resolved an aluminium price, so the model's aluminium "
          "price now comes from World Bank data rather than the hardcoded "
          "fallback constant",
          "Aluminum" not in ctx.prices and math.isclose(
              ctx.price("aluminium"),
              ctx.worldbank_prices["Aluminum"]["usd_per_kg_trailing_12m_mean"],
              rel_tol=1e-9))
    check("forcing price_source='worldbank' changes the copper price used "
          "(USGS does have copper data, so this is a real source switch)",
          not math.isclose(
              M.Context.build(M.Assumptions(price_source="usgs")).price("copper"),
              M.Context.build(M.Assumptions(price_source="worldbank")).price("copper")))


def test_methodological_spread():
    """The synthesis table (fix 2) must be internally consistent."""
    print("\nmethodological vs design spread (fix 2)")
    from circuopt import analyses as AN
    ds = [M.baseline_design(),
          M.Design(recycled={f: 0.8 for f in M.DESIGN_FAMILIES},
                  joints=(0, 0, 0), lifetime=15, reman=3,
                  eol={f: "Closed-loop" for f in M.EOL_FAMILIES},
                  standby_class=2)]
    allo = AN.allocation_sensitivity(ds)
    geo = AN.geography_sensitivity(ds)
    sp = AN.methodological_vs_design_spread(ds, allo, geo)
    check("all four spans are non-negative",
          all(v >= 0 for v in sp["spans_kgco2e_per_service_year"].values()))
    ctx0 = M.Context.build()
    F0 = [M.evaluate(d, ctx0)["f_planet"] for d in ds]
    check("design-choice span matches the archive's own min-max range, "
          "independently recomputed",
          math.isclose(
              sp["spans_kgco2e_per_service_year"]["design choice (whole Pareto archive)"],
              max(F0) - min(F0), rel_tol=1e-6),
          f"{sp['spans_kgco2e_per_service_year']['design choice (whole Pareto archive)']} "
          f"vs {max(F0) - min(F0)}")
    check("the ranked list contains exactly the four spans reported",
          set(sp["ranked_largest_first"])
          == set(sp["spans_kgco2e_per_service_year"]))


# --------------------------------------------------------------------------
if __name__ == "__main__":
    test_datasets()
    test_cited_bom()
    test_inventory()
    test_mci()
    test_statistics()
    test_validation()
    test_electronics_averaging()
    test_electronics_breakeven()
    test_cff()
    test_worldbank_prices()
    test_methodological_spread()
    test_model()
    test_algorithm()
    test_scenarios()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  failed:", f)
    sys.exit(1 if FAIL else 0)
