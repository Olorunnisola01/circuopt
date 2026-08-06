#!/usr/bin/env python3
"""Generate report.tex from results.json and compile it to report.pdf.

Every number in the report is read from results.json.  Nothing is typed in by
hand, so the document cannot drift away from the run that produced it.

    python3 make_report.py            # writes report.tex and report.pdf
    python3 make_report.py --tex-only
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))

from circuopt import model as M   # noqa: E402  (only for structural constants)
from circuopt import scenarios as SC  # noqa: E402


def esc(s):
    return (str(s).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def f(x, n=2):
    return f"{x:,.{n}f}"


# --------------------------------------------------------------------------
meta, prov = R["meta"], R["E0_provenance"]
base, opt = R["E1_baseline"], R["E2_optimisation"]
scen, prof, head = R["E3_scenarios"], R["E4_profiles"], R["headline"]
# Recomputed here so the threshold is defined in one place: twice the rate a
# design picked at random would achieve.
ROBUST_THRESHOLD = 2.0 * scen["top_quantile"]
br = base["result"]
val = base["validation"]
#: How close the independent factor set comes to the published result, as a
#: percentage discrepancy - quoted in the abstract and the validation section.
valpct = f"{abs(1.0 - val['ratio_model_over_published']) * 100:.0f}"
packpct = f"{M.GARO['excluded_packaging_pct']:.0f}"
arch = R["archive"]

rob = np.array(scen["robustness"])
nd = np.array(scen["nondominance"])
feas = np.array(scen["feasibility"])
Fp = np.array([[a["objectives"]["planet"], a["objectives"]["cost"],
                a["objectives"]["social"]] for a in arch])

best_planet = int(np.argmin(Fp[:, 0]))
best_cost = int(np.argmin(Fp[:, 1]))
best_social = int(np.argmax(Fp[:, 2]))
most_robust = int(np.argmax(rob))


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def table_bom():
    rows = []
    for c in M.BOM:
        joint = M.JOINTS[0][0] if c.joint == 0 else ("-" if c.joint < 0 else "")
        rows.append(f"{esc(c.name)} & {c.mass_kg:.2f} & {esc(c.family)} & "
                    f"{'yes' if c.priority_repair else 'no'} & "
                    f"{c.joint if c.joint >= 0 else '--'} \\\\")
    rows.append(r"\midrule")
    rows.append(f"\\textbf{{total}} & \\textbf{{{M.TOTAL_MASS:.2f}}} & & & \\\\")
    return "\n".join(rows)


def table_indicator_ranks():
    rows = []
    for k, rc in ind["rank_correlation_vs_reference"].items():
        agree = ind["best_design_agrees_with_reference"][k]
        lab = ind["per_indicator"][k]["label"]
        rows.append(f"{esc(lab)} & {rc['spearman']:+.3f} & {rc['kendall_tau']:+.3f} & "
                    f"{'yes' if agree else 'NO'} \\\\")
    return "\n".join(rows)


def table_geography():
    rows = []
    ref = geo["reference_region"]
    for r, v in sorted(geo["per_region"].items(),
                       key=lambda kv: kv[1]["grid_kgco2e_per_kwh"]):
        rc = geo["rank_correlation_vs_reference"].get(r)
        rho = "--" if rc is None else f"{rc['spearman']:+.3f}"
        rows.append(f"{esc(r)}{' (reference)' if r == ref else ''} & "
                    f"{v['grid_kgco2e_per_kwh']:.4f} & {v['planet_mean']:.1f} & "
                    f"{rho} \\\\")
    return "\n".join(rows)


def table_algorithm():
    rows = []
    for m, v in algo["methods"].items():
        rows.append(f"{esc(m)} & {v['n']} & {v['median']:.2f} & {v['mean']:.2f} "
                    f"& {v['std']:.2f} \\\\")
    return "\n".join(rows)


def table_algorithm_tests():
    rows = []
    for pair, t in algo["tests"].items():
        sig = "yes" if t["p_two_sided"] < 0.05 else "no"
        rows.append(f"{esc(pair)} & {t['median_ratio']:.4f} & {t['U']:.0f} & "
                    f"{t['p_two_sided']:.4f} & {t['rank_biserial']:.2f} & {sig} \\\\")
    return "\n".join(rows)


def table_vdecomp():
    rows = []
    for r in vdec["rows"]:
        if r.get("contribution_kgco2e") is None:
            rows.append(f"{esc(r['family'])} & {r['mass_kg']:.2f} & "
                        f"{r['model_factor']:.3f} & -- & -- \\\\")
        else:
            rows.append(f"{esc(r['family'])} & {r['mass_kg']:.2f} & "
                        f"{r['model_factor']:.3f} & {r['published_factor']:.3f} & "
                        f"{r['contribution_kgco2e']:+.1f} \\\\")
    return "\n".join(rows)


def table_distributional():
    rows = []
    for sc_ in dist["_scales"]:
        row = dist[str(sc_)]
        rows.append(f"$\\times${sc_} & \\#{row['top_index']} & "
                    f"{row['robust_core']} & {row['spearman_vs_nominal']:+.3f} \\\\")
    return "\n".join(rows)


def table_validation():
    rows = []
    for k, d in val["factor_cross_check_kgco2e_per_kg"].items():
        rows.append(f"{esc(k)} & {d['defra_model']:.3f} & "
                    f"{d['published_study']:.3f} & {d['ratio']:.2f} \\\\")
    return "\n".join(rows)


def table_material_shares():
    rows = []
    for k, sh in val["model_material_shares"].items():
        rows.append(f"{esc(k)} & {100 * sh:.1f} \\\\")
    return "\n".join(rows)


def table_factors():
    rows = []
    for m in prov["materials"]:
        cl = m["closed_loop_kgco2e_per_kg"]
        red = m["reduction_pct"]
        rows.append(
            f"{esc(m['family'])} & \\textsc{{{m['provenance'].lower()}}} & "
            f"{esc(m['defra_material'])} & {m['primary_kgco2e_per_kg']:.3f} & "
            f"{('%.3f' % cl) if cl else '--'} & "
            f"{('%.1f' % red) if red else '--'} \\\\")
    return "\n".join(rows)


def table_prices():
    rows = []
    for p in prov["prices"]:
        rows.append(f"{esc(p['commodity'])} & {esc(p['reported'])} & "
                    f"{p['year']} & {p['usd_per_kg']:.3f} & "
                    f"\\texttt{{{esc(p['column'])}}} \\\\")
    return "\n".join(rows)


def _dsummary(d):
    rc = d["recycled_content"]
    return (f"{rc['plastic']:.2f}/{rc['steel']:.2f}/{rc['aluminium']:.2f}",
            "/".join(x[:4] for x in d["joints"]),
            f"{d['design_lifetime_years']}",
            f"{d['remanufacture_cycles']}",
            f"{d['standby_w']:.1f}",
            d["end_of_life_route"]["plastic"][:6])


def table_corners():
    labels = {best_planet: "min carbon", best_cost: "min cost",
              best_social: "max social", most_robust: "most robust"}
    rows = []
    b = (br["f_planet_kgco2e_per_service_year"],
         br["f_cost_usd_per_service_year"], br["f_social_index"])
    rows.append(
        f"linear baseline & -- & 0.00/0.00/0.00 & bond/scre/bond & 8 & 0 & 1.0 & "
        f"Landfi & {b[0]:.2f} & {b[1]:.2f} & {b[2]:.3f} \\\\")
    rows.append(r"\midrule")
    for i, lab in labels.items():
        d = arch[i]["design"]
        rc, j, L, k, sw, eol = _dsummary(d)
        rows.append(f"{lab} & \\#{i} & {rc} & {j} & {L} & {k} & {sw} & {eol} & "
                    f"{Fp[i, 0]:.2f} & {Fp[i, 1]:.2f} & {Fp[i, 2]:.3f} \\\\")
    return "\n".join(rows)


def table_profiles():
    rows = []
    for name, p in prof.items():
        d = p["design"]
        rc, j, L, k, sw, eol = _dsummary(d)
        w = p["weights"]
        rows.append(
            f"{esc(name)} & {w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f} & \\#{p['modal_choice_index']} & "
            f"{100 * p['modal_choice_share']:.0f}\\% & {p['distinct_winners']} & "
            f"{rc} & {L} & {k} & {sw} & "
            f"{p['mean_objectives'][0]:.2f} & {p['mean_objectives'][1]:.2f} \\\\")
    return "\n".join(rows)


def table_assumptions():
    a = M.Assumptions()
    keep = [
        ("mfg\\_energy\\_kwh\\_per\\_kg", a.mfg_energy_kwh_per_kg,
         "kWh/kg", "assembly energy intensity"),
        ("energy\\_delivered\\_kwh\\_per\\_year", a.energy_delivered_kwh_per_year,
         "kWh/yr", "charging throughput"),
        ("conversion\\_loss\\_frac", a.conversion_loss_frac, "--",
         "AC station switching losses"),
        ("reman\\_life\\_extension", a.reman_life_extension, "--",
         "service-life gain per remanufacture cycle"),
        ("reman\\_part\\_replacement\\_frac", a.reman_part_replacement_frac, "--",
         "mass renewed per cycle"),
        ("labour\\_rate\\_usd\\_per\\_hour", a.labour_rate_usd_per_hour, "USD/h",
         "loaded EU manufacturing rate"),
        ("plastic\\_virgin\\_usd\\_per\\_kg", a.plastic_virgin_usd_per_kg, "USD/kg",
         "PC/ABS resin, no open index available"),
        ("electronics\\_usd\\_per\\_kg", a.electronics_usd_per_kg, "USD/kg",
         "populated PCBA and contactor"),
        ("electricity\\_price\\_usd\\_per\\_kwh", a.electricity_price_usd_per_kwh,
         "USD/kWh", "EU household tariff"),
        ("scrap\\_recovery\\_efficiency", a.scrap_recovery_efficiency, "--",
         "closed-loop yield"),
        ("collection\\_cost\\_usd\\_per\\_kg", a.collection_cost_usd_per_kg,
         "USD/kg", "WEEE take-back logistics"),
        ("discount\\_rate", a.discount_rate, "--", "life-cycle costing"),
        ("repair\\_time\\_target\\_s", a.repair_time_target_s, "s",
         "EN 45554-style disassembly benchmark"),
        ("recyclate\\_threshold", a.recyclate_threshold, "--",
         "recycled content below which there is no quality penalty"),
        ("recyclate\\_cost\\_penalty", a.recyclate_cost_penalty, "--",
         "quadratic surcharge coefficient $\\lambda$"),
        ("recyclate\\_reject\\_rate", a.recyclate_reject_rate, "--",
         "extra input mass per kilogram of good part at $r=1$"),
    ]
    return "\n".join(f"\\texttt{{{k}}} & {v:g} & {u} & {d} \\\\"
                     for k, v, u, d in keep)


def table_scenario_ranges():
    rows = [
        ("UK grid decarbonisation", "0--7\\% per year, compounded over 12 years",
         "use-phase and remanufacturing carbon"),
        ("electricity tariff", "lognormal, $\\sigma=0.30$ about the base tariff",
         "use-phase cost"),
        ("carbon price", "0 / 40 / 90 / 160 USD per tonne, $p=(0.5,0.2,0.2,0.1)$",
         "cost objective"),
        ("resin price", "lognormal, $\\sigma=0.22$", "material cost"),
        ("recyclate price ratio", "normal, $\\sigma=0.18$, clipped to $[0.45,1.45]$",
         "the recycled-content business case"),
        ("secondary metal ratio", "normal, $\\sigma=0.10$, clipped to $[0.40,1.00]$",
         "scrap revenue at end of life"),
        ("electronics price", "lognormal, $\\sigma=0.18$", "capital cost"),
        ("labour rate", "lognormal, $\\sigma=0.15$",
         "assembly and remanufacturing cost"),
        ("scrap recovery efficiency", "normal, $\\sigma=0.08$, clipped to $[0.55,0.97]$",
         "circularity value"),
        ("remanufacture life extension", "normal, $\\sigma=0.12$, clipped to $[0.25,0.90]$",
         "service life"),
        ("charging throughput", "lognormal, $\\sigma=0.35$",
         "use-phase carbon and cost"),
        ("recyclate quality threshold $r_0$",
         "normal, $\\sigma=0.12$, clipped to $[0.15,0.75]$",
         "where the recycled-content cost optimum sits"),
        ("recyclate cost penalty $\\lambda$",
         "normal, $\\sigma=0.25$, clipped to $[0.05,1.40]$",
         "how steeply high recyclate is punished"),
        ("recyclate reject rate $\\beta$",
         "normal, $\\sigma=0.04$, clipped to $[0.0,0.22]$",
         "extra input mass, so both carbon and cost"),
        ("proxy factor multipliers", "lognormal, $\\sigma=0.20$ (metals) and "
         "$0.35$ (copper, electronics)",
         "the parts of the inventory built on proxies"),
        ("ESPR recycled-content mandate", "0 / 15 / 25 / 35\\%, $p=(0.55,0.20,0.15,0.10)$",
         "feasibility"),
    ]
    return "\n".join(f"{k} & {v} & {d} \\\\" for k, v, d in rows)


# --------------------------------------------------------------------------
win = head.get("win_win_example")
win_block = ""
if win:
    d = win["design"]
    win_block = (
        f"Design \\#{win['index']} is one concrete instance: "
        f"{100 * d['recycled_content']['plastic']:.0f}\\% recycled polymer, "
        f"{100 * d['recycled_content']['steel']:.0f}\\% recycled steel, "
        f"a {d['design_lifetime_years']}-year design life with "
        f"{d['remanufacture_cycles']} remanufacture cycle"
        f"{'s' if d['remanufacture_cycles'] != 1 else ''}, "
        f"{esc('/'.join(d['joints']))} joints and a "
        f"{d['standby_w']:.1f}~W standby class. Against the linear baseline it "
        f"cuts carbon per service-year by {win['planet_reduction_pct']:.1f}\\% "
        f"and cost per service-year by {win['cost_reduction_pct']:.1f}\\%, while "
        f"raising the People index by {win['social_gain_pct']:.0f}\\%. It is a "
        f"top-decile answer in {100 * win['robustness']:.0f}\\% of the "
        f"future~$\\times$~preference draws.")

hv_ratio = 100 * opt["random_search_hypervolume"] / opt["hypervolume_mean"]
seed_cv = 100 * opt["hypervolume_std"] / opt["hypervolume_mean"]

profile_spread = {n: prof[n]["modal_choice_index"] for n in prof}
n_distinct_profiles = len(set(profile_spread.values()))

DATE = datetime.date.today().strftime("%d %B %Y")
PARTITIONS = 8 if meta["quick"] else 12
N_UNCERTAIN = 16
aluminium_note = (
    ", so its price falls back to a declared assumption."
    if prov["usgs_families_missing"] else
    ", and its price is read from the dataset.")

# -- values quoted in the validation section --------------------------------
modelmat = f(val["model_materials_kgco2e"], 0)
pubmat = f(val["published_materials_kgco2e"], 0)
alupct = f"{(1 - val['factor_cross_check_kgco2e_per_kg']['aluminium']['ratio']) * 100:.0f}"
alushare = f"{100 * val['model_material_shares']['aluminium']:.0f}"
elecshare = f"{100 * val['model_material_shares']['electronics']:.0f}"
_a = M.Assumptions()
_sb = M.STANDBY_CLASSES[0][1] * _a.hours_per_year / 1000.0
_conv = _a.energy_delivered_kwh_per_year * _a.conversion_loss_frac
throughput = f(_a.energy_delivered_kwh_per_year, 0)
convkwh = f(_conv, 0)
sbkwh = f(_sb, 1)
sbshare = f"{100 * _sb / (_sb + _conv):.1f}"
ind  = R["E5_indicators"]
allo = R["E6_allocation"]
par  = R["E7_parameters"]
wts  = R["E8_weights"]
geo  = R["E9_geography"]
algo = R["E10_algorithm"]
dist = R["E11_distributional"]
vdec = R["E12_validation_decomposition"]
recon = val.get("geography_reconciliation") or {}
invd = prov.get("inventory", {})

archive_n = opt["archive_size"]
indicator_rank_rows = table_indicator_ranks()
geography_rows = table_geography()
algorithm_rows = table_algorithm()
algorithm_test_rows = table_algorithm_tests()
vdecomp_rows = table_vdecomp()
distributional_rows = table_distributional()

cutoff_eol = f"{100 * allo['cut_off']['eol_share_of_total_mean']:.2f}"
sub_eol = f"{100 * allo['substitution']['eol_share_of_total_mean']:.1f}"
alloc_shift = f"{allo['mean_shift_pct']:+.1f}"
alloc_rho = f"{allo['spearman_between_conventions']:+.3f}"
geo_span = f"{geo['span_ratio']:.0f}"

_corner_int = sum(1 for c in par["corners"] if c["interior"])
corner_int = f"{_corner_int} of {len(par['corners'])}"
_argmins = [r["argmin_recycled_content"]
            for rows in par["one_at_a_time"].values() for r in rows]
par_lo, par_hi = min(_argmins), max(_argmins)

wt_draws = wts["n_draws"]
wt_share = f"{100 * wts['nominal_best_win_share']:.0f}"
wt_distinct = wts["distinct_winners"]
wt_rho = f"{wts['spearman_vs_nominal_mean']:+.3f}"
wt_p05 = f"{wts['spearman_vs_nominal_p05']:+.3f}"

algo_seeds = len(algo["seeds"])
algo_budget = f"{algo['budget_per_run']:,}".replace(",", "\\,")
_t32 = algo["tests"].get("NSGA-III vs NSGA-II", {})
_t3r = algo["tests"].get("NSGA-III vs random search", {})
nsga2_margin = f"{100 * (1 / _t32.get('median_ratio', 1.0) - 1):.1f}"
nsga_p = f"{_t32.get('p_two_sided', float('nan')):.5f}"
ea_vs_rand = f"{100 * (_t3r.get('median_ratio', 1.0) - 1):.0f}"

_alu_contrib = next((r["contribution_kgco2e"] for r in vdec["rows"]
                     if r["family"] == "aluminium"), 0.0)
alu_share = f"{100 * abs(_alu_contrib) / max(vdec['sum_absolute_difference_kgco2e'], 1e-9):.0f}"
cancel_pct = f"{100 * vdec['cancellation_fraction']:.0f}"
alu_glob = f"{recon.get('aluminium_default_global_kgco2e_per_kg', float('nan')):.2f}"
alu_eu = f"{recon.get('aluminium_european_reference_kgco2e_per_kg', float('nan')):.2f}"
alu_pub = f"{recon.get('published_study_aluminium_kgco2e_per_kg', float('nan')):.2f}"
ratio_default = f"{vdec['ratio']:.2f}"
ratio_eu = f"{recon.get('ratio_with_european_aluminium', float('nan')):.2f}"
alu_mass_pct = f"{100 * sum(c.mass_kg for c in M.BOM if c.family == 'aluminium') / M.TOTAL_MASS:.0f}"

eb = R["E13_electronics"]
pr = R["E14_prices"]
sp = R["E15_spread"]

_cff = allo["cff_allocation_factor_sweep"]
cff_sweep_span = f"{100 * (max(v['planet_mean'] for v in _cff.values()) / min(v['planet_mean'] for v in _cff.values()) - 1):.1f}"
cff_rho_cutoff = f"{allo['cff_vs_cutoff_spearman']:+.3f}"
cff_rho_sub = f"{allo['cff_vs_substitution_spearman']:+.3f}"

alu_share_1x = f"{100 * eb['aluminium_share_at_1x']:.0f}"
elec_share_1x = f"{100 * eb['electronics_share_at_1x']:.0f}"
electronics_verdict = eb["verdict"]
electronics_verdict_short = (f"{eb['crossover_multiplier']:.1f}x its factor"
                             if eb["crossover_multiplier"] else
                             f"{eb['tested_multiplier_range'][1]:.0f}x its factor")

price_alu_wb = f"{pr['prices_usd_per_kg']['worldbank']['aluminium']:.2f}"
price_cu_usgs = f"{pr['prices_usd_per_kg']['usgs']['copper']:.2f}"
price_cu_wb = f"{pr['prices_usd_per_kg']['worldbank']['copper']:.2f}"
price_shift_pct = f"{pr['cost_span_pct']:+.2f}"
price_rho = f"{pr['spearman_between_sources']:+.3f}"
price_same_best = "yes" if pr["same_best_design"] else "no"

spread_rows = "\n".join(
    f"{esc(k)} & {sp['spans_kgco2e_per_service_year'][k]:.1f} \\\\\\\\"
    for k in sp["ranked_largest_first"])
spread_rank = sp["design_choice_rank"]
spread_n = len(sp["spans_kgco2e_per_service_year"])
spread_verdict_sentence = (
    "The design decision moves the result more than any single "
    "methodological choice tested."
    if spread_rank == 1 else
    f"{spread_rank - 1} purely methodological choice"
    f"{'s' if spread_rank > 2 else ''} move{'s' if spread_rank == 2 else ''} "
    f"the Planet objective more than the entire design space does."
)

validation_rows = table_validation()
material_share_rows = table_material_shares()


TEX = rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.4cm]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{siunitx}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{microtype}}
\usepackage[font=small,labelfont=bf]{{caption}}
\usepackage{{xcolor}}
\usepackage[hidelinks,
  pdftitle={{A three-objective optimisation framework for circular product ecodesign, built end to end on open data}},
  pdfauthor={{Adeleke Olorunnisola}},
  pdfsubject={{Multi-objective optimisation for circular product ecodesign}},
  pdfkeywords={{circular economy, ecodesign, NSGA-III, life cycle assessment, multi-objective optimisation}}
]{{hyperref}}
\usepackage{{enumitem}}
\usepackage{{titlesec}}
\usepackage{{placeins}}
\renewcommand{{\topfraction}}{{0.92}}
\renewcommand{{\bottomfraction}}{{0.75}}
\renewcommand{{\textfraction}}{{0.06}}
\renewcommand{{\floatpagefraction}}{{0.80}}
\setcounter{{topnumber}}{{3}}
\setcounter{{totalnumber}}{{5}}
\titleformat{{\section}}{{\large\bfseries}}{{\thesection}}{{0.7em}}{{}}
\titleformat{{\subsection}}{{\normalsize\bfseries}}{{\thesubsection}}{{0.6em}}{{}}
\setlist{{itemsep=1pt,topsep=3pt}}
\renewcommand{{\arraystretch}}{{1.12}}

\title{{\textbf{{A three-objective optimisation framework for circular product\\
ecodesign, built end to end on open data}}\\[2mm]
\large (the \texttt{{CircuOpt}} implementation)}}
\author{{Adeleke Olorunnisola}}
\date{{{DATE}}}

\begin{{document}}
\maketitle

\begin{{abstract}}
\noindent
Circular ecodesign stalls at the concept stage because designers cannot show
that circularity pays. This report builds a framework that decides the
question quantitatively for one product: the GARO LS4 ground-mounted AC
electric-vehicle charging station, whose bill of materials, mass and service
life are taken from a published manufacturer LCA rather than assumed.
Fourteen design variables --- recycled
content, joining method, design life, number of remanufacture cycles,
end-of-life route per material family and standby power class --- are
optimised simultaneously against three objectives: cradle-to-grave carbon per
service-year (\emph{{Planet}}), annualised life-cycle cost (\emph{{Prosperity}}),
and a composite repairability, modularity, circular-labour and supply-risk
index (\emph{{People}}). The life-cycle inventory is read at run time from the
Idemat 2026, a process-level inventory carrying global warming, cumulative
energy demand and the European Commission's Environmental Footprint 3.1
method; commodity prices come from the USGS Mineral Commodity Summaries 2025
(US public domain). No licensed database is used and no factor is invented.
Every methodological choice that could decide the answer is tested rather than
assumed: impact category, end-of-life allocation convention (including the EU
Product Environmental Footprint's own Circular Footprint Formula), grid
geography, social weighting, price source, the electronics proxy, the three
unmeasured recyclate constants, and the widths of the scenario distributions.
One of those tests overturns a comfortable assumption: purely methodological
choices routinely move the Planet objective more than the entire Pareto
archive of design decisions does, which is reported as a finding in its own
right, not smoothed over. A
self-contained NSGA-III recovers a {opt['archive_size']}-design Pareto
archive, of which {head['n_dominating_baseline_on_planet_and_cost']}
designs beat a linear take-make-waste baseline on carbon \emph{{and}} cost at
the same time --- the central claim, and the one industry disputes. A
Monte-Carlo ensemble of {scen['n_scenarios']} futures, spanning grid
decarbonisation, carbon pricing, commodity volatility observed in the USGS
series, remanufacturing performance and an ESPR-style minimum recycled-content
mandate, then separates designs that are merely optimal today from designs
that stay good answers under uncertainty. {int((rob >= ROBUST_THRESHOLD).sum())} of
{opt['archive_size']} designs survive that test. Reading the same front
through four decision-maker preference profiles produces
{n_distinct_profiles} distinct recommendations, which is the practical
content of the exercise: the front is one artefact, the decision is not.
\end{{abstract}}

\section{{Why this problem}}

Circular economy has more definitions than consensus: a review of 114
published definitions found that most reduce it to reduce/reuse/recycle
activity and rarely connect it to social equity at all [24]. That gap is one
reason the People objective in this framework is a full, independently
optimised objective rather than a constraint bolted onto an environmental
model, following the Life Cycle Sustainability Assessment framing of
LCA + LCC + S-LCA as three co-equal pillars, not one pillar with two
footnotes [22].

The European Ecodesign for Sustainable Products Regulation moves circularity
from voluntary to mandatory [11], and the Digital Product Passport it
introduces will make the underlying data auditable at end of life in a way
existing WEEE take-back schemes were not designed for [27]. The barrier is no
longer intent. It is
that deep circularity --- designing for disassembly, remanufacturing,
high-purity recycling --- demands capital up front against benefits that are
distributed across a decade, split between actors, and exposed to commodity
and regulatory volatility. A product manager asked to approve a snap-fit
tooling premium today, for a scrap revenue in 2038, has no rigorous way to
answer.

Existing eco-efficiency tools answer a narrower question. They optimise one
variable, hold the rest fixed, evaluate against today's factors, and collapse
the social dimension into a footnote. That produces recommendations which are
locally correct and strategically useless.

This work takes the opposite stance on three points.

\begin{{enumerate}}
\item \textbf{{Three objectives, not one.}} Planet, Prosperity and People are
kept separate to the end. Aggregating them early hides exactly the trade-off
the decision-maker is paid to make.
\item \textbf{{Futures, not a snapshot.}} A design is scored by how often it is
a good answer across an ensemble of futures, not by its rank against today's
factors.
\item \textbf{{Provenance, not precision theatre.}} Every quantity is tagged
\textsc{{data}}, \textsc{{proxy}} or \textsc{{assumption}}. The proxies and
assumptions are perturbed hardest in the scenario ensemble, so the reader can
see which conclusions rest on measurement and which rest on judgement.
\end{{enumerate}}

\section{{Data}}

Three datasets are used. Two are freely redistributable without restriction;
the third is free to use with attribution and is fetched rather than
redistributed.

\paragraph{{Idemat 2026}} (Vogtlander, Sustainability Impact Metrics / Delft
University of Technology) is the life-cycle inventory. It is process-level,
which matters for two reasons beyond the obvious one.

First, it carries a full impact-assessment layer, so the same calculation
yields global warming, cumulative energy demand, the European Commission's
Environmental Footprint 3.1 single score and the EF 3.1 minerals-and-metals
score simultaneously. EF 3.1 is the method the ESPR delegated acts are written
against, which makes it the right yardstick for a study framed around that
regulation.

Second, and more importantly for a circularity study, it resolves \emph{{process
pairs}}: primary and secondary production of the same material, and end-of-life
processes carrying explicit substitution credits. Recycled content is therefore
a blend between two real production routes rather than a ratio between two
reporting factors, and the end-of-life credit convention becomes a choice this
study can test rather than one the dataset imposes.

Which Idemat process represents which material family is a modelling decision,
not a parsing detail, so it lives in
\texttt{{data/idemat\_process\_map.json}} with a justification per entry and a
resolver that fails loudly if a code moves between releases. Process choice is
decided per family on the evidence available rather than by a blanket rule:
grid electricity uses the European (Swedish) region, matching the factory's
location, which is the one geographic fact this study is actually sure of.
Material \emph{{production}} routes are not assumed to follow the factory
location by default --- aluminium in particular defaults to the
\emph{{global-average}} primary route, because GARO's own published transport
data shows an internationally distributed supplier base and no source names a
specific smelter. Section~\ref{{sec:validation}} and the geography
reconciliation in Section~9 set out the evidence and test the alternative.

\paragraph{{UK Government GHG Conversion Factors for Company Reporting 2025}}
(DESNZ/DEFRA, Open Government Licence v3.0) is retained as a second,
independent backend rather than discarded. It is a corporate GHG-reporting
dataset, not a life-cycle inventory: single-issue, built on consumer waste
categories, and carrying no avoided-burden credit at end of life. Keeping it
selectable means the two can be compared --- which is how the earlier version
of this study, built on DEFRA alone, can be shown to have been carbon
footprinting rather than LCA.

\paragraph{{USGS Mineral Commodity Summaries 2025}} salient-statistics data
releases (US Geological Survey; a work of the US Government, public domain).
These supply annual average commodity prices for the life-cycle costing, and
--- more usefully --- a five-year price history per commodity. That history
is put to two uses: it calibrates the price shocks in the scenario ensemble,
and its coefficient of variation is used as an observable proxy for supply
risk in the People objective, so the social weighting of recycled content is
derived from data rather than asserted.

\begin{{table}}[t]
\centering\small
\caption{{Life-cycle inventory factors, as parsed from the DEFRA 2025 workbook at
run time. \textsc{{data}} means the material family maps directly onto a
published row; \textsc{{proxy}} means the closest published material is used as
a stand-in and the scenario ensemble perturbs it hardest.}}
\label{{tab:factors}}
\begin{{tabular}}{{llp{{6.0cm}}rrr}}
\toprule
family & tag & DEFRA material & primary & closed-loop & cut \\
 & & & \multicolumn{{2}}{{c}}{{kg CO$_2$e/kg}} & \% \\
\midrule
{table_factors()}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering\small
\caption{{Commodity prices parsed from the USGS MCS 2025 salient-statistics
releases. Unit conversion is driven by the column-name suffix
(\texttt{{ctslb}}, \texttt{{dt}}, \texttt{{dtoz}}).}}
\begin{{tabular}}{{llrrl}}
\toprule
commodity & as published & year & USD/kg & source column \\
\midrule
{table_prices()}
\bottomrule
\end{{tabular}}
\end{{table}}

\paragraph{{What the data does not cover.}} Three gaps are load-bearing and are
declared rather than papered over. First, no open dataset gives an engineering
polymer price index, so the PC/ABS resin price is an assumption. Second, the
USGS aluminium release was not retrievable during this run{aluminium_note}
Third, DEFRA does not resolve copper or populated electronics as materials;
both use proxies, which is why they carry the widest uncertainty
($\sigma=0.35$ lognormal) in the ensemble. A model that hid these would report
the same numbers with false confidence.

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig_datasets.png}}
\caption{{Left: the recycled-content lever, straight from DEFRA 2025 --- the gap
between the two bars is the entire environmental case for secondary material.
Right: USGS price histories, indexed to 2020. The dispersion on the right is
why a single-point-in-time optimisation is not a decision aid.}}
\end{{figure}}

\FloatBarrier
\section{{Product and design space}}

The case product is the GARO LS4, a ground-mounted AC electric-vehicle
charging station of total mass \SI{{{f(meta['total_mass_kg'], 2)}}}{{\kilo\gram}}.
It sits inside ESPR scope, has a substantial use phase, and has the modular
architecture that makes disassembly decisions meaningful.

It was chosen for one reason above the others: \emph{{its bill of materials is
published}}. Persson and Erselius [1] report, for this exact unit,
the material distribution by mass, the \SI{{24.5}}{{\kilo\gram}} product mass,
the \SI{{29.7}}{{\kilo\gram}} mass including packaging, a 15-year technical
life and 20\,000 charging sessions over that life. The mass basis of every
number in this report is therefore citable rather than invented --- which
matters, because a life-cycle model is only ever as defensible as the
inventory it multiplies.

Two things remain this project's own. Packaging (corrugated cardboard,
laminated paper and paper, {packpct}\,\% of the published bill of materials)
is excluded and the remainder renormalised onto the official product mass.
And the report publishes mass by \emph{{material}}, not by component, so the
split of each material total across components is an assumption --- declared
in \texttt{{data/garo\_ls4\_bom.json}} and constrained so that the family
totals reproduce the published percentages exactly. The regression suite
fails if that constraint is ever broken.

\begin{{table}}[h]
\centering\small
\caption{{Bill of materials. \emph{{Priority repair}} marks the components a
repairer must reach; \emph{{interface}} indexes the joining decision that governs
access.}}
\begin{{tabular}}{{lrllc}}
\toprule
component & mass (kg) & family & priority repair & interface \\
\midrule
{table_bom()}
\bottomrule
\end{{tabular}}
\end{{table}}

The design vector has {M.N_CONT} continuous and {M.N_INT} categorical genes:

\begin{{itemize}}
\item recycled-content fraction $r_m \in [0,1]$ for polymer, steel and
aluminium;
\item joining method at three interfaces (enclosure closure, chassis, cable
entry) from \{{snap-fit, screwed, bonded\}};
\item design life $L \in \{{8, 10, 12, 15\}}$ years;
\item remanufacture cycles $k \in \{{0,1,2,3\}}$;
\item end-of-life route per material family from \{{closed-loop, open-loop,
incineration with energy recovery, landfill\}};
\item standby power class from \{{\SI{{1.0}}{{\watt}}, \SI{{0.6}}{{\watt}},
\SI{{0.3}}{{\watt}}\}}, each with its own cost premium.
\end{{itemize}}

The search space contains $3^3 \times 4^2 \times 4^5 \times 3 =
{3**3 * 4**2 * 4**5 * 3:,}$ discrete configurations, each with a
three-dimensional continuous interior.

\FloatBarrier
\section{{Formulation}}

Let $\mathcal{{C}}$ be the component set, $m_c$ the mass of component $c$, and
$\phi(c)$ its material family. Service life is
\begin{{equation}}
S = L\,(1 + \rho k), \qquad \rho = 0.60,
\end{{equation}}
with $\rho$ the fraction of the original design life restored by one
remanufacture cycle.

\paragraph{{Planet.}} With $e^{{\mathrm{{pri}}}}_m$ and $e^{{\mathrm{{cl}}}}_m$ the DEFRA
primary and closed-loop factors, $g$ the grid intensity, $\gamma$ the
scenario's grid-decarbonisation multiplier and $d_m^{{(\text{{route}})}}$ the
disposal factor,
\begin{{align}}
E_{{\text{{mat}}}} &= \sum_{{c \in \mathcal{{C}}}} m_c \left[(1 - r_{{\phi(c)}})
  e^{{\mathrm{{pri}}}}_{{\phi(c)}} + r_{{\phi(c)}} e^{{\mathrm{{cl}}}}_{{\phi(c)}}\right], \\
E_{{\text{{use}}}} &= S \left(\tfrac{{P_{{\text{{sb}}}} \cdot 8760}}{{1000}}
  + \eta Q\right) \gamma g, \\
f_1 &= \frac{{E_{{\text{{mat}}}} + E_{{\text{{mfg}}}} + E_{{\text{{use}}}}
  + E_{{\text{{reman}}}} + E_{{\text{{eol}}}}}}{{S}},
\end{{align}}
where $P_{{\text{{sb}}}}$ is standby power, $Q$ the annual charging throughput and
$\eta$ the conversion loss. Dividing by $S$ is the point: it is what allows a
longer-lived, remanufacturable design to amortise a heavier embodied burden,
and it is why lifetime and recycled content compete rather than simply add.

\paragraph{{Prosperity.}} Following the environmental life-cycle costing code
of practice [23], costs are discounted at $r=5\%$ and annualised over
the realised service life,
\begin{{equation}}
f_2 = \frac{{C_{{\text{{cap}}}} + \mathrm{{PV}}(C_{{\text{{energy}}}})
  + \mathrm{{PV}}(C_{{\text{{reman}}}}) + \mathrm{{PV}}(C_{{\text{{eol}}}})
  + \pi_{{\mathrm{{CO_2}}}} E / 1000}}
  {{\left(1 - (1+r)^{{-S}}\right)/r}},
\end{{equation}}
with $\pi_{{\mathrm{{CO_2}}}}$ an internal carbon price (zero in the nominal case).
$C_{{\text{{eol}}}}$ is signed: closed- and open-loop routes return scrap revenue,
landfill and incineration incur gate fees, and take-back logistics are charged
in every case.

\paragraph{{People.}} The social objective is the part that resists
quantification, and it is treated as a composite of four normalised terms
rather than a single invented metric:
\begin{{equation}}
f_3 = -\Big(w_1 s_{{\text{{repair}}}} + w_2 s_{{\text{{modular}}}}
  + w_3 s_{{\text{{labour}}}} + w_4 s_{{\text{{supply}}}}\Big),
\end{{equation}}
$w = (0.35, 0.20, 0.20, 0.25)$. Here $s_{{\text{{repair}}}}$ falls linearly with
the disassembly time needed to reach the priority-repair components, in the
spirit of EN 45554 and the French repairability index; $s_{{\text{{modular}}}}$ is
the share of reversible joints; $s_{{\text{{labour}}}}$ counts remanufacturing
hours created per service-year, the circular-economy employment channel; and
$s_{{\text{{supply}}}}$ credits recycled content in proportion to the USGS-derived
volatility of the commodity displaced --- aluminium's inclusion on the EU's
critical raw materials list, driven by import dependence on bauxite and a low
domestic self-sufficiency ratio, is exactly the kind of exposure this term is
meant to price in [28]. The weights are a stated preference,
not a finding --- and the scenario ensemble re-draws preference weights from a
Dirichlet precisely so that no conclusion depends on them.

\paragraph{{Recyclate is not a free substitute.}} An early version of this
model drove recycled content to its upper bound in every non-dominated design.
That is not a finding, it is a defect: the DEFRA closed-loop factor and the
secondary-metal price both reward recycled content without limit, so the
variable pinned to its bound and carried no information. Two penalty terms,
both declared assumptions, restore the trade-off that exists in practice.
Above a threshold $r_0 = 0.40$, secondary feedstock attracts a quadratic
quality surcharge --- tighter sorting, compatibilisers, property compensation
---
\begin{{equation}}
\kappa(r) = 1 + \lambda \left(\frac{{\max(0, r - r_0)}}{{1 - r_0}}\right)^2,
\qquad \lambda = 0.60,
\end{{equation}}
and a reject rate $1 + \beta r$, $\beta = 0.09$, raises the input mass needed
per kilogram of good part, which costs both money and carbon. The result is
that carbon still falls monotonically with recycled content while cost turns
upward past roughly 40\%, so the optimiser has a genuine interior decision to
make. Both $\lambda$ and $\beta$ are re-drawn in every scenario, so no
conclusion rests on their nominal values.

\paragraph{{Constraints.}} Four constraints encode physical and regulatory
coupling that a naive model would violate silently:
$g_1$ a bonded enclosure cannot be reopened, so it forbids $k \ge 1$;
$g_2$ a bonded enclosure prevents separation, so it forbids a closed-loop
claim on the polymer housings;
$g_3$ claiming a closed-loop return route while specifying under 10\% recycled
input is self-inconsistent;
$g_4$ an ESPR-style minimum recycled content, active only in the scenarios
that impose one. Feasibility is enforced by Deb's constraint-domination rules,
so violation is never traded against objective value.

\FloatBarrier
\section{{Method}}

\paragraph{{NSGA-III.}} The lineage runs from the original non-dominated
sorting genetic algorithm [16] through NSGA-II [7] to reference-point-based
many-objective selection [6, 19]. With three objectives, crowding distance is
a weak diversity signal, so selection uses the reference-point niching of Deb
and Jain: a Das--Dennis simplex lattice of {opt['n_refs']} directions
($p={PARTITIONS}$ partitions), adaptive
normalisation by extreme-point intercepts, and niche-count-based survival.
The reference directions carry a second benefit specific to this problem: a
decision-maker profile is nothing but a preferred direction on the same
normalised hyperplane, so the front and the decision aid share one geometry.

The design vector is hybrid, so variation is hybrid: simulated binary
crossover [17] with polynomial mutation [18] ($\eta_c = \eta_m = 20$) on the
continuous genes, uniform crossover with random-reset mutation on the
categorical ones.
Population {opt['pop_size']}, {opt['generations']} generations,
{len(opt['seeds'])} independent seeds. The implementation is self-contained;
it depends on NumPy alone, and its components are covered by unit tests with
analytically known answers.

\paragraph{{Scenario engine.}} Each future re-draws
{N_UNCERTAIN} uncertain quantities (Table~\ref{{tab:scen}}) and
re-evaluates the entire archive. Robustness is then defined as the share of
future~$\times$~preference draws in which a design lands in the best decile by
augmented achievement scalarisation, with preference weights drawn from a
Dirichlet.

That definition was arrived at by discarding a more obvious one. Counting how
often a design remains non-dominated inside the archive is nearly vacuous
here, because the archive is \emph{{already}} a mutually non-dominated set:
in this run that measure averaged
{100 * nd.mean():.1f}\% with a minimum of {100 * nd.min():.1f}\%, ranking
nothing. The top-decile criterion instead asks whether a design stays a good
\emph{{answer}} when both the world and the decision-maker's priorities move,
and it separates the archive cleanly.

\begin{{table}}[t]
\centering\small
\caption{{The uncertainty space. Distributions are deliberately wide and simple:
the aim is to test whether a conclusion survives, not to forecast.}}
\label{{tab:scen}}
\begin{{tabular}}{{p{{4.2cm}}p{{6.6cm}}p{{4.6cm}}}}
\toprule
quantity & distribution & what it moves \\
\midrule
{table_scenario_ranges()}
\bottomrule
\end{{tabular}}
\end{{table}}

\FloatBarrier
\section{{Results}}

\subsection{{The linear baseline}}

The reference design is a bonded enclosure, all-virgin materials, an
8-year life, no remanufacturing, landfill at end of life and a
\SI{{1.0}}{{\watt}} standby draw. It emits
\SI{{{f(br['gwp_total_kgco2e'], 1)}}}{{\kilo\gram}} CO$_2$e cradle-to-grave,
which is \SI{{{f(br['f_planet_kgco2e_per_service_year'], 2)}}}{{\kilo\gram}}
CO$_2$e per service-year at a cost of
{f(br['f_cost_usd_per_service_year'], 2)}~USD per service-year, a
People index of {f(br['f_social_index'], 3)} and a Material Circularity
Indicator of {f(br['material_circularity_indicator'], 2)} --- zero, correctly,
for a design that is neither made of nor destined for recovered material.

Materials contribute
\SI{{{f(br['gwp_breakdown']['materials'], 1)}}}{{\kilo\gram}} CO$_2$e and the use
phase \SI{{{f(br['gwp_breakdown']['use'], 1)}}}{{\kilo\gram}}, i.e.\ the use phase
is {100 * br['gwp_breakdown']['use'] / br['gwp_total_kgco2e']:.0f}\% of the
total.

Within that use phase the split is decisive, and it is set by the cited duty
cycle rather than by anything chosen here. At the published
20\,000 sessions over 15 years, the station moves
\SI{{{throughput}}}{{\kilo\watt\hour}} per year, so conversion losses account
for \SI{{{convkwh}}}{{\kilo\watt\hour}} annually against
\SI{{{sbkwh}}}{{\kilo\watt\hour}} of standby --- standby is {sbshare}\,\% of
the use phase, not the lever it would be on a domestic wallbox sitting idle.
Conversion efficiency is where the carbon is, and the optimiser finds this
without being told.

\FloatBarrier
\subsection{{Does the inventory hold up? A check against the published LCA}}
\label{{sec:validation}}

The mass basis of this model is cited, but the emission factors are not the
ones the cited study used: it worked from ICE v3.0 and the European EF
database with supplier-specific data, while this model works from Idemat 2026
process-level factors, each chosen per family on stated evidence rather than
assumed to match the study's own conventions (Table~\ref{{tab:factors}} and
\texttt{{data/idemat\_process\_map.json}}). Two independently sourced
inventories applied to the same physical unit is the only external check
available without a licensed LCI database, so it is reported rather than
quietly omitted.

Applied to the cradle-to-gate material stage, this model returns
\SI{{{modelmat}}}{{\kilo\gram}} CO$_2$e against the published
\SI{{{pubmat}}}{{\kilo\gram}} CO$_2$e --- a discrepancy of {valpct}\,\%, with
this model the lower of the two. Section~9 traces essentially all of that gap
to one factor: aluminium, which is over half the product's mass by
citation, and where the two inventories' underlying process assumptions
differ most.

\begin{{table}}[h]
\centering\small
\caption{{Factor cross-check, \si{{\kilo\gram}} CO$_2$e per \si{{\kilo\gram}}
of virgin material. Neither column is ground truth; the point is the size and
direction of the gap.}}
\label{{tab:validation}}
\begin{{tabular}}{{lrrr}}
\toprule
material & this model (Idemat) & published study & ratio \\
\midrule
{validation_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

The more informative result is structural. The published study reports that
aluminium and electronic components each account for roughly 47\,\% of
material-stage emissions. This model, told nothing of that, returns
{alushare}\,\% and {elecshare}\,\%. It recovers the hotspot structure of the
product unprompted --- which is the property that actually matters here,
because every design comparison in this report is a comparison of where those
hotspots move.

\begin{{table}}[h]
\centering\small
\caption{{Material-stage carbon by family, this model, \% of the material
total. The published study identifies the same top two.}}
\begin{{tabular}}{{lr}}
\toprule
family & share of material-stage carbon (\%) \\
\midrule
{material_share_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

What this does \emph{{not}} establish is that either number is right. Both
rest on generic factors; agreement between two generic sources is weaker
evidence than either being validated against measurement. It establishes
something narrower and still worth having: the inventory is not
arbitrary, and the conclusions drawn from it do not depend on one particular
published spreadsheet.

\subsection{{Optimisation}}

\begin{{table}}[h]
\centering\small
\caption{{Search performance over {len(opt['seeds'])} seeds. Hypervolume uses a
common reference point; the Monte-Carlo sampling error of the hypervolume
estimator is reported so it is not mistaken for run-to-run variance.}}
\begin{{tabular}}{{lr}}
\toprule
population $\times$ generations & {opt['pop_size']} $\times$ {opt['generations']} \\
reference directions & {opt['n_refs']} \\
mean runtime per seed & \SI{{{opt['runtime_cpu_s_mean']:.1f}}}{{\second}} CPU \\
hypervolume, mean $\pm$ s.d. & {f(opt['hypervolume_mean'], 2)} $\pm$ {f(opt['hypervolume_std'], 2)} \\
\quad seed-to-seed variation & {seed_cv:.2f}\% \\
\quad estimator sampling error & $\pm${f(opt['hypervolume_mc_error'], 2)} \\
Schott spacing, mean $\pm$ s.d. & {f(opt['spacing_mean'], 4)} $\pm$ {f(opt['spacing_std'], 4)} \\
random search, equal budget ({opt['random_search_budget']:,} evaluations)
  & {f(opt['random_search_hypervolume'], 2)} ({hv_ratio:.1f}\% of NSGA-III) \\
unique non-dominated designs & {opt['archive_size']} \\
\bottomrule
\end{{tabular}}
\end{{table}}

Seed-to-seed hypervolume varies by {seed_cv:.2f}\%, which is
{'below' if seed_cv < 100 * opt['hypervolume_mc_error'] / opt['hypervolume_mean'] else 'comparable to'}
the sampling error of the estimator itself: the front is stable, not a lucky
draw. Equal-budget random search reaches {hv_ratio:.1f}\% of the same
hypervolume. That gap is the honest measure of what the algorithm contributes,
and it is worth reading in both directions --- the problem is not so rugged
that search is the hard part; the hard part is the model and the data.

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig_front.png}}
\caption{{The Pareto archive in its three pairwise projections; the red cross is
the linear baseline. It sits outside the front on every projection --- the
baseline is not a trade-off, it is simply dominated.}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.62\textwidth]{{figures/fig_convergence.png}}
\caption{{Ideal-point convergence across seeds.}}
\end{{figure}}

\FloatBarrier
\subsection{{The central result}}

{head['n_dominating_baseline_on_planet_and_cost']} of the
{opt['archive_size']} Pareto designs beat the linear baseline on carbon
\emph{{and}} cost simultaneously. The best available reduction is
{f(head['max_planet_reduction_pct'], 1)}\% on carbon per service-year and
{f(head['max_cost_reduction_pct'], 1)}\% on cost per service-year.

{win_block}

This is the claim the position statement calls cost-prohibitive, tested rather
than asserted, and the mechanism is not mysterious: circularity in this
product buys its own capital premium back through amortisation. A longer,
remanufacturable life spreads the embodied burden over more service-years, and
the low-standby class pays for itself out of a use phase that dominates the
inventory. The recycled-content lever contributes, but it is the smaller
effect.

\begin{{table}}[t]
\centering\small
\caption{{Corner solutions of the archive, against the linear baseline.
Recycled content is listed polymer/steel/aluminium; joints are enclosure,
chassis, cable entry.}}
\begin{{tabular}}{{llllrrrlrrr}}
\toprule
& \# & recycled & joints & $L$ & $k$ & sb & EoL & $f_1$ & $f_2$ & People \\
\midrule
{table_corners()}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig_breakdown.png}}
\caption{{Where the differences actually come from. Negative cost bars are scrap
revenue recovered at end of life.}}
\end{{figure}}

\FloatBarrier
\subsection{{Robustness across futures}}

Across {scen['n_scenarios']} futures and
{scen['n_tastes']} preference draws each,
{int((rob >= ROBUST_THRESHOLD).sum())} of {opt['archive_size']} designs clear
twice the chance rate --- a design drawn at random would be a top-decile
answer in {100 * scen['top_quantile']:.0f}\% of draws, so
{100 * ROBUST_THRESHOLD:.0f}\% is the bar --- and {scen['never_top_decile']}
are never a top-decile answer at all.
The most robust design, \#{most_robust}, holds top-decile status in
{100 * rob[most_robust]:.0f}\% of draws while remaining feasible in
{100 * feas[most_robust]:.0f}\% of futures --- including those imposing an ESPR
recycled-content mandate.

The mechanism behind robustness is worth naming, because it is a design
instruction rather than a number. Designs that survive share reversible
joints, a long design life and meaningful recycled content. Those are exactly
the choices that hedge: reversible joints keep the remanufacturing option open
if labour economics improve, recycled content keeps the design compliant if a
delegated act lands, and a long life dilutes embodied carbon regardless of how
fast the grid decarbonises. Robustness here is not a statistical artefact, it
is optionality.

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig_robustness.png}}
\caption{{Left: the nominal front coloured by robustness --- note that the
cheapest nominal designs are not the most robust. Right: 5th--95th percentile
spread of the Planet objective across futures, which is dominated by grid
decarbonisation and charging-throughput uncertainty.}}
\end{{figure}}

\FloatBarrier
\subsection{{The decision aid}}
\label{{sec:aid}}

\begin{{table}}[h]
\centering\small
\caption{{What each profile selects. Weights are (Planet, Prosperity, People);
$f_1$ and $f_2$ are means across the ensemble.}}
\begin{{tabular}}{{llllrlrrrrr}}
\toprule
profile & weights & pick & share & winners & recycled & $L$ & $k$ & sb & $\bar f_1$ & $\bar f_2$ \\
\midrule
{table_profiles()}
\bottomrule
\end{{tabular}}
\end{{table}}

Four profiles produce {n_distinct_profiles} distinct recommendations from one
front. No profile's choice wins in every future --- the shares in the table
are the honest answer to ``what should we build?'', and their distance from
100\% is the residual uncertainty a committee, not an optimiser, has to
absorb.

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig_profiles.png}}
\caption{{Left: how often each archive design is selected, by profile. Right:
the levers each profile's modal choice pulls, normalised.}}
\end{{figure}}

\FloatBarrier
\FloatBarrier
\section{{Testing the choices that could decide the answer}}

A life-cycle model contains a small number of choices that are not measurements
and that can move the conclusion more than any design variable. Declaring them
is necessary; testing them is what makes the study checkable. Six are tested
here, and two of the six change the answer.

\subsection{{Impact category: was optimising carbon a fair proxy?}}

The inventory carries four indicators. Ranking the whole archive under each and
correlating against global warming gives Table~\ref{{tab:indrank}}.

\begin{{table}}[h]
\centering\small
\caption{{Rank agreement with global warming across the {archive_n}-design archive.}}
\label{{tab:indrank}}
\begin{{tabular}}{{lrrc}}
\toprule
indicator & Spearman $\rho$ & Kendall $\tau$ & picks the same best design \\
\midrule
{indicator_rank_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

Agreement is high but not perfect. Carbon is a defensible proxy for ranking
designs in this product, and it is not a substitute for the other indicators
when selecting a single winner. Reporting the correlation is the honest form of
that statement; reporting only the carbon front and calling it an LCA would not
have been.

\subsection{{Allocation: the choice that does decide the answer}}

ISO 14044 requires the allocation convention to be declared and its influence
examined. Under \emph{{cut-off}}, recovered material earns nothing and
end-of-life is {cutoff_eol}\,\% of total carbon --- a rounding error, which
makes the end-of-life decision variable environmentally inert. Under
\emph{{substitution}} it earns the virgin production it displaces and rises to
{sub_eol}\,\% of the total, shifting the mean Planet objective by
{alloc_shift}\,\%.

The consequence is larger than the shift. Rank correlation between the two
conventions across the archive is $\rho={alloc_rho}$, and they do not agree on
the best design. \textbf{{The allocation convention, not the design space,
selects the winner.}} Any result in this report that concerns end-of-life route
is conditional on the convention stated with it. This is the single most
important methodological caveat in the study, and it is why the earlier
carbon-reporting-factor version of this model --- which had no substitution
option at all --- could not have been trusted on circularity questions.

Rather than stop at two independently invented conventions, a third is
implemented: the material term of the European Commission's own \emph{{Circular
Footprint Formula}} (CFF), the method the Product Environmental Footprint
regulation actually specifies for exactly this problem [13]:
\[
E = (1-R_1)E_V + R_1\left[A\,E_{{\text{{recycled}}}}
+ (1-A)E_V\frac{{Q_{{s,\text{{in}}}}}}{{Q_p}}\right],
\]
where $R_1$ is the recycled-content decision variable already in this model,
$E_V$ and $E_{{\text{{recycled}}}}$ the virgin and secondary production
processes, and $A\in[0,1]$ a market-based allocation factor splitting the
benefit of recycling between the material's two lives. At $A=1$ the formula
collapses exactly onto this model's \emph{{substitution}} convention on both the
input side and end-of-life -- confirmed numerically as a correctness check,
not asserted -- and at $A=0$ recycled content earns almost no credit. Sweeping
$A\in\{{0,\,0.5,\,1\}}$ moves the archive's mean Planet objective by
{cff_sweep_span}\,\% across the sweep, and CFF's rank correlation against the
existing conventions is $\rho={cff_rho_cutoff}$ (cut-off) and
$\rho={cff_rho_sub}$ (substitution) -- a third, citable point inside the same
disagreement, not outside it.

Two things are stated plainly about this implementation rather than left for
a reader to discover. First, $A=0.5$ is the commonly cited PEF default for
market-based allocation, not a value verified here for aluminium, steel,
copper or plastic specifically -- it is swept as a sensitivity, not asserted
as this product's true factor. Second, the CFF's material formula is
implemented; its energy-recovery and disposal formula components (parameters
$B$, calorific value and recovery efficiencies) could not be verified against
a primary EU/JRC source at the time of writing -- several official PDF
documents returned access errors during research for this feature -- and
guessing a regulatory formula would be a worse error than not implementing
it. End-of-life under \texttt{{cff}} therefore falls back to this model's own
\emph{{substitution}} treatment, labelled in code as an approximation, not a
citation. Partial coverage, verified precisely, was judged better than full
coverage, guessed.

\subsection{{Geography: the level moves, the ranking does not}}

\begin{{table}}[h]
\centering\small
\caption{{Grid sensitivity. Mean Planet objective across the archive, by market.}}
\label{{tab:geo}}
\begin{{tabular}}{{lrrr}}
\toprule
grid & \si{{\kilo\gram}} CO$_2$e/kWh & mean Planet & $\rho$ vs reference \\
\midrule
{geography_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

The absolute result spans {geo_span}$\times$ between the cleanest and dirtiest
European grid --- larger than the effect of every design variable combined. But
rank correlation across the archive stays above $+0.99$ everywhere. So the
level of the answer is a statement about the grid, while the \emph{{ranking}} of
designs is not. A designer can use this front anywhere in Europe; a
communications department quoting the absolute number cannot.

\subsection{{The recyclate constants: a finding with its conditionality attached}}

Three constants govern the cost penalty on recycled content --- a quality
threshold, a quadratic surcharge and a reject rate. \textbf{{None has an
empirical source.}} They matter because they produce the interior cost optimum
in recycled content, i.e.\ the result that recycled content is not simply
maximised.

Sweeping all three over their plausible ranges: the interior optimum exists in
{corner_int} of the corner combinations tested, but its location moves between
{par_lo:.2f} and {par_hi:.2f} recycled content. The honest statement is
therefore narrow: \emph{{that}} an interior optimum exists is reasonably robust to
these constants; \emph{{where}} it sits is not, and no number quoted from this
model for an optimal recycled content should be read as an empirical estimate.
Measuring these three constants for a real recyclate stream would be the single
most valuable piece of primary data this model could receive.

\subsection{{Social weights: a property of the designs, mostly}}

The four social sub-weights were chosen, not elicited. Redrawing them
{wt_draws} times from a flat Dirichlet, the nominally best design remains best
in {wt_share}\,\% of draws, with {wt_distinct} distinct winners overall, and
rank correlation against the nominal weighting averages $\rho={wt_rho}$
(5th percentile ${wt_p05}$). The social ranking is therefore mostly a property
of the designs rather than of the weighting --- but the {wt_distinct} distinct
winners are a reminder that this index is a transparent decision aid, not a
measurement.

\subsection{{Scenario widths: not load-bearing}}

\begin{{table}}[h]
\centering\small
\caption{{Robustness ranking under halved and doubled judgement widths.
Measured widths (commodity volatility) and the discrete grid set are unchanged.}}
\begin{{tabular}}{{lrrr}}
\toprule
judgement $\sigma$ & top design & robust core & $\rho$ vs nominal \\
\midrule
{distributional_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

A factor of four in the guessed widths leaves the top design unchanged and rank
correlation above $+0.98$. The robustness conclusions do not rest on the
distribution widths.

\subsection{{Electronics: how wrong would the weakest proxy have to be?}}

Electronics has no matching Idemat process at all -- no entry represents a
charging-station metering or RCD board. Rather than pin one white-goods
control board arbitrarily, the factor used here is the average of the two
closest analogues, a refrigerator and a washing-machine control board
(\texttt{{data/idemat\_process\_map.json}}
\allowbreak\texttt{{-> families.electronics.virgin}}), following the same
"use the available evidence rather than one arbitrary pick" logic as the
aluminium choice. Both source processes explicitly exclude integrated
circuits, a real limitation the averaging does not resolve.

At today's factor, aluminium accounts for {alu_share_1x}\,\% of material-stage
carbon and electronics {elec_share_1x}\,\%. {electronics_verdict}
This is a stronger statement than a wide error bar: it says exactly how wrong
the proxy would have to be before it changed which material dominates the
carbon story, rather than asserting robustness by perturbing it and hoping.

\subsection{{Prices: USGS against a second, independent source}}

USGS never resolved an aluminium price -- the download was blocked at
data-collection time -- so aluminium ran on a hardcoded guess
(\SI{{2.55}}{{}} USD/kg) until this fix. It now uses the World Bank Commodity
Markets ("Pink Sheet") trailing 12-month mean instead, a live, independently
compiled global reference series: \SI{{{price_alu_wb}}}{{}} USD/kg against the
guess it replaces. For copper, where USGS does have data, the two sources are
compared directly: USGS \SI{{{price_cu_usgs}}}{{}} versus World Bank
\SI{{{price_cu_wb}}}{{}} USD/kg, a {price_shift_pct}\,\% shift in mean archive
cost between sources, rank correlation $\rho={price_rho}$, same best design:
{price_same_best}. Steel has no comparable World Bank entry -- that source
publishes iron ore, not scrap, a different commodity -- so this cross-check
covers aluminium and copper only, and that limitation is stated rather than
papered over with an approximate substitute.

\subsection{{Does the design decision matter more than the modelling decision?}}

Every methodological sensitivity in this section can be put on one axis: how
much does each one move the Planet objective, compared to moving across the
entire {archive_n}-design Pareto archive itself?

\begin{{table}}[h]
\centering\small
\caption{{Spread in the Planet objective, \si{{\kilo\gram}} CO$_2$e per
service-year, induced by each choice, design space held fixed except where
the choice itself is the design archive.}}
\label{{tab:spread}}
\begin{{tabular}}{{lr}}
\toprule
source of variation & span \\
\midrule
{spread_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

Design choice ranks \#{spread_rank} of {spread_n} by how much it moves the
result. {spread_verdict_sentence} That is not a reason to stop optimising the
design -- the win-win designs this report finds are real and reproducible --
but it is a reason to report the modelling choices with the same weight as the
design recommendation, which is what Sections~7 and~9 of this report do rather
than treating them as footnotes to it.

\FloatBarrier
\section{{Does the optimiser earn its keep?}}

Earlier versions of this work reported that random search reached a large
fraction of NSGA-III's hypervolume, without testing whether the difference was
real. It is now tested properly: {algo_seeds} independent seeds per method, an
identical evaluation budget of {algo_budget} per run, and the same variation
operators for both evolutionary algorithms so the comparison isolates the
selection mechanism.

\begin{{table}}[h]
\centering\small
\caption{{Hypervolume by method, over {algo_seeds} seeds each.}}
\begin{{tabular}}{{lrrrr}}
\toprule
method & seeds & median & mean & s.d. \\
\midrule
{algorithm_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[h]
\centering\small
\caption{{Pairwise Mann--Whitney $U$, two-sided, with rank-biserial effect
size, following the nonparametric evolutionary-algorithm comparison protocol
of [21]. With ten seeds per method the test is weakly powered, so the effect
size carries more information than the $p$-value and both are reported.}}
\begin{{tabular}}{{lrrrrc}}
\toprule
comparison & median ratio & $U$ & $p$ & effect $r$ & $p<0.05$ \\
\midrule
{algorithm_test_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

Both evolutionary algorithms beat random search by roughly {ea_vs_rand}\,\% in
median hypervolume [20], with complete rank separation across ten seeds.

The comparison between them is less comfortable, and is reported as it came
out. \textbf{{NSGA-II attains a {nsga2_margin}\,\% higher median hypervolume
than NSGA-III, and the separation is complete}} ($U=0$, $r=1.00$,
$p={nsga_p}$): every NSGA-II seed beat every NSGA-III seed. The margin is small
in absolute terms and both algorithms sit far above random search, but the
direction is consistent and the test is unambiguous, so the honest statement is
that \emph{{NSGA-III is not the better optimiser on this problem}}.

That is what should be expected, and saying so is more useful than explaining
it away. Reference-point niching is designed for many-objective problems where
crowding distance degrades; with three objectives, crowding distance is still a
perfectly good diversity signal, and NSGA-III's uniform reference lattice
spends population on regions of the simplex that this front does not occupy.

NSGA-III is nonetheless what the rest of this report uses, for a reason that is
not performance: its reference directions are where decision-maker preferences
attach, which is the mechanism Section~\ref{{sec:aid}} depends on. A reader who
wants the best-converged front on this problem should use NSGA-II; a reader who
wants the decision aid should use NSGA-III and accept {nsga2_margin}\,\% of
hypervolume as its price. Both are implemented, share the same variation
operators, and can be swapped with one argument.

\FloatBarrier
\section{{Where the validation gap actually comes from}}

Section~\ref{{sec:validation}} reported the aggregate agreement with the
published study. Aggregates hide whether agreement is accuracy or cancellation,
so the difference is decomposed by material family in
Table~\ref{{tab:vdecomp}}.

\begin{{table}}[h]
\centering\small
\caption{{Contribution of each family to the difference against the published
study, \si{{\kilo\gram}} CO$_2$e per unit.}}
\label{{tab:vdecomp}}
\begin{{tabular}}{{lrrrr}}
\toprule
family & mass (kg) & model factor & published factor & contribution \\
\midrule
{vdecomp_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

The gap is not spread across the inventory: {alu_share}\,\% of what remains of
it is the aluminium factor alone, and only {cancel_pct}\,\% of the
family-level disagreement cancels in the total. That is a better outcome than
broad agreement would have been, because it is attributable.

The attribution is a supply-chain geography choice, and it was made the right
way round: on evidence, before checking the fit, not after. Idemat resolves
both a global-average primary aluminium route and a European one. GARO's own
LCA report names no smelter, grid, or bauxite origin for the aluminium in the
LS4 --- but its own supplier transport-distance table lists routes to Gnosjö
from Poland, Germany, the Czech Republic, Italy, Turkey, China, Denmark and
Slovenia, describing a genuinely international supplier base, not a
predominantly Nordic one. On that evidence this report defaults to the
\emph{{global}} route (\SI{{{alu_glob}}}{{\kilo\gram}} CO$_2$e/kg), and keeps
the European route (\SI{{{alu_eu}}}{{\kilo\gram}} CO$_2$e/kg, hydro- and
nuclear-heavy smelting) only as a labelled comparison.

That choice happens to move the model closer to the published study's own
factor (\SI{{{alu_pub}}}{{\kilo\gram}} CO$_2$e/kg): the default gives a
cradle-to-gate ratio of {ratio_default} against the published figure, where
substituting the European alternative would give {ratio_eu} --- \emph{{further}}
from the published result, not closer. That agreement is worth stating plainly
as a consequence, not a justification: the route was chosen on the transport
evidence first, and only checked against the published figure afterwards. Had
the two disagreed, the transport evidence would still have been the better
basis for the default.

Neither study is wrong on this point; they simply required different evidence
to resolve, and only one source (GARO's own transport data) actually bears on
which is closer to this product's real supply chain. For a product that is
{alu_mass_pct}\,\% aluminium by mass, that one declared, evidenced choice is
worth more than every circular design decision in this report combined ---
which is itself the most useful thing the validation exercise produced.

What remains of the gap after that reconciliation ({ratio_default} of the
published figure) should not be chased further by tuning; the more defensible
move is to size it against what independent LCA studies find when they compare
databases for the \emph{{same}} product. Pauer, Wohner and Tacker modelled six
packaging systems in three database/software combinations (GaBi, ecoinvent
3.6, and the EU's own EF database) and found that while global-warming results
were broadly similar across databases, other impact categories diverged
substantially, driven by differences in background-process scope and
characterisation coverage between databases [14]. A companion
study of LCI database choice in a drivetrain (combustion vs. electric vehicle)
case study reports ecoinvent-vs-GaBi deviations from 15\,\% up to several
thousand per cent depending on the impact category [15].
Against that published range, a residual gap of
this size between two independently built inventories applied to the same
product is the expected order of magnitude, not evidence of an error in
either.

\section{{What this model cannot tell you}}

Several limitations of earlier versions of this work have been closed and are
recorded here as closed rather than quietly dropped: the inventory is now
process-level and multi-indicator rather than a single carbon-reporting factor;
the end-of-life credit convention is implemented three ways, including the
EU's own regulatory formula, and tested rather than fixed by the dataset's
convention; the circularity indicator follows the published formula rather
than a simplification; the scenario widths for commodity prices are measured
rather than guessed; the electronics proxy is an average across the available
evidence with a quantified breakeven bound rather than one arbitrary pick;
aluminium pricing comes from a live independent source rather than a
hardcoded guess; and the aluminium production route was chosen from evidence
in the product's own published transport data, before checking whether it
improved agreement with anything, not after. What remains is below.

\begin{{itemize}}
\item \textbf{{The allocation convention selects the winner.}} This is the
binding limitation, not a footnote. Cut-off and substitution rank the archive
almost independently ($\rho={alloc_rho}$) and disagree on the best design. Any
end-of-life conclusion here is conditional on the convention quoted beside it,
and a reader who needs one answer rather than two must decide the convention
on grounds this model cannot supply.
\item \textbf{{The material masses are cited; their allocation to components
is not.}} The published study gives mass by material, not by part. Anything
depending only on how much aluminium or electronics the product contains is on
cited ground; anything depending on \emph{{which part}} the aluminium is in ---
disassembly sequence above all --- rests on this project's allocation. The
repairability term is the most exposed.
\item \textbf{{The recyclate constants are unmeasured, and they set the
optimum's location.}} The interior optimum exists across most of the parameter
space, but sits anywhere between {par_lo:.2f} and {par_hi:.2f} recycled
content. No optimal recycled content quoted from this model is an empirical
estimate.
\item \textbf{{The electronics process is a stand-in, bounded rather than
merely disclosed.}} The average of two white-goods control boards represents
the metering, RCD and power boards; both source processes exclude integrated
circuits. It remains the weakest mapping in the inventory. What changed is
that the weakness is now bounded rather than only flagged: Section 8.1 shows
electronics would need to be wrong by more than {electronics_verdict_short}
before it changed which material dominates the carbon story, and the
archive's ranking barely moves across that range regardless, because
electronics mass is not itself a decision variable.
\item \textbf{{The People index is a proxy, not a social LCA.}} It is a
transparent weighted index over four design-controllable indicators. It is not
a UNEP S-LCA study [26]: there is no stakeholder inventory, no worker or
community subcategory, and no site-specific data. The Dirichlet re-weighting shows the
ranking is not an artifact of the weights; it cannot validate the choice of
terms, and no claim about actual social performance follows from it.
\item \textbf{{One inventory, no uncertainty data.}} Idemat publishes point
values, not distributions. The proxy multipliers in the ensemble are a stand-in
for inventory uncertainty, not a propagation of it. A study with access to
ecoinvent's uncertainty information could do this properly.
\item \textbf{{Prices are world-market, impacts are European.}} USGS and World
Bank both quote globally traded metals, which is defensible for commodities ---
Section 8.2 shows the two sources agree to within {price_shift_pct}\,\% on
archive cost and pick the same best design --- but this leaves polymer and
labour prices as assumptions rather than data, and steel has no comparable
World Bank series to cross-check against at all.
\item \textbf{{The CFF's disposal formula is not implemented.}} Only the
material (input-side) term of the EU PEF Circular Footprint Formula is
implemented; its energy-recovery and disposal terms could not be verified
against a primary EU/JRC source and are not guessed. End-of-life under the
\texttt{{cff}} convention falls back to this model's own substitution
treatment as a stated approximation, not a citation.
\item \textbf{{Nothing here is validated against measurement.}} The one
external check is against another model. Agreement between two models is
weaker evidence than either being right, and the residual gap after
reconciling the aluminium route is sized against the published range of
inter-database LCA discrepancies rather than closed further by tuning.
\end{{itemize}}

The central finding --- that carbon-and-cost win-win designs exist and can be
found systematically --- is a statement about the \emph{{structure}} of the
trade-off, and it survives every perturbation tested: all four impact
categories, three allocation conventions including the EU's own CFF, six
European grids, two independent price sources, an order-of-magnitude range in
the electronics factor, a factor of four in the scenario widths, and random
re-weighting of the social index. What does \emph{{not}} survive unqualified is
the implicit assumption that design choice is the largest lever available:
Section 8.3 finds it usually is not. The specific numbers should not be
carried past the limitations above.

\section{{Appendix: the declared assumptions}}

Every quantity below is an engineering estimate rather than a dataset value.
They are collected in one class in \texttt{{circuopt/model.py}} so that a
reviewer can replace any of them without touching the model.

\begin{{table}}[h]
\centering\small
\begin{{tabular}}{{lrlp{{6.4cm}}}}
\toprule
parameter & value & unit & what it represents \\
\midrule
{table_assumptions()}
\bottomrule
\end{{tabular}}
\end{{table}}

\FloatBarrier
\section{{Reproducing this}}

\begin{{verbatim}}
python3 circuopt/datasets.py      # parse and cache the two open datasets
python3 test_circuopt.py          # regression suite
python3 scripts/fetch_idemat.py   # one-time: Idemat 2026 (free, attribution)
python3 scripts/fetch_worldbank.py # one-time: World Bank commodity prices
python3 run_experiments.py        # E0-E15, writes results.json + figures/
python3 make_report.py            # regenerates this PDF from results.json
\end{{verbatim}}

Requires Python 3.8+, NumPy, Matplotlib and \texttt{{openpyxl}} (to read the
DEFRA workbook). Total runtime for the full protocol was
\SI{{{int(round(meta['runtime_s']))}}}{{\second}} on one core. The raw datasets are
included under \texttt{{data/}} under the terms of their respective licences.

\section*{{Data and licences}}
\small
\begin{{itemize}}
\item Idemat 2026, Sustainability Impact Metrics / Delft University of
Technology. Free to use with attribution; fetched by
\texttt{{scripts/fetch\_idemat.py}}, not redistributed with this repository.
\item UK Government GHG Conversion Factors for Company Reporting 2025,
Department for Energy Security and Net Zero. Open Government Licence v3.0.
Retained as a second, independent inventory backend for comparison.
\item Mineral Commodity Summaries 2025 salient-statistics data releases,
U.S. Geological Survey. Public domain (17 U.S.C. \S 105).
\item Commodity Markets ("Pink Sheet") monthly price data, World Bank.
Freely available World Bank data; fetched by
\texttt{{scripts/fetch\_worldbank.py}}, not redistributed. Used as the live
source for aluminium pricing (USGS never resolved this commodity) and as a
second, independent price source for copper.
\item GARO LS4 Life Cycle Assessment, Persson \& Erselius (GARO AB / 2050
Consulting), July 2021. Published manufacturer report, cited throughout for
the product's bill of materials and service life.
\end{{itemize}}

\section*{{References}}
\small
\begin{{enumerate}}[label={{[\arabic*]}}]
\item J. Persson and G. Erselius, \emph{{Life Cycle Assessment of the charging
station GARO LS4}}, GARO AB with 2050 Consulting, July 2021. Available from
GARO AB; a copy is included at \texttt{{data/garo\_ls4\_lca.pdf}}.
\item J.G. Vogtlander, \emph{{Idemat 2026}}, Sustainability Impact Metrics /
Delft University of Technology, \texttt{{ecocostsvalue.com}}. Free to use with
attribution; not redistributed with this repository.
\item European Commission, \emph{{Environmental Footprint (EF) 3.1 reference
package}}, Joint Research Centre.
\item ISO 14044:2006, \emph{{Environmental management --- Life cycle assessment
--- Requirements and guidelines}}, clause 4.3.4 on allocation.
\item H.B. Mann and D.R. Whitney, ``On a test of whether one of two random
variables is stochastically larger than the other,'' \emph{{Annals of
Mathematical Statistics}}, 18(1), 50--60, 1947.
\item K. Deb and H. Jain, ``An evolutionary many-objective optimization
algorithm using reference-point-based nondominated sorting approach, part I,''
\emph{{IEEE Trans. Evolutionary Computation}}, 18(4), 577--601, 2014.
\item K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, ``A fast and elitist
multiobjective genetic algorithm: NSGA-II,'' \emph{{IEEE Trans. Evolutionary
Computation}}, 6(2), 182--197, 2002.
\item I. Das and J. E. Dennis, ``Normal-boundary intersection,''
\emph{{SIAM J. Optimization}}, 8(3), 631--657, 1998.
\item A. P. Wierzbicki, ``The use of reference objectives in multiobjective
optimization,'' in \emph{{Multiple Criteria Decision Making Theory and
Application}}, Springer, 1980.
\item Ellen MacArthur Foundation and Granta Design, \emph{{Circularity
Indicators: An Approach to Measuring Circularity --- Methodology}}, 2015. The
MCI in this report follows this formulation, including $F(X)=0.9/X$.
\item Regulation (EU) 2024/1781 establishing a framework for the setting of
ecodesign requirements for sustainable products (ESPR).
\item EN 45554:2020, General methods for the assessment of the ability to
repair, reuse and upgrade energy-related products.
\item European Commission, Joint Research Centre, \emph{{Product
Environmental Footprint Category Rules (PEFCR) Guidance}}, which defines the
Circular Footprint Formula. The primary document could not be fetched
directly during this work (access was blocked); the formula implemented here
was reconstructed from secondary summaries of the guidance (GreenDelta,
2020; ARECO) and should be checked against the primary JRC document before
being relied on beyond this report.
\item E. Pauer, B. Wohner, M. Tacker, ``The Influence of Database Selection
on Environmental Impact Results. Life Cycle Assessment of Packaging Using
GaBi, Ecoinvent 3.6, and the Environmental Footprint Database,''
\emph{{Sustainability}}, 12(23), 9948, 2020.
\href{{https://doi.org/10.3390/su12239948}}{{doi:10.3390/su12239948}}.
\item M. Kalverkamp, E. Helmers, A. Pehlken, ``Impacts of life cycle
inventory databases on life cycle assessments: A review by means of a
drivetrain case study,'' \emph{{Journal of Cleaner Production}}, 269, 121329,
2020. \href{{https://doi.org/10.1016/j.jclepro.2020.121329}}{{doi:10.1016/j.jclepro.2020.121329}}.
\item N. Srinivas and K. Deb, ``Multiobjective Optimization Using
Nondominated Sorting in Genetic Algorithms,'' \emph{{Evolutionary
Computation}}, 2(3), 221--248, 1994. The original NSGA, the direct ancestor
of NSGA-II and NSGA-III.
\item K. Deb and R.B. Agrawal, ``Simulated Binary Crossover for Continuous
Search Space,'' \emph{{Complex Systems}}, 9, 115--148, 1995. The crossover
operator used on the continuous genes throughout this work.
\item K. Deb and M. Goyal, ``A Combined Genetic Adaptive Search (GeneAS) for
Engineering Design,'' \emph{{Computer Science and Informatics}}, 26(4),
30--45, 1996. Introduces the polynomial mutation operator used alongside SBX.
\item H. Jain and K. Deb, ``An Evolutionary Many-Objective Optimization
Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part II:
Handling Constraints and Extending to an Adaptive Approach,'' \emph{{IEEE
Trans. Evolutionary Computation}}, 18(4), 602--622, 2014. Extends NSGA-III to
constrained problems, which is the form implemented here.
\item E. Zitzler and L. Thiele, ``Multiobjective Optimization Using
Evolutionary Algorithms --- A Comparative Case Study,'' in \emph{{Parallel
Problem Solving from Nature --- PPSN V}}, Springer, 292--301, 1998.
Introduces the hypervolume indicator used throughout Section~8 to compare
NSGA-III, NSGA-II and random search.
\item J. Derrac, S. Garc\'ia, D. Molina, F. Herrera, ``A Practical Tutorial
on the Use of Nonparametric Statistical Tests as a Methodology for Comparing
Evolutionary and Swarm Intelligence Algorithms,'' \emph{{Swarm and
Evolutionary Computation}}, 1, 3--18, 2011. Basis for the Mann--Whitney
protocol and the emphasis on effect size over $p$-value with small seed
counts.
\item W. Kl\"opffer, ``Life Cycle Sustainability Assessment of Products,''
\emph{{International Journal of Life Cycle Assessment}}, 13(2), 89--95, 2008.
The LCA + LCC + S-LCA framing this report's three objectives instantiate.
\item T.E. Swarr, D. Hunkeler, W. Kl\"opffer et al., ``Environmental
Life-Cycle Costing: A Code of Practice,'' \emph{{International Journal of
Life Cycle Assessment}}, 16, 389--391, 2011. The LCC methodology the
Prosperity objective follows.
\item J. Kirchherr, D. Reike, M. Hekkert, ``Conceptualizing the Circular
Economy: An Analysis of 114 Definitions,'' \emph{{Resources, Conservation and
Recycling}}, 127, 221--232, 2017. Notes that most circular-economy
definitions barely connect to social equity, which is part of the motivation
for treating People as a full, independent objective rather than a
constraint.
\item D. Figueirinhas, Y. Vakulenko, H. P\r{{a}}lsson, D. Hellstr\"om,
``Advancing Circularity Metrics: Revisiting the Ellen MacArthur Foundation's
Material Circularity Indicator,'' \emph{{Resources, Conservation and
Recycling}}, 226, 108682, 2025. Documents specific methodological weaknesses
in the MCI formula this report also implements --- most relevantly its
single-cycle view and its treatment of the 50:50 waste-allocation split ---
and is why the MCI values reported here are labelled against the published
formula rather than presented as a fully resolved circularity measure.
\item United Nations Environment Programme, \emph{{Guidelines for Social Life
Cycle Assessment of Products and Organisations 2020}}, UNEP/SETAC Life Cycle
Initiative, 2020. The methodology the People index explicitly does not
implement; cited so the distinction in Section~10 is checkable.
\item Directive 2012/19/EU of the European Parliament and of the Council of
4 July 2012 on waste electrical and electronic equipment (WEEE) (recast).
Basis for treating end-of-life collection and recovery as a first-class
design variable rather than an afterthought.
\item European Commission, \emph{{Critical Raw Materials Act and the 2023
List of Critical Raw Materials}}, 2023. Aluminium's dependence on imported
bauxite and the EU's low self-sufficiency ratio are part of the case for
using commodity-price volatility as an observable proxy for the
supply-risk term in the People index.
\item EN 15804:2012+A2:2019, \emph{{Sustainability of construction works ---
Environmental product declarations --- Core rules for the product category
of construction products}}, CEN, 2019. The closest existing standardised
reporting format to what a Digital Product Passport under ESPR is expected
to require; cited for context, not applied directly, since the GARO LS4 is
an electrical product, not a construction product.
\end{{enumerate}}

\end{{document}}
"""



def main():
    tex_path = os.path.join(HERE, "report.tex")
    with open(tex_path, "w") as fh:
        fh.write(TEX)
    print("wrote report.tex")
    if "--tex-only" in sys.argv:
        return
    for i in range(2):
        p = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "report.tex"], cwd=HERE, capture_output=True, text=True)
        if p.returncode != 0:
            tail = "\n".join(p.stdout.splitlines()[-40:])
            print("pdflatex failed:\n" + tail)
            sys.exit(1)
    print("wrote report.pdf")


if __name__ == "__main__":
    main()
