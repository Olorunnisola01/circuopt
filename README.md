# CircuOpt — three-objective optimisation for circular product ecodesign

A multi-objective optimisation framework that decides, quantitatively, whether
circular design choices pay — for one concrete product, on data anyone can
download, with every assumption labelled.

The case product is the **GARO LS4 ground-mounted AC electric-vehicle charging
station** — chosen because its bill of materials is *published*. Persson &
Erselius (2021) report the material distribution by mass, the 24.5 kg product
mass, a 15-year technical life and 20 000 charging sessions for this exact
unit, so the mass basis of every result here is citable rather than invented.

Fourteen design variables are optimised simultaneously against **Planet**
(cradle-to-grave carbon per service-year), **Prosperity** (annualised
life-cycle cost) and **People** (a composite repairability, modularity,
circular-labour and supply-risk index), then stress-tested across a
Monte-Carlo ensemble of futures.

**Headline:** most of the Pareto archive beats a linear take-make-waste
baseline on carbon *and* cost at the same time. See `report.pdf` for the run
that produced that number, and for the limitations that bound it.

---

## Contents

| File | Purpose |
|---|---|
| `circuopt/datasets.py` | Parsers for the two open datasets; caches to `data/factors_cache.json` |
| `circuopt/model.py` | Bill of materials, design encoding, the three objective functions, constraints |
| `circuopt/nsga3.py` | Self-contained NSGA-III (Das–Dennis references, hybrid operators, constraint domination) + hypervolume and spacing |
| `circuopt/scenarios.py` | Monte-Carlo future sampling, robustness scoring, decision-maker profiles |
| `run_experiments.py` | Experimental protocol E0–E4; writes `results.json` and `figures/` |
| `make_report.py` | Generates `report.tex` → `report.pdf` **from `results.json`** |
| `test_circuopt.py` | 56 regression tests, no pytest required |
| `report.pdf` | The write-up: data, formulation, method, results, limitations |
| `data/` | The raw open datasets, as downloaded |

## Quick start

```bash
python3 test_circuopt.py
```

```bash
python3 run_experiments.py
```

```bash
python3 make_report.py
```

Add `--quick` to `run_experiments.py` for a reduced-budget run (~2.5 min)
instead of the full protocol.

Requires Python 3.8+, NumPy, Matplotlib, and `openpyxl` (to read the DEFRA
workbook). `pdflatex` is needed only for `make_report.py`.

---

## The data

Both datasets are freely redistributable, with no registration and no licence
fee. That was a hard requirement: a framework nobody can re-run is not a
framework.

**UK Government GHG Conversion Factors for Company Reporting 2025**
(DESNZ/DEFRA, Open Government Licence v3.0) — `data/ghg-conversion-factors-2025-flat-format.xlsx`

The *Material use* category publishes both a **primary material production**
factor and a **closed-loop source** factor per material. That pairing is what
makes recycled content an optimisable variable with a government-published
environmental effect, rather than a qualitative claim. The *Waste disposal*
category gives a factor per end-of-life route, and the electricity tables give
the grid intensity for the use phase.

**USGS Mineral Commodity Summaries 2025** salient-statistics data releases
(public domain) — `data/usgs/mcs2025-*_salient.csv`

Annual commodity prices for the life-cycle costing, plus a five-year price
history that is used twice: to calibrate price shocks in the scenario
ensemble, and — through its coefficient of variation — as an observable
supply-risk weight in the People objective.

### Provenance discipline

Every quantity carries one of three tags:

- **DATA** — read out of an open dataset at run time
- **PROXY** — the closest published material used as a stand-in (copper and
  populated electronics have no DEFRA row); perturbed hardest in the ensemble
- **ASSUMPTION** — an engineering estimate, declared in `Assumptions` so a
  reviewer can replace it

### Datasets

| | Dataset | Licence | Role |
|---|---|---|---|
| LCI/LCIA | **Idemat 2026** (TU Delft) | free with attribution, *fetched not redistributed* | process-level inventory; GWP, CED, EF 3.1, EF minerals+metals; primary/secondary process pairs; end-of-life substitution credits |
| GHG | UK DESNZ/DEFRA 2025 | Open Government Licence v3.0 | second, independent backend kept for comparison |
| Prices | USGS MCS 2025 | US public domain | commodity prices and the measured volatility used in the ensemble |
| BOM | GARO LS4 LCA (Persson & Erselius 2021) | published manufacturer LCA | the cited mass basis |

Run `python3 scripts/fetch_idemat.py` once before `run_experiments.py`.

### The methodological choices are tested, not assumed

| Choice | Tested by | Does it change the answer? |
|---|---|---|
| impact category (4 of them) | E5, rank correlation | ranking mostly agrees; best design sometimes differs |
| **end-of-life allocation** | E6, cut-off vs substitution | **yes — the conventions rank the archive almost independently** |
| grid geography (6 markets) | E9 | level moves 13×; *ranking* is stable |
| recyclate constants (3, unmeasured) | E7, OAT + corners | optimum exists broadly; its *location* is not identified |
| social weights | E8, Dirichlet redraw | ranking mostly stable |
| scenario widths | E11, ×0.5 to ×2 | not load-bearing |
| algorithm choice | E10, Mann–Whitney + effect size | beats random search; NSGA-II not separated |

**The binding limitation** is the allocation convention: cut-off and
substitution disagree on which design is best, so every end-of-life conclusion
is conditional on the convention quoted with it.

Other gaps, stated rather than hidden: the published study gives mass by
material, not by component, so the allocation of each material total across
parts is this project's assumption (constrained to reproduce the published
family totals exactly, and tested); the People index is a transparent proxy and
explicitly *not* a UNEP S-LCA; nothing is validated against measurement; no open index exists
for engineering-polymer prices; the USGS aluminium release was not retrievable
during the recorded run and falls back to a declared assumption; USGS prices
are US while DEFRA factors are UK.

---

## Method

**NSGA-III**, not NSGA-II: with three objectives crowding distance is a weak
diversity signal, and reference-point niching gives a natural place to hang
decision-maker preferences — each stakeholder profile is just a different
direction on the same normalised hyperplane.

The design vector is hybrid (3 continuous recycled-content fractions, 11
categorical genes), so variation is hybrid: SBX + polynomial mutation on the
continuous genes, uniform crossover + random reset on the categorical ones.
Constraints use Deb's feasibility rules, so violation is never traded against
objective value.

**Robustness** is the share of *future × preference* draws in which a design
lands in the best decile by augmented achievement scalarisation, with
preference weights redrawn from a Dirichlet. The more obvious metric — how
often a design stays non-dominated inside the archive — is nearly vacuous
here, because the archive is already a mutually non-dominated set; it
saturates near 100% and ranks nothing. `report.pdf` shows both.

**Constraints** encode couplings a naive model would violate silently: a
bonded enclosure cannot be reopened (so it forbids remanufacturing) and cannot
be separated (so it forbids a closed-loop claim on the housings); a
closed-loop route with under 10% recycled input is self-inconsistent; and an
ESPR-style minimum recycled content binds in the scenarios that impose one.

---

## Reproducibility

`make_report.py` reads `results.json` and writes every number into the PDF.
Nothing is transcribed by hand, so the document cannot drift from the run that
produced it. Re-run the protocol and the report updates itself.
