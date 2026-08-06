"""The CircuOpt product model: bill of materials, design space, and the three
objective functions (Planet, Prosperity, People).

Case product
------------
The GARO LS4, a ground-mounted AC electric-vehicle charging station: 24.5 kg
of product, a 15-year technical life and 20 000 charging sessions over that
life.  It is the class of low-voltage electrical product that falls squarely
inside the scope of the EU Ecodesign for Sustainable Products Regulation and
its Digital Product Passport requirements.

The product was chosen because its bill of materials is *published*.  Persson
& Erselius (2021) report the material distribution by mass, the product mass,
the technical lifetime and the cradle-to-grave result for this exact unit, so
the mass basis of every number below is citable rather than invented.

Provenance of every number
--------------------------
Each quantity below carries one of four provenance tags:

  DATA        read out of an open dataset at run time (DEFRA 2025, USGS MCS 2025)
  CITED       taken from the published LCA of the case product
  PROXY       an open-dataset value used to stand in for a material the dataset
              does not resolve; the scenario engine perturbs these hardest
  ASSUMPTION  an engineering estimate, stated here so a reviewer can replace it

No quantity is presented as measured when it is not.  The material masses are
CITED; their allocation across components is an ASSUMPTION constrained to
reproduce the published totals exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .datasets import (load_defra, load_usgs_prices, load_worldbank_prices,
                       load_garo_bom, expand_bom)
from .model_families import FAMILY_DEFRA, FAMILY_USGS
from .inventory import (IMPACT_INDICATORS, Inventory, add, scale, zero_vector,
                        material_circularity_indicator)

# --------------------------------------------------------------------------
# Material families
# --------------------------------------------------------------------------

#: Families whose recycled-content fraction is a decision variable.  Copper
#: conductors in a charging cable are specified to IEC 62893 conductivity and
#: are not, in practice, a recycled-content lever at the design stage; the
#: electronics family is an assembly, not a material.
DESIGN_FAMILIES = ("plastic", "steel", "aluminium")

#: End-of-life routes, in DEFRA's own vocabulary.
EOL_ROUTES = ("Closed-loop", "Open-loop",
              "Incineration with Energy Recovery", "Landfill")

#: Joining methods per interface: (label, assembly_s, disassembly_s, cost_usd,
#: reversible)
JOINTS = (
    ("snap-fit",   12.0,  25.0, 1.85, True),
    ("screwed",    38.0,  55.0, 0.60, True),
    ("bonded",     20.0, 300.0, 0.35, False),
)

LIFETIMES = (8, 10, 12, 15)          # years, design lifetime
REMAN_CYCLES = (0, 1, 2, 3)          # number of remanufacture interventions
#: Standby power class: (label, watts, unit cost premium USD, extra mass kg)
STANDBY_CLASSES = (("baseline", 1.0, 0.0, 0.000),
                   ("improved", 0.6, 3.50, 0.010),
                   ("best",     0.3, 9.00, 0.030))


@dataclass(frozen=True)
class Component:
    name: str
    mass_kg: float
    family: str
    priority_repair: bool = False   # counts toward the repairability index
    joint: int = -1                 # index into the joint decision vector, -1 = none


#: The bill of materials, built from a published manufacturer LCA rather than
#: invented here.  ``data/garo_ls4_bom.json`` records the material
#: distribution reported for the GARO LS4 charging station (Persson &
#: Erselius, 2021); the split of each material total across components is this
#: project's assumption, declared in that file and constrained so the family
#: totals reproduce the published percentages exactly.
GARO = load_garo_bom()

#: The Idemat process map, needed by the validation reconciliation below.
from .idemat import load_process_map as _load_pmap   # noqa: E402
GARO_MAP = _load_pmap()

#: (25y serviced life / 15y technical life) - 1, from GARO's own report.
#: See Assumptions.reman_life_extension and data/garo_ls4_bom.json ->
#: "service_and_lifetime" for the citation and its residual assumption.
_GARO_LIFE_EXTENSION_RATIO = (
    GARO["service_and_lifetime"]["serviced_operating_lifetime_years"]
    / GARO["service_and_lifetime"]["technical_lifetime_years"] - 1.0
)

BOM = tuple(
    Component(name, mass, family,
              name in GARO["priority_repair_components"],
              GARO["component_interfaces"].get(name, -1))
    for name, mass, family in expand_bom(GARO)
)

TOTAL_MASS = sum(c.mass_kg for c in BOM)


# --------------------------------------------------------------------------
# Assumptions, gathered in one place so they can be audited or overridden
# --------------------------------------------------------------------------

@dataclass
class Assumptions:
    # --- energy and use phase --------------------------------------------
    mfg_energy_kwh_per_kg: float = 2.4      # ASSUMPTION: assembly and forming
    joint_energy_kwh: tuple = (0.010, 0.004, 0.030)   # ASSUMPTION, per joint made
    #: DATA-anchored: the cited LCA reports 20 000 charging sessions over a
    #: 15-year technical life.  The energy per session is an ASSUMPTION.
    charging_sessions_over_life: float = 20000.0
    kwh_per_charging_session: float = 25.0  # ASSUMPTION
    conversion_loss_frac: float = 0.010     # ASSUMPTION: AC station switching losses
    hours_per_year: float = 8760.0

    # --- remanufacturing --------------------------------------------------
    #: CITED ratio, ASSUMED cycle count. GARO's own LCA report states the LS4's
    #: technical lifetime is 15 years but "with proper service" (periodic RCD
    #: replacement) its operating lifetime reaches about 25 years --
    #: (25-15)/15 = 0.667. The report does not say how many service
    #: interventions that assumes; this project maps it onto one modelled
    #: remanufacture cycle, so only the base ratio is cited, not the mapping.
    reman_life_extension: float = _GARO_LIFE_EXTENSION_RATIO
    reman_energy_kwh: float = 6.0           # ASSUMPTION, per intervention
    reman_part_replacement_frac: float = 0.18   # ASSUMPTION: mass renewed per cycle
    reman_labour_hours: float = 1.6         # ASSUMPTION, per intervention

    # --- cost -------------------------------------------------------------
    labour_rate_usd_per_hour: float = 34.0  # ASSUMPTION: EU manufacturing loaded rate
    plastic_virgin_usd_per_kg: float = 2.40      # ASSUMPTION: PC/ABS resin
    plastic_recyclate_price_ratio: float = 0.85  # ASSUMPTION
    metal_secondary_price_ratio: float = 0.72    # ASSUMPTION vs primary metal price
    steel_primary_multiple: float = 2.6     # ASSUMPTION: sheet vs No.1 heavy scrap
    #: Last-resort constant, used only if neither USGS nor World Bank data is
    #: available (e.g. offline with no cached files). In normal operation
    #: aluminium price comes from World Bank data -- see Context._market_price
    #: -- because the USGS download for aluminium was blocked at
    #: data-collection time and never actually resolved.
    aluminium_fallback_usd_per_kg: float = 2.55  # ASSUMPTION, last resort only
    #: Which live source Context._market_price prefers for globally-traded
    #: metals. "usgs" (default) uses USGS where available, falling back to
    #: World Bank; "worldbank" forces World Bank even where USGS has data.
    #: Used by the price cross-check sensitivity to test whether cost
    #: conclusions depend on which source is read.
    price_source: str = "usgs"
    electronics_usd_per_kg: float = 68.0    # ASSUMPTION: populated PCBA / contactor
    electricity_price_usd_per_kwh: float = 0.31  # ASSUMPTION: EU household tariff
    landfill_gate_fee_usd_per_kg: float = 0.11   # ASSUMPTION
    incineration_gate_fee_usd_per_kg: float = 0.09   # ASSUMPTION
    collection_cost_usd_per_kg: float = 0.35     # ASSUMPTION: WEEE take-back
    scrap_recovery_efficiency: float = 0.85 # ASSUMPTION: closed-loop yield

    #: Recyclate is not a free substitute for virgin feedstock.  Above a
    #: threshold, secondary material needs sorting to tighter tolerances,
    #: compatibilisers and property compensation, and it drives a higher
    #: process reject rate.  Without these two terms the optimiser drives
    #: recycled content to 1.0 in every design, because the DEFRA factor and
    #: the scrap price both reward it without limit - a bound-pinned variable
    #: that carries no information.  Both terms are ASSUMPTIONS and both are
    #: perturbed in the scenario ensemble.
    recyclate_threshold: float = 0.40       # ASSUMPTION: below this, no penalty
    recyclate_cost_penalty: float = 0.60    # ASSUMPTION: quadratic surcharge
    recyclate_reject_rate: float = 0.09     # ASSUMPTION: extra input mass at r=1
    open_loop_value_ratio: float = 0.45     # ASSUMPTION: downcycled value
    discount_rate: float = 0.05
    carbon_price_usd_per_tonne: float = 0.0 # nominal case: no internal carbon price

    # --- methodological choices, declared rather than assumed -------------
    #: Which inventory backend supplies the environmental numbers.
    #: "idemat" is process-level LCI with a full LCIA layer; "defra" is the
    #: single-indicator corporate-reporting factor set, kept for comparison.
    backend: str = "idemat"
    #: How the benefit of recycling is credited.  ISO 14044 requires this to
    #: be declared and its influence tested; experiment E6 does the testing.
    #: "cff" is the EU PEF Circular Footprint Formula's material term (see
    #: inventory.ALLOCATIONS for the formula and its stated scope limit).
    allocation: str = "cut_off"
    #: EU PEF CFF parameter A, used only when allocation="cff". 0.5 is the
    #: commonly cited PEF default, treated as a swept sensitivity value, not
    #: a verified material-specific figure.
    cff_allocation_factor: float = 0.5
    cff_quality_ratio: float = 1.0
    #: Grid the station is plugged into.  For a use-phase-dominated product
    #: this is one of the most consequential numbers in the whole model, so it
    #: is a declared parameter and not a constant.
    grid_region: str = "Sweden"
    #: Indicators carried through the calculation.
    indicators: tuple = IMPACT_INDICATORS
    #: Which of them the optimiser minimises as the Planet objective.
    impact_indicator: str = "gwp_kgco2e"

    # --- social performance proxy ------------------------------------------
    #: NOT a social LCA.  See the module docstring and the report: this is a
    #: transparent weighted index over four design-controllable proxies, of
    #: the kind used in ecodesign decision support, not a UNEP S-LCA study.
    repair_time_target_s: float = 300.0     # ASSUMPTION: EN 45554 style benchmark
    social_weights: tuple = (0.35, 0.20, 0.20, 0.25)  # repair, modularity,
                                                      # labour, supply-risk relief

    # --- environmental proxies -------------------------------------------
    proxy_multiplier: dict = field(default_factory=dict)  # family -> multiplier

    #: Ratio of the average grid carbon intensity over the service life to
    #: today's published DEFRA factor.  1.0 = a frozen grid; the scenario
    #: engine draws it from a decarbonisation-rate distribution.
    use_grid_multiplier: float = 1.0

    #: Minimum recycled content a design must carry to be placed on the
    #: market, as an ESPR-style delegated-act requirement.  0.0 = today.
    mandated_recycled_content: float = 0.0

    #: Scales the published duty cycle when the scenario engine varies how
    #: hard the station is actually worked.
    duty_multiplier: float = 1.0

    # --- Material Circularity Indicator (EMF 2015) -------------------------
    #: Efficiency of the recycling process that produces the recycled
    #: feedstock, and of the one that treats the product at end of life.
    mci_feedstock_efficiency: float = 0.90   # ASSUMPTION
    #: Industry-average lifetime for the utility term.  The published LCA
    #: gives 15 years for this product; a 10-year average for the product
    #: class is an ASSUMPTION and is varied in the ensemble.
    mci_industry_average_life_years: float = 10.0

    @property
    def energy_delivered_kwh_per_year(self) -> float:
        """Throughput per year, from the cited 20 000 sessions / 15 years."""
        published_life = GARO["product"]["technical_lifetime_years"]
        return (self.charging_sessions_over_life * self.kwh_per_charging_session
                / published_life) * self.duty_multiplier


@dataclass
class Context:
    """Everything the objective functions read, assembled once."""
    defra: object
    prices: dict
    worldbank_prices: dict = field(default_factory=dict)
    a: Assumptions = None
    criticality: dict = field(default_factory=dict)
    inv: object = None

    _cache = {}

    @staticmethod
    def build(assumptions: Assumptions | None = None) -> "Context":
        """Assemble a context.  The datasets are parsed once per process: the
        scenario engine builds thousands of contexts and re-reading a 3.5 MB
        workbook each time would dominate the runtime."""
        a = assumptions or Assumptions()
        if "defra" not in Context._cache:
            prices = load_usgs_prices()
            Context._cache["defra"] = load_defra()
            Context._cache["prices"] = prices
            Context._cache["worldbank_prices"] = load_worldbank_prices()
            Context._cache["criticality"] = _criticality_from_usgs(prices)
        key = (a.backend, a.allocation, a.grid_region, a.indicators,
              a.cff_allocation_factor, a.cff_quality_ratio)
        if key not in Context._cache:
            Context._cache[key] = Inventory.build(
                backend=a.backend, allocation=a.allocation,
                grid_region=a.grid_region, indicators=a.indicators,
                cff_allocation_factor=a.cff_allocation_factor,
                cff_quality_ratio=a.cff_quality_ratio)
        return Context(defra=Context._cache["defra"],
                       prices=Context._cache["prices"],
                       worldbank_prices=Context._cache["worldbank_prices"], a=a,
                       criticality=Context._cache["criticality"],
                       inv=Context._cache[key])

    # -- recyclate penalties -------------------------------------------------
    def reject_multiplier(self, recycled_frac: float) -> float:
        """Extra input mass needed to yield one kilogram of good part."""
        return 1.0 + self.a.recyclate_reject_rate * recycled_frac

    def quality_surcharge(self, recycled_frac: float) -> float:
        """Multiplier on the secondary-feedstock price above the threshold."""
        r0 = self.a.recyclate_threshold
        if recycled_frac <= r0:
            return 1.0
        x = (recycled_frac - r0) / max(1.0 - r0, 1e-9)
        return 1.0 + self.a.recyclate_cost_penalty * x * x

    # -- material factors ---------------------------------------------------
    def gwp_input(self, family: str, recycled_frac: float) -> float:
        """kg CO2e per kg of input material at the chosen recycled content."""
        name, _ = FAMILY_DEFRA[family]
        prim = self.defra.primary[name]
        closed = self.defra.closed_loop.get(name, prim)
        mult = self.a.proxy_multiplier.get(family, 1.0)
        return mult * ((1.0 - recycled_frac) * prim + recycled_frac * closed)

    def gwp_disposal(self, family: str, route: str) -> float:
        name, _ = FAMILY_DEFRA[family]
        table = self.defra.disposal.get(name, {})
        if route in table:
            return table[route]
        # DEFRA does not publish a closed-loop row for every material; where it
        # is absent the open-loop treatment burden is the closest published
        # figure, and the two are identical wherever both exist.
        return table.get("Open-loop", table.get("Landfill", 0.0))

    @property
    def grid(self) -> float:
        return self.defra.electricity

    def price(self, family: str, secondary: bool = False) -> float:
        """USD per kg of feedstock."""
        a = self.a
        if family == "plastic":
            p = a.plastic_virgin_usd_per_kg
            return p * a.plastic_recyclate_price_ratio if secondary else p
        if family == "electronics":
            return a.electronics_usd_per_kg
        if family == "steel":
            scrap = self.prices.get("Iron and Steel Scrap", {}).get("usd_per_kg")
            if scrap is None:
                scrap = 0.33
            return scrap if secondary else scrap * a.steel_primary_multiple
        if family == "copper":
            return self._market_price("Copper", 9.48) \
                * (a.metal_secondary_price_ratio if secondary else 1.0)
        if family == "aluminium":
            return self._market_price("Aluminum", a.aluminium_fallback_usd_per_kg) \
                * (a.metal_secondary_price_ratio if secondary else 1.0)
        raise KeyError(family)

    def _market_price(self, commodity: str, hardcoded_fallback: float) -> float:
        """USD/kg for a globally-traded commodity, preferring in order:

          1. USGS Mineral Commodity Summaries (this project's primary price
             source elsewhere)
          2. World Bank Commodity Markets, trailing 12-month mean (used here
             because USGS never actually resolved an aluminium price -- the
             download was blocked at data-collection time -- so aluminium ran
             on ``aluminium_fallback_usd_per_kg``, a bare guess, until this
             fallback was added)
          3. the hardcoded constant, only if both live sources are absent

        ``price_source`` on Assumptions can force World Bank even when USGS
        has data, which is what the price cross-check sensitivity (E-prices)
        uses to test whether cost conclusions depend on which source is read.
        """
        if self.a.price_source == "worldbank":
            wb = self.worldbank_prices.get(commodity)
            if wb is not None:
                return wb["usd_per_kg_trailing_12m_mean"]
        usgs = self.prices.get(commodity, {}).get("usd_per_kg")
        if usgs is not None:
            return usgs
        wb = self.worldbank_prices.get(commodity)
        if wb is not None:
            return wb["usd_per_kg_trailing_12m_mean"]
        return hardcoded_fallback


def _criticality_from_usgs(prices: dict) -> dict:
    """Derive a 0-1 supply-risk weight per family from USGS price volatility.

    The coefficient of variation of the published annual price series is used
    as an observable proxy for market/supply instability: the more violently a
    commodity has repriced over 2020-2024, the more a design that reduces its
    virgin intake is worth socially.  This is a data-driven stand-in for a
    formal criticality assessment, not a substitute for one.
    """
    out = {}
    for family, (commodity, _) in FAMILY_USGS.items():
        hist = prices.get(commodity, {}).get("history", {})
        vals = np.array(list(hist.values()), dtype=float)
        if vals.size >= 3 and vals.mean() > 0:
            out[family] = float(np.std(vals) / np.mean(vals))
        else:
            out[family] = 0.15   # ASSUMPTION where the series is unavailable
    out["plastic"] = 0.10        # ASSUMPTION: resin, not a critical raw material
    out["electronics"] = 0.30    # ASSUMPTION: assembly of several critical inputs
    # normalise to 0-1 across families
    hi = max(out.values())
    return {k: v / hi for k, v in out.items()}


# --------------------------------------------------------------------------
# Design encoding
# --------------------------------------------------------------------------
# x_cont : 3 recycled-content fractions, in DESIGN_FAMILIES order
# x_int  : [j0, j1, j2, lifetime, reman, eol_plastic, eol_steel, eol_alu,
#           eol_copper, eol_electronics, standby]
N_CONT = 3
INT_CARDINALITY = (3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3)
N_INT = len(INT_CARDINALITY)
EOL_FAMILIES = ("plastic", "steel", "aluminium", "copper", "electronics")


@dataclass
class Design:
    recycled: dict
    joints: tuple
    lifetime: int
    reman: int
    eol: dict
    standby_class: int

    @staticmethod
    def decode(xc, xi) -> "Design":
        xi = [int(v) for v in xi]
        return Design(
            recycled={f: float(np.clip(xc[i], 0.0, 1.0))
                      for i, f in enumerate(DESIGN_FAMILIES)},
            joints=tuple(xi[0:3]),
            lifetime=LIFETIMES[xi[3]],
            reman=REMAN_CYCLES[xi[4]],
            eol={f: EOL_ROUTES[xi[5 + i]] for i, f in enumerate(EOL_FAMILIES)},
            standby_class=xi[10],
        )

    def recycled_frac(self, family: str) -> float:
        return self.recycled.get(family, 0.0)

    def as_dict(self) -> dict:
        return {
            "recycled_content": {k: round(v, 4) for k, v in self.recycled.items()},
            "joints": [JOINTS[j][0] for j in self.joints],
            "design_lifetime_years": self.lifetime,
            "remanufacture_cycles": self.reman,
            "end_of_life_route": dict(self.eol),
            "standby_class": STANDBY_CLASSES[self.standby_class][0],
            "standby_w": STANDBY_CLASSES[self.standby_class][1],
        }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def service_years(d: Design, a: Assumptions) -> float:
    return d.lifetime * (1.0 + a.reman_life_extension * d.reman)


def disassembly_time_to_priority_parts(d: Design) -> float:
    """Seconds to reach the components a repairer actually needs to reach."""
    interfaces = {c.joint for c in BOM if c.priority_repair and c.joint >= 0}
    # the enclosure (interface 0) must always be opened first
    interfaces.add(0)
    return sum(JOINTS[d.joints[i]][2] for i in sorted(interfaces))


def constraint_violation(d: Design, mandate: float = 0.0) -> float:
    """Total violation, 0.0 when feasible.

    g1  A bonded enclosure cannot be reopened, so it is incompatible with any
        remanufacture cycle.
    g2  A bonded enclosure also prevents material separation, so closed-loop
        recycling of the plastic housings cannot be claimed.
    g3  Closed-loop recycling of a family with essentially no recycled content
        designed in is self-inconsistent for the families where content is a
        lever: claiming closed-loop return while specifying <10% recycled input
        breaks the loop the route presumes.
    """
    v = 0.0
    bonded_enclosure = JOINTS[d.joints[0]][0] == "bonded"
    if bonded_enclosure and d.reman > 0:
        v += float(d.reman)
    if bonded_enclosure and d.eol["plastic"] == "Closed-loop":
        v += 1.0
    for f in DESIGN_FAMILIES:
        if d.eol[f] == "Closed-loop" and d.recycled_frac(f) < 0.10:
            v += (0.10 - d.recycled_frac(f)) * 10.0
    if mandate > 0.0:
        # g4  ESPR-style minimum recycled content on the polymer housings,
        #     the part such delegated acts actually target.
        deficit = mandate - d.recycled_frac("plastic")
        if deficit > 0:
            v += deficit * 10.0
    return v


def evaluate(d: Design, ctx: Context) -> dict:
    """Full multi-indicator evaluation of one design.

    Every environmental quantity is an indicator *vector*, so the same
    calculation yields global warming, cumulative energy demand, the EF 3.1
    single score and the EF 3.1 minerals-and-metals score simultaneously.
    Which one becomes the Planet objective is a declared parameter, and
    experiment E5 tests whether the Pareto front reorders between them.
    """
    a = ctx.a
    inv = ctx.inv
    S = service_years(d, a)
    grid = inv.electricity()
    grid_use = scale(grid, a.use_grid_multiplier)

    def proxy(family, vec):
        """Apply the family's scenario proxy multiplier, if any."""
        m = a.proxy_multiplier.get(family, 1.0)
        return scale(vec, m) if m != 1.0 else vec

    # ---------------- Planet: materials ------------------------------------
    imp_materials = zero_vector(a.indicators)
    for c in BOM:
        r = d.recycled_frac(c.family)
        v = proxy(c.family, inv.material_input(c.family, r))
        imp_materials = add(imp_materials, v,
                            c.mass_kg * ctx.reject_multiplier(r))
        imp_materials = add(imp_materials, proxy(c.family, inv.forming(c.family)),
                            c.mass_kg)

    joints_made = sum(1 for c in BOM if c.joint >= 0)
    mfg_kwh = (a.mfg_energy_kwh_per_kg * TOTAL_MASS
               + sum(a.joint_energy_kwh[d.joints[c.joint]]
                     for c in BOM if c.joint >= 0))
    imp_manufacture = scale(grid, mfg_kwh)

    # ---------------- Planet: use phase ------------------------------------
    standby_w = STANDBY_CLASSES[d.standby_class][1]
    standby_kwh_yr = standby_w * a.hours_per_year / 1000.0
    throughput_loss_kwh_yr = a.energy_delivered_kwh_per_year * a.conversion_loss_frac
    use_kwh = S * (standby_kwh_yr + throughput_loss_kwh_yr)
    imp_use = scale(grid_use, use_kwh)

    # ---------------- Planet: remanufacture --------------------------------
    reman_mass = a.reman_part_replacement_frac * TOTAL_MASS
    imp_reman = zero_vector(a.indicators)
    if d.reman:
        per_cycle = scale(grid_use, a.reman_energy_kwh)
        per_cycle = add(per_cycle,
                        proxy("electronics", inv.material_input("electronics", 0.0)),
                        reman_mass * 0.35)      # ASSUMPTION: replacement mix
        per_cycle = add(per_cycle,
                        proxy("plastic",
                              inv.material_input("plastic", d.recycled_frac("plastic"))),
                        reman_mass * 0.65)
        imp_reman = scale(per_cycle, float(d.reman))

    # ---------------- Planet: end of life ----------------------------------
    imp_eol = zero_vector(a.indicators)
    for c in BOM:
        v = inv.end_of_life(c.family, d.eol[c.family], a.scrap_recovery_efficiency)
        imp_eol = add(imp_eol, proxy(c.family, v), c.mass_kg)

    impact_total = zero_vector(a.indicators)
    for part in (imp_materials, imp_manufacture, imp_use, imp_reman, imp_eol):
        impact_total = add(impact_total, part)

    gwp_total = impact_total.get("gwp_kgco2e", 0.0)
    planet = impact_total[a.impact_indicator] / S

    # ---------------- Prosperity -------------------------------------------
    material_cost = 0.0
    for c in BOM:
        r = d.recycled_frac(c.family)
        material_cost += c.mass_kg * ctx.reject_multiplier(r) * (
            (1.0 - r) * ctx.price(c.family, secondary=False)
            + r * ctx.price(c.family, secondary=True) * ctx.quality_surcharge(r)
        )

    assembly_s = sum(JOINTS[d.joints[c.joint]][1] for c in BOM if c.joint >= 0)
    joint_cost = sum(JOINTS[d.joints[c.joint]][3] for c in BOM if c.joint >= 0)
    assembly_cost = assembly_s / 3600.0 * a.labour_rate_usd_per_hour + joint_cost
    standby_premium = STANDBY_CLASSES[d.standby_class][2]

    capex = material_cost + assembly_cost + standby_premium

    r_disc = a.discount_rate

    def pv(amount, year):
        return amount / ((1.0 + r_disc) ** year)

    energy_cost_yr = (standby_kwh_yr + throughput_loss_kwh_yr) \
        * a.electricity_price_usd_per_kwh
    pv_energy = sum(pv(energy_cost_yr, t) for t in range(1, int(round(S)) + 1))

    pv_reman = 0.0
    for i in range(1, d.reman + 1):
        year = S * i / (d.reman + 1)
        cost = (a.reman_labour_hours * a.labour_rate_usd_per_hour
                + reman_mass * ctx.price("electronics") * 0.35
                + reman_mass * ctx.price("plastic") * 0.65
                + disassembly_time_to_priority_parts(d) / 3600.0
                * a.labour_rate_usd_per_hour)
        pv_reman += pv(cost, year)

    eol_cash = a.collection_cost_usd_per_kg * TOTAL_MASS
    for c in BOM:
        route = d.eol[c.family]
        sec = ctx.price(c.family, secondary=True)
        if route == "Closed-loop":
            eol_cash -= c.mass_kg * sec * a.scrap_recovery_efficiency
        elif route == "Open-loop":
            eol_cash -= c.mass_kg * sec * a.open_loop_value_ratio
        elif route == "Incineration with Energy Recovery":
            eol_cash += c.mass_kg * a.incineration_gate_fee_usd_per_kg
        else:
            eol_cash += c.mass_kg * a.landfill_gate_fee_usd_per_kg
    pv_eol = pv(eol_cash, S)

    carbon_cost = gwp_total / 1000.0 * a.carbon_price_usd_per_tonne

    npv_cost = capex + pv_energy + pv_reman + pv_eol + carbon_cost
    annuity = (1.0 - (1.0 + r_disc) ** (-S)) / r_disc
    lcc_annual = npv_cost / annuity

    # ---------------- People (a proxy index, not a social LCA) -------------
    t_dis = disassembly_time_to_priority_parts(d)
    s_repair = float(np.clip(1.0 - t_dis / (2.0 * a.repair_time_target_s), 0.0, 1.0))
    s_modular = float(np.mean([1.0 if JOINTS[j][4] else 0.0 for j in d.joints]))
    reman_hours_per_service_year = d.reman * a.reman_labour_hours / S
    s_labour = float(np.clip(reman_hours_per_service_year / 0.25, 0.0, 1.0))
    s_supply = float(np.clip(
        sum(ctx.criticality[f] * d.recycled_frac(f) for f in DESIGN_FAMILIES)
        / max(sum(ctx.criticality[f] for f in DESIGN_FAMILIES), 1e-9),
        0.0, 1.0))

    w = a.social_weights
    social = (w[0] * s_repair + w[1] * s_modular
              + w[2] * s_labour + w[3] * s_supply) / sum(w)

    # ---------------- Material Circularity Indicator (EMF 2015) ------------
    mass_recycled_feedstock = sum(c.mass_kg * d.recycled_frac(c.family)
                                  for c in BOM)
    FR = mass_recycled_feedstock / TOTAL_MASS
    mass_collected = sum(
        c.mass_kg for c in BOM
        if d.eol[c.family] in ("Closed-loop", "Open-loop"))
    CR = mass_collected / TOTAL_MASS
    mci = material_circularity_indicator(
        mass_kg=TOTAL_MASS,
        recycled_feedstock=FR,
        reused_feedstock=0.0,
        collected_for_recycling=CR,
        collected_for_reuse=0.0,
        recycling_efficiency_eol=a.scrap_recovery_efficiency,
        recycling_efficiency_feedstock=a.mci_feedstock_efficiency,
        lifetime_years=S,
        industry_average_lifetime_years=a.mci_industry_average_life_years,
    )

    return {
        "f_planet": planet,
        "f_planet_indicator": a.impact_indicator,
        "f_planet_kgco2e_per_service_year": gwp_total / S,
        "f_cost_usd_per_service_year": lcc_annual,
        "f_social_index": social,
        "service_years": S,
        "impact_total": impact_total,
        "impact_per_service_year": {k: v / S for k, v in impact_total.items()},
        "impact_breakdown": {
            "materials": imp_materials, "manufacture": imp_manufacture,
            "use": imp_use, "remanufacture": imp_reman, "end_of_life": imp_eol,
        },
        "gwp_total_kgco2e": gwp_total,
        "gwp_breakdown": {
            "materials": imp_materials.get("gwp_kgco2e", 0.0),
            "manufacture": imp_manufacture.get("gwp_kgco2e", 0.0),
            "use": imp_use.get("gwp_kgco2e", 0.0),
            "remanufacture": imp_reman.get("gwp_kgco2e", 0.0),
            "end_of_life": imp_eol.get("gwp_kgco2e", 0.0),
        },
        "npv_cost_usd": npv_cost,
        "cost_breakdown": {
            "capex": capex, "material": material_cost, "assembly": assembly_cost,
            "standby_premium": standby_premium, "pv_energy": pv_energy,
            "pv_remanufacture": pv_reman, "pv_end_of_life": pv_eol,
            "carbon": carbon_cost,
        },
        "social_breakdown": {"repairability": s_repair, "modularity": s_modular,
                             "reman_labour": s_labour, "supply_risk_relief": s_supply},
        "disassembly_time_s": t_dis,
        "material_circularity_indicator": mci["mci"],
        "mci_components": mci,
        "constraint_violation": constraint_violation(d, a.mandated_recycled_content),
        "joints_made": joints_made,
    }


def objectives(xc, xi, ctx: Context):
    """Return (f, violation) with all three objectives posed for minimisation."""
    d = Design.decode(xc, xi)
    r = evaluate(d, ctx)
    f = np.array([r["f_planet"],
                  r["f_cost_usd_per_service_year"],
                  -r["f_social_index"]], dtype=float)
    return f, r["constraint_violation"]


def baseline_design() -> Design:
    """A linear-economy reference: bonded enclosure, virgin materials, landfill.

    This is the 'take-make-waste' incumbent every result is quoted against.
    """
    return Design(
        recycled={f: 0.0 for f in DESIGN_FAMILIES},
        joints=(2, 1, 2),
        lifetime=8,
        reman=0,
        eol={f: "Landfill" for f in EOL_FAMILIES},
        standby_class=0,
    )


# --------------------------------------------------------------------------
# Validation against the published LCA of the same product
# --------------------------------------------------------------------------

def validate_against_published(ctx: Context) -> dict:
    """Compare this model's cradle-to-gate material result with the published
    one for the same physical unit.

    The two are *not* expected to agree exactly, and the disagreement is the
    informative part.  The published study used ICE v3.0 and the European EF
    database with supplier-specific data; this model uses DEFRA 2025 company
    reporting factors with documented proxies.  Two independent factor sets
    applied to the same cited mass basis is the only external check available
    without a licensed LCI database, so it is reported rather than hidden.
    """
    pub = GARO["published_results"]
    a = ctx.a

    per_family = {}
    for c in BOM:
        f = ctx.inv.material_input(c.family, 0.0)["gwp_kgco2e"]
        per_family[c.family] = per_family.get(c.family, 0.0) + c.mass_kg * f
    model_materials = sum(per_family.values())

    # factor-level cross-check: the published study's own factors vs DEFRA's
    pubf = GARO["published_emission_factors_gco2e_per_kg"]
    def mf(fam):
        return ctx.inv.material_input(fam, 0.0)["gwp_kgco2e"]

    cross = {
        "aluminium": (mf("aluminium"), pubf["Aluminium"] / 1000.0),
        "steel": (mf("steel"), pubf["Steel"] / 1000.0),
        "cable (copper/plastic)": (0.5 * mf("copper") + 0.5 * mf("plastic"),
                                   pubf["Copper/plastic"] / 1000.0),
        "plastic": (mf("plastic"), pubf["General plastic"] / 1000.0),
    }

    # -- geography reconciliation ------------------------------------------
    # The single largest disagreement is the aluminium factor.  Idemat resolves
    # both a European and a global primary route; the published study's value
    # sits close to the global one.  Recomputing with the global route isolates
    # how much of the gap is smelting geography rather than error.
    recon = None
    if ctx.a.backend == "idemat":
        # The model defaults to the global aluminium route (see
        # data/idemat_process_map.json -> families.aluminium.virgin for the
        # evidence). The European route is kept only as a labelled
        # comparison, not discarded.
        alu_default = ctx.inv.material_input("aluminium", 0.0)["gwp_kgco2e"]
        alu_eu_ref = ctx.inv._vec(
            GARO_MAP["families"]["aluminium"]["virgin_european"]["code"])["gwp_kgco2e"]
        alu_mass = sum(c.mass_kg for c in BOM if c.family == "aluminium")
        with_european = model_materials + alu_mass * (alu_eu_ref - alu_default)
        recon = {
            "aluminium_default_global_kgco2e_per_kg": alu_default,
            "aluminium_european_reference_kgco2e_per_kg": alu_eu_ref,
            "published_study_aluminium_kgco2e_per_kg": pubf["Aluminium"] / 1000.0,
            "model_materials_with_european_aluminium_kgco2e": with_european,
            "ratio_with_european_aluminium": with_european
            / pub["raw_material_and_component_production_kgco2e"],
            "note": ("The model defaults to the global-average primary "
                     "aluminium route because GARO's own LCA report shows an "
                     "internationally distributed supplier base and names no "
                     "specific smelter for the aluminium itself -- an "
                     "evidence-based choice, made before checking whether it "
                     "would improve agreement with the published figure. That "
                     "it also sits closer to the published study's own "
                     "aluminium factor than the European route would is a "
                     "consequence of that choice, not the reason for it. The "
                     "European route is shown here as a labelled alternative: "
                     "using it instead would move the model further from the "
                     "published figure, not closer."),
        }

    return {
        "geography_reconciliation": recon,
        "published_materials_kgco2e": pub["raw_material_and_component_production_kgco2e"],
        "model_materials_kgco2e": model_materials,
        "ratio_model_over_published": model_materials
        / pub["raw_material_and_component_production_kgco2e"],
        "model_material_shares": {k: v / model_materials
                                  for k, v in sorted(per_family.items(),
                                                     key=lambda x: -x[1])},
        "published_hotspot_note": pub["note_material_hotspots"],
        "backend": ctx.a.backend,
        "factor_cross_check_kgco2e_per_kg": {
            k: {"defra_model": round(m, 3), "published_study": round(p, 3),
                "ratio": round(m / p, 3)}
            for k, (m, p) in cross.items()
        },
        "published_total_kgco2e": pub["total_kgco2e_per_unit"],
        "published_kgco2e_per_kg": pub["kgco2e_per_kg_of_product"],
        "model_kgco2e_per_kg_materials_only": model_materials / TOTAL_MASS,
    }
