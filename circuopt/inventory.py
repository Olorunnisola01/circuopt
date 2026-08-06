"""The life-cycle inventory and impact-assessment layer.

This module is the only place that knows where an environmental number comes
from.  Everything above it works in terms of *indicator vectors*: a dict from
indicator name to value, added and scaled like any other quantity.

Two things it makes explicit that the earlier single-factor version could not.

Allocation convention
---------------------
How the benefit of recycling is credited is a *choice*, not a fact, and ISO
14044 requires it to be declared and tested.  Two conventions are implemented:

``cut_off``      Recycled input carries only the burden of the recycling
                 process.  Material leaving the system at end of life earns
                 nothing.  Benefit appears entirely on the input side.  This
                 is the convention DEFRA's factors are built on.

``substitution`` Material recovered at end of life earns a credit equal to the
                 virgin production it displaces, scaled by recovery
                 efficiency.  Benefit appears on the output side, and
                 end-of-life route choice becomes environmentally live rather
                 than a rounding error.

Under ``cut_off`` the end-of-life decision variable barely moves the
environmental objective, which is a structural distortion in a study about
circularity.  Reporting both, and showing whether the Pareto front reorders
between them, is the honest treatment.

Backend
-------
``idemat``  process-level LCI with a full LCIA layer (default)
``defra``   the single-indicator corporate-reporting factors, kept so the two
            can be compared rather than one silently replacing the other
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .idemat import (IMPACT_INDICATORS, INDICATOR_LABELS, load_idemat,
                     load_process_map, resolve, augment_with_synthetics)
from .datasets import load_defra

MJ_PER_KWH = 3.6

#: Families whose recycled content is a design variable.
DESIGN_FAMILIES = ("plastic", "steel", "aluminium")
#: Every family appearing in the bill of materials.
FAMILIES = ("plastic", "steel", "aluminium", "copper", "electronics")

EOL_ROUTES = ("Closed-loop", "Open-loop",
              "Incineration with Energy Recovery", "Landfill")

#: "cff" implements the material (input-side) term of the EU PEF Circular
#: Footprint Formula (European Commission JRC, Product Environmental
#: Footprint method) rather than a third independently invented convention:
#:
#:   E = (1 - R1)*Ev + R1*[A*Erecycled + (1-A)*Ev*(Qsin/Qp)]
#:
#: R1 = recycled_frac (the model's own decision variable), Ev/Erecycled = the
#: virgin/secondary production processes already in this module, A = the
#: market-based allocation factor (Inventory.cff_allocation_factor),
#: Qsin/Qp = the secondary-to-primary quality ratio (Inventory.cff_quality_ratio,
#: default 1.0: quality-equivalent, since Idemat's secondary-production
#: factors already represent realistic output, not degraded material).
#:
#: SCOPE LIMIT, stated rather than silently exceeded: this implements only
#: the CFF's material formula.  The formula's energy-recovery and disposal
#: terms (parameters B, LHV, XER, ESE, EER) could not be verified against a
#: primary EU/JRC source at the time of writing -- several official PDFs
#: returned 403 errors during research for this feature -- and guessing a
#: regulatory formula would be worse than not implementing it.  End-of-life
#: treatment under "cff" therefore falls back to the "substitution"
#: convention, which is documented here as an approximation, not a citation.
ALLOCATIONS = ("cut_off", "substitution", "cff")


def zero_vector(indicators=IMPACT_INDICATORS) -> dict:
    return {k: 0.0 for k in indicators}


def add(a: dict, b: dict, scale: float = 1.0) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v * scale
    return out


def scale(a: dict, s: float) -> dict:
    return {k: v * s for k, v in a.items()}


@dataclass
class Inventory:
    """Indicator vectors per kilogram of material and per kWh of electricity."""

    backend: str = "idemat"
    allocation: str = "cut_off"
    grid_region: str = "Sweden"
    indicators: tuple = IMPACT_INDICATORS
    #: EU PEF Circular Footprint Formula parameter A, used only when
    #: allocation="cff".  0.5 is the commonly cited PEF default for
    #: market-based allocation when material-specific data is unavailable;
    #: it is not verified here as the correct value for aluminium, steel,
    #: copper or plastic specifically, so it is treated as a swept sensitivity
    #: parameter (A in {0, 0.5, 1}) rather than asserted as this product's
    #: true allocation factor.
    cff_allocation_factor: float = 0.5
    #: EU PEF Qsin/Qp quality ratio, used only when allocation="cff".
    cff_quality_ratio: float = 1.0

    _proc: dict = field(default_factory=dict, repr=False)
    _pmap: dict = field(default_factory=dict, repr=False)
    _defra: object = None
    _cache: dict = field(default_factory=dict, repr=False)

    # -- construction -------------------------------------------------------
    _shared = {}

    @staticmethod
    def build(backend="idemat", allocation="cut_off", grid_region="Sweden",
              indicators=IMPACT_INDICATORS, cff_allocation_factor=0.5,
              cff_quality_ratio=1.0) -> "Inventory":
        if allocation not in ALLOCATIONS:
            raise ValueError(f"allocation must be one of {ALLOCATIONS}")
        if backend not in ("idemat", "defra"):
            raise ValueError("backend must be 'idemat' or 'defra'")

        s = Inventory._shared
        if "idemat" not in s:
            procs = load_idemat()
            # No Idemat process represents a charging-station metering/RCD
            # board; see data/idemat_process_map.json ->
            # families.electronics.virgin for the citation and the
            # "excluding ICs" caveat.
            augment_with_synthetics(procs)
            pmap = load_process_map()
            resolve(procs, pmap)          # fail loudly if the map is stale
            s["idemat"] = procs
            s["pmap"] = pmap
            s["defra"] = load_defra()

        return Inventory(backend=backend, allocation=allocation,
                         grid_region=grid_region, indicators=tuple(indicators),
                         cff_allocation_factor=cff_allocation_factor,
                         cff_quality_ratio=cff_quality_ratio,
                         _proc=s["idemat"], _pmap=s["pmap"], _defra=s["defra"])

    # -- helpers ------------------------------------------------------------
    def _vec(self, code: str) -> dict:
        p = self._proc[code]
        return {k: p.indicators.get(k, 0.0) for k in self.indicators}

    def _role(self, family: str, role: str) -> dict:
        spec = self._pmap["families"][family][role]
        return self._vec(spec["code"])

    @property
    def grid_regions(self) -> list:
        return [k for k in self._pmap["grids"] if not k.startswith("_")]

    # -- public API ---------------------------------------------------------
    def electricity(self, region: str | None = None) -> dict:
        """Indicator vector per kWh delivered."""
        region = region or self.grid_region
        grids = self._pmap["grids"]
        if region not in grids:
            raise KeyError(f"unknown grid region {region!r}; "
                           f"have {self.grid_regions}")
        if self.backend == "defra":
            # DEFRA publishes only UK, and only GWP.
            v = zero_vector(self.indicators)
            if "gwp_kgco2e" in v:
                v["gwp_kgco2e"] = self._defra.electricity
            return v
        return scale(self._vec(grids[region]["code"]), MJ_PER_KWH)

    def material_input(self, family: str, recycled_frac: float) -> dict:
        """Indicator vector per kg of feedstock at the chosen recycled content.

        A linear blend of the primary and secondary production processes.  The
        blend is linear because the two processes are alternative routes to the
        same kilogram; the *non*-linearity of recyclate use is a quality and
        cost effect and lives in the model, not here.
        """
        r = min(max(recycled_frac, 0.0), 1.0)
        if self.backend == "defra":
            return self._defra_material(family, r)

        fam = self._pmap["families"][family]
        virgin = self._role(family, "virgin")
        if "secondary" not in fam:
            return virgin
        secondary = self._role(family, "secondary")

        if self.allocation == "cff":
            # EU PEF Circular Footprint Formula, material term:
            # E = (1-R1)*Ev + R1*[A*Erecycled + (1-A)*Ev*(Qsin/Qp)]
            A = self.cff_allocation_factor
            Q = self.cff_quality_ratio
            return {k: (1.0 - r) * virgin[k]
                    + r * (A * secondary[k] + (1.0 - A) * virgin[k] * Q)
                    for k in virgin}

        # cut_off and substitution share the same input-side treatment
        # (their difference is entirely at end-of-life, below); this is the
        # CFF's A=1 special case.
        return {k: (1.0 - r) * virgin[k] + r * secondary[k] for k in virgin}

    def _defra_material(self, family: str, r: float) -> dict:
        from .model_families import FAMILY_DEFRA
        name, _tag = FAMILY_DEFRA[family]
        prim = self._defra.primary[name]
        closed = self._defra.closed_loop.get(name, prim)
        v = zero_vector(self.indicators)
        if "gwp_kgco2e" in v:
            v["gwp_kgco2e"] = (1.0 - r) * prim + r * closed
        return v

    def end_of_life(self, family: str, route: str,
                    recovery_efficiency: float = 0.85) -> dict:
        """Indicator vector per kg leaving the system by ``route``.

        Under ``cut_off`` this is the treatment burden only.  Under
        ``substitution`` a closed-loop route additionally earns the recycling
        credit, scaled by how much material the process actually recovers.
        Under ``cff`` end-of-life falls back to the ``substitution``
        treatment: the CFF's own disposal formula (parameters B, LHV, XER,
        ESE, EER) could not be verified against a primary EU/JRC source, so
        it is not implemented, and this fallback is an approximation rather
        than a citation -- see the ALLOCATIONS module docstring.
        """
        if route not in EOL_ROUTES:
            raise KeyError(route)
        if self.backend == "defra":
            return self._defra_eol(family, route)

        waste = self._pmap["families"]["waste"]
        fam = self._pmap["families"][family]
        v = zero_vector(self.indicators)

        if route in ("Closed-loop", "Open-loop"):
            v = add(v, self._vec(waste["scrap_collection_metal"]["code"]))
        if route == "Incineration with Energy Recovery":
            if "incineration" in fam:
                v = add(v, self._role(family, "incineration"))
            else:
                # Metals are not combusted; they report to slag and are
                # treated as landfilled inert material.
                v = add(v, self._vec(waste["landfill_inert"]["code"]))
        if route == "Landfill":
            v = add(v, self._vec(waste["landfill_inert"]["code"]))

        if self.allocation in ("substitution", "cff"):
            if route == "Closed-loop" and "recycling_credit" in fam:
                v = add(v, self._role(family, "recycling_credit"),
                        scale=recovery_efficiency)
            elif route == "Open-loop":
                key = ("plastics_open_loop_credit" if family == "plastic"
                       else "metals_open_loop_credit")
                v = add(v, self._vec(waste[key]["code"]))
        return v

    def _defra_eol(self, family: str, route: str) -> dict:
        from .model_families import FAMILY_DEFRA
        name, _ = FAMILY_DEFRA[family]
        table = self._defra.disposal.get(name, {})
        val = table.get(route, table.get("Open-loop", table.get("Landfill", 0.0)))
        v = zero_vector(self.indicators)
        if "gwp_kgco2e" in v:
            v["gwp_kgco2e"] = val
        return v

    def forming(self, family: str) -> dict:
        fam = self._pmap["families"].get(family, {})
        if "forming" in fam and self.backend == "idemat":
            return self._role(family, "forming")
        return zero_vector(self.indicators)

    # -- provenance ---------------------------------------------------------
    def describe(self) -> dict:
        out = {"backend": self.backend, "allocation": self.allocation,
               "grid_region": self.grid_region,
               "indicators": list(self.indicators),
               "indicator_labels": {k: INDICATOR_LABELS.get(k, k)
                                    for k in self.indicators}}
        if self.backend == "idemat":
            rows = []
            for family, roles in self._pmap["families"].items():
                for role, spec in roles.items():
                    if role.startswith("_"):
                        continue
                    p = self._proc[spec["code"]]
                    rows.append({"family": family, "role": role,
                                 "code": spec["code"], "process": p.name,
                                 "unit": p.unit, "note": spec.get("note", ""),
                                 **{k: p.indicators.get(k) for k in self.indicators}})
            out["processes"] = rows
            g = self._pmap["grids"][self.grid_region]
            out["grid_process"] = {"region": self.grid_region,
                                   "code": g["code"],
                                   "process": self._proc[g["code"]].name}
            out["grid_alternatives"] = {
                r: scale(self._vec(self._pmap["grids"][r]["code"]), MJ_PER_KWH)
                for r in self.grid_regions}
        return out


# --------------------------------------------------------------------------
# Material Circularity Indicator, as published
# --------------------------------------------------------------------------

def material_circularity_indicator(mass_kg: float, recycled_feedstock: float,
                                   reused_feedstock: float,
                                   collected_for_recycling: float,
                                   collected_for_reuse: float,
                                   recycling_efficiency_eol: float,
                                   recycling_efficiency_feedstock: float,
                                   lifetime_years: float,
                                   industry_average_lifetime_years: float,
                                   functional_units: float = 1.0,
                                   industry_average_units: float = 1.0) -> dict:
    """Ellen MacArthur Foundation Material Circularity Indicator.

    Implemented to the published formulation (Ellen MacArthur Foundation and
    Granta Design, *Circularity Indicators: An Approach to Measuring
    Circularity -- Methodology*, 2015) rather than to a simplification:

        V   = M (1 - F_R - F_U)                     virgin feedstock
        W_0 = M (1 - C_R - C_U)                     unrecoverable at end of life
        W_C = M C_R (1 - E_C)                       losses in EoL recycling
        W_F = M F_R (1 - E_F) / E_F                 losses making the feedstock
        W   = W_0 + (W_F + W_C) / 2
        LFI = (V + W) / (2M + (W_F - W_C) / 2)
        X   = (L / L_av) (U / U_av)
        F(X)= 0.9 / X
        MCI = max(0, 1 - LFI * F(X))

    Returns the components as well as the index, because a single number
    between 0 and 1 tells a reader nothing about which term drove it.
    """
    M = float(mass_kg)
    FR, FU = float(recycled_feedstock), float(reused_feedstock)
    CR, CU = float(collected_for_recycling), float(collected_for_reuse)
    EC, EF = float(recycling_efficiency_eol), float(recycling_efficiency_feedstock)

    if FR + FU > 1.0 + 1e-9:
        raise ValueError("recycled + reused feedstock exceeds 1")
    if CR + CU > 1.0 + 1e-9:
        raise ValueError("collected for recycling + reuse exceeds 1")
    EF = max(EF, 1e-6)

    V = M * (1.0 - FR - FU)
    W0 = M * (1.0 - CR - CU)
    WC = M * CR * (1.0 - EC)
    WF = M * FR * (1.0 - EF) / EF
    W = W0 + (WF + WC) / 2.0

    denom = 2.0 * M + (WF - WC) / 2.0
    LFI = (V + W) / denom if denom > 0 else 1.0

    X = (lifetime_years / max(industry_average_lifetime_years, 1e-9)) * \
        (functional_units / max(industry_average_units, 1e-9))
    FX = 0.9 / max(X, 1e-9)

    mci = max(0.0, 1.0 - LFI * FX)
    return {"mci": mci, "linear_flow_index": LFI, "utility_X": X,
            "virgin_kg": V, "unrecoverable_kg": W,
            "waste_eol_recycling_kg": WC, "waste_feedstock_kg": WF}
