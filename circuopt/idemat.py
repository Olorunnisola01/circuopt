"""Idemat 2026 parser: the multi-indicator life-cycle inventory backend.

Why this exists
---------------
The DEFRA backend gives one number per material: a global-warming factor from
a *corporate GHG reporting* dataset.  That is enough to compute a carbon
footprint and not enough to call anything an LCA.  Idemat 2026 (Vogtlander,
Delft University of Technology) is a process-level LCI with a full LCIA layer,
so it supplies:

  * global warming, kg CO2e per kg
  * cumulative energy demand, MJ per kg
  * Environmental Footprint 3.1 single score, Pt  -- the European Commission's
    own method, and therefore the one ESPR delegated acts are written against
  * EF 3.1 resource use, minerals and metals, Pt  -- an abiotic-resource
    indicator, which behaves very differently from carbon for this product
  * ReCiPe 2016 endpoint, Pt
  * eco-costs, EUR

and, critically for a circularity study, *process pairs*: primary and
secondary production of the same material, and end-of-life processes carrying
explicit substitution credits.  That is what lets this project compare the
cut-off and substitution allocation conventions rather than silently assume
one.

Licence
-------
Idemat is (c) Eco Cost Value / SSIM, free to use with attribution.  Unlike the
DEFRA and USGS files it is NOT redistributed with this repository: it is
listed in .gitignore and fetched by ``scripts/fetch_idemat.py``.  Cite it as

  Vogtlander, J.G., Idemat 2026, Sustainability Impact Metrics / Delft
  University of Technology, https://www.ecocostsvalue.com

Process selection
-----------------
Which Idemat process stands for which material family is a modelling choice,
not a parsing detail, so it lives in ``data/idemat_process_map.json`` with a
justification per entry rather than being buried here.  Processes are keyed by
their Idemat code truncated to four groups (e.g. ``A.100.24.101``); the fifth
group is a release date that changes between versions and is deliberately not
matched on.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

IDEMAT_XLSX = os.path.join(DATA, "Idemat_2026RevB1.xlsx")
PROCESS_MAP = os.path.join(DATA, "idemat_process_map.json")

SHEET = "Idemat2026 midpoints"

#: Column index -> indicator key and unit.  Read from the three-row header of
#: the midpoints sheet; see the module docstring for what each one is.
COLUMNS = {
    6:  ("gwp_kgco2e", "kg CO2e"),
    9:  ("ecocost_eur", "EUR"),
    35: ("ced_mj", "MJ"),
    43: ("recipe2016_pt", "Pt"),
    73: ("ef31_total_pt", "Pt"),
    75: ("ef31_climate_pt", "Pt"),
    87: ("ef31_resource_fossil_pt", "Pt"),
    88: ("ef31_resource_minerals_metals_pt", "Pt"),
}

#: The indicators the optimiser may be asked to minimise.  ``gwp_kgco2e`` is
#: the default so results stay comparable with the DEFRA backend.
IMPACT_INDICATORS = ("gwp_kgco2e", "ced_mj", "ef31_total_pt",
                     "ef31_resource_minerals_metals_pt")

INDICATOR_LABELS = {
    "gwp_kgco2e": "global warming (kg CO$_2$e)",
    "ced_mj": "cumulative energy demand (MJ)",
    "ef31_total_pt": "EF 3.1 single score (Pt)",
    "ef31_resource_minerals_metals_pt": "EF 3.1 resource use, minerals and metals (Pt)",
    "recipe2016_pt": "ReCiPe 2016 endpoint (Pt)",
    "ecocost_eur": "eco-costs (EUR)",
}

_CODE = re.compile(r"^\s*([A-Z]\.\d{3}\.\d{2}\.\d{3})\.\d+\s*(.*)$")


@dataclass
class IdematProcess:
    code: str
    name: str
    unit: str
    indicators: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.indicators[key]


def load_idemat(path: str = IDEMAT_XLSX) -> dict:
    """Return ``{code: IdematProcess}`` for every row with a parseable code."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.  Idemat is not redistributed with this "
            f"repository; run scripts/fetch_idemat.py to download it."
        )
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[SHEET]

    out = {}
    for row in ws.iter_rows(min_row=4, values_only=True):
        if not row or row[0] is None:
            continue
        m = _CODE.match(str(row[0]))
        if not m:
            continue
        code, name = m.group(1), m.group(2).strip()
        unit = str(row[4]).strip() if row[4] is not None else ""
        ind = {}
        for col, (key, _unit) in COLUMNS.items():
            v = row[col] if col < len(row) else None
            if isinstance(v, (int, float)):
                ind[key] = float(v)
        if "gwp_kgco2e" not in ind:
            continue
        out[code] = IdematProcess(code=code, name=name, unit=unit, indicators=ind)

    if len(out) < 500:
        raise RuntimeError(f"Idemat parsed only {len(out)} processes; "
                           f"the sheet layout has probably changed")
    return out


def average_process(processes: dict, codes: list, synthetic_code: str,
                    name: str, unit: str) -> IdematProcess:
    """A synthetic process whose indicators are the arithmetic mean of ``codes``.

    Used where no single Idemat process is a good match for a component and
    averaging across the closest available analogues is more defensible than
    pinning one arbitrarily.  The synthetic process is inserted into the
    process table under ``synthetic_code`` so everything downstream (resolve,
    Inventory) treats it exactly like a database entry -- its derivation is
    recorded in the process map's note field, not hidden in code.
    """
    picks = [processes[c] for c in codes]
    keys = set()
    for p in picks:
        keys |= set(p.indicators)
    avg = {k: sum(p.indicators.get(k, 0.0) for p in picks) / len(picks)
           for k in keys}
    return IdematProcess(code=synthetic_code, name=name, unit=unit,
                         indicators=avg)


def load_process_map(path: str = PROCESS_MAP) -> dict:
    with open(path) as fh:
        return json.load(fh)


def resolve(processes: dict, pmap: dict) -> dict:
    """Check every mapped code exists and return a flat ``{role: process}``.

    Fails loudly rather than silently substituting: a missing code means the
    Idemat release has moved a process, and results computed around that would
    be wrong in a way nobody would notice.
    """
    resolved, missing = {}, []
    for family, roles in pmap["families"].items():
        for role, spec in roles.items():
            if role.startswith("_"):
                continue
            code = spec["code"]
            if code not in processes:
                missing.append(f"{family}.{role} -> {code}")
                continue
            p = processes[code]
            if spec.get("expect_unit") and p.unit != spec["expect_unit"]:
                missing.append(
                    f"{family}.{role} -> {code} unit {p.unit!r} != "
                    f"{spec['expect_unit']!r}")
                continue
            resolved[f"{family}.{role}"] = p
    if missing:
        raise RuntimeError("Idemat process map does not resolve:\n  "
                           + "\n  ".join(missing))
    return resolved


#: Synthetic processes the process map references but the database does not
#: contain outright.  Registered here once so every caller that builds a
#: process table (Inventory.build, this module's self-test) stays in sync;
#: see data/idemat_process_map.json -> families.electronics.virgin.
def augment_with_synthetics(processes: dict) -> None:
    elec_avg = average_process(
        processes, ["A.050.06.304", "A.050.06.305"],
        "SYN.ELECTRONICS.WHITEGOODS.AVG",
        "Synthetic average: PCB of refrigerator + PCB of washing machine",
        "kg")
    processes[elec_avg.code] = elec_avg


if __name__ == "__main__":
    procs = load_idemat()
    augment_with_synthetics(procs)
    print(f"parsed {len(procs)} Idemat processes ({len(procs) - 1} native + "
          f"1 synthetic)")
    pmap = load_process_map()
    res = resolve(procs, pmap)
    print(f"resolved {len(res)} mapped roles\n")
    hdr = f"{'role':34s} {'gwp':>9s} {'CED':>9s} {'EF3.1':>11s} {'EF min/met':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for role, p in res.items():
        print(f"{role:34s} {p['gwp_kgco2e']:9.4f} {p['ced_mj']:9.2f} "
              f"{p['ef31_total_pt']:11.3e} "
              f"{p['ef31_resource_minerals_metals_pt']:11.3e}")
