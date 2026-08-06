#!/usr/bin/env python3
"""Download Idemat 2026 into data/.

Idemat is (c) Eco Cost Value / SSIM and free to use with attribution, but it
is not an open-licensed dataset like DEFRA or USGS, so this repository cites
it and fetches it rather than redistributing it.  Run this once before
run_experiments.py.

    python3 scripts/fetch_idemat.py
"""
import os
import sys
import urllib.request

URL = "https://www.ecocostsvalue.com/EVR/img/Idemat_2026RevB1.xlsx"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "Idemat_2026RevB1.xlsx")

if os.path.exists(DEST):
    print(f"already present: {DEST}")
    sys.exit(0)

print(f"downloading {URL}")
req = urllib.request.Request(URL, headers={"User-Agent": "CircuOpt/1.0"})
with urllib.request.urlopen(req, timeout=300) as r, open(DEST, "wb") as fh:
    fh.write(r.read())
print(f"wrote {DEST} ({os.path.getsize(DEST) / 1e6:.1f} MB)")
print("Cite as: Vogtlander, J.G., Idemat 2026, Sustainability Impact Metrics /")
print("Delft University of Technology, https://www.ecocostsvalue.com")
