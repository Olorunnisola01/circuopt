#!/usr/bin/env python3
"""Download the World Bank Commodity Markets ("Pink Sheet") monthly price
file into data/.

World Bank commodity price data is a work product the Bank makes freely
available; unlike Idemat it carries no attribution requirement, but the file
updates monthly and is fetched here rather than committed so the repository
does not go stale.

    python3 scripts/fetch_worldbank.py
"""
import os
import sys
import urllib.request

URL = ("https://thedocs.worldbank.org/en/doc/"
      "74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/"
      "CMO-Historical-Data-Monthly.xlsx")
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "worldbank_cmo_monthly.xlsx")

if os.path.exists(DEST):
    print(f"already present: {DEST}")
    sys.exit(0)

print(f"downloading {URL}")
req = urllib.request.Request(URL, headers={"User-Agent": "CircuOpt/1.0"})
with urllib.request.urlopen(req, timeout=300) as r, open(DEST, "wb") as fh:
    fh.write(r.read())
print(f"wrote {DEST} ({os.path.getsize(DEST) / 1e6:.1f} MB)")
print("Source: World Bank Commodity Markets (Pink Sheet), "
      "https://www.worldbank.org/en/research/commodity-markets")
