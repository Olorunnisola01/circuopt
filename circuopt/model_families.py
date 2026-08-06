"""Family-to-DEFRA mappings, kept separate so inventory.py and model.py can
both use them without importing each other."""

# family -> (DEFRA Level-3 material name, provenance tag)
FAMILY_DEFRA = {
    "plastic":     ("Plastics: average plastic rigid", "DATA"),
    "steel":       ("Metal: steel cans", "PROXY"),
    "aluminium":   ("Metal: aluminium cans and foil (excl. forming)", "PROXY"),
    "copper":      ("Metal: scrap metal", "PROXY"),
    "electronics": ("Electrical items - IT", "PROXY"),
}

# family -> (USGS commodity name, provenance tag)
FAMILY_USGS = {
    "steel":     ("Iron and Steel Scrap", "DATA"),
    "copper":    ("Copper", "DATA"),
    "aluminium": ("Aluminum", "DATA"),
}
