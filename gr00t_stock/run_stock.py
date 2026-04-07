#!/usr/bin/env python3
"""Wrapper to run any script using the STOCK gr00t code, not our modified version.

Usage:
    python run_stock.py <script.py> [args...]

This inserts the stock repo at the front of sys.path before any imports,
overriding the editable install from our modified repo.
"""
import sys
import os

# Force stock gr00t to be found first
STOCK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STOCK_ROOT)

# Also add Eagle2.5 if available
eagle = os.path.join(STOCK_ROOT, "..", "Eagle", "Eagle2_5")
if os.path.isdir(eagle):
    sys.path.insert(1, os.path.abspath(eagle))

# Remove the editable install finder so it doesn't override us
mods_to_remove = [k for k in sys.modules if "editable" in k and "gr00t" in k]
for k in mods_to_remove:
    del sys.modules[k]

# Also remove any already-imported gr00t modules
mods_to_remove = [k for k in sys.modules if k == "gr00t" or k.startswith("gr00t.")]
for k in mods_to_remove:
    del sys.modules[k]

if len(sys.argv) < 2:
    print("Usage: python run_stock.py <script.py> [args...]")
    sys.exit(1)

script = sys.argv[1]
sys.argv = sys.argv[1:]  # shift so the target script sees correct argv

# Verify we're loading stock code
import gr00t
assert STOCK_ROOT in gr00t.__file__, f"ERROR: loaded gr00t from {gr00t.__file__}, expected {STOCK_ROOT}"
print(f"[run_stock] Using gr00t from: {gr00t.__file__}")

exec(open(script).read(), {"__name__": "__main__", "__file__": script})
