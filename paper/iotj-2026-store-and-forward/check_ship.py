#!/usr/bin/env python3
"""Fails (exit 1) if the source or the built PDF could ship a placeholder. Run after the build, before packaging.
    python3 sim/check_ship.py"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
bad = []
if os.path.exists(os.path.join(ROOT, "macros_pending.tex")):
    bad.append("macros_pending.tex exists")
for f in ["macros_p6.tex", "main.tex", "cover_letter.md"] + [os.path.relpath(x, ROOT) for x in glob.glob(os.path.join(ROOT, "sections", "*.tex"))]:
    t = open(os.path.join(ROOT, f), encoding="utf-8").read()
    if "??" in t or "macros_pending.tex}" in t and f != "main.tex":
        bad.append(f"?? in {f}")
txt = subprocess.run(["pdftotext", os.path.join(ROOT, "main.pdf"), "-"], capture_output=True, text=True).stdout
if "??" in txt:
    bad.append("?? in main.pdf: " + re.search(r".{30}\?\?.{30}", txt.replace("\n", " ")).group(0))
log = open(os.path.join(ROOT, "main.log"), errors="ignore").read()
if re.search(r"^!", log, re.M):
    bad.append("hard error in main.log")
if re.search(r"undefined", log, re.I):
    bad.append("undefined reference or macro in main.log")
print("\n".join(bad) or "ship check OK")
sys.exit(1 if bad else 0)
