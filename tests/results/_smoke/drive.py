"""Smoke driver: sets KIRBY_SMOKE_N then runs an OPIG experiment main() with argv.

Usage: python3 tests/results/_smoke/drive.py <smoke_n> <module> [args...]
"""
import os
import sys

smoke_n = sys.argv[1]
module = sys.argv[2]
argv_rest = sys.argv[3:]

os.environ["KIRBY_SMOKE_N"] = smoke_n
os.environ.setdefault("PYTHONUNBUFFERED", "1")

repo = "/Users/apunt/repos/KIRBy"
sys.path.insert(0, repo + "/src")
sys.path.insert(0, repo + "/tests")
sys.path.insert(0, repo + "/tests/experiments")
os.chdir(repo)

import importlib
sys.argv = [module] + argv_rest
mod = importlib.import_module(module)
mod.main()
