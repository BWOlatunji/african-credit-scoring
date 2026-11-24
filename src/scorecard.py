"""Compatibility shim: the scorecard now lives in the production package `credit_scoring.scorecard`.

Notebooks add `src` to sys.path and `import scorecard`; this re-exports the package module so that the notebooks and
the API use one implementation.
"""
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

from credit_scoring.scorecard import *                     # noqa: F401,F403
from credit_scoring.scorecard import Band, OVERLAYS, PD_FLOOR, PD_CAP   # noqa: F401
