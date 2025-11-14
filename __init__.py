"""Top‑level package for parking lot monitoring.

This package bundles together the detection, forecasting and API modules for
building an end‑to‑end smart parking solution.  It does not import heavy
dependencies at import time; users should import submodules directly.
"""

__all__ = [
    "detection",
    "forecasting",
    "app",
]