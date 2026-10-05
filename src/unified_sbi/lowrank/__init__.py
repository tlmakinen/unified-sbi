"""Rank-deficient benchmark: a hidden 2D Rosenbrock inside 8 parameters, with a (k, m) screen.

Ported from ``unified_sbi_rosenbrock_2d_in_8d.ipynb`` and extended with automatic selection of
the active dimension k and the summary dimension m. The full-rank scaling study in
``unified_sbi.study`` is unchanged and does not use the screen.
"""

from .maps import StructuredMapConfig, StructuredParameterMap, fit_structured_maps
from .screen import candidate_grid, run_screen
from .simulator import HiddenRosenbrock
from .study import LowRankConfig, run_cell

__all__ = ["HiddenRosenbrock", "StructuredParameterMap", "StructuredMapConfig", "fit_structured_maps",
           "candidate_grid", "run_screen", "LowRankConfig", "run_cell"]
