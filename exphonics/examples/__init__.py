"""
Examples module demonstrating the theory implementation.

Contains complete reproductions of the paper's Section IV results
and convergence analysis tools.
"""

from .section_iv import *
from .convergence import *

__all__ = [
    "run_section_iv_example",
    "plot_band_structure", 
    "plot_exciton_maps",
    "analyze_temperature_dependence",
    "convergence_analysis",
]