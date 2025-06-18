"""
Models module for specific material systems and theoretical frameworks.

Contains implementations of:
- Two-band triangular lattice model for TMDs
- Bethe-Salpeter equation solver
- Phonon models and dispersions
"""

from .tight_binding import *
from .bse import *
from .phonons import *

__all__ = [
    "TightBindingModel",
    "BSESolver", 
    "PhononModel",
]