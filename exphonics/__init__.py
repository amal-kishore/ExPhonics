"""
ExPhonics: Theory of Exciton-Phonon Coupling

A comprehensive Python package for calculating exciton-phonon interactions
in 2D materials and computing temperature-dependent optical properties.
"""

__version__ = "1.0.0"

from .core import *
from .models import *
from .utils import *

__all__ = ["core", "models", "utils", "examples"]