"""
Core module for exciton-phonon coupling calculations.

Contains the fundamental algorithms for computing self-energies,
solving the Bethe-Salpeter equation, and handling electron-phonon interactions.
"""

from .constants import *
from .hamiltonian import *
from .self_energy import *
from .electron_phonon import *

__all__ = [
    "PhysicalConstants",
    "TightBindingHamiltonian", 
    "BSEHamiltonian",
    "ExcitonPhononSelfEnergy",
    "ElectronPhononCoupling",
]