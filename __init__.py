#!/usr/bin/env python3
"""
ExPhonics: Exciton-Phonon Interaction Calculator

A comprehensive Python package for calculating exciton-phonon interactions 
and self-energies using many-body perturbation theory and the Bethe-Salpeter equation.

Main functions:
- create_electronic_band_structure: Electronic band structure from tight-binding
- create_exciton_band_structure: Exciton bands from BSE calculations  
- create_exciton_wavefunctions: Real-space wavefunction visualization
- create_self_energy_convergence: Convergence and temperature analysis
- create_frequency_dependent_self_energy: Self-energy with/without e-h interaction
"""

__version__ = "1.0.0"
__author__ = "Exciton-Phonon Research Team"
__email__ = "exphonics@research.edu"

# Import main functions for easy access
try:
    from .run_exphonics_demo import (
        create_electronic_band_structure,
        create_exciton_band_structure, 
        create_exciton_wavefunctions,
        create_self_energy_convergence,
        create_frequency_dependent_self_energy
    )
    
    __all__ = [
        'create_electronic_band_structure',
        'create_exciton_band_structure',
        'create_exciton_wavefunctions', 
        'create_self_energy_convergence',
        'create_frequency_dependent_self_energy'
    ]
    
except ImportError:
    # Handle case where dependencies aren't installed
    __all__ = []

# Package metadata
__description__ = "Exciton-Phonon Interaction Calculator using Many-Body Perturbation Theory"
__url__ = "https://github.com/research-team/exphonics"
__license__ = "MIT"

# Physical constants used throughout the package
PHYSICAL_CONSTANTS = {
    'k_B': 8.617333e-5,  # Boltzmann constant (eV/K)
    'hbar': 6.582119e-16,  # Reduced Planck constant (eV⋅s)
    'e': 1.602176e-19,   # Elementary charge (C)
}

# Default parameters from the theoretical model
DEFAULT_PARAMETERS = {
    'effective_mass': 0.49,      # m* 
    'lattice_constant': 3.13,    # a (Bohr)
    'band_gap': 2.5,            # E_g (eV)
    'spin_orbit': 0.425,        # Δ (eV)
    'coupling_strength': 0.250,  # g (eV)
    'phonon_energy': 0.050,     # ω₀ (eV)
    'broadening': 0.010,        # η (eV)
    'coulomb_strength': 1.6,    # Δv₀ (eV)
}

def get_version():
    """Return package version."""
    return __version__

def get_default_parameters():
    """Return dictionary of default physical parameters."""
    return DEFAULT_PARAMETERS.copy()

def print_info():
    """Print package information."""
    print(f"ExPhonics v{__version__}")
    print(f"{__description__}")
    print(f"Author: {__author__}")
    print(f"URL: {__url__}")
    print(f"License: {__license__}")