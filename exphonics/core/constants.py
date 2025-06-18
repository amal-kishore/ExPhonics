"""
Physical constants and unit conversions.
"""

import numpy as np

class PhysicalConstants:
    """Collection of physical constants in appropriate units."""
    
    # Fundamental constants
    hbar = 1.0  # Set ℏ = 1 in natural units
    e = 1.0     # Elementary charge (natural units)
    
    # Conversion factors
    bohr_to_angstrom = 0.52917721092  # Bohr radius in Å
    hartree_to_eV = 27.211386245      # Hartree to eV
    hbar2_2m0 = 3.809982              # ℏ²/2m₀ in eV·Å²
    
    # Boltzmann constant
    k_B = 8.617333e-5  # eV/K
    
    @staticmethod
    def fermi_dirac(energy, mu, T):
        """Fermi-Dirac distribution function."""
        if T == 0:
            return 1.0 if energy <= mu else 0.0
        else:
            beta = 1.0 / (PhysicalConstants.k_B * T)
            return 1.0 / (np.exp(beta * (energy - mu)) + 1.0)
    
    @staticmethod
    def bose_einstein(energy, T):
        """Bose-Einstein distribution function."""
        if T == 0:
            return 0.0
        elif energy <= 0:
            return np.inf
        else:
            beta = 1.0 / (PhysicalConstants.k_B * T)
            return 1.0 / (np.exp(beta * energy) - 1.0)
    
    @staticmethod
    def phonon_population(omega, T):
        """Phonon occupation number at temperature T."""
        return PhysicalConstants.bose_einstein(omega, T)

# Convenience aliases
k_B = PhysicalConstants.k_B
bohr_to_ang = PhysicalConstants.bohr_to_angstrom
hartree_to_eV = PhysicalConstants.hartree_to_eV