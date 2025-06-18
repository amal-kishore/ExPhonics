"""
Electron-phonon coupling calculations.

Handles the calculation of electron-phonon matrix elements and 
the transformation to the exciton basis.
"""

import numpy as np
from .constants import PhysicalConstants

# Try to import numba, fall back to regular functions if not available
try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class ElectronPhononCoupling:
    """
    Calculator for electron-phonon coupling matrix elements.
    
    Handles both first-order and second-order coupling terms
    and their transformation to the exciton basis.
    """
    
    def __init__(self, g_c=0.250, g_v=0.250, omega_phonon=0.050):
        """
        Initialize electron-phonon coupling.
        
        Parameters:
        -----------
        g_c : float
            Conduction band coupling strength (eV)
        g_v : float  
            Valence band coupling strength (eV)
        omega_phonon : float
            Phonon frequency (eV)
        """
        self.g_c = g_c
        self.g_v = g_v
        self.omega_ph = omega_phonon
    
    def first_order_coupling(self, k1, k2, band1, band2, q):
        """
        Calculate first-order electron-phonon coupling matrix element.
        
        g_{k1n1,k2n2}(q,ν) for scattering from (k1,n1) to (k2,n2) 
        with phonon momentum q.
        
        Parameters:
        -----------
        k1, k2 : array
            Initial and final electron wavevectors
        band1, band2 : str
            Initial and final bands ('c' or 'v')
        q : array
            Phonon wavevector
            
        Returns:
        --------
        complex
            Coupling matrix element
        """
        # For the simplified model, coupling is momentum-independent
        # and only intraband terms are non-zero
        
        if band1 == band2:
            if band1 == 'c':
                return self.g_c
            else:  # valence
                return self.g_v
        else:
            # Interband coupling assumed zero for simplicity
            return 0.0
    
    def second_order_coupling(self, k1, k2, band1, band2, q1, q2):
        """
        Calculate second-order electron-phonon coupling (Debye-Waller term).
        
        g^{(2)}_{k1n1,k2n2}(q1,ν1; q2,ν2)
        
        Parameters:
        -----------
        k1, k2 : array
            Initial and final electron wavevectors
        band1, band2 : str
            Initial and final bands
        q1, q2 : array
            Phonon wavevectors
            
        Returns:
        --------
        complex
            Second-order coupling matrix element
        """
        # Simplified model: only diagonal terms
        if band1 == band2 and np.allclose(k1, k2):
            if band1 == 'c':
                return self.g_c**2 / (2 * self.omega_ph)
            else:
                return self.g_v**2 / (2 * self.omega_ph)
        else:
            return 0.0
    
    def transform_to_exciton_basis(self, A_S, A_Sp, Q, q, L, site_indices):
        """
        Transform electron-phonon coupling to exciton basis.
        
        Calculates g_{SS'λ} from Eq. (64):
        g_{SS'λ} = Σ_{vc,v'c'} A*_{vc}^S A_{v'c'}^{S'} [g_{cc'λ} δ_{vv'} - g_{v'vλ} δ_{cc'}]
        
        Parameters:
        -----------
        A_S, A_Sp : array
            Exciton wavefunction coefficients
        Q : array  
            Exciton center-of-mass momentum
        q : array
            Phonon momentum
        L : int
            Supercell half-size
        site_indices : dict
            Mapping from lattice coordinates to linear indices
            
        Returns:
        --------
        complex
            Exciton-phonon coupling matrix element
        """
        # For the two-band model, this simplifies considerably
        # In the real-space representation, A_S represents the 
        # electron-hole wavefunction amplitude at each site
        
        g_SS_prime = 0.0j
        
        # First term: conduction band coupling
        # g_{cc'λ} δ_{vv'} contribution
        overlap_c = np.vdot(A_S, A_Sp)
        g_SS_prime += self.g_c * overlap_c
        
        # Second term: valence band coupling  
        # -g_{v'vλ} δ_{cc'} contribution
        # For q ≠ 0, this involves momentum-shifted states
        if np.linalg.norm(q) < 1e-12:
            overlap_v = np.vdot(A_S, A_Sp)
        else:
            # Approximate momentum shift effect
            phase_factor = np.exp(1j * np.dot(q, [0, 0]))  # Simplified
            overlap_v = phase_factor * np.vdot(A_S, A_Sp)
        
        g_SS_prime -= self.g_v * overlap_v
        
        return g_SS_prime
    
    def momentum_conservation_check(self, Q_S, Q_Sp, q):
        """
        Check momentum conservation: Q_S = Q_S' + q + G
        
        Parameters:
        -----------
        Q_S, Q_Sp : array
            Exciton center-of-mass momenta
        q : array
            Phonon momentum
            
        Returns:
        --------
        bool
            True if momentum is conserved (within numerical tolerance)
        """
        delta_Q = Q_S - Q_Sp - q
        
        # Check if delta_Q is a reciprocal lattice vector
        # For simplicity, we only check if it's close to zero
        return np.linalg.norm(delta_Q) < 1e-10
    
    def calculate_coupling_matrix(self, exciton_energies, exciton_wavefunctions, 
                                 Q_vectors, q_grid):
        """
        Calculate full matrix of exciton-phonon coupling elements.
        
        Parameters:
        -----------
        exciton_energies : array
            Exciton energies for all Q-points
        exciton_wavefunctions : array  
            Exciton wavefunctions for all Q-points
        Q_vectors : array
            Array of exciton center-of-mass momenta
        q_grid : array
            Grid of phonon momenta
            
        Returns:
        --------
        dict
            Dictionary with coupling matrices for each q-point
        """
        n_states = exciton_wavefunctions.shape[1]
        n_q = len(q_grid)
        
        coupling_matrices = {}
        
        for iq, q in enumerate(q_grid):
            coupling_matrix = np.zeros((n_states, n_states), dtype=complex)
            
            for S in range(n_states):
                for Sp in range(n_states):
                    # For Q = 0 excitons (optical transitions)
                    Q_S = np.zeros(2)
                    Q_Sp = q  # Target state has momentum q
                    
                    if self.momentum_conservation_check(Q_S, Q_Sp, q):
                        A_S = exciton_wavefunctions[:, S]
                        A_Sp = exciton_wavefunctions[:, Sp]
                        
                        g_element = self.transform_to_exciton_basis(
                            A_S, A_Sp, Q_S, q, L=25, site_indices={}
                        )
                        
                        coupling_matrix[S, Sp] = g_element
            
            coupling_matrices[f'q_{iq}'] = coupling_matrix
        
        return coupling_matrices
    
    def effective_coupling_strength(self, exciton_wavefunction, site_weights=None):
        """
        Calculate effective coupling strength for a given exciton state.
        
        Takes into account the spatial distribution of the exciton wavefunction
        and the local coupling strengths.
        
        Parameters:
        -----------
        exciton_wavefunction : array
            Exciton wavefunction in real space
        site_weights : array, optional
            Weights for different lattice sites
            
        Returns:
        --------
        float
            Effective coupling strength
        """
        if site_weights is None:
            site_weights = np.ones_like(exciton_wavefunction)
        
        # Calculate weighted average
        probability_density = np.abs(exciton_wavefunction)**2
        normalized_weights = site_weights / np.sum(site_weights)
        
        g_eff_c = self.g_c * np.sum(probability_density * normalized_weights)
        g_eff_v = self.g_v * np.sum(probability_density * normalized_weights)
        
        # Combined effective coupling
        g_eff = np.sqrt(g_eff_c**2 + g_eff_v**2)
        
        return g_eff
    
    def phonon_scattering_rate(self, exciton_energy, target_energies, T, eta=1e-3):
        """
        Calculate phonon scattering rate from initial to final exciton states.
        
        Uses Fermi's golden rule with proper temperature dependence.
        
        Parameters:
        -----------
        exciton_energy : float
            Initial exciton energy
        target_energies : array
            Final exciton energies
        T : float
            Temperature (K)
        eta : float
            Broadening parameter (eV)
            
        Returns:
        --------
        float
            Total scattering rate (1/time)
        """
        n_ph = PhysicalConstants.phonon_population(self.omega_ph, T)
        
        total_rate = 0.0
        
        for E_final in target_energies:
            # Emission process
            if exciton_energy > E_final + self.omega_ph:
                delta_E = exciton_energy - E_final - self.omega_ph
                rate_emission = (n_ph + 1) * eta / (delta_E**2 + eta**2)
                total_rate += rate_emission
            
            # Absorption process (if T > 0)
            if T > 0 and exciton_energy + self.omega_ph > E_final:
                delta_E = exciton_energy + self.omega_ph - E_final
                rate_absorption = n_ph * eta / (delta_E**2 + eta**2)
                total_rate += rate_absorption
        
        # Include coupling strength
        g_eff = np.sqrt(self.g_c**2 + self.g_v**2)
        prefactor = 2 * np.pi * g_eff**2 / PhysicalConstants.hbar
        
        return prefactor * total_rate