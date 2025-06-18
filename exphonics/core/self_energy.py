"""
Exciton-phonon self-energy calculations.

Implements the complete self-energy expression from the paper:
Ξ_{SS'}(ω,T) = Ξ^{FMd}_{SS'}(ω,T) + Ξ^{FMs}_{SS'}(T) + Ξ^{DW}_{SS'}(T) + Ξ^{C}_{SS'}(ω,T)
"""

import numpy as np
from .constants import PhysicalConstants

# Try to import numba, fall back to regular functions if not available
try:
    from numba import jit, prange
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(x):
        return range(x)


class ExcitonPhononSelfEnergy:
    """
    Calculator for exciton-phonon self-energy with all contributing terms.
    
    Implements the theory from the paper with:
    - Fan-Migdal terms (dynamic and static)
    - Debye-Waller terms
    - Completion terms for finite basis sets
    """
    
    def __init__(self, omega_phonon=0.050, g_coupling=0.250):
        """
        Initialize self-energy calculator.
        
        Parameters:
        -----------
        omega_phonon : float
            Phonon frequency (eV)
        g_coupling : float
            Electron-phonon coupling strength (eV)
        """
        self.omega_ph = omega_phonon
        self.g = g_coupling
        self.k_B = PhysicalConstants.k_B
    
    def phonon_occupation(self, T):
        """Calculate phonon occupation number at temperature T."""
        if T == 0:
            return 0.0
        else:
            return PhysicalConstants.phonon_population(self.omega_ph, T)
    
    def coupling_matrix_element(self, A_S, A_Sp, g_c, g_v, Q, q):
        """
        Calculate exciton-phonon coupling matrix element g_{SS'}.
        
        From Eq. (64) in the paper:
        g_{SS'λ} = Σ_{vc,v'c'} A*_{vc}^S A_{v'c'}^{S'} [g_{cc'λ} δ_{vv'} - g_{v'vλ} δ_{cc'}]
        
        Parameters:
        -----------
        A_S : array
            Exciton wavefunction coefficients for state S
        A_Sp : array  
            Exciton wavefunction coefficients for state S'
        g_c : float
            Conduction band coupling
        g_v : float
            Valence band coupling
        Q : array
            Exciton center-of-mass momentum
        q : array
            Phonon momentum
            
        Returns:
        --------
        complex
            Coupling matrix element
        """
        # For the two-band model, this simplifies considerably
        # since we only have one valence and one conduction band
        
        # First term: conduction band coupling
        overlap_c = np.vdot(A_S, A_Sp)  # δ_{vv'} = 1 for same valence band
        term1 = g_c * overlap_c
        
        # Second term: valence band coupling  
        # This involves momentum shift due to phonon
        # For simplicity in the model, we approximate this
        overlap_v = np.vdot(A_S, A_Sp)  # Approximate for q-shifted states
        term2 = -g_v * overlap_v
        
        return term1 + term2
    
    def fan_migdal_dynamic(self, omega, T, exciton_energies, coupling_elements):
        """
        Calculate dynamic Fan-Migdal self-energy term.
        
        From Eq. (63):
        Ξ^{FMd}_{SS'}(iω_n) = Σ_λ Σ_{S''} g_{SS''λ} g*_{S'S''λ} × 
                              [N_B(ω_λ) - N_B(Ω_{S''}) / (iω_n - Ω_{S''} + ω_λ) + 
                               N_B(ω_λ) + 1 + N_B(Ω_{S''}) / (iω_n - Ω_{S''} - ω_λ)]
        
        Parameters:
        -----------
        omega : float or complex
            Frequency (can be complex for analytic continuation)
        T : float
            Temperature (K)
        exciton_energies : array
            Exciton energies Ω_{S''}
        coupling_elements : array
            Coupling matrix elements g_{SS''λ}
            
        Returns:
        --------
        complex
            Dynamic Fan-Migdal self-energy
        """
        n_ph = self.phonon_occupation(T)
        eta = 1e-6  # Small imaginary part for convergence
        
        Sigma_FMd = 0.0j
        
        for S_pp, (Omega_Spp, g_element) in enumerate(zip(exciton_energies, coupling_elements)):
            g_squared = np.abs(g_element)**2
            
            # Phonon emission term
            denom1 = omega - Omega_Spp + self.omega_ph + 1j*eta
            term1 = (n_ph - PhysicalConstants.fermi_dirac(Omega_Spp, 0, T)) / denom1
            
            # Phonon absorption term  
            denom2 = omega - Omega_Spp - self.omega_ph + 1j*eta
            term2 = (n_ph + 1 + PhysicalConstants.fermi_dirac(Omega_Spp, 0, T)) / denom2
            
            Sigma_FMd += g_squared * (term1 + term2)
        
        return Sigma_FMd
    
    def fan_migdal_static(self, T, exciton_energies, coupling_elements):
        """
        Calculate static Fan-Migdal self-energy term.
        
        This term includes virtual transitions that don't conserve energy.
        """
        n_ph = self.phonon_occupation(T)
        
        Sigma_FMs = 0.0
        
        for Omega_Spp, g_element in zip(exciton_energies, coupling_elements):
            g_squared = np.abs(g_element)**2
            
            # Symmetrized form of static contributions
            # This is an approximation for the two-band model
            denom_plus = Omega_Spp + self.omega_ph
            denom_minus = Omega_Spp - self.omega_ph
            
            if abs(denom_plus) > 1e-12:
                term_plus = (n_ph + 1) / denom_plus
            else:
                term_plus = 0.0
                
            if abs(denom_minus) > 1e-12:
                term_minus = n_ph / denom_minus
            else:
                term_minus = 0.0
            
            Sigma_FMs += g_squared * (term_plus + term_minus)
        
        return Sigma_FMs
    
    def debye_waller(self, T):
        """
        Calculate Debye-Waller self-energy term.
        
        From the paper:
        Ξ^{DW}_{SS'} = Σ_λ g^{(2)}_{SS'λλ} (2N_B(ω_λ) + 1)
        """
        n_ph = self.phonon_occupation(T)
        
        # Second-order coupling (approximated as constant for the model)
        g_squared = self.g**2 / self.omega_ph  # Dimensional analysis
        
        return g_squared * (2*n_ph + 1)
    
    def completion_term(self, omega, T, exciton_energies, n_computed):
        """
        Calculate completion term for finite basis.
        
        This accounts for exciton states not explicitly computed.
        """
        # For the model system, we use the approximation that
        # high-energy states behave like free electron-hole pairs
        
        n_ph = self.phonon_occupation(T)
        eta = 1e-6
        
        # Estimate contribution from uncomputed states
        # This is a simplified model
        if len(exciton_energies) > n_computed:
            high_energy_scale = np.mean(exciton_energies[n_computed:])
        else:
            high_energy_scale = 2 * np.max(exciton_energies)
        
        g_eff_squared = self.g**2 * 0.1  # Reduced coupling for high states
        
        denom = omega - high_energy_scale + 1j*eta
        completion = g_eff_squared * (n_ph + 1) / denom
        
        return completion
    
    def calculate_self_energy(self, omega, T, exciton_energies, exciton_wavefunctions, 
                            S_index=0, Sp_index=None, include_completion=True):
        """
        Calculate complete exciton-phonon self-energy.
        
        Implements Eq. (68):
        Ξ_{SS'}(ω,T) = Ξ^{FMd}_{SS'}(ω,T) + Ξ^{FMs}_{SS'}(T) + Ξ^{DW}_{SS'}(T) + Ξ^{C}_{SS'}(ω,T)
        
        Parameters:
        -----------
        omega : float or complex
            Frequency
        T : float
            Temperature (K)
        exciton_energies : array
            Exciton energies
        exciton_wavefunctions : array
            Exciton wavefunctions
        S_index : int
            Initial exciton state index
        Sp_index : int or None
            Final exciton state index (if None, use S_index for diagonal element)
        include_completion : bool
            Whether to include completion term
            
        Returns:
        --------
        complex
            Total self-energy Ξ_{SS'}(ω,T)
        """
        if Sp_index is None:
            Sp_index = S_index
        
        # Get wavefunctions for states S and S'
        A_S = exciton_wavefunctions[:, S_index]
        A_Sp = exciton_wavefunctions[:, Sp_index]
        
        # Calculate coupling matrix elements (simplified for model)
        coupling_elements = []
        for S_pp in range(len(exciton_energies)):
            A_Spp = exciton_wavefunctions[:, S_pp]
            
            # Simplified coupling (constant g for intraband processes)
            if S_pp == S_index:
                g_element = self.g
            else:
                # Approximate inter-exciton coupling
                overlap = np.abs(np.vdot(A_S, A_Spp))**2
                g_element = self.g * overlap
            
            coupling_elements.append(g_element)
        
        coupling_elements = np.array(coupling_elements)
        
        # Calculate all terms
        Sigma_FMd = self.fan_migdal_dynamic(omega, T, exciton_energies, coupling_elements)
        Sigma_FMs = self.fan_migdal_static(T, exciton_energies, coupling_elements)
        Sigma_DW = self.debye_waller(T)
        
        if include_completion:
            Sigma_C = self.completion_term(omega, T, exciton_energies, len(exciton_energies)//2)
        else:
            Sigma_C = 0.0
        
        # Total self-energy
        Sigma_total = Sigma_FMd + Sigma_FMs + Sigma_DW + Sigma_C
        
        return Sigma_total
    
    def calculate_renormalized_energy(self, T, exciton_energies, exciton_wavefunctions, S_index=0):
        """
        Calculate temperature-dependent renormalized exciton energy.
        
        E_S(T) = Ω_S + Re[Ξ_{SS}(Ω_S, T)]
        """
        omega = exciton_energies[S_index]
        self_energy = self.calculate_self_energy(omega, T, exciton_energies, 
                                                exciton_wavefunctions, S_index)
        
        return omega + np.real(self_energy)
    
    def calculate_lifetime_broadening(self, T, exciton_energies, exciton_wavefunctions, S_index=0):
        """
        Calculate exciton lifetime broadening.
        
        Γ_S(T) = 2|Im[Ξ_{SS}(Ω_S, T)]|
        """
        omega = exciton_energies[S_index]
        self_energy = self.calculate_self_energy(omega, T, exciton_energies,
                                                exciton_wavefunctions, S_index)
        
        return 2 * np.abs(np.imag(self_energy))
    
    def temperature_dependence(self, T_array, exciton_energies, exciton_wavefunctions, 
                             n_states=4):
        """
        Calculate temperature dependence of exciton energies and lifetimes.
        
        Returns:
        --------
        dict
            Dictionary with 'energies', 'lifetimes', 'temperatures'
        """
        n_temps = len(T_array)
        energies = np.zeros((n_temps, n_states))
        lifetimes = np.zeros((n_temps, n_states))
        
        for i, T in enumerate(T_array):
            for S in range(n_states):
                energies[i, S] = self.calculate_renormalized_energy(
                    T, exciton_energies, exciton_wavefunctions, S
                )
                lifetimes[i, S] = self.calculate_lifetime_broadening(
                    T, exciton_energies, exciton_wavefunctions, S
                )
        
        return {
            'temperatures': T_array,
            'energies': energies,
            'lifetimes': lifetimes,
            'bare_energies': exciton_energies[:n_states]
        }