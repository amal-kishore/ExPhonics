"""
Hamiltonian construction for tight-binding and BSE calculations.

Implements the two-band triangular lattice model with spin-orbit coupling
and the Bethe-Salpeter equation for exciton states.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .constants import PhysicalConstants, bohr_to_ang

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


class TightBindingHamiltonian:
    """
    Two-band tight-binding Hamiltonian for triangular lattice.
    
    Implements the model from the paper with spin-orbit coupling:
    H_{nσ} = Σ_{R,δ} t_{nσδ} c†_{nσ,R+δ} c_{nσ,R} + Σ_R ε_n c†_{nσ,R} c_{nσ,R}
    
    With hopping parameters:
    t_{nσδ} = t_n + 4iσ t̃_n sin(K⋅δ)
    """
    
    def __init__(self, m_star=0.49, a_bohr=3.13, E_gap=2.5, Delta=0.425):
        """
        Initialize tight-binding parameters.
        
        Parameters:
        -----------
        m_star : float
            Effective mass in units of electron mass
        a_bohr : float  
            Lattice constant in Bohr radii
        E_gap : float
            Band gap in eV
        Delta : float
            Spin-orbit splitting in eV
        """
        self.m_star = m_star
        self.a_bohr = a_bohr
        self.E_gap = E_gap
        self.Delta = Delta
        
        # Convert to Angstrom
        self.a = a_bohr * bohr_to_ang
        
        # Calculate hopping parameter
        self.t = (4.0 / (3.0 * m_star)) * PhysicalConstants.hbar2_2m0 / self.a**2
        
        # On-site energies
        self.eps_c = 3 * self.t + E_gap
        self.eps_v = -3 * self.t - Delta / 2
        
        # Hopping parameters
        self.t_c = self.t
        self.t_v = -self.t
        self.t_tilde_c = 0.0
        self.t_tilde_v = Delta / 18.0
        
        # Set up lattice vectors for triangular lattice
        self._setup_lattice()
    
    def _setup_lattice(self):
        """Set up lattice vectors and nearest neighbor vectors."""
        # Primitive lattice vectors  
        self.a1 = np.array([self.a, 0.0])
        self.a2 = np.array([self.a/2, np.sqrt(3)*self.a/2])
        
        # Nearest neighbor vectors (6 for triangular lattice)
        self.delta = np.array([
            [ self.a, 0],
            [ self.a/2,  np.sqrt(3)*self.a/2],
            [-self.a/2,  np.sqrt(3)*self.a/2],
            [-self.a, 0],
            [-self.a/2, -np.sqrt(3)*self.a/2],
            [ self.a/2, -np.sqrt(3)*self.a/2]
        ])
        
        # K point for spin-orbit coupling
        self.K = np.array([4*np.pi/(3*self.a), 0.0])
        
        # Reciprocal lattice vectors
        self.b1 = np.array([2*np.pi/self.a, 2*np.pi/(self.a*np.sqrt(3))])
        self.b2 = np.array([0.0, 4*np.pi/(self.a*np.sqrt(3))])
    
    def hopping_amplitude(self, band, spin, delta_vec):
        """
        Calculate hopping amplitude for given band, spin, and lattice vector.
        
        Parameters:
        -----------
        band : str
            'c' for conduction, 'v' for valence
        spin : float
            Spin value (+0.5 or -0.5)
        delta_vec : array
            Lattice vector connecting two sites
            
        Returns:
        --------
        complex
            Hopping amplitude t_{nσδ}
        """
        if band == 'c':
            t_base = self.t_c
            t_tilde = self.t_tilde_c
        else:  # valence
            t_base = self.t_v
            t_tilde = self.t_tilde_v
        
        # Spin-orbit coupling term
        K_dot_delta = np.dot(self.K, delta_vec)
        soc_term = 4j * spin * t_tilde * np.sin(K_dot_delta)
        
        return t_base + soc_term
    
    def band_energy(self, k, band='c', spin=0.5):
        """
        Calculate band energy at wavevector k.
        
        Parameters:
        -----------
        k : array
            Wavevector
        band : str
            'c' for conduction, 'v' for valence  
        spin : float
            Spin value
            
        Returns:
        --------
        float
            Band energy
        """
        if band == 'c':
            eps_base = self.eps_c
        else:
            eps_base = self.eps_v
        
        # Sum over nearest neighbors
        hopping_sum = 0.0
        for delta in self.delta:
            t_hop = self.hopping_amplitude(band, spin, delta)
            hopping_sum += t_hop * np.exp(-1j * np.dot(k, delta))
        
        return eps_base + hopping_sum.real
    
    def calculate_band_structure(self, k_path, bands=['v', 'c'], spins=[0.5, -0.5]):
        """
        Calculate band structure along k-path.
        
        Parameters:
        -----------
        k_path : array
            Array of k-points along high-symmetry path
        bands : list
            List of bands to calculate
        spins : list  
            List of spin values
            
        Returns:
        --------
        dict
            Dictionary with band energies
        """
        results = {}
        
        for band in bands:
            for spin in spins:
                key = f"{band}_{'up' if spin > 0 else 'dn'}"
                energies = []
                
                for k in k_path:
                    energy = self.band_energy(k, band, spin)
                    energies.append(energy)
                
                results[key] = np.array(energies)
        
        return results


class BSEHamiltonian:
    """
    Bethe-Salpeter equation Hamiltonian for excitons.
    
    Implements the two-particle Hamiltonian:
    H^{2p}_{vc,v'c'} = (ε_c - ε_v) δ_{cc'} δ_{vv'} + K_{vc,v'c'}
    """
    
    def __init__(self, tb_model, v0=1.6, epsilon_r=4.5):
        """
        Initialize BSE Hamiltonian.
        
        Parameters:
        -----------
        tb_model : TightBindingHamiltonian
            Tight-binding model instance
        v0 : float
            On-site Coulomb interaction (eV)
        epsilon_r : float
            Relative dielectric constant
        """
        self.tb = tb_model
        self.v0 = v0
        self.epsilon_r = epsilon_r
    
    def coulomb_interaction(self, R):
        """
        Calculate Coulomb interaction V(R).
        
        Parameters:
        -----------
        R : array
            Relative position vector
            
        Returns:
        --------
        float
            Coulomb interaction energy
        """
        R_norm = np.linalg.norm(R)
        
        if R_norm < 1e-12:  # On-site
            return self.v0
        else:
            # 14.4 eV⋅Å is e²/(4πε₀)
            return 14.4 / (self.epsilon_r * R_norm)
    
    def setup_supercell(self, L):
        """
        Set up real-space supercell for BSE calculation.
        
        Parameters:
        -----------
        L : int
            Half-size of supercell (total size is (2L+1)×(2L+1))
            
        Returns:
        --------
        tuple
            (R_vectors, site_indices)
        """
        R_vectors = []
        site_indices = {}
        
        for i in range(-L, L+1):
            for j in range(-L, L+1):
                R = i * self.tb.a1 + j * self.tb.a2
                site_indices[(i, j)] = len(R_vectors)
                R_vectors.append(R)
        
        return np.array(R_vectors), site_indices
    
    def build_bse_hamiltonian(self, Q, L, spin=0.5):
        """
        Build BSE Hamiltonian matrix for exciton center-of-mass momentum Q.
        
        Parameters:
        -----------
        Q : array
            Exciton center-of-mass momentum
        L : int
            Supercell half-size
        spin : float
            Spin value
            
        Returns:
        --------
        sparse matrix
            BSE Hamiltonian matrix
        """
        R_vectors, site_indices = self.setup_supercell(L)
        N = len(R_vectors)
        
        # Initialize Hamiltonian matrix
        H = sp.lil_matrix((N, N), dtype=complex)
        
        # Diagonal terms: kinetic energy + Coulomb interaction
        for idx, R in enumerate(R_vectors):
            # Kinetic energy: ε_c - ε_v  
            E_kinetic = self.tb.eps_c - self.tb.eps_v
            
            # Coulomb interaction: -V(R)
            V_coul = -self.coulomb_interaction(R)
            
            H[idx, idx] = E_kinetic + V_coul
        
        # Off-diagonal terms: hopping with Q-dependence
        shift_vectors = np.array([
            [+1, 0], [0, +1], [-1, +1], [-1, 0], [0, -1], [+1, -1]
        ])
        
        for (i, j), idx in site_indices.items():
            for delta, (di, dj) in zip(self.tb.delta, shift_vectors):
                ii, jj = i + di, j + dj
                
                # Check if target site is within supercell
                if abs(ii) > L or abs(jj) > L:
                    continue
                
                target_idx = site_indices[(ii, jj)]
                
                # BSE hopping parameter with Q-dependence
                phase_factor_e = np.exp(-1j * np.dot(Q, delta) / 2)
                phase_factor_h = np.exp(+1j * np.dot(Q, delta) / 2)
                
                t_c = self.tb.hopping_amplitude('c', spin, delta)
                t_v = self.tb.hopping_amplitude('v', spin, delta)
                
                T_BSE = t_c * phase_factor_e - t_v * phase_factor_h
                
                H[idx, target_idx] = T_BSE
                H[target_idx, idx] = np.conj(T_BSE)
        
        return H.tocsr()
    
    def solve_bse(self, Q, L, n_states=10, spin=0.5):
        """
        Solve BSE to get exciton energies and wavefunctions.
        
        Parameters:
        -----------
        Q : array
            Exciton center-of-mass momentum
        L : int
            Supercell half-size
        n_states : int
            Number of lowest exciton states to compute
        spin : float
            Spin value
            
        Returns:
        --------
        tuple
            (exciton_energies, exciton_wavefunctions)
        """
        H = self.build_bse_hamiltonian(Q, L, spin)
        
        # Solve eigenvalue problem for lowest states
        energies, wavefunctions = spla.eigsh(
            H, k=n_states, which='SA', tol=1e-9
        )
        
        # Sort by energy
        sort_idx = np.argsort(energies)
        energies = energies[sort_idx]
        wavefunctions = wavefunctions[:, sort_idx]
        
        return energies, wavefunctions