"""
Two-band triangular lattice tight-binding model.

Implements the specific model from Section IV of the paper for 
transition metal dichalcogenides with spin-orbit coupling.
"""

import numpy as np
import matplotlib.pyplot as plt
from ..core.hamiltonian import TightBindingHamiltonian
from ..core.constants import PhysicalConstants


class TightBindingModel:
    """
    Complete two-band triangular lattice model for TMD monolayers.
    
    This implements the full model from Section IV, including:
    - Spin-orbit coupling in valence band
    - Realistic band structure 
    - High-symmetry k-path definitions
    """
    
    def __init__(self, m_star=0.49, a_bohr=3.13, E_gap=2.5, Delta=0.425):
        """
        Initialize the TMD tight-binding model.
        
        Parameters (Section IV values):
        -----------
        m_star : float
            Effective mass (0.49 m_e)
        a_bohr : float
            Lattice constant (3.13 Bohr)
        E_gap : float  
            Band gap (2.5 eV)
        Delta : float
            Spin-orbit splitting (425 meV)
        """
        self.tb = TightBindingHamiltonian(m_star, a_bohr, E_gap, Delta)
        
        # Store parameters for easy access
        self.m_star = m_star
        self.a_bohr = a_bohr  
        self.E_gap = E_gap
        self.Delta = Delta
        
        # Derived quantities
        self.a = self.tb.a  # Lattice constant in Angstrom
        self.t = self.tb.t  # Hopping parameter
        
        # Define high-symmetry points
        self._setup_brillouin_zone()
    
    def _setup_brillouin_zone(self):
        """Set up high-symmetry points and paths in the Brillouin zone."""
        # High-symmetry points
        self.Gamma = np.array([0.0, 0.0])
        self.M = np.array([np.pi/self.a, np.pi/(np.sqrt(3)*self.a)])
        self.K = np.array([4*np.pi/(3*self.a), 0.0])
        
        # K' point using reciprocal lattice vector
        self.b1 = self.tb.b1
        self.K_prime = -self.K + self.b1
        
        # Standard path for TMDs: Γ-M-K-Γ-K'-M-Γ
        self.high_sym_points = {
            'Γ': self.Gamma,
            'M': self.M, 
            'K': self.K,
            'K\'': self.K_prime
        }
        
        self.standard_path = [
            self.Gamma, self.M, self.K, self.Gamma, self.K_prime, self.M, self.Gamma
        ]
        self.path_labels = ['Γ', 'M', 'K', 'Γ', 'K\'', 'M', 'Γ']
    
    def generate_k_path(self, n_points=200):
        """
        Generate k-points along the standard high-symmetry path.
        
        Parameters:
        -----------
        n_points : int
            Number of points per segment
            
        Returns:
        --------
        tuple
            (k_points, distances, tick_positions, labels)
        """
        k_points = []
        distances = [0.0]
        
        # Generate points along each segment
        for i in range(len(self.standard_path) - 1):
            start_point = self.standard_path[i]
            end_point = self.standard_path[i + 1]
            
            # Linear interpolation between points
            segment_points = np.linspace(start_point, end_point, n_points, endpoint=False)
            k_points.extend(segment_points)
            
            # Calculate cumulative distances
            segment_length = np.linalg.norm(end_point - start_point)
            segment_distances = np.linspace(0, segment_length, n_points, endpoint=False)
            if len(distances) > 1:
                distances.extend(distances[-1] + segment_distances[1:])
            else:
                distances.extend(segment_distances[1:])
        
        # Add final point
        k_points.append(self.standard_path[-1])
        if len(distances) > 0:
            final_segment_length = np.linalg.norm(self.standard_path[-1] - self.standard_path[-2])
            distances.append(distances[-1] + final_segment_length)
        else:
            distances.append(0.0)
        
        k_points = np.array(k_points)
        distances = np.array(distances)
        
        # Tick positions for high-symmetry points
        tick_positions = [i * n_points for i in range(len(self.standard_path))]
        tick_positions[-1] = len(k_points) - 1
        
        return k_points, distances, tick_positions, self.path_labels
    
    def calculate_band_structure(self, n_points=200):
        """
        Calculate electronic band structure along high-symmetry path.
        
        Returns:
        --------
        dict
            Dictionary with band structure data
        """
        k_points, distances, tick_positions, labels = self.generate_k_path(n_points)
        
        # Calculate bands for both spins
        bands = self.tb.calculate_band_structure(
            k_points, 
            bands=['v', 'c'],
            spins=[0.5, -0.5]
        )
        
        return {
            'k_points': k_points,
            'distances': distances,
            'tick_positions': tick_positions,
            'labels': labels,
            'bands': bands
        }
    
    def plot_band_structure(self, n_points=200, save_path=None):
        """
        Plot the electronic band structure.
        
        Reproduces Figure 6(a) from the paper.
        """
        data = self.calculate_band_structure(n_points)
        
        plt.figure(figsize=(8, 6))
        
        # Plot valence bands
        plt.plot(data['distances'], data['bands']['v_up'], 
                label='Valence ↑', color='#D95F02', linewidth=2)
        plt.plot(data['distances'], data['bands']['v_dn'], 
                label='Valence ↓', color='#1B9E77', linewidth=2)
        
        # Plot conduction bands  
        plt.plot(data['distances'], data['bands']['c_up'], 
                label='Conduction ↑', color='#7570B3', linewidth=2)
        plt.plot(data['distances'], data['bands']['c_dn'], 
                label='Conduction ↓', color='#7570B3', linewidth=2, linestyle='--')
        
        # Formatting
        tick_distances = [data['distances'][i] for i in data['tick_positions']]
        plt.xticks(tick_distances, data['labels'])
        plt.xlim(0, data['distances'][-1])
        
        plt.ylabel('Energy (eV)')
        plt.title('Two-band TB Model - Electronic Band Structure')
        plt.legend(frameon=False, fontsize=10)
        plt.grid(alpha=0.3, linestyle='--', linewidth=0.4)
        
        # Add vertical lines at high-symmetry points
        for tick_pos in tick_distances:
            plt.axvline(tick_pos, color='gray', alpha=0.5, linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def get_band_gap(self):
        """Calculate the direct band gap at K and K' points."""
        gaps = []
        
        for k_point in [self.K, self.K_prime]:
            E_v_up = self.tb.band_energy(k_point, 'v', 0.5)
            E_v_dn = self.tb.band_energy(k_point, 'v', -0.5)
            E_c_up = self.tb.band_energy(k_point, 'c', 0.5)
            E_c_dn = self.tb.band_energy(k_point, 'c', -0.5)
            
            # Gap is between highest valence and lowest conduction
            gap = min(E_c_up, E_c_dn) - max(E_v_up, E_v_dn)
            gaps.append(gap)
        
        return np.min(gaps)
    
    def get_spin_splitting(self):
        """Calculate spin splitting in valence band at K points."""
        splittings = []
        
        for k_point in [self.K, self.K_prime]:
            E_v_up = self.tb.band_energy(k_point, 'v', 0.5)
            E_v_dn = self.tb.band_energy(k_point, 'v', -0.5)
            
            splitting = abs(E_v_up - E_v_dn)
            splittings.append(splitting)
        
        return np.max(splittings)
    
    def export_parameters(self):
        """
        Export all model parameters in a dictionary.
        
        Useful for reproducing calculations and parameter studies.
        """
        return {
            'material_parameters': {
                'm_star': self.m_star,
                'a_bohr': self.a_bohr,
                'E_gap': self.E_gap,
                'Delta': self.Delta
            },
            'derived_parameters': {
                'a_angstrom': self.a,
                't': self.t,
                'eps_c': self.tb.eps_c,
                'eps_v': self.tb.eps_v,
                't_tilde_v': self.tb.t_tilde_v
            },
            'calculated_properties': {
                'band_gap': self.get_band_gap(),
                'spin_splitting': self.get_spin_splitting()
            },
            'high_symmetry_points': self.high_sym_points
        }