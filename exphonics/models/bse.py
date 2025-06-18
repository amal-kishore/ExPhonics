"""
Bethe-Salpeter equation solver for exciton calculations.

Implements the full BSE solution for the two-band model including
real-space visualization and momentum-space analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from ..core.hamiltonian import BSEHamiltonian
from ..core.constants import PhysicalConstants


class BSESolver:
    """
    Complete BSE solver for exciton states in the two-band model.
    
    Handles both Γ-point (optical) excitons and finite-momentum excitons
    needed for phonon scattering calculations.
    """
    
    def __init__(self, tb_model, v0=1.6, epsilon_r=4.5):
        """
        Initialize BSE solver.
        
        Parameters:
        -----------
        tb_model : TightBindingModel
            Tight-binding model instance
        v0 : float
            On-site Coulomb interaction (eV)
        epsilon_r : float
            Relative dielectric constant
        """
        self.tb_model = tb_model
        self.bse = BSEHamiltonian(tb_model.tb, v0, epsilon_r)
        self.v0 = v0
        self.epsilon_r = epsilon_r
        
        # Storage for computed results
        self.exciton_results = {}
    
    def solve_optical_excitons(self, L=25, n_states=10):
        """
        Solve for optical excitons at Γ point (Q = 0).
        
        Parameters:
        -----------
        L : int
            Supercell half-size (total size = (2L+1)²)
        n_states : int
            Number of lowest exciton states to compute
            
        Returns:
        --------
        dict
            Dictionary with exciton energies and wavefunctions
        """
        Q_gamma = np.array([0.0, 0.0])
        
        # Solve BSE
        energies, wavefunctions = self.bse.solve_bse(Q_gamma, L, n_states)
        
        # Convert to real-space grid
        R_vectors, site_indices = self.bse.setup_supercell(L)
        
        results = {
            'Q': Q_gamma,
            'L': L,
            'energies': energies,
            'wavefunctions': wavefunctions,
            'R_vectors': R_vectors,
            'site_indices': site_indices,
            'n_states': n_states
        }
        
        self.exciton_results['gamma'] = results
        return results
    
    def solve_finite_momentum_excitons(self, Q_points=None, Q=None, L=25, n_states=4, sigma=0.5):
        """
        Solve for finite-momentum excitons needed for phonon scattering.
        
        Parameters:
        -----------
        Q_points : array, optional
            Array of exciton center-of-mass momenta (for batch processing)
        Q : array, optional  
            Single exciton center-of-mass momentum (for individual calculation)
        L : int
            Supercell half-size
        n_states : int
            Number of states per Q-point
        sigma : float
            Spin value (+0.5 or -0.5)
            
        Returns:
        --------
        dict
            Dictionary with results for each Q-point or single Q-point
        """
        # Handle single Q-point case (new functionality)
        if Q is not None:
            energies, wavefunctions = self.bse.solve_bse(Q, L, n_states, spin=sigma)
            return {
                'Q': Q,
                'energies': energies,
                'wavefunctions': wavefunctions,
                'sigma': sigma
            }
        
        # Handle multiple Q-points case (original functionality)  
        if Q_points is None:
            raise ValueError("Must provide either Q_points or Q")
            
        finite_q_results = {}
        
        for i, Q_vec in enumerate(Q_points):
            energies, wavefunctions = self.bse.solve_bse(Q_vec, L, n_states, spin=sigma)
            
            finite_q_results[f'Q_{i}'] = {
                'Q': Q_vec,
                'energies': energies,
                'wavefunctions': wavefunctions,
                'sigma': sigma
            }
        
        self.exciton_results['finite_q'] = finite_q_results
        return finite_q_results
    
    def calculate_binding_energies(self, optical_results=None):
        """
        Calculate exciton binding energies.
        
        Binding energy = (E_gap - Ω_S) where Ω_S is exciton energy.
        """
        if optical_results is None:
            if 'gamma' not in self.exciton_results:
                optical_results = self.solve_optical_excitons()
            else:
                optical_results = self.exciton_results['gamma']
        
        # Get band gap
        band_gap = self.tb_model.get_band_gap()
        
        # Calculate binding energies
        exciton_energies = optical_results['energies']
        binding_energies = band_gap - exciton_energies
        
        return binding_energies
    
    def plot_exciton_wavefunctions(self, optical_results=None, states=[0, 1, 2, 3], 
                                  save_path=None):
        """
        Plot real-space exciton wavefunctions |A^S(R)|².
        
        Reproduces Figure 6(c-f) from the paper.
        """
        if optical_results is None:
            if 'gamma' not in self.exciton_results:
                optical_results = self.solve_optical_excitons()
            else:
                optical_results = self.exciton_results['gamma']
        
        L = optical_results['L']
        wavefunctions = optical_results['wavefunctions']
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        
        for i, state in enumerate(states):
            if i >= 4:  # Only plot first 4 states
                break
                
            ax = axes[i]
            
            # Reshape wavefunction to 2D grid
            wf_2d = np.abs(wavefunctions[:, state])**2
            wf_grid = wf_2d.reshape(2*L+1, 2*L+1)
            
            # Plot
            im = ax.imshow(wf_grid, origin='lower', cmap='inferno',
                          extent=[-L, L, -L, L])
            
            ax.set_title(f'Exciton S={state+1}')
            ax.set_xlabel('x (lattice units)')
            ax.set_ylabel('y (lattice units)')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label=r'$|A^S(R)|^2$')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def calculate_oscillator_strengths(self, optical_results=None):
        """
        Calculate oscillator strengths for optical transitions.
        
        This determines which excitons are bright (optically active).
        """
        if optical_results is None:
            optical_results = self.exciton_results.get('gamma')
            if optical_results is None:
                raise ValueError("Need to solve optical excitons first")
        
        wavefunctions = optical_results['wavefunctions']
        n_states = optical_results['n_states']
        
        # For the model, oscillator strength is related to 
        # the wavefunction amplitude at R=0 (on-site)
        L = optical_results['L']
        center_index = L * (2*L + 1) + L  # Center of the grid
        
        oscillator_strengths = []
        for s in range(n_states):
            # Oscillator strength ∝ |A^S(0)|²
            f_osc = np.abs(wavefunctions[center_index, s])**2
            oscillator_strengths.append(f_osc)
        
        return np.array(oscillator_strengths)
    
    def analyze_exciton_sizes(self, optical_results=None):
        """
        Analyze the spatial extent of excitons.
        
        Calculates average radius and other size measures.
        """
        if optical_results is None:
            optical_results = self.exciton_results.get('gamma')
            if optical_results is None:
                raise ValueError("Need to solve optical excitons first")
        
        R_vectors = optical_results['R_vectors']
        wavefunctions = optical_results['wavefunctions']
        n_states = optical_results['n_states']
        
        exciton_sizes = []
        
        for s in range(n_states):
            wf = wavefunctions[:, s]
            prob_density = np.abs(wf)**2
            prob_density /= np.sum(prob_density)  # Normalize
            
            # Calculate average radius
            radii = np.linalg.norm(R_vectors, axis=1)
            avg_radius = np.sum(prob_density * radii)
            
            # Calculate RMS radius
            rms_radius = np.sqrt(np.sum(prob_density * radii**2))
            
            exciton_sizes.append({
                'average_radius': avg_radius,
                'rms_radius': rms_radius
            })
        
        return exciton_sizes
    
    def export_for_phonon_coupling(self):
        """
        Export exciton data in format needed for phonon coupling calculations.
        
        Returns:
        --------
        dict
            Processed exciton data for self-energy calculations
        """
        if 'gamma' not in self.exciton_results:
            raise ValueError("Need to solve optical excitons first")
        
        optical_results = self.exciton_results['gamma']
        
        export_data = {
            'energies': optical_results['energies'],
            'wavefunctions': optical_results['wavefunctions'],
            'binding_energies': self.calculate_binding_energies(),
            'oscillator_strengths': self.calculate_oscillator_strengths(),
            'exciton_sizes': self.analyze_exciton_sizes(),
            'parameters': {
                'L': optical_results['L'],
                'v0': self.v0,
                'epsilon_r': self.epsilon_r,
                'n_states': optical_results['n_states']
            }
        }
        
        return export_data
    
    def convergence_test(self, L_values=[15, 20, 25, 30], n_states=4):
        """
        Test convergence with respect to supercell size.
        
        Parameters:
        -----------
        L_values : list
            List of supercell half-sizes to test
        n_states : int
            Number of states to track
            
        Returns:
        --------
        dict
            Convergence data
        """
        convergence_data = {
            'L_values': L_values,
            'energies': [],
            'binding_energies': []
        }
        
        for L in L_values:
            results = self.solve_optical_excitons(L=L, n_states=n_states)
            binding_energies = self.calculate_binding_energies(results)
            
            convergence_data['energies'].append(results['energies'][:n_states])
            convergence_data['binding_energies'].append(binding_energies[:n_states])
        
        convergence_data['energies'] = np.array(convergence_data['energies'])
        convergence_data['binding_energies'] = np.array(convergence_data['binding_energies'])
        
        return convergence_data
    
    def plot_convergence(self, convergence_data, save_path=None):
        """Plot convergence of exciton energies vs supercell size."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        L_values = convergence_data['L_values']
        energies = convergence_data['energies']
        binding_energies = convergence_data['binding_energies']
        
        # Plot exciton energies
        for s in range(energies.shape[1]):
            ax1.plot(L_values, energies[:, s], 'o-', label=f'S={s+1}')
        
        ax1.set_xlabel('Supercell half-size L')
        ax1.set_ylabel('Exciton energy (eV)')
        ax1.set_title('Convergence of Exciton Energies')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot binding energies
        for s in range(binding_energies.shape[1]):
            ax2.plot(L_values, binding_energies[:, s], 's-', label=f'S={s+1}')
        
        ax2.set_xlabel('Supercell half-size L')
        ax2.set_ylabel('Binding energy (eV)')
        ax2.set_title('Convergence of Binding Energies')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig