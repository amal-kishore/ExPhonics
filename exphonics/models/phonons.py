"""
Phonon models for exciton-phonon coupling calculations.

Implements simplified phonon dispersions and coupling mechanisms
as used in the Section IV model system.
"""

import numpy as np
from ..core.constants import PhysicalConstants


class PhononModel:
    """
    Simple phonon model for the two-band system.
    
    Uses a dispersionless optical phonon as in the paper's model,
    with constant electron-phonon coupling strength.
    """
    
    def __init__(self, omega_phonon=0.050, phonon_type='optical'):
        """
        Initialize phonon model.
        
        Parameters:
        -----------
        omega_phonon : float
            Phonon frequency (eV)
        phonon_type : str
            Type of phonon ('optical' or 'acoustic')
        """
        self.omega_ph = omega_phonon
        self.phonon_type = phonon_type
    
    def phonon_frequency(self, q):
        """
        Calculate phonon frequency at wavevector q.
        
        For the model system, we use a dispersionless optical phonon.
        
        Parameters:
        -----------
        q : array
            Phonon wavevector
            
        Returns:
        --------
        float
            Phonon frequency
        """
        if self.phonon_type == 'optical':
            # Dispersionless optical phonon
            return self.omega_ph
        elif self.phonon_type == 'acoustic':
            # Linear acoustic phonon dispersion
            v_sound = 5000.0  # Sound velocity in m/s (typical for TMDs)
            # Convert to appropriate units
            q_norm = np.linalg.norm(q)
            return v_sound * q_norm * 1e-15  # Rough conversion to eV
        else:
            raise ValueError(f"Unknown phonon type: {self.phonon_type}")
    
    def phonon_occupation(self, q, T):
        """
        Calculate phonon occupation number at temperature T.
        
        Parameters:
        -----------
        q : array
            Phonon wavevector
        T : float
            Temperature (K)
            
        Returns:
        --------
        float
            Phonon occupation number
        """
        omega = self.phonon_frequency(q)
        return PhysicalConstants.phonon_population(omega, T)
    
    def generate_q_grid(self, N_q=48, q_max=None):
        """
        Generate a grid of phonon wavevectors for integration.
        
        Parameters:
        -----------
        N_q : int
            Number of q-points along each direction
        q_max : float, optional
            Maximum q-value (if None, use Brillouin zone boundary)
            
        Returns:
        --------
        array
            Array of q-vectors
        """
        if q_max is None:
            # Use a reasonable fraction of the Brillouin zone
            q_max = np.pi / 3.0  # Roughly 1/3 of BZ edge
        
        # Create uniform grid in 2D
        q_1d = np.linspace(-q_max, q_max, N_q)
        q_x, q_y = np.meshgrid(q_1d, q_1d)
        
        # Flatten to get list of q-vectors
        q_grid = np.column_stack([q_x.flatten(), q_y.flatten()])
        
        # Remove q=0 point to avoid singularities
        q_norms = np.linalg.norm(q_grid, axis=1)
        q_grid = q_grid[q_norms > 1e-12]
        
        return q_grid
    
    def calculate_phase_space(self, q_grid):
        """
        Calculate 2D phase space factor for q-integration.
        
        For 2D systems: d²q/(2π)² with appropriate normalization.
        
        Parameters:
        -----------
        q_grid : array
            Grid of q-vectors
            
        Returns:
        --------
        array
            Phase space weights for each q-point
        """
        # For uniform grid, each point has equal weight
        N_q = len(q_grid)
        if N_q > 0:
            # Estimate grid spacing
            q_max = np.max(np.linalg.norm(q_grid, axis=1))
            grid_area = (2 * q_max)**2
            phase_space_element = grid_area / N_q / (2 * np.pi)**2
            
            return np.full(N_q, phase_space_element)
        else:
            return np.array([])
    
    def thermal_average(self, quantity_func, q_grid, T):
        """
        Calculate thermal average of a quantity over phonon modes.
        
        <quantity> = Σ_q quantity(q) * n_B(ω_q, T) * phase_space(q)
        
        Parameters:
        -----------
        quantity_func : callable
            Function that takes q and returns quantity to average
        q_grid : array
            Grid of q-vectors
        T : float
            Temperature (K)
            
        Returns:
        --------
        float
            Thermally averaged quantity
        """
        phase_space = self.calculate_phase_space(q_grid)
        total = 0.0
        
        for i, q in enumerate(q_grid):
            n_q = self.phonon_occupation(q, T)
            quantity = quantity_func(q)
            total += quantity * n_q * phase_space[i]
        
        return total


class OpticalPhononModel(PhononModel):
    """
    Specialized model for optical phonons in TMDs.
    
    Includes more realistic features like:
    - LO-TO splitting
    - Infrared activity
    - Temperature-dependent frequency shifts
    """
    
    def __init__(self, omega_LO=0.050, omega_TO=0.048, dielectric_contrast=True):
        """
        Initialize optical phonon model.
        
        Parameters:
        -----------
        omega_LO : float
            Longitudinal optical phonon frequency (eV)
        omega_TO : float
            Transverse optical phonon frequency (eV)
        dielectric_contrast : bool
            Whether to include long-range dielectric effects
        """
        super().__init__(omega_LO, 'optical')
        self.omega_LO = omega_LO
        self.omega_TO = omega_TO
        self.dielectric_contrast = dielectric_contrast
    
    def phonon_frequency(self, q):
        """
        Calculate optical phonon frequency with LO-TO splitting.
        
        For q → 0: ω → ω_LO (longitudinal) or ω_TO (transverse)
        For finite q: interpolation between LO and TO frequencies
        """
        q_norm = np.linalg.norm(q)
        
        if not self.dielectric_contrast or q_norm < 1e-12:
            # Use average frequency for simplicity
            return (self.omega_LO + self.omega_TO) / 2
        else:
            # Simple interpolation (more sophisticated models exist)
            # At large q, approach TO frequency
            q_characteristic = 0.1  # Characteristic q-scale
            weight = np.exp(-q_norm / q_characteristic)
            return weight * self.omega_LO + (1 - weight) * self.omega_TO
    
    def temperature_shift(self, T, alpha=1e-5):
        """
        Calculate temperature-dependent phonon frequency shift.
        
        ω(T) = ω(0) - α*T for simple linear model
        
        Parameters:
        -----------
        T : float
            Temperature (K)
        alpha : float
            Temperature coefficient (eV/K)
            
        Returns:
        --------
        float
            Frequency shift
        """
        return -alpha * T
    
    def effective_frequency(self, q, T):
        """Get temperature-dependent phonon frequency."""
        omega_0 = self.phonon_frequency(q)
        shift = self.temperature_shift(T)
        return omega_0 + shift


class AcousticPhononModel(PhononModel):
    """
    Model for acoustic phonons with linear dispersion.
    
    Important for low-temperature physics and certain scattering processes.
    """
    
    def __init__(self, v_sound=5000.0, mass_density=3.0e-6):
        """
        Initialize acoustic phonon model.
        
        Parameters:
        -----------
        v_sound : float
            Sound velocity (m/s)
        mass_density : float
            2D mass density (kg/m²)
        """
        super().__init__(0.0, 'acoustic')  # Zero frequency at q=0
        self.v_sound = v_sound
        self.mass_density = mass_density
        
        # Convert sound velocity to appropriate units for calculations
        # v_sound in eV⋅Å⋅ps units
        self.v_sound_atomic = v_sound * 1e-10 * 6.58e-4  # Rough conversion
    
    def phonon_frequency(self, q):
        """Linear acoustic phonon dispersion: ω = v_s |q|"""
        q_norm = np.linalg.norm(q)
        return self.v_sound_atomic * q_norm
    
    def deformation_potential_coupling(self, band='valence'):
        """
        Estimate deformation potential coupling for acoustic phonons.
        
        This gives the coupling strength for acoustic phonon scattering.
        
        Parameters:
        -----------
        band : str
            'valence' or 'conduction'
            
        Returns:
        --------
        float
            Deformation potential (eV)
        """
        # Typical values for TMDs
        if band == 'valence':
            return 3.0  # eV
        else:  # conduction
            return 1.5  # eV