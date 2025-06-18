"""
Brillouin zone utilities for k-point generation and symmetry operations.

Handles k-point paths, grids, and Brillouin zone operations for 2D triangular lattice.
"""

import numpy as np
import matplotlib.pyplot as plt


class BrillouinZone:
    """
    Brillouin zone handler for 2D triangular lattice.
    
    Manages k-point generation, high-symmetry paths, and zone operations.
    """
    
    def __init__(self, a):
        """
        Initialize Brillouin zone for triangular lattice.
        
        Parameters:
        -----------
        a : float
            Lattice constant (Angstrom)
        """
        self.a = a
        
        # Reciprocal lattice vectors
        self.b1 = np.array([2*np.pi/a, 2*np.pi/(a*np.sqrt(3))])
        self.b2 = np.array([0.0, 4*np.pi/(a*np.sqrt(3))])
        
        # High-symmetry points
        self.setup_high_symmetry_points()
        
        # Standard paths
        self.setup_standard_paths()
    
    def setup_high_symmetry_points(self):
        """Set up high-symmetry points in the Brillouin zone."""
        self.Gamma = np.array([0.0, 0.0])
        self.M = np.array([np.pi/self.a, np.pi/(np.sqrt(3)*self.a)])
        self.K = np.array([4*np.pi/(3*self.a), 0.0])
        self.K_prime = -self.K + self.b1
        
        self.high_sym_points = {
            'Γ': self.Gamma,
            'G': self.Gamma,  # Alternative notation
            'M': self.M,
            'K': self.K,
            'K\'': self.K_prime,
            'Kp': self.K_prime  # Alternative notation
        }
    
    def setup_standard_paths(self):
        """Define standard high-symmetry paths."""
        self.paths = {
            'standard': {
                'points': [self.Gamma, self.M, self.K, self.Gamma, self.K_prime, self.M, self.Gamma],
                'labels': ['Γ', 'M', 'K', 'Γ', 'K\'', 'M', 'Γ']
            },
            'triangle': {
                'points': [self.Gamma, self.M, self.K, self.Gamma],
                'labels': ['Γ', 'M', 'K', 'Γ']
            },
            'hexagon': {
                'points': [self.K, self.Gamma, self.M, self.K_prime, self.Gamma],
                'labels': ['K', 'Γ', 'M', 'K\'', 'Γ']
            }
        }
    
    def generate_k_path(self, path_name='standard', n_points=200):
        """
        Generate k-points along a high-symmetry path.
        
        Parameters:
        -----------
        path_name : str
            Name of path ('standard', 'triangle', 'hexagon') or custom
        n_points : int
            Number of points per segment
            
        Returns:
        --------
        tuple
            (k_points, distances, tick_positions, labels)
        """
        if isinstance(path_name, str):
            if path_name not in self.paths:
                raise ValueError(f"Unknown path: {path_name}")
            path_info = self.paths[path_name]
            points = path_info['points']
            labels = path_info['labels']
        else:
            # Custom path provided as list of points
            points = path_name
            labels = [f'P{i}' for i in range(len(points))]
        
        k_points = []
        distances = [0.0]
        
        # Generate points along each segment
        for i in range(len(points) - 1):
            start_point = points[i]
            end_point = points[i + 1]
            
            # Linear interpolation
            segment_points = np.linspace(start_point, end_point, n_points, endpoint=False)
            k_points.extend(segment_points)
            
            # Calculate distances
            segment_length = np.linalg.norm(end_point - start_point)
            segment_distances = np.linspace(0, segment_length, n_points, endpoint=False)
            distances.extend(distances[-1] + segment_distances[1:])
        
        # Add final point
        k_points.append(points[-1])
        distances.append(distances[-1])
        
        # Tick positions
        tick_positions = [i * n_points for i in range(len(points))]
        tick_positions[-1] = len(k_points) - 1
        
        return np.array(k_points), np.array(distances), tick_positions, labels
    
    def generate_k_grid(self, n_k, k_max=None, grid_type='square'):
        """
        Generate uniform k-point grid.
        
        Parameters:
        -----------
        n_k : int
            Number of k-points along each direction
        k_max : float, optional
            Maximum k-value (if None, use BZ boundary)
        grid_type : str
            Type of grid ('square', 'circular', 'hexagonal')
            
        Returns:
        --------
        array
            Array of k-points
        """
        if k_max is None:
            k_max = np.linalg.norm(self.K)  # Use distance to K point
        
        if grid_type == 'square':
            k_1d = np.linspace(-k_max, k_max, n_k)
            k_x, k_y = np.meshgrid(k_1d, k_1d)
            k_points = np.column_stack([k_x.flatten(), k_y.flatten()])
            
        elif grid_type == 'circular':
            # Polar grid
            r_values = np.linspace(0, k_max, n_k//2)
            theta_values = np.linspace(0, 2*np.pi, n_k, endpoint=False)
            
            k_points = []
            for r in r_values:
                for theta in theta_values:
                    k_x = r * np.cos(theta)
                    k_y = r * np.sin(theta)
                    k_points.append([k_x, k_y])
            
            k_points = np.array(k_points)
            
        elif grid_type == 'hexagonal':
            # Hexagonal grid aligned with BZ
            k_points = self._generate_hexagonal_grid(n_k, k_max)
            
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        return k_points
    
    def _generate_hexagonal_grid(self, n_k, k_max):
        """Generate hexagonal k-point grid aligned with BZ."""
        # This is a simplified implementation
        # In practice, would use proper hexagonal lattice generation
        k_points = []
        
        # Generate points in hexagonal pattern
        for i in range(-n_k//2, n_k//2 + 1):
            for j in range(-n_k//2, n_k//2 + 1):
                # Hexagonal coordinates
                k = i * self.b1/n_k + j * self.b2/n_k
                
                if np.linalg.norm(k) <= k_max:
                    k_points.append(k)
        
        return np.array(k_points)
    
    def is_in_first_bz(self, k_point):
        """
        Check if k-point is in the first Brillouin zone.
        
        For triangular lattice, this is a hexagon.
        """
        # Simplified check - use circular approximation
        k_max = np.linalg.norm(self.K)
        return np.linalg.norm(k_point) <= k_max
    
    def fold_to_first_bz(self, k_point):
        """
        Fold k-point back to first Brillouin zone.
        
        Uses reciprocal lattice vectors to find equivalent point.
        """
        # This is a simplified implementation
        # Proper implementation would use Wigner-Seitz cell
        
        k_folded = k_point.copy()
        
        # Try different reciprocal lattice vector combinations
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                k_candidate = k_point - n1*self.b1 - n2*self.b2
                
                if (np.linalg.norm(k_candidate) < np.linalg.norm(k_folded) and
                    self.is_in_first_bz(k_candidate)):
                    k_folded = k_candidate
        
        return k_folded
    
    def plot_brillouin_zone(self, show_paths=True, show_grid=False, n_grid=20):
        """
        Plot the Brillouin zone with high-symmetry points and paths.
        
        Parameters:
        -----------
        show_paths : bool
            Whether to show high-symmetry paths
        show_grid : bool
            Whether to show k-point grid
        n_grid : int
            Grid size if show_grid=True
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw BZ boundary (hexagon for triangular lattice)
        theta = np.linspace(0, 2*np.pi, 7)
        k_boundary = np.linalg.norm(self.K)
        bz_x = k_boundary * np.cos(theta + np.pi/6)  # Rotate for proper orientation
        bz_y = k_boundary * np.sin(theta + np.pi/6)
        
        ax.plot(bz_x, bz_y, 'k-', linewidth=2, label='BZ boundary')
        
        # Plot high-symmetry points
        for label, point in self.high_sym_points.items():
            ax.plot(point[0], point[1], 'ro', markersize=8)
            ax.annotate(label, point, xytext=(5, 5), textcoords='offset points',
                       fontsize=12, fontweight='bold')
        
        # Plot high-symmetry paths
        if show_paths:
            for path_name, path_info in self.paths.items():
                if path_name == 'standard':  # Only show main path
                    points = path_info['points']
                    for i in range(len(points) - 1):
                        ax.plot([points[i][0], points[i+1][0]], 
                               [points[i][1], points[i+1][1]], 
                               'b-', linewidth=1, alpha=0.7)
        
        # Plot k-point grid
        if show_grid:
            k_grid = self.generate_k_grid(n_grid, grid_type='hexagonal')
            ax.plot(k_grid[:, 0], k_grid[:, 1], 'g.', markersize=2, alpha=0.5)
        
        ax.set_xlabel('k_x (Å⁻¹)')
        ax.set_ylabel('k_y (Å⁻¹)')
        ax.set_title('Brillouin Zone - Triangular Lattice')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig


def generate_k_path(high_sym_points, n_points=200):
    """
    Generate k-path from list of high-symmetry points.
    
    Convenience function for custom paths.
    
    Parameters:
    -----------
    high_sym_points : list
        List of k-points as [k1, k2, k3, ...]
    n_points : int
        Number of points per segment
        
    Returns:
    --------
    tuple
        (k_points, distances, tick_positions)
    """
    k_points = []
    distances = [0.0]
    
    for i in range(len(high_sym_points) - 1):
        start = high_sym_points[i]
        end = high_sym_points[i + 1]
        
        segment = np.linspace(start, end, n_points, endpoint=False)
        k_points.extend(segment)
        
        segment_length = np.linalg.norm(end - start)
        segment_distances = np.linspace(0, segment_length, n_points, endpoint=False)
        distances.extend(distances[-1] + segment_distances[1:])
    
    k_points.append(high_sym_points[-1])
    distances.append(distances[-1])
    
    tick_positions = [i * n_points for i in range(len(high_sym_points))]
    tick_positions[-1] = len(k_points) - 1
    
    return np.array(k_points), np.array(distances), tick_positions