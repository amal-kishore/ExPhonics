"""
Plotting utilities for exciton-phonon coupling visualizations.

Contains specialized plotting functions for band structures, self-energies,
temperature dependence, and convergence analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


class PlotManager:
    """
    Centralized plotting manager with consistent styling and layouts.
    
    Handles all publication-quality plots for the package.
    """
    
    def __init__(self, style='paper', figsize_scale=1.0):
        """
        Initialize plot manager.
        
        Parameters:
        -----------
        style : str
            Plot style ('paper', 'presentation', 'notebook')
        figsize_scale : float
            Scale factor for figure sizes
        """
        self.style = style
        self.figsize_scale = figsize_scale
        
        # Set up plot style
        self.setup_style()
        
        # Define color schemes
        self.setup_colors()
        
        # Define default figure sizes
        self.setup_figure_sizes()
    
    def setup_style(self):
        """Set up matplotlib style parameters."""
        if self.style == 'paper':
            try:
                plt.style.use('seaborn-v0_8-paper')
            except:
                plt.style.use('default')
            self.font_size = 10
            self.line_width = 1.5
            self.marker_size = 6
        elif self.style == 'presentation':
            try:
                plt.style.use('seaborn-v0_8-talk')
            except:
                plt.style.use('default')
            self.font_size = 14
            self.line_width = 2.5
            self.marker_size = 8
        else:  # notebook
            try:
                plt.style.use('seaborn-v0_8-notebook')
            except:
                plt.style.use('default')
            self.font_size = 12
            self.line_width = 2.0
            self.marker_size = 7
        
        # Set universal parameters
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.linewidth': 1.0,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'legend.frameon': False,
            'legend.fontsize': self.font_size - 1,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.top': True,
            'ytick.right': True,
        })
    
    def setup_colors(self):
        """Define color schemes for different plot types."""
        # Band structure colors
        self.band_colors = {
            'valence_up': '#D95F02',    # Orange
            'valence_dn': '#1B9E77',    # Teal
            'conduction_up': '#7570B3', # Purple
            'conduction_dn': '#E7298A'  # Pink
        }
        
        # Exciton state colors (for up to 10 states)
        self.exciton_colors = [
            '#8B008B', '#0000CD', '#006400', '#FF8C00',
            '#DC143C', '#4B0082', '#FFD700', '#FF1493',
            '#32CD32', '#FF6347'
        ]
        
        # Convergence analysis colors
        self.convergence_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        self.convergence_markers = ['o', 's', '^', 'v', 'D', 'P']
        
        # Colormaps
        self.wavefunction_cmap = 'inferno'
        self.density_cmap = 'viridis'
    
    def setup_figure_sizes(self):
        """Define standard figure sizes."""
        scale = self.figsize_scale
        
        self.fig_sizes = {
            'single': (6*scale, 4*scale),
            'double': (12*scale, 4*scale),
            'triple': (15*scale, 4*scale),
            'square': (6*scale, 6*scale),
            'tall': (6*scale, 8*scale),
            'wide': (10*scale, 3*scale)
        }
    
    def plot_band_structure(self, k_path_data, title=None, save_path=None):
        """
        Plot electronic band structure.
        
        Parameters:
        -----------
        k_path_data : dict
            Band structure data with keys: 'distances', 'bands', 'labels', 'tick_positions'
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.fig_sizes['single'])
        
        distances = k_path_data['distances']
        bands = k_path_data['bands']
        labels = k_path_data['labels']
        tick_positions = k_path_data['tick_positions']
        
        # Plot valence bands
        if 'v_up' in bands:
            ax.plot(distances, bands['v_up'], color=self.band_colors['valence_up'],
                   linewidth=self.line_width, label='Valence ↑')
        if 'v_dn' in bands:
            ax.plot(distances, bands['v_dn'], color=self.band_colors['valence_dn'],
                   linewidth=self.line_width, label='Valence ↓')
        
        # Plot conduction bands
        if 'c_up' in bands:
            ax.plot(distances, bands['c_up'], color=self.band_colors['conduction_up'],
                   linewidth=self.line_width, label='Conduction ↑')
        if 'c_dn' in bands:
            ax.plot(distances, bands['c_dn'], color=self.band_colors['conduction_dn'],
                   linewidth=self.line_width, label='Conduction ↓', linestyle='--')
        
        # Formatting
        tick_distances = [distances[i] for i in tick_positions]
        ax.set_xticks(tick_distances)
        ax.set_xticklabels(labels)
        ax.set_xlim(0, distances[-1])
        
        ax.set_ylabel('Energy (eV)')
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Add vertical lines at high-symmetry points
        for tick_dist in tick_distances:
            ax.axvline(tick_dist, color='gray', alpha=0.5, linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_exciton_wavefunctions(self, wavefunction_data, states=[0, 1, 2, 3], 
                                  save_path=None):
        """
        Plot exciton wavefunctions in real space.
        
        Parameters:
        -----------
        wavefunction_data : dict
            Data with keys: 'wavefunctions', 'L', 'energies'
        states : list
            List of exciton state indices to plot
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        n_states = len(states)
        if n_states <= 2:
            fig, axes = plt.subplots(1, n_states, figsize=self.fig_sizes['double'])
        else:
            fig, axes = plt.subplots(2, 2, figsize=self.fig_sizes['square'])
        
        if n_states == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        L = wavefunction_data['L']
        wavefunctions = wavefunction_data['wavefunctions']
        energies = wavefunction_data.get('energies', None)
        
        for i, state in enumerate(states):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get wavefunction probability density
            wf_prob = np.abs(wavefunctions[:, state])**2
            wf_grid = wf_prob.reshape(2*L+1, 2*L+1)
            
            # Plot
            im = ax.imshow(wf_grid, origin='lower', cmap=self.wavefunction_cmap,
                          extent=[-L, L, -L, L])
            
            # Title with energy if available
            if energies is not None:
                title = f'S={state+1}, E={energies[state]:.3f} eV'
            else:
                title = f'Exciton S={state+1}'
            ax.set_title(title)
            
            ax.set_xlabel('x (lattice units)')
            ax.set_ylabel('y (lattice units)')
            
            # Colorbar
            plt.colorbar(im, ax=ax, label=r'$|A^S(R)|^2$', shrink=0.8)
        
        # Hide unused subplots
        for i in range(len(states), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_self_energy_convergence(self, convergence_data, save_path=None):
        """
        Plot self-energy convergence analysis.
        
        Reproduces Figure 7(a-b) style plots.
        
        Parameters:
        -----------
        convergence_data : dict
            Convergence analysis data
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        n_states = len(convergence_data.get('results', {}))
        
        fig, axes = plt.subplots(1, n_states, figsize=self.fig_sizes['double'])
        if n_states == 1:
            axes = [axes]
        
        eta_values = convergence_data.get('eta_values', [])
        
        for s, (state_key, state_data) in enumerate(convergence_data['results'].items()):
            ax = axes[s]
            
            for i, (eta_key, eta_data) in enumerate(state_data.items()):
                eta_meV = int(eta_key.split('_')[1].replace('meV', ''))
                
                x_vals = [point['inv_N_q_squared'] for point in eta_data]
                y_vals = [point['Im_Sigma_meV'] for point in eta_data]
                
                color = self.convergence_colors[i % len(self.convergence_colors)]
                marker = self.convergence_markers[i % len(self.convergence_markers)]
                
                ax.plot(x_vals, y_vals, color=color, marker=marker,
                       markersize=self.marker_size, linewidth=self.line_width,
                       label=f'η = {eta_meV} meV')
            
            ax.set_xlabel('1/N²_q')
            ax.set_ylabel(f'Im Σ_{s+1}(T=0) [meV]')
            ax.set_title(f'({chr(97+s)}) {state_key} Convergence')
            ax.legend()
            ax.grid(True)
            
            if s == 0:  # S=1 should converge to 0
                ax.axhline(0, color='k', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_temperature_dependence(self, temp_data, plot_type='both', save_path=None):
        """
        Plot temperature-dependent exciton properties.
        
        Parameters:
        -----------
        temp_data : dict
            Temperature dependence data
        plot_type : str
            'energies', 'linewidths', or 'both'
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if plot_type == 'both':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.fig_sizes['double'])
        else:
            fig, ax1 = plt.subplots(figsize=self.fig_sizes['single'])
            ax2 = None
        
        T_array = temp_data['temperatures']
        energies = temp_data['energies']
        linewidths = temp_data['lifetimes']
        bare_energies = temp_data.get('bare_energies', None)
        
        n_states = energies.shape[1]
        
        # Plot energies
        if plot_type in ['energies', 'both']:
            for s in range(min(n_states, len(self.exciton_colors))):
                color = self.exciton_colors[s]
                
                # Bare energy (dashed line)
                if bare_energies is not None:
                    ax1.axhline(bare_energies[s], color=color, linestyle='--', 
                               alpha=0.5, linewidth=self.line_width)
                
                # Renormalized energy
                ax1.plot(T_array, energies[:, s], color=color, 
                        linewidth=self.line_width, label=f'S={s+1}')
            
            ax1.set_xlabel('Temperature [K]')
            ax1.set_ylabel('Energy [eV]')
            ax1.set_title('Temperature-dependent Energies')
            ax1.grid(True)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot linewidths
        if plot_type in ['linewidths', 'both']:
            ax_linewidth = ax2 if ax2 is not None else ax1
            
            for s in range(min(n_states, len(self.exciton_colors))):
                color = self.exciton_colors[s]
                Gamma = linewidths[:, s] * 1000  # Convert to meV
                
                ax_linewidth.plot(T_array, Gamma, color=color,
                                linewidth=self.line_width, label=f'S={s+1}')
            
            ax_linewidth.set_xlabel('Temperature [K]')
            ax_linewidth.set_ylabel('Linewidth Γ [meV]')
            ax_linewidth.set_title('Temperature-dependent Linewidths')
            ax_linewidth.legend()
            ax_linewidth.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_convergence_analysis(self, convergence_data, parameter_name, save_path=None):
        """
        Generic convergence analysis plot.
        
        Parameters:
        -----------
        convergence_data : dict
            Convergence data
        parameter_name : str
            Name of parameter being varied
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.fig_sizes['single'])
        
        param_values = convergence_data['param_values']
        quantities = convergence_data['quantities']
        
        if quantities.ndim == 1:
            ax.plot(param_values, quantities, 'o-', linewidth=self.line_width,
                   markersize=self.marker_size)
        else:
            # Multiple quantities to plot
            for i in range(quantities.shape[1]):
                ax.plot(param_values, quantities[:, i], 'o-', 
                       linewidth=self.line_width, markersize=self.marker_size,
                       label=f'Quantity {i+1}')
            ax.legend()
        
        ax.set_xlabel(parameter_name.replace('_', ' ').title())
        ax.set_ylabel('Quantity')
        ax.set_title(f'Convergence vs {parameter_name.replace("_", " ").title()}')
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_figure_summary(self, figures_dict, save_path=None):
        """
        Create a summary figure combining multiple plots.
        
        Parameters:
        -----------
        figures_dict : dict
            Dictionary of figures to combine
        save_path : str, optional
            Path to save combined figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Combined figure
        """
        n_figs = len(figures_dict)
        
        if n_figs <= 2:
            fig, axes = plt.subplots(1, n_figs, figsize=self.fig_sizes['triple'])
        elif n_figs <= 4:
            fig, axes = plt.subplots(2, 2, figsize=self.fig_sizes['square'])
        else:
            fig, axes = plt.subplots(3, 2, figsize=self.fig_sizes['tall'])
        
        if n_figs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # This is a simplified version - in practice would need to 
        # extract and recreate plots from the figures
        for i, (title, figure) in enumerate(figures_dict.items()):
            if i >= len(axes):
                break
            
            axes[i].text(0.5, 0.5, f'{title}\n(Figure placeholder)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(title)
        
        # Hide unused subplots
        for i in range(len(figures_dict), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Convenience functions for quick plotting
def quick_band_plot(k_path_data, save_path=None):
    """Quick band structure plot with default styling."""
    plotter = PlotManager()
    return plotter.plot_band_structure(k_path_data, save_path=save_path)


def quick_wavefunction_plot(wf_data, save_path=None):
    """Quick exciton wavefunction plot."""
    plotter = PlotManager()
    return plotter.plot_exciton_wavefunctions(wf_data, save_path=save_path)


def quick_temperature_plot(temp_data, save_path=None):
    """Quick temperature dependence plot."""
    plotter = PlotManager()
    return plotter.plot_temperature_dependence(temp_data, save_path=save_path)