# ExPhonics User Manual

## Table of Contents
1. [Quick Start](#quick-start)
2. [Command Line Interface](#command-line-interface)
3. [Python API](#python-api)
4. [Physical Parameters](#physical-parameters)
5. [Output Files](#output-files)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Documentation

📖 **[Complete Parameter Guide](docs/user_parameters.md)** - Detailed documentation of all parameters with usage examples and material-specific recommendations.

## Quick Start

### Installation
```bash
cd exphonics
pip install -e .
```

### Generate All Plots
```bash
python run_exphonics_demo.py
```

## Command Line Interface

### Basic Usage

#### Method 1: Direct Script (requires python prefix)
```bash
python exphonics_cli.py --help       # Show help
python exphonics_cli.py --list       # List available plots
python exphonics_cli.py --all        # Generate all plots
python exphonics_cli.py --plot electronic_bands  # Specific plot
```

#### Method 2: Executable Script (no python needed)
```bash
chmod +x exphonics_cli.py           # Make executable (one time)
./exphonics_cli.py --help           # Show help
./exphonics_cli.py --list           # List available plots  
./exphonics_cli.py --all            # Generate all plots
```

#### Method 3: Package Installation (cleanest)
```bash
pip install -e .                    # Install package (one time)
exphonics --help                    # Show help
exphonics --list                    # List available plots
exphonics --all                     # Generate all plots
```

### Available Plot Types
- `electronic_bands`: Electronic band structure from tight-binding model
- `exciton_bands`: Exciton band structure from BSE calculations
- `wavefunctions`: Real-space exciton wavefunction visualization  
- `self_energy`: Self-energy convergence and temperature dependence
- `frequency_dependent`: Frequency-dependent self-energy comparison

### Advanced Options
```bash
# Custom output directory
python exphonics_cli.py --all --output ./results/

# Modify physical parameters
python exphonics_cli.py --plot frequency_dependent --coupling 0.3 --temperature 300

# GPU acceleration
python exphonics_cli.py --all --gpu

# Verbose output
python exphonics_cli.py --all --verbose
```

## Python API

### Import Functions
```python
from run_exphonics_demo import (
    create_electronic_band_structure,
    create_exciton_band_structure,
    create_exciton_wavefunctions,
    create_self_energy_convergence,
    create_frequency_dependent_self_energy
)
```

### Generate Individual Plots
```python
import matplotlib.pyplot as plt

# Electronic band structure
fig1 = create_electronic_band_structure()
plt.show()

# Exciton band structure (requires GPU for large calculations)
fig2 = create_exciton_band_structure()
plt.show()

# Exciton wavefunctions
fig3 = create_exciton_wavefunctions()
plt.show()

# Self-energy analysis
fig4 = create_self_energy_convergence()
plt.show()

# Frequency-dependent self-energy
fig5 = create_frequency_dependent_self_energy()
plt.show()
```

### Using the Package Import
```python
import exphonics

# Print package info
exphonics.print_info()

# Get default parameters
params = exphonics.get_default_parameters()
print(params)

# Generate plots
fig = exphonics.create_frequency_dependent_self_energy()
```

## Physical Parameters

### Default Values (from theoretical model)
```python
# System parameters
effective_mass = 0.49      # m*
lattice_constant = 3.13    # a (Bohr)
band_gap = 2.5            # E_g (eV)
spin_orbit = 0.425        # Δ (eV)

# Interaction parameters  
coupling_strength = 0.250  # g (eV) - electron-phonon coupling
phonon_energy = 0.050     # ω₀ (eV)
broadening = 0.010        # η (eV)
coulomb_strength = 1.6    # Δv₀ (eV)
```

### Modifying Parameters
To change parameters, edit the relevant sections in `run_exphonics_demo.py`:

```python
# Model parameters
omega_ph = 0.050  # Optical phonon (eV) - ω₀ = 50 meV
g_base = 0.250    # Base coupling strength (eV) - g = 250 meV
E_gap = 2.5       # Band gap (eV) - E_g = 2.5 eV
eta = 0.010       # Broadening parameter (eV) - η = 10 meV
```

## Output Files

### Generated Plots
| Filename | Description | Function |
|----------|-------------|----------|
| `electronic_band_structure.png` | Two-band electronic dispersion | `create_electronic_band_structure()` |
| `exciton_band_structure.png` | BSE exciton bands along Γ-M-K path | `create_exciton_band_structure()` |
| `exciton_wavefunctions.png` | Real-space \|A^S(R)\|² for S=1,2,3,4 | `create_exciton_wavefunctions()` |
| `self_energy_convergence.png` | Convergence vs k-points and temperature | `create_self_energy_convergence()` |
| `frequency_dependent_self_energy.png` | Self-energy with/without e-h interaction | `create_frequency_dependent_self_energy()` |

### Plot Details

#### Electronic Band Structure
- Shows valence and conduction bands with spin-orbit splitting
- High-symmetry path: Γ-M-K-Γ-K'-M-Γ
- Energy range: -2.2 to 4.2 eV

#### Exciton Band Structure  
- First 4 exciton states (S=1,2,3,4)
- Exciton binding energies: 100-500 meV
- Requires BSE solution (computationally intensive)

#### Exciton Wavefunctions
- Real-space probability density |A^S(R)|²
- Shows localization of different exciton states
- S=1: Localized, S=2: Extended, S=3,4: More complex patterns

#### Self-Energy Convergence
- Panel (a): Convergence vs k-point grid
- Panel (b): Temperature dependence 0-500K
- Shows energy shifts and broadening

#### Frequency-Dependent Self-Energy
- Imaginary part comparison: with vs without e-h interaction
- Key physics: spectral weight transfer
- IEHPP vs full exciton treatment

## Advanced Usage

### Custom Temperature Range
```python
# Modify temperature array in create_self_energy_convergence()
T_array = np.linspace(0, 800, 41)  # 0-800K instead of 0-500K
```

### Different Exciton State
```python
# Change from S=2 to S=1 in create_frequency_dependent_self_energy()
Omega_S = 2.35  # S=1 exciton energy
```

### Higher Resolution
```python
# Increase frequency grid resolution
omega_array = np.linspace(1.0, 4.5, 1000)  # Higher resolution
```

### GPU Acceleration
```python
try:
    import cupy as cp
    print("Using GPU acceleration")
    # BSE calculations will automatically use GPU
except ImportError:
    print("Using CPU (install cupy for GPU acceleration)")
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors
**Problem**: Out of memory during BSE calculations
**Solution**: Reduce supercell size
```python
# In create_exciton_band_structure()
exciton_energies = bse_solver.solve_optical_excitons(L=15)  # Smaller supercell
```

#### 2. Slow Performance
**Problem**: Calculations taking too long
**Solutions**:
- Install CuPy for GPU acceleration
- Reduce k-point grid density
- Use fewer exciton states

#### 3. Import Errors
**Problem**: Cannot import package modules
**Solution**: Check Python path
```python
import sys
sys.path.insert(0, '/path/to/exphonics')
```

#### 4. Plot Display Issues
**Problem**: Plots not showing
**Solutions**:
```python
import matplotlib
matplotlib.use('Agg')  # For headless systems
# or
matplotlib.use('TkAgg')  # For interactive display
```

### Performance Tips

1. **Use GPU**: Install CuPy for 10-100x speedup in BSE calculations
2. **Optimize grids**: Balance accuracy vs speed in k-point sampling
3. **Parallel execution**: Use multiple cores where possible
4. **Memory management**: Close plots with `plt.close()` to free memory

### Getting Help

1. **Check verbose output**: Use `--verbose` flag for detailed information
2. **Review parameters**: Ensure physical parameters are reasonable
3. **Test individual plots**: Generate one plot at a time to isolate issues
4. **Check dependencies**: Ensure numpy, matplotlib, scipy are installed

### Example Workflow
```bash
# 1. Quick test
python exphonics_cli.py --plot electronic_bands --verbose

# 2. Check GPU availability  
python -c "import cupy; print('GPU available')"

# 3. Generate specific analysis
python exphonics_cli.py --plot frequency_dependent --output ./results/

# 4. Full calculation
python exphonics_cli.py --all --gpu --verbose
```