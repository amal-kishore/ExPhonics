# ExPhonics: Exciton-Phonon Interaction Calculator

A comprehensive Python package for calculating exciton-phonon interactions and self-energies using many-body perturbation theory and the Bethe-Salpeter equation.

## Overview

ExPhonics implements the theoretical framework for computing:
- Electronic band structures from tight-binding models
- Exciton band structures via the Bethe-Salpeter Equation (BSE)
- Exciton-phonon self-energies with and without electron-hole interaction
- Temperature-dependent exciton properties and lifetimes
- Convergence analysis for computational parameters

## Key Features

### 🔬 **Physical Models**
- Two-band tight-binding model for 2D materials
- Full BSE treatment of excitonic effects
- Fan-Migdal dynamic and static self-energy terms
- Debye-Waller temperature effects
- Completion correction for missing high-energy states

### 📊 **Visualizations**
- Electronic band structure plots
- Exciton band dispersions along high-symmetry paths
- Real-space exciton wavefunction visualization
- Self-energy convergence analysis
- Frequency-dependent self-energy comparisons

### ⚡ **Performance**
- GPU acceleration support with CuPy
- Optimized matrix operations
- Parallel computation capabilities

## Installation

### Prerequisites
```bash
# Required dependencies
pip install numpy matplotlib scipy

# Optional GPU acceleration
pip install cupy  # For NVIDIA GPUs
```

### Package Installation
```bash
# Clone the repository
git clone <repository_url>
cd exphonics

# Install in development mode
pip install -e .
```

## Quick Start

### Command Line Interface

#### Option 1: Direct Script Execution
```bash
# Generate all plots
python run_exphonics_demo.py

# Using the CLI tool directly
python exphonics_cli.py --all
python exphonics_cli.py --plot frequency_dependent
```

#### Option 2: Executable Script (after chmod +x)
```bash
# Make executable
chmod +x exphonics_cli.py

# Run without python prefix
./exphonics_cli.py --all
./exphonics_cli.py --plot electronic_bands
```

#### Option 3: Package Installation (recommended)
```bash
# Install package
pip install -e .

# Use as command (if entry points work)
exphonics --all
exphonics --plot frequency_dependent
```

### Python API

```python
import sys
sys.path.insert(0, '/path/to/exphonics')

# Import main functions
from run_exphonics_demo import (
    create_electronic_band_structure,
    create_exciton_band_structure,
    create_exciton_wavefunctions,
    create_self_energy_convergence,
    create_frequency_dependent_self_energy
)

# Generate electronic band structure
fig = create_electronic_band_structure()

# Create exciton analysis
fig = create_frequency_dependent_self_energy()
```

## Physical Parameters

The package uses the following parameters based on the theoretical model:

### System Parameters
- **Effective mass**: m* = 0.49
- **Lattice constant**: a = 3.13 Bohr
- **Band gap**: E_g = 2.5 eV
- **Spin-orbit coupling**: Δ = 425 meV

### Interaction Parameters
- **Electron-phonon coupling**: g = 250 meV
- **Phonon frequency**: ω₀ = 50 meV
- **Broadening parameter**: η = 10 meV
- **Screened Coulomb**: Δv₀ = 1.6 eV

## Generated Output Files

When running the complete demo, the following files are generated:

| File | Description |
|------|-------------|
| `electronic_band_structure.png` | Two-band electronic dispersion |
| `exciton_band_structure.png` | BSE exciton bands along Γ-M-K path |
| `exciton_wavefunctions.png` | Real-space \|A^S(R)\|² for S=1,2,3,4 |
| `self_energy_convergence.png` | Convergence vs k-points and temperature |
| `frequency_dependent_self_energy.png` | Self-energy with/without e-h interaction |

## Scientific Background

### Exciton-Phonon Self-Energy

The package implements the complete self-energy formalism:

**Full Interacting Case:**
```
Σ_{SS'} = Σ^{FMd}_{SS'} + Σ^{FMs}_{SS'} + Σ^{DW}_{SS'} + Σ^C_{SS'}
```

**Uncorrelated Case (IEHPP):**
```
Σ^0_{vcv'c'} = Σ^{0,FM}_{vcv'c'} + Σ^{0,X}_{vcv'c'} + Σ^{0,DW}_{vcv'c'}
```

### Key Physics

1. **Bound vs Unbound States**: Excitons (bound e-h pairs) vs independent e-h transitions
2. **Energy Shifts**: Real part of self-energy gives temperature-dependent energy renormalization
3. **Lifetimes**: Imaginary part determines inverse lifetimes and broadening
4. **Spectral Weight Transfer**: e-h interaction moves spectral weight to lower frequencies

## Documentation

📖 **[Complete Parameter Guide](docs/user_parameters.md)** - Comprehensive guide to all user parameters, their meanings, and how to modify them.

## Advanced Usage

### Custom Parameters

Most physics parameters can be controlled via CLI:

```bash
# Modify physics parameters
exphonics --plot frequency_dependent \
  --coupling 0.35 \
  --phonon-energy 0.06 \
  --temperature 300 \
  --broadening 0.015
```

For tight-binding model parameters, see the [Parameter Guide](docs/user_parameters.md).

### GPU Acceleration

For large calculations, ensure CuPy is installed:
```python
try:
    import cupy as cp
    print("GPU acceleration available")
except ImportError:
    print("Using CPU computation")
```

### Memory Optimization

For memory-intensive calculations:
- Reduce k-point grid density
- Limit number of exciton states
- Use smaller real-space grids for wavefunctions

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce computational grid sizes
2. **Slow convergence**: Increase broadening parameter η
3. **GPU issues**: Ensure CuPy version matches CUDA installation

### Performance Tips

- Use GPU acceleration for BSE calculations
- Optimize k-point sampling for your system
- Consider parallel execution for parameter sweeps

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{exphonics,
  title={ExPhonics: Exciton-Phonon Interaction Calculator},
  author={Amal Kishore},
  year={2024},
  url={https://github.com/amal-kishore/ExPhonics}
}
```

## References

The theoretical framework is based on:
- Many-body perturbation theory for exciton-phonon interactions
- Bethe-Salpeter equation for optical excitations
- Fan-Migdal self-energy formalism
- Temperature-dependent exciton properties

## Support

For questions or issues:
- Open a GitHub issue at: https://github.com/amal-kishore/ExPhonics/issues
- Contact: amalk4905@gmail.com
- Documentation: [docs/](docs/)