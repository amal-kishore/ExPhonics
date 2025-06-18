# ExPhonics Documentation

Welcome to the ExPhonics documentation! This directory contains comprehensive guides for using the ExPhonics package.

## Available Documentation

### 📋 [User Parameters Guide](user_parameters.md)
**Complete guide to all parameters in ExPhonics**
- Physics parameters (coupling, temperature, phonon energy)
- Computational parameters (k-points, supercell size, GPU)
- Tight-binding model parameters (band gap, effective mass, etc.)
- Material-specific recommendations (MoS₂, WSe₂)
- Usage examples and troubleshooting

## Quick Links

### **Physics Parameters** (CLI controllable)
```bash
--coupling 0.25          # Electron-phonon coupling (eV)
--phonon-energy 0.05     # Optical phonon energy (eV) 
--temperature 0          # Sample temperature (K)
--broadening 0.01        # Numerical broadening (eV)
```

### **Tight-Binding Parameters** (require code modification)
- Band gap: 2.5 eV
- Effective mass: 0.49 m₀
- Lattice constant: 3.13 Bohr
- Spin-orbit coupling: 425 meV
- Coulomb strength: 1.6 eV

### **Common Usage Patterns**
```bash
# Quick start with defaults
exphonics --all

# Custom material study  
exphonics --plot frequency_dependent --coupling 0.3 --temperature 300

# High-quality figures
exphonics --all --output ./figures/ --format pdf --dpi 600 --gpu

# Parameter sweeps
exphonics --sweep coupling --verbose
```

## Getting Help

1. **Command line help**: `exphonics --help`
2. **List available options**: `exphonics --list`
3. **Parameter documentation**: [user_parameters.md](user_parameters.md)
4. **Main README**: [../README.md](../README.md)
5. **User manual**: [../MANUAL.md](../MANUAL.md)

## Contributing to Documentation

Documentation improvements are welcome! Please:
1. Keep examples practical and tested
2. Include both physics and computational perspectives
3. Provide material-specific guidance where possible
4. Update cross-references when adding new content