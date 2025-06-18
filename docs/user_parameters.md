# ExPhonics User Parameters Guide

## Overview

ExPhonics requires various parameters to perform exciton-phonon interaction calculations. These parameters are divided into several categories based on their physical meaning and computational role.

## Parameter Categories

### 1. Physics Parameters (User-Controllable via CLI)

These parameters control the physical aspects of exciton-phonon interactions and can be easily modified through command-line arguments.

#### **Electron-Phonon Coupling**
- **Parameter**: `--coupling`, `-g` 
- **Default**: 0.250 eV
- **Range**: 0.1 - 0.5 eV
- **Physical meaning**: Strength of interaction between electrons and optical phonons
- **Usage**: `exphonics --plot frequency_dependent --coupling 0.3`

#### **Phonon Energy**
- **Parameter**: `--phonon-energy`
- **Default**: 0.050 eV (50 meV)
- **Range**: 0.02 - 0.08 eV
- **Physical meaning**: Energy of optical phonon modes
- **Usage**: `exphonics --calculate lifetimes --phonon-energy 0.04`

#### **Temperature**
- **Parameter**: `--temperature`, `-T`
- **Default**: 0.0 K
- **Range**: 0 - 500 K
- **Physical meaning**: Sample temperature affecting phonon populations
- **Usage**: `exphonics --sweep temperature --coupling 0.25`

#### **Broadening Parameter**
- **Parameter**: `--broadening`
- **Default**: 0.010 eV (10 meV)
- **Range**: 0.005 - 0.050 eV
- **Physical meaning**: Phenomenological broadening for numerical convergence
- **Usage**: `exphonics --plot self_energy --broadening 0.015`

### 2. Computational Parameters (User-Controllable via CLI)

These parameters control numerical accuracy and computational performance.

#### **Supercell Size**
- **Parameter**: `--supercell-size`, `-L`
- **Default**: 25
- **Range**: 15 - 50
- **Physical meaning**: Half-size of supercell for BSE calculations
- **Impact**: Larger = more accurate but slower
- **Usage**: `exphonics --plot exciton_bands --supercell-size 30`

#### **K-Point Sampling**
- **Parameter**: `--k-points`
- **Default**: 48
- **Range**: 24 - 100
- **Physical meaning**: Number of k-points for band structure sampling
- **Impact**: More points = smoother curves but slower
- **Usage**: `exphonics --plot electronic_bands --k-points 72`

#### **GPU Acceleration**
- **Parameter**: `--gpu`
- **Default**: False
- **Requirement**: CuPy installation
- **Impact**: 10-100x speedup for BSE calculations
- **Usage**: `exphonics --all --gpu`

### 3. Tight-Binding Model Parameters (Fixed in Current Implementation)

These parameters define the underlying electronic structure via the two-band tight-binding model. They are **hardcoded** and require code modification to change.

#### **Band Gap**
- **Value**: 2.5 eV
- **Physical meaning**: Energy difference between valence and conduction band edges
- **Material dependence**: Varies with 2D material (MoS₂ ≈ 1.8 eV, WSe₂ ≈ 1.6 eV)
- **Code location**: `run_exphonics_demo.py`, multiple functions

#### **Effective Mass**
- **Value**: 0.49 m₀ (where m₀ is free electron mass)
- **Physical meaning**: Curvature of bands near extrema
- **Material dependence**: Electron mass in 2D materials (MoS₂ ≈ 0.48 m₀)
- **Code location**: Tight-binding model definition

#### **Lattice Constant**
- **Value**: 3.13 Bohr (≈ 1.66 Å)
- **Physical meaning**: In-plane lattice spacing of 2D material
- **Material dependence**: MoS₂ ≈ 3.19 Å, WS₂ ≈ 3.18 Å
- **Code location**: Real-space grid calculations

#### **Spin-Orbit Coupling**
- **Value**: 425 meV
- **Physical meaning**: Spin-orbit splitting of valence bands
- **Material dependence**: Heavy transition metals (Mo, W) have large SOC
- **Code location**: Band structure calculation

#### **Coulomb Interaction Strength**
- **Value**: 1.6 eV
- **Physical meaning**: Screened electron-hole Coulomb interaction (Δv₀)
- **Material dependence**: Depends on dielectric screening
- **Code location**: BSE interaction matrix elements

### 4. Output Control Parameters

#### **Output Directory**
- **Parameter**: `--output`, `-o`
- **Default**: Current directory (`.`)
- **Usage**: `exphonics --all --output ./results/`

#### **Image Format**
- **Parameter**: `--format`
- **Options**: png, pdf, svg
- **Default**: png
- **Usage**: `exphonics --plot frequency_dependent --format pdf`

#### **Resolution**
- **Parameter**: `--dpi`
- **Default**: 300
- **Range**: 150 - 600
- **Usage**: `exphonics --all --dpi 600`

#### **Display Options**
- **Parameter**: `--show`
- **Effect**: Display plots on screen in addition to saving
- **Usage**: `exphonics --plot electronic_bands --show`

#### **Verbose Output**
- **Parameter**: `--verbose`, `-v`
- **Effect**: Show detailed calculation progress
- **Usage**: `exphonics --all --verbose`

## How to Provide Parameters

### Method 1: Command Line Arguments (Recommended)

```bash
# Basic usage with defaults
exphonics --plot frequency_dependent

# Custom physics parameters
exphonics --plot frequency_dependent \
  --coupling 0.35 \
  --phonon-energy 0.06 \
  --temperature 300 \
  --broadening 0.015

# Computational parameters
exphonics --plot exciton_bands \
  --supercell-size 30 \
  --k-points 72 \
  --gpu

# Output control
exphonics --all \
  --output ./publication_figures/ \
  --format pdf \
  --dpi 600 \
  --show \
  --verbose

# Parameter sweeps
exphonics --sweep coupling --verbose
exphonics --sweep temperature --coupling 0.3
exphonics --calculate binding_energies --temperature 200
exphonics --calculate oscillator_strengths
exphonics --plot oscillator_strengths
```

### Method 2: Code Modification (For Tight-Binding Parameters)

To change the fixed tight-binding parameters, edit `run_exphonics_demo.py`:

```python
# Example: Modify for different 2D material
# In create_electronic_band_structure():
E_gap = 1.8        # eV - for MoS₂
m_eff = 0.48       # m₀ - electron effective mass
a_lattice = 3.19   # Å - MoS₂ lattice constant
Delta_SO = 150     # meV - MoS₂ spin-orbit coupling

# In create_exciton_band_structure():
Delta_v0 = 1.2     # eV - reduced screening in MoS₂
```

### Method 3: Python API (Advanced Users)

```python
import sys
sys.path.insert(0, '/path/to/exphonics')

# Direct function calls (parameters currently hardcoded)
from run_exphonics_demo import (
    create_electronic_band_structure,
    create_frequency_dependent_self_energy
)

# Generate plots with current hardcoded parameters
fig1 = create_electronic_band_structure()
fig2 = create_frequency_dependent_self_energy()
```

## Parameter Recommendations

### **Material-Specific Values**

#### **MoS₂ (Molybdenum Disulfide)**
```bash
# Tight-binding parameters (require code modification):
# Band gap: 1.8 eV
# Effective mass: 0.48 m₀
# Lattice constant: 3.19 Å
# Spin-orbit coupling: 150 meV
# Coulomb strength: 1.2 eV

# CLI parameters:
exphonics --plot frequency_dependent \
  --coupling 0.2 \
  --phonon-energy 0.047 \
  --temperature 300
```

#### **WSe₂ (Tungsten Diselenide)**
```bash
# Tight-binding parameters (require code modification):
# Band gap: 1.6 eV
# Effective mass: 0.35 m₀  
# Lattice constant: 3.28 Å
# Spin-orbit coupling: 460 meV
# Coulomb strength: 1.1 eV

# CLI parameters:
exphonics --plot frequency_dependent \
  --coupling 0.25 \
  --phonon-energy 0.032 \
  --temperature 300
```

### **Performance Optimization**

#### **Fast Calculations** (for testing)
```bash
exphonics --plot electronic_bands \
  --k-points 24 \
  --supercell-size 15
```

#### **High Accuracy** (for publication)
```bash
exphonics --all \
  --k-points 100 \
  --supercell-size 40 \
  --gpu \
  --dpi 600
```

#### **Parameter Studies**
```bash
# Systematic coupling study
for g in 0.1 0.2 0.3 0.4 0.5; do
  exphonics --plot frequency_dependent \
    --coupling $g \
    --output ./coupling_study/ \
    --format pdf
done

# Temperature sweep
exphonics --sweep temperature \
  --coupling 0.25 \
  --output ./temp_study/
```

## Common Use Cases

### **1. Quick Material Exploration**
```bash
# Use defaults to get quick overview
exphonics --all --verbose
```

### **2. Custom Material Study**
```bash
# Modify physics parameters for your material
exphonics --plot frequency_dependent \
  --coupling 0.3 \
  --phonon-energy 0.04 \
  --temperature 77  # Liquid nitrogen temperature
```

### **3. High-Quality Figures**
```bash
# Publication-ready figures
exphonics --all \
  --output ./figures/ \
  --format pdf \
  --dpi 600 \
  --gpu
```

### **4. Convergence Testing**
```bash
# Test numerical convergence
exphonics --plot convergence \
  --supercell-size 35 \
  --k-points 80 \
  --verbose
```

## Parameter Dependencies

### **Physics Dependencies**
- **Coupling strength** affects all self-energy magnitudes
- **Phonon energy** determines resonance positions
- **Temperature** controls phonon populations and linewidths
- **Broadening** affects peak widths and numerical stability

### **Computational Dependencies**
- **Supercell size** affects BSE accuracy (exciton binding energies)
- **K-points** affect band structure smoothness
- **GPU** dramatically speeds up BSE calculations

### **Material Dependencies**
- **Band gap** sets exciton energy scale
- **Effective mass** affects exciton sizes and binding energies
- **Lattice constant** determines real-space scales
- **Spin-orbit coupling** affects band splittings
- **Coulomb strength** controls exciton binding

## Troubleshooting

### **Memory Issues**
```bash
# Reduce memory usage
exphonics --plot exciton_bands --supercell-size 20
```

### **Slow Calculations**
```bash
# Speed up calculations
exphonics --all --gpu --k-points 36
```

### **Convergence Problems**
```bash
# Increase broadening for stability
exphonics --plot self_energy --broadening 0.02
```

### **GPU Issues**
```bash
# Check GPU availability
python -c "import cupy; print('GPU available')"

# Fallback to CPU
exphonics --all  # (omit --gpu flag)
```

This parameter guide provides complete control over ExPhonics calculations, from quick exploratory runs to high-accuracy production calculations for research and publication.