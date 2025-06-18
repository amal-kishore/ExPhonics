#!/usr/bin/env python3
"""
Setup script for ExPhonics package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
README_PATH = Path(__file__).parent / "README.md"
long_description = README_PATH.read_text() if README_PATH.exists() else ""

# Package requirements
requirements = [
    "numpy>=1.20.0",
    "matplotlib>=3.5.0",
    "scipy>=1.7.0",
]

# Optional GPU requirements
gpu_requirements = [
    "cupy",
]

# Development requirements  
dev_requirements = [
    "pytest>=6.0",
    "pytest-cov",
    "black",
    "flake8",
    "mypy",
]

setup(
    name="exphonics",
    version="1.0.0",
    author="Amal Kishore",
    author_email="amalk4905@gmail.com",
    description="Exciton-Phonon Interaction Calculator using Many-Body Perturbation Theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amal-kishore/ExPhonics",
    
    packages=find_packages(),
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    python_requires=">=3.8",
    
    install_requires=requirements,
    
    extras_require={
        "gpu": gpu_requirements,
        "dev": dev_requirements,
        "all": requirements + gpu_requirements + dev_requirements,
    },
    
    entry_points={
        "console_scripts": [
            "exphonics=exphonics_cli:main",
        ],
    },
    
    include_package_data=True,
    
    package_data={
        "exphonics": [
            "*.py",
            "**/*.py",
            "README.md",
            "LICENSE",
        ],
    },
    
    project_urls={
        "Bug Reports": "https://github.com/amal-kishore/ExPhonics/issues",
        "Source": "https://github.com/amal-kishore/ExPhonics",
        "Documentation": "https://exphonics.readthedocs.io/",
    },
    
    keywords=[
        "excitons",
        "phonons", 
        "many-body theory",
        "bethe-salpeter equation",
        "self-energy",
        "2D materials",
        "computational physics",
        "condensed matter",
    ],
    
    zip_safe=False,
)