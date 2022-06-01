#!/usr/bin/env python

import setuptools

VER = "0.1"

setuptools.setup(
    name="deepid",
    version=VER,
    author="Stefano Roberto Soleti",
    author_email="roberto@lbl.gov",
    description="Deep neural network model for cosmic-ray rejection in the Mu2e experiment",
    url="https://github.com/soleti/DeePID",
    packages=setuptools.find_packages(),
    install_requires=["tensorflow", "tqdm", "numpy", "pandas", "matplotlib", "scikit-learn", "uproot", "awkward"],
    python_requires='>=3.7',
)