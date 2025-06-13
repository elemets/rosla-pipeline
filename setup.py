# setup.py
"""Setup script for the diabetes_ha1c project."""
import setuptools

setuptools.setup(
    name="rapid_overdose_classification",  # Should match your main package directory
    version="0.1.0",  # Set an initial version
    packages=setuptools.find_packages(),  # Automatically find your 'diabetes_ha1c' package
    # You can optionally list dependencies here, often mirroring requirements.txt
    # install_requires=[ 'pandas', 'scikit-learn', ... ],
)
