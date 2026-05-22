import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="diffint",
    version="0.1.0",
    description="Differentiable interval bottlenecks for interpretable anomaly detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DiffInt/diffint",
    author="DiffInt authors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12",
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "pandas>=1.3",
    ],
    extras_require={"explain": ["seaborn>=0.11", "matplotlib>=3.4"]},
    keywords=[
        "anomaly detection",
        "interpretability",
        "interval patterns",
        "autoencoder",
        "tabular data",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
