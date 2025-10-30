from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fractional-dynamics-neuroimaging",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Physics-informed neural networks with fractional calculus for neuroimaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csw-research/fractional-dynamics-neuroimaging",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "jax>=0.4.13",
        "nibabel>=5.0.0",
        "nilearn>=0.10.0",
        "dipy>=1.7.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.75.0",
            "sphinx>=6.2.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
