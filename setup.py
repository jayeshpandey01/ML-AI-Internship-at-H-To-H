"""
Setup script for Algo Trading System
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="algo-trading-system",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive algorithmic trading system with ML predictions and automated reporting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/algo-trading-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=5.0.0",
            "pylint>=2.15.0",
            "mypy>=0.991",
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
        ],
        "docs": [
            "sphinx>=5.3.0",
            "sphinx-rtd-theme>=1.1.0",
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "algo-trading=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
    },
)