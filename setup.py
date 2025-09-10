"""
Setup script for Nifty Options Trader
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nifty-options-trader",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated options trading system for NIFTY 50 index",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nifty-options-trader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nifty-trader=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json"],
    },
)
