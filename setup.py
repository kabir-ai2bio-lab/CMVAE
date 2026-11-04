"""Installable package metadata for CMVAE."""
from setuptools import setup, find_packages
from CMVAE.version import __version__

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", "r") as f:
    INSTALL_REQUIRES = [l.strip() for l in f if l.strip()]

setup(
    name="cmvae",
    version=__version__,
    description="CMVAE: Contrastive Supervision Improves Single-Omic Latent Manifold for Breast Cancer Survival Prognosis",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    author="Akash Singh, Anowarul Kabir",
    author_email="akashsingh@usf.edu, akabir@usf.edu",
    url="https://github.com/kabir-ai2bio-lab/CMVAE",
    license="MIT",
)
