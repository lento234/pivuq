# PIVUQ: PIV Uncertainty Quantification

[![Docs](https://img.shields.io/readthedocs/pivuq?style=flat-square&labelColor=000000)](https://pivuq.readthedocs.io/)
[![PyPi Version](https://img.shields.io/pypi/v/pivuq.svg?style=flat-square&labelColor=000000)](https://pypi.org/project/pivuq/)
[![PyPi Python versions](https://img.shields.io/pypi/pyversions/pivuq.svg?style=flat-square&labelColor=000000)](https://pypi.org/project/pivuq/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square&labelColor=000000)](#license)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.6458153-blue?style=flat-square&labelColor=000000)](https://doi.org/10.5281/zenodo.6458153)

## Description

This package contains python implementations of uncertainty quantification (UQ) for Particle Image Velocimetry (PIV). Primary aim is to implement UQ algorithms for PIV techniques. Future goals include possible extensions to other domains including but not limited to optical flow and BOS.

List of approachs:

- [`pivuq.diparity.ilk`](https://pivuq.readthedocs.io/en/latest/api/disparity.html#pivuq.disparity.ilk): Iterative Lucas-Kanade based disparity estimation. [[scikit-image](https://scikit-image.org/docs/dev/api/skimage.registration.html#skimage.registration.optical_flow_ilk)]
- [`pivuq.disparity.sws`](https://pivuq.readthedocs.io/en/latest/api/disparity.html#pivuq.disparity.ilk): Python implementation of Sciacchitano, A., Wieneke, B., & Scarano, F. (2013). PIV uncertainty quantification by image matching. *Measurement Science and Technology, 24* (4). [https://doi.org/10.1088/0957-0233/24/4/045302](https://doi.org/10.1088/0957-0233/24/4/045302). [[piv.de](http://piv.de/uncertainty/)]


## Installation

Install using pip

```bash
pip install pivuq
```

### Development mode

Initialize [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) environment

```bash
conda env create -f environment.yml
```

Install packages using [`poetry`](https://python-poetry.org/docs/):

```bash
poetry install
```

## How to cite

*Work in progress* version: https://doi.org/10.5281/zenodo.6458153

In future, please cite the following paper:

> Manickathan et al. (2022). PIVUQ: Uncertainty Quantification Toolkit for Quantitative Flow Visualization. *in prep*.
