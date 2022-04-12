# PIV-UQ: PIV Uncertainty Quantification

`This project is still under active development.`

## Description

This package contains python implementations of uncertainty quantification (UQ) for Particle Image Velocimetry (PIV). Primary aim is to implement UQ algorithms for PIV techniques. Future goals include possible extensions to other domains including but not limited to optical flow and BOS.

List of approachs:

- [x]: [`pivuq.diparity.ilk`](/src/pivuq/disparity.py#L7): Iterative Lucas-Kanade based disparity estimation. [[scikit-image](https://scikit-image.org/docs/dev/api/skimage.registration.html#skimage.registration.optical_flow_ilk)]
- [ ]: [`pivuq.disparity.sws`](/src/pivuq/disparity.py#L87): Python implementation of Sciacchitano, A., Wieneke, B., & Scarano, F. (2013). PIV uncertainty quantification by image matching. *Measurement Science and Technology, 24* (4). [https://doi.org/10.1088/0957-0233/24/4/045302](https://doi.org/10.1088/0957-0233/24/4/045302). [[piv.de](http://piv.de/uncertainty/)]


## Installation

Install using pip

```bash
pip install pivuq
```

## How to cite

*Work in progress*. In future, please cite the following paper:

> Manickathan et al. (2022). PIVUQ: Uncertainty Quantification Toolkit for Quantitative Flow Visualization. *in prep*.
