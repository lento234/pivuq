# PIV-UQ: PIV Uncertainty Quantification

`Note: Primary aim is to implement UQ algorithms for PIV techniques. Future goals include possible extensions to other domains including but not limited to optical flow and BOS.`

## Description

This package contains python implementations of uncertainty quantification (UQ) for Particle Image Velocimetry (PIV). Implements:

* `pivuq.diparity.ilk`: Iterative Lucas-Kanade based disparity estimation. [[scikit-image](https://scikit-image.org/docs/dev/api/skimage.registration.html#skimage.registration.optical_flow_ilk)]
* `pivuq.disparity.sws`: Python implementation of Sciacchitano, A., Wieneke, B., & Scarano, F. (2013). PIV uncertainty quantification by image matching. *Measurement Science and Technology, 24* (4). [https://doi.org/10.1088/0957-0233/24/4/045302](https://doi.org/10.1088/0957-0233/24/4/045302). [[piv.de](http://piv.de/uncertainty/)]


## Installation

Install using pip

```bash
pip install pivuq
```
