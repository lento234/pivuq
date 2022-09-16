import numpy as np
import scipy.signal
import skimage.registration

from . import lib, warping


def ilk(
    image_pair,
    U,
    window_size=16,
    prefilter=True,
    window="gaussian",
    velocity_upsample_kind="linear",
    warp_direction="center",
    warp_order=1,
    warp_nsteps=1,
):
    r"""Disparity map calculation using iterative Lucas Kanade ("ilk").

    Parameters
    ----------
    image_pair : np.ndarray
        Image pairs :math:`\mathbf{I} = (I_0, I_1)^{\top}` of size (2 x rows x cols).
    U : np.ndarray
        Sparse or dense 2D velocity field :math:`\mathbf{U} = (u, v)^{\top}` of (2 x U_rows x U_cols).
    window_size : int, default: 16
        Window size around the pixel to consider the disparity for optical flow estimator.
    window : {"gaussian", "tophat"}, default: "gaussian"
        Windowing kernel type for integration around the pixel.
    prefilter : bool, default: True
        Whether to prefilter the estimated optical flow before each image warp. When True, a median filter with window
        size 3 along each axis is applied. This helps to remove potential outliers.
    velocity_upsample_kind : {"linear", "cubic", "quintic"}, default: "linear"
        Velocity upsampling kind for spline interpolation `scipy.interpolation.interp2d`.
    warp_direction : {"forward", "center", "backward"}, default: "center"
        Warping direction.
    warp_order : 1-5, default: 1
        The order of interpolation for `skimage.transform.warp`.
    warp_nsteps : int, default: 5
        Number of sub-steps to use for warping to improve accuracy.

    Returns
    -------
    X, Y : np.ndarray
        `x` and `y` coordinates of disparity map.
    D : np.ndarray
        pixel-wise 2D disparity map :math:`\mathbf{D} = (d_x, d_y)^\top` of size (2 x rows x cols).

    See Also
    --------
    skimage.registration.optical_flow_ilk : Coarse to fine optical flow estimator.
    skimage.transform.warp : Warp an image according to a given coordinate transformation.
    """
    # Image dimensions
    nr, nc = image_pair[0].shape

    # Warp image: $\hat{\mathbf{I}}$
    warped_frame_a, warped_frame_b = warping.warp(
        image_pair,
        U,
        velocity_upsample_kind=velocity_upsample_kind,
        direction=warp_direction,
        nsteps=warp_nsteps,
        order=warp_order,
    )

    # windowing type
    gaussian = True if window == "gaussian" else False

    # Disparity map using Lucas Kanade
    optical_flow = skimage.registration.optical_flow_ilk(
        warped_frame_a,
        warped_frame_b,
        radius=window_size // 2,
        gaussian=gaussian,
        prefilter=prefilter,
    )

    # Disparity map
    D = np.abs(optical_flow[::-1])

    # Image coordinates
    Y, X = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")

    return X, Y, D


def sws(
    image_pair,
    U,
    window_size=16,
    grid_size=4,
    window="gaussian",
    radius=1,
    sliding_window_subtraction=True,
    ROI=None,
    velocity_upsample_kind="linear",
    warp_direction="center",
    warp_order=1,
    warp_nsteps=1,
):
    r"""Python implementation of `Sciacchitano-Wieneke-Scarano` algorithm of PIV Uncertainty Quantification by image
    matching [1]_.

    Parameters
    ----------
    image_pair : np.ndarray
        Image pairs :math:`\mathbf{I} = (I_0, I_1)^{\top}` of size (2 x rows x cols).
    U : np.ndarray
        Sparse or dense 2D velocity field :math:`\mathbf{U} = (u, v)^{\top}` of (2 x U_rows x U_cols).
    window_size : int, default: 16
        Window size around the pixel to consider the disparity ensemble.
    grid_size: int, default: 4
        Disparity ensemble grid resolution in pixels.
    window : {"gaussian", "tophat"}, default: "gaussian"
        Window type for the disparity statistics.
    radius : int, default: 1
        Search radius for particle peak position.
    sliding_window_subtraction : bool, default: False
        Whether to use the sliding window subtraction before disparity vector calculation.
    ROI : tuple, default: None
        Region of interest to use for calculating the disparity ensemble (`i_min`, `i_max`, `j_min`, `j_max`).
    velocity_upsample_kind : {"linear", "cubic", "quintic"}, default: "linear"
        Velocity upsampling kind for spline interpolation `scipy.interpolation.interp2d`.
    warp_direction : {"forward", "center", "backward"}, default: "center"
        Warping direction.
    warp_order : 1-5, default: 1
        The order of interpolation for `skimage.transform.warp`.
    warp_nsteps : int, default: 5
        Number of sub-steps to use for warping to improve accuracy.

    Returns
    -------
    X, Y : np.ndarray
        `x` and `y` coordinates of disparity map.
    delta : np.ndarray
        Instantaneous error estimation map of size :math:`2 \times N \times M` defined by Eq. (3) [1]_.
    N : np.ndarray
        Number of peaks inside the window.
    mu : np.ndarray
        Mean disparity map of size :math:`2 \times N \times M` defined by Eq. (3) [1]_.
    sigma : np.ndarray
        Standard deviation disparity map of size :math:`2 \times N \times M` defined by Eq. (3) [1]_.

    References
    ----------
    .. [1] Sciacchitano, A., Wieneke, B., & Scarano, F. (2013). PIV uncertainty quantification by image matching.
        Measurement Science and Technology, 24 (4). https://doi.org/10.1088/0957-0233/24/4/045302
    """

    # Image dimensions
    nr, nc = image_pair[0].shape

    # Step 1: Warp image: $\hat{\mathbf{I}}$
    warped_image_pair = warping.warp(
        image_pair,
        U,
        velocity_upsample_kind=velocity_upsample_kind,
        direction=warp_direction,
        nsteps=warp_nsteps,
        order=warp_order,
    )

    # Step 2: Disparity vector computation
    if sliding_window_subtraction:
        sliding_window_size = window_size
    else:
        sliding_window_size = None

    D, c = lib.disparity_vector_computation(
        warped_image_pair,
        radius=radius,
        sliding_window_size=sliding_window_size,
    )

    # Step 3: Disparity statistics calculation

    # Gaussian windowing
    wr = int(np.round(window_size / 2))
    if window == "gaussian":
        coeff = 1.75
        weights = scipy.signal.windows.gaussian(int(np.round(wr * 2 * coeff)) + 1, int(np.round(wr / 2 * coeff)))
    elif window == "tophat":
        coeff = 1
        weights = np.ones(wr * 2 * coeff + 1)
    else:
        raise ValueError(f"Window type `{window}` not valid.")

    weights = np.outer(weights, weights)  # 2D windowing weights

    if ROI is None:
        ROI = (0, nr, 0, nc)
    else:
        ROI = tuple(ROI)

    # Accumulate disparity statistics within the window (numba accelerated loop)
    delta, N, mu, sigma = lib.disparity_ensemble_statistics(D, c, weights, wr, grid_size, coeff, ROI)

    # Coordinates
    Y, X = np.meshgrid(
        np.arange(grid_size // 2, nr, grid_size), np.arange(grid_size // 2, nc, grid_size), indexing="ij"
    )

    return X, Y, delta, N, mu, sigma
