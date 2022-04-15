import numpy as np
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

    # Warp image: $\hat{\mathbf{I}}$
    warped_frame_a, warped_frame_b = warping.warp(
        image_pair,
        U,
        velocity_upsample_kind=velocity_upsample_kind,
        direction=warp_direction,
        nsteps=warp_nsteps,
        order=warp_order,
    )

    # Warped image coordinates
    nr, nc = warped_frame_a.shape
    Y, X = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")

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

    return X, Y, D


def sws(
    image_pair,
    U,
    window_size=16,
    window="gaussian",
    threshold_ratio=1,
    radius=1,
    sigma=None,
    sliding_window_subtraction=False,
    velocity_upsample_kind="linear",
    warp_direction="center",
    warp_order=1,
    warp_nsteps=1,
):
    """Python implementation of `Sciacchitano-Wieneke-Scarano` algorithm of PIV Uncertainty Quantification by image
    matching [1]_.

    Raises
    ------
    NotImplementedError

    References
    ----------
    .. [1] Sciacchitano, A., Wieneke, B., & Scarano, F. (2013). PIV uncertainty quantification by image matching.
        Measurement Science and Technology, 24 (4). https://doi.org/10.1088/0957-0233/24/4/045302
    """

    # Warp image: $\hat{\mathbf{I}}$
    warped_image_pair = warping.warp(
        image_pair,
        U,
        velocity_upsample_kind=velocity_upsample_kind,
        direction=warp_direction,
        nsteps=warp_nsteps,
        order=warp_order,
    )

    # Warped image coordinates
    nr, nc = warped_image_pair.shape[1:]
    X, Y = np.meshgrid(np.arange(nr), np.arange(nc))

    # Disparity vector computation
    if sliding_window_subtraction:
        sliding_window_size = window_size
    else:
        sliding_window_size = None
    D, c, peaks = lib.disparity_vector_computation(
        warped_image_pair,
        threshold_ratio=threshold_ratio,
        radius=radius,
        sigma=sigma,
        sliding_window_size=sliding_window_size,
    )

    N, mu, sigma, delta = lib.disparity_statistics(
        D, c, window_size=window_size, window=window
    )

    return X, Y, delta, N, mu, sigma
