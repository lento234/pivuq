from functools import partial

import numpy as np
import scipy.interpolate
import skimage.transform
from numba import jit, prange


def warp_skimage(frame, U, coords, order=1, mode="edge") -> np.ndarray:
    """Warp image frame pixel-wise using `skimage.transform.warp`.

    Parameters
    ----------
    frame : np.ndarray
        image frame.
    U : np.ndarray
        pixel-wise 2D velocity field.
    coords : np.ndarray
        2D image coordinates (row, cols).
    order : 1-5, default: 1
        The order of interpolation for `skimage.transform.warp`.
    mode: {"constant", "edge", "symmetric", "reflect", "wrap"}, default: "edge"
        Points outside the boundaries of the input are filled according to the given mode.

    Returns
    -------
    np.ndarray
        Warped image frame

    See Also
    --------
    skimage.transform.warp : Warp an image according to a given coordinate transformation.
    """

    row_coords, col_coords = coords

    u, v = U

    warped_frame = skimage.transform.warp(frame, np.array([row_coords - v, col_coords - u]), order=order, mode=mode)

    return warped_frame


@jit(nopython=True, parallel=True, cache=True)
def whittaker_interpolation(im, xi, yi, r=3):
    r"""Whittaker-Shannon interpolation [1]_.

    Parameters
    ----------
    im : np.ndarray
        Image frame of shape (rows, cols).
    xi, yi : np.ndarray
        The `x` and `y`-coordinates at which to evaluate the interpolated values of shape (rows, cols).
    r : int, default: 3
        Radius of the interpolation stencil.

    Returns
    -------
    np.ndarray
        Interpolated image frame.

    References
    ----------
    .. [1] Wikipedia contributors. (2022, March 18). Whittaker-Shannon interpolation formula. In Wikipedia, The Free
        Encyclopedia. Retrieved 12:34, April 20, 2022, from
        https://en.wikipedia.org/w/index.php?title=Whittaker%E2%80%93Shannon_interpolation_formula&oldid=1077909297
    """
    n, m = im.shape

    im_interp = np.zeros((n, m), dtype=np.float64)

    for i in prange(n):
        for j in prange(m):

            xn = xi[i, j]
            yn = yi[i, j]
            xn_a = int(xn)
            yn_a = int(yn)

            i0 = max(xn_a - r, 0)
            i1 = min(xn_a + r, n - 1)
            j0 = max(yn_a - r, 0)
            j1 = min(yn_a + r, m - 1)

            for k in range(i0, i1 + 1):
                sincx_a = np.sinc(k - xn)
                for h in range(j0, j1 + 1):
                    sincy_a = np.sinc(h - yn)
                    im_interp[i, j] += im[k, h] * sincx_a * sincy_a

            im_interp[i, j] = max(im_interp[i, j], 0)  # strictly positive

    return im_interp


def warp_whittaker(frame, U, coords, radius=3) -> np.ndarray:
    """Warp image using Whittaker-Shannon interpolation

    Parameters
    ----------
    frame : np.ndarray
        image frame.
    U : np.ndarray
        pixel-wise 2D velocity field.
    coords : np.ndarray
        2D image coordinates (row, cols).
    radius : int, default: 3
        Radius of the interpolation stencil.

    Returns
    -------
    np.ndarray
        Warped image frame

    See Also
    --------
    whittaker_interpolation : Whittaker-Shannon interpolation.
    """

    row_coords, col_coords = coords

    u, v = U

    warped_frame = whittaker_interpolation(frame, row_coords - v, col_coords - u, r=radius)

    return warped_frame


def interpolate_to_pixel(U, imshape, kind="linear") -> np.ndarray:
    """Interpolate velocity field to pixel level.

    Parameters
    ----------
    U : np.ndarray
        Sparse 2D velocity field.
    imshape: tuple
        Image frame dimension (rows, cols).
    kind : {"linear", "cubic", "quintic"}, default: "linear"
        The kind of spline interpolation for `scipy.interpolation.interp2d`.

    Returns
    -------
    np.ndarray
        Pixel-wise 2D velocity field.
    """
    # Velocity components
    u, v = U
    nr, nc = u.shape

    ws_x = int(np.round(imshape[0] / nr))
    ws_y = int(np.round(imshape[1] / nc))

    x, y = np.arange(nr) * ws_x + ws_x // 2, np.arange(nc) * ws_y + ws_y // 2
    xi, yi = np.arange(imshape[0]), np.arange(imshape[1])

    # Interpolate to pixel level
    u_px = scipy.interpolate.interp2d(y, x, u, kind=kind)(yi, xi)
    v_px = scipy.interpolate.interp2d(y, x, v, kind=kind)(yi, xi)

    return np.stack((u_px, v_px))


def warp(
    image_pair, U, velocity_upsample_kind="linear", direction="center", nsteps=1, order=-1, radius=2
) -> np.ndarray:
    r"""Warp image pair pixel-wise to each other using `skimage.transform.warp`.

    Parameters
    ----------
    image_pair : np.ndarray
        Image pairs :math:`\mathbf{I} = (I_0, I_1)^{\top}` of size :math:`2 \times N \times M`.
    U : np.ndarray
        Sparse or dense 2D velocity field :math:`\mathbf{U} = (u, v)^{\top}` of size :math:`2 \times U_N \times U_M`.
    warp_direction : {"forward", "center", "backward"}, default: "center"
        Warping warp_direction.
    velocity_upsample_kind : {"linear", "cubic", "quintic"}, default: "linear"
        Velocity upsampling kind for spline interpolation `scipy.interpolation.interp2d`.
    nsteps : int, default: 1
        Number of sub-steps to use for warping to improve accuracy. Although, the original flow estimator (e.g. PIV)
        most likely uses :math:`n_{\mathrm{steps}}=1`.
    order : -1, 1, 2 (bug), 3, default: -1
        The order of interpolation for `skimage.transform.warp`. If order is negative, using `Whittaker-Shannon
        interpolation`.
    radius : int, default: 3
        Radius of the Whittaker-Shannon interpolation stencil.

    Returns
    -------
    np.ndarray
        Warped image pair :math:`\hat{\mathbf{I}} = (\hat{I}_0, \hat{I}_1)^{\top}` of size :math:`2 \times N \times M`.
    """

    # warping image pairs
    warped_frame_a, warped_frame_b = image_pair
    nr, nc = warped_frame_a.shape

    # interpolate velocity to pixel level
    if U.shape[1] != nr or U.shape[2] != nc:
        U = interpolate_to_pixel(U, warped_frame_a.shape, kind=velocity_upsample_kind)
    U_substep = U / nsteps

    # generate mapping grid
    image_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")

    if order == -1:
        warp_func = partial(warp_whittaker, coords=image_coords, radius=radius)
    elif order == 1 or order == 3:
        warp_func = partial(warp_skimage, coords=image_coords, order=order)
    else:
        raise ValueError(
            "Use order -1 for Whittaker-shannon interpolation and 1 or 3 for bi-linear and bi-cubic respectively."
        )

    # warp images in nsteps
    for istep in range(nsteps):
        if direction == "forward":
            warped_frame_a = warp_func(warped_frame_a, U_substep)
        elif direction == "backward":
            warped_frame_b = warp_func(warped_frame_b, -U_substep)
        elif direction == "center":
            warped_frame_a = warp_func(warped_frame_a, 0.5 * U_substep)
            warped_frame_b = warp_func(warped_frame_b, -0.5 * U_substep)
        else:
            raise ValueError(f"Unknown warping direction: {direction}.")

    return np.stack((warped_frame_a, warped_frame_b))
