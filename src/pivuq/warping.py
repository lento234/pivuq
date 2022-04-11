import numpy as np
import scipy.interpolate as spinterp
import skimage


def warp_skimage(I, U, coords, order=1, mode="edge") -> np.ndarray:
    """Warp image frame pixel-wise using skimage.transform.warp

    Args:
        I (np.ndarray): image frame.
        U (np.ndarray): pixel-wise 2D velocity field.
        coords (np.ndarray): 2D image coordinates (row, cols).
        order (int, optional): The order of interpolation. The order has to be in the range 0-5. Defaults to 1.
        mode (str, optional): Points outside the boundaries of the input are filled according to the given mode. Defaults to "edge".

    Returns:
        np.ndarray: warped image frame
    """
    row_coords, col_coords = coords
    # Warp
    u, v = U

    warped_frame = skimage.transform.warp(
        I, np.array([row_coords - v, col_coords - u]), order=order, mode=mode
    )

    return warped_frame


def interpolate_to_pixel(U, imshape, kind="linear") -> np.ndarray:
    """Interpolate velocity field to pixel level

    Args:
        U (np.ndarray): Sparse 2D velocity field.
        imshape (tuple): Image frame dimension (rows, cols).
        kind (str, optional): The kind of spline interpolation to use. Defaults to "linear".

    Returns:
        np.ndarray: Pixel-wise 2D velocity field.
    """
    # Velocity components
    u, v = U
    nr, nc = u.shape

    ws_x = int(np.round(imshape[0] / nr))
    ws_y = int(np.round(imshape[1] / nc))

    x, y = np.arange(nr) * ws_x + ws_x // 2, np.arange(nc) * ws_y + ws_y / 2
    xi, yi = np.arange(imshape[0]), np.arange(imshape[1])

    # Interpolate to pixel level
    u_px = spinterp.interp2d(y, x, u, kind=kind)(yi, xi)
    v_px = spinterp.interp2d(y, x, v, kind=kind)(yi, xi)

    return np.stack((u_px, v_px))


def warp(
    image_pair, U, direction="center", upsample_kind="linear", order=1, nsteps=5
) -> np.ndarray:
    r"""Warp image pair pixel-wise to each other using `skimage.transform.warp`

    Args:
        image_pair (np.ndarray): Image pairs ($\mathbf{I} = (I_0, I_1)^{\top}$) (2 x rows x cols).
        U (np.ndarray): Sparse or dense 2D velocity field ($\mathbf{U} = (u, v)^{\top}$) (2 x U_rows x U_cols).
        direction (str, optional): Warping direction. Defaults to "center".
        upsample_kind (str, optional): Velocity upsampling kind for spline interpolation in `interp2d`. Defaults to "linear".
        order (int, optional): The order of interpolation. The order has to be in the range 0-5. Defaults to 1.
        nsteps (int, optional): Number of sub-steps to use for warping to improve accuracy. Defaults to 5.

    Returns:
        np.ndarray: Warped image pair
    """

    # warping image pairs
    I1hat, I2hat = image_pair
    nr, nc = I1hat.shape

    # interpolate velocity to pixel level
    if U.shape[1] != nr or U.shape[2] != nc:
        U = interpolate_to_pixel(U, I1hat.shape, kind=upsample_kind)
    U_substep = U / nsteps

    # generate mapping grid
    image_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")

    # warp images in nsteps
    for istep in range(nsteps):
        if direction == "forward":
            I1hat = warp_skimage(I1hat, U_substep, image_coords, order=order)
        elif direction == "backward":
            I2hat = warp_skimage(I2hat, -U_substep, image_coords, order=order)
        elif direction == "center":
            I1hat = warp_skimage(I1hat, 0.5 * U_substep, image_coords, order=order)
            I2hat = warp_skimage(I2hat, -0.5 * U_substep, image_coords, order=order)

    return np.stack((I1hat, I2hat))
