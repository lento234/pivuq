import numpy as np
import scipy.interpolate as spinterp
import skimage


def warp_skimage(I, U, coords, order=3, mode="edge"):
    row_coords, col_coords = coords
    # Warp
    u, v = U

    warped_frame = skimage.transform.warp(
        I, np.array([row_coords - v, col_coords - u]), order=order, mode=mode
    )

    return warped_frame


def interpolate_to_pixel(U, imshape, kind="linear"):

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


def warp(image_pair, U, direction="center", upsample_kind="linear", order=1, nsteps=5):

    # warping image pairs
    I1hat, I2hat = image_pair
    nr, nc = I1hat.shape

    # interpolate velocity to pixel level
    U_interp = interpolate_to_pixel(U, (nr, nc), kind=upsample_kind)
    U_substep = U_interp / nsteps

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
