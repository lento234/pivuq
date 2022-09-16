import numpy as np
import scipy.ndimage
from numba import jit, prange


def sliding_avg_subtract(im, window_size) -> np.ndarray:
    r"""Perform sliding window average subtraction.

    Parameters
    ----------
    im : np.ndarray
        Image of size :math:`N \times M`.
    window_size : int
        Window size.

    Returns
    -------
    np.ndarray
        Average subtracted image of size :math:`N \times M`.
    """
    im_avg = scipy.ndimage.gaussian_filter(im, sigma=window_size // 2 + 1)
    return im - im_avg


def construct_subpixel_position_map(im):
    r"""Construct sub-pixel position map based on 3-point Gaussian fit stencil.

    Parameters
    ----------
    im : np.ndarray
        Image of size :math:`N \times M`.

    Returns
    -------
    X_sub, Y_sub : np.ndarray
        Subpixel position map of size :math:`N \times M`.
    """
    eps = np.finfo(np.float64).eps

    # Log of image
    im = np.log(im)

    # Shifted images
    im_E = np.zeros_like(im) + eps
    im_W = np.zeros_like(im) + eps
    im_S = np.zeros_like(im) + eps
    im_N = np.zeros_like(im) + eps

    im_E[:, :-1] = im[:, 1:]
    im_W[:, 1:] = im[:, :-1]
    im_N[1:, :] = im[:-1, :]
    im_S[:-1, :] = im[1:, :]

    # Subpixel position
    X_sub = (im_W - im_E) / 2.0 / (im_E + im_W - 2 * im)
    Y_sub = (im_S - im_N) / 2.0 / (im_S + im_N - 2 * im)

    X_sub[~np.isfinite(X_sub)] = 0
    Y_sub[~np.isfinite(Y_sub)] = 0

    return X_sub, Y_sub


@jit(nopython=True, cache=True)
def find_particle(im, ic, jc, radius=1):
    r"""Particle peak position finder around the radius of centroid.

    Parameters
    ----------
    im : np.ndarray
        Image array of size :math:`N \times M`.
    ic, jc : int
        Row and column index of centroid.
    radius : int
        Search radius of centroid.

    Returns
    -------
    int, int
        Row index and column index of peak.
    """
    n, m = im.shape
    r, iPeak, jPeak, imax = 0, 0, 0, 0

    while r <= radius and imax == 0:
        for i in range(max(1, ic - r), min(ic + r, n - 2)):
            for j in range(max(1, jc - r), min(jc + r, m - 2)):
                if (
                    (im[i, j] > im[i, j - 1])
                    & (im[i, j] > im[i - 1, j])
                    & (im[i, j] > im[i, j + 1])
                    & (im[i, j] > im[i + 1, j])
                    & (im[i, j] > imax)
                ):
                    iPeak = i
                    jPeak = j
                    imax = im[i, j]
        r += 1
    return iPeak, jPeak


def find_peaks(imgPI) -> np.ndarray:
    r"""Particle peak position detection according to Eq. (1) [1]_.

    Parameters
    ----------
    imgPI : np.ndarray
        Image intensity product :math:`\Pi` of size :math:`N \times M`.

    Returns
    -------
    np.ndarray
        Peak map :math:`\varphi` of size :math:`N \times M`.

    """

    # Calculate peaks values
    imgPI_C = imgPI[1:-1, 1:-1]

    # Neighbors
    imgPI_E = imgPI[1:-1, 2:]
    imgPI_W = imgPI[1:-1, :-2]
    imgPI_N = imgPI[:-2, 1:-1]
    imgPI_S = imgPI[2:, 1:-1]

    # Locate peaks
    peaks = np.zeros_like(imgPI)
    peaks[1:-1, 1:-1] = (imgPI_C > imgPI_E) & (imgPI_C > imgPI_W) & (imgPI_C > imgPI_N) & (imgPI_C > imgPI_S)

    # Threshold background
    peaks *= np.sqrt(np.abs(imgPI))

    return peaks


@jit(nopython=True, parallel=True, cache=True)
def disparity_ensemble_statistics(D, c, weights, wr, grid_size, coeff, ROI):
    r"""Numba accelerated loop for computing the disparity statistics inside a window of radius `wr`.

    Parameters
    ----------
    D : np.ndarray
        Disparity map :math:`D` of size :math:`2 \times N \times M` defined by Eq. (2) [1]_.
    c : np.ndarray
        Disparity weight map :math:`c` of size :math:`N \times M` defined by Eq. (3) [1]_.
    weights : np.ndarray
        Windowing weights of size :math:`N \times M` defined by Gaussian or tophat filter.
    wr : int
        Window radius.
    ws : int
        Disparity resolution size.
    coeff : float
        Confidence interval coefficient.
    ROI : tuple
        Row and column indices of the ROI: (`i_min`, `i_max`, `j_min`, `j_max`).

    Returns
    -------
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
        Measurement Science and Technology, 24 (4). https://doi.org/10.1088/0957-0233/24/4/045302.
    """
    n, m = D.shape[1] // grid_size, D.shape[2] // grid_size

    wr_eff = int(np.round(coeff * wr))

    # Uncertainty statistics
    N = np.zeros((n, m))
    mu = np.zeros((2, n, m))
    sigma = np.zeros((2, n, m))
    delta = np.zeros((2, n, m))

    for ii in prange(n):
        i = ii * grid_size + grid_size // 2
        # Row bounds of windw
        i0 = max(i - wr_eff, 0)
        i1 = min(i + wr_eff, n * grid_size - 1)

        for jj in prange(m):
            j = jj * grid_size + grid_size // 2
            # Only calculate inside ROI
            if (i >= ROI[0]) and (i <= ROI[1]) and (j >= ROI[2]) and (j <= ROI[3]):

                # Column bounds of windw
                j0 = max(j - wr_eff, 0)
                j1 = min(j + wr_eff, m * grid_size - 1)

                # Filter windowed
                weights_w = weights[
                    wr_eff - (i - i0) : wr_eff + (i1 - i),
                    wr_eff - (j - j0) : wr_eff + (j1 - j),
                ]

                # Peaks windowed
                c_w = c[i0:i1, j0:j1] * weights_w

                # Number of peaks inside window # Bug in original code?
                N[ii, jj] = np.maximum(np.sum((c_w > 0) * weights_w), 1)

                for k in range(2):
                    # Disparity windowed
                    d_w = D[k, i0:i1, j0:j1].ravel()
                    c_w = (c[i0:i1, j0:j1] * weights_w).ravel()

                    d_w = d_w[np.nonzero(d_w)]
                    c_w = c_w[np.nonzero(c_w)]

                    # Outlier removal
                    valid_mask = np.where(np.abs(d_w - np.mean(d_w)) <= 3 * np.std(d_w))
                    d_w = d_w[valid_mask]
                    c_w = c_w[valid_mask]

                    # Mean disparity (bias): Eq. (3) (left)
                    mu[k, ii, jj] = np.sum(c_w * d_w) / np.sum(c_w)

                    # Std. dev. disparity (rms): Eq. (3) (right)
                    sigma[k, ii, jj] = np.sqrt(np.sum(c_w * (d_w - mu[k, ii, jj]) ** 2) / np.sum(c_w))

                    # Instantanous error estimation
                    delta[k, ii, jj] = np.sqrt(mu[k, ii, jj] ** 2 + (sigma[k, ii, jj] ** 2 / N[ii, jj]))

    return delta, N, mu, sigma


def disparity_vector_computation(warped_image_pair, radius=2.0, sliding_window_size=16):
    r"""Python implementation of `Sciacchitano-Wieneke-Scarano` disparity vector computation algorithm for PIV
    Uncertainty Quantification by image matching [1]_.

    Parameters
    ----------
    warped_image_pair : np.ndarray
        Warped image pair :math:`\hat{\mathbf{I}} = (\hat{I}_0, \hat{I}_1)^{\top}` of size :math:`2 \times N \times M`.
    radius : int, default: 2
        Discrete particle position search radius from the centroid defined by :math:`\varphi`.
    sliding_window_size : int, default: 16
        Sliding window average subtraction window size.

    Returns
    -------
    D : np.ndarray
        Disparity map :math:`D` of size :math:`2 \times N \times M` defined by Eq. (2).
    c : np.ndarray
        Disparity weight map :math:`c` of size :math:`N \times M` defined by Eq. (3).
    """

    frame_a, frame_b = warped_image_pair

    # Ensure images are float
    frame_a = frame_a.astype("float")
    frame_b = frame_b.astype("float")

    if sliding_window_size:
        frame_a = sliding_avg_subtract(frame_a, sliding_window_size)
        frame_b = sliding_avg_subtract(frame_b, sliding_window_size)

    # Image intensity product (Eq. 1): :math:`\Pi = \hat{I}_1\hat{I}_2`
    imgPI = frame_a * frame_b

    # Positive regions
    img_pos = (frame_a > 0) & (frame_b > 0)

    # Ensure frames are positive
    frame_a = frame_a - np.min(frame_a) + np.finfo(np.float64).eps
    frame_b = frame_b - np.min(frame_b) + np.finfo(np.float64).eps

    # Find peaks
    peaks = find_peaks(imgPI)

    # Calculate indices of peaks
    coords = np.stack(np.where(peaks))

    # # Construct coordinates
    Xo, Yo = np.meshgrid(
        np.arange(imgPI.shape[1], dtype="float"),
        np.arange(imgPI.shape[0], dtype="float"),
    )

    # Construct coordinates
    X = np.tile(Xo, (2, 1, 1))
    Y = np.tile(Yo, (2, 1, 1))
    X_A_sub, Y_A_sub = construct_subpixel_position_map(frame_a)
    X_B_sub, Y_B_sub = construct_subpixel_position_map(frame_b)

    for i, j in coords.T:
        iAp, jAp = find_particle(frame_a, i, j, radius=radius)
        iBp, jBp = find_particle(frame_b, i, j, radius=radius)
        if (iAp * jAp * iBp * jBp) > 0:
            X[0, i, j] = Xo[iAp, jAp] + X_A_sub[iAp, jAp]
            Y[0, i, j] = Yo[iAp, jAp] - Y_A_sub[iAp, jAp]
            X[1, i, j] = Xo[iBp, jBp] + X_B_sub[iBp, jBp]
            Y[1, i, j] = Yo[iBp, jBp] - Y_B_sub[iBp, jBp]
        else:
            peaks[i, j] = 0

    # (Eq. 3): disparity weights $c$
    c = peaks * (imgPI > 0) * img_pos

    # Disparity map: $D$
    D = np.stack((X[1] - X[0], Y[1] - Y[0])) * (imgPI > 0) * img_pos

    return D, c
