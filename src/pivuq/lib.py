import numpy as np
import scipy.ndimage
import skimage.filters
from numba import jit, prange


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
    im = im.astype("float") + eps
    im = np.log(im)

    # Shifted images
    im_E = np.zeros_like(im)
    im_W = np.zeros_like(im)
    im_S = np.zeros_like(im)
    im_N = np.zeros_like(im)

    im_E[:, :-1] = im[:, 1:]
    im_W[:, 1:] = im[:, :-1]
    im_N[1:, :] = im[:-1, :]
    im_S[:-1, :] = im[1:, :]

    # Subpixel position
    Y_sub = (im_W - im_E) / 2 / (im_E + im_W - 2 * im + eps)
    X_sub = (im_S - im_N) / 2 / (im_S + im_N - 2 * im + eps)

    Y_sub[~np.isfinite(Y_sub)] = 0
    X_sub[~np.isfinite(X_sub)] = 0

    return X_sub, Y_sub


@jit(nopython=True, cache=True)
def find_peak_position(im, ic, jc, radius=1):
    r"""Peak position finder around the radius of centroid.

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
        for j in range(max(0, jc - r), min(jc + r, m - 1)):
            for i in range(max(0, ic - r), min(ic + r, n - 1)):
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


def disparity_vector_computation(
    warped_image_pair, threshold_ratio=0.5, radius=2, sigma=None
):
    r"""Python implementation of `Sciacchitano-Wieneke-Scarano` disparity vector computation algorithm for PIV
    Uncertainty Quantification by image matching [1]_.

    Parameters
    ----------
    warped_image_pair : np.ndarray
        Warped image pair :math:`\hat{\mathbf{I}} = (\hat{I}_0, \hat{I}_1)^{\top}` of size :math:`2 \times N \times M`.
    threshold_ratio : float, default: 0.5
        Threshold ratio multiplier for threshold value based on Otsu's method.
    radius : int, default: 2
        Discrete particle position search radius from the centroid defined by :math:`\varphi`.
    sigma : int or None, default: None
        To perform gaussian smoothing of the image intensity product :math:`\Pi`.

    Returns
    -------
    D : np.ndarray
        Disparity map :math:`D` of size :math:`2 \times N \times M` defined by Eq. (2) [1]_.
    c : np.ndarray
        Disparity weight map :math:`c` of size :math:`N \times M` defined by Eq. (3) [1]_.
    peaks : np.ndarray
        Binary peak map :math:`\varphi` of size :math:`N \times M` containing the peak positions of the particle
        positions.

    References
    ----------
    .. [1] Sciacchitano, A., Wieneke, B., & Scarano, F. (2013). PIV uncertainty quantification by image matching.
        Measurement Science and Technology, 24 (4). https://doi.org/10.1088/0957-0233/24/4/045302.
    """

    frame_a, frame_b = warped_image_pair

    # Ensure images are float
    frame_a = frame_a.astype("float")
    frame_b = frame_b.astype("float")

    # Smoothing to suppress noise
    if sigma:
        frame_a = scipy.ndimage.gaussian_filter(frame_a, sigma)
        frame_b = scipy.ndimage.gaussian_filter(frame_b, sigma)

    # Image intensity product (Eq. 1): :math:`\Pi = \hat{I}_1\hat{I}_2`
    imgPI = frame_a * frame_b

    # Generate binary image for thresholding
    thres = skimage.filters.threshold_otsu(imgPI)
    imgPI_b = imgPI > (thres * threshold_ratio)

    # Calculate peaks values
    imgPI_C = imgPI[1:-1, 1:-1]
    imgPI_W = imgPI[1:-1, :-2]
    imgPI_E = imgPI[1:-1, 2:]
    imgPI_N = imgPI[:-2, 1:-1]
    imgPI_S = imgPI[2:, 1:-1]

    # Locate peaks
    peaks = np.zeros_like(imgPI)
    peaks[1:-1, 1:-1] = (
        (imgPI_C > imgPI_E)
        & (imgPI_C > imgPI_W)
        & (imgPI_C > imgPI_S)
        & (imgPI_C > imgPI_N)
    )

    # Threshold background
    peaks *= imgPI_b

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

    nan_indices = []
    for k, (i, j) in enumerate(coords.T):
        iAp, jAp = find_peak_position(frame_a, i, j, radius=radius)
        iBp, jBp = find_peak_position(frame_b, i, j, radius=radius)
        if (iAp * jAp * iBp * jBp) > 0:
            X[0, i, j] = Xo[iAp, jAp] + X_A_sub[iAp, jAp]
            Y[0, i, j] = Yo[iAp, jAp] - Y_A_sub[iAp, jAp]
            X[1, i, j] = Xo[iBp, jBp] + X_B_sub[iBp, jBp]
            Y[1, i, j] = Yo[iBp, jBp] - Y_B_sub[iBp, jBp]
        else:
            peaks[i, j] = 0
            nan_indices.append(k)

    # (Eq. 3): disparity weights $c$
    c = (peaks).astype("float") * np.sqrt(np.abs(imgPI)) * (imgPI > 0)
    coords = np.delete(coords, nan_indices, axis=1)

    # Disparity map
    D = np.stack((X[1] - X[0], Y[1] - Y[0]))

    return D, c, peaks


@jit(nopython=True, parallel=True, cache=True)
def accumulate_windowed_statistics(D, c, weights, wr, N, mu, sigma, delta):
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
    N : np.ndarray
        Number of particle peaks in the window.
    mu : np.ndarray
        Mean of the disparity map :math:`\mu` in the window.
    sigma : np.ndarray
        Standard deviation of the disparity map :math:`\sigma`.
    delta : np.ndarray
         Instantaneous error estimation :math:`\hat{\delta}` defined by Eq. (4) [1]_.

    References
    ----------
    .. [1] Sciacchitano, A., Wieneke, B., & Scarano, F. (2013). PIV uncertainty quantification by image matching.
        Measurement Science and Technology, 24 (4). https://doi.org/10.1088/0957-0233/24/4/045302.
    """
    n, m = D.shape[1:]

    for i in prange(n):
        for j in prange(m):
            i0, i1 = max(0, i - wr), min(i + wr, n - 1)
            j0, j1 = max(0, j - wr), min(j + wr, m - 1)

            # Filter windowed
            weights_w = weights[
                wr - (i - i0) : wr + (i1 - i), wr - (j - j0) : wr + (j1 - j)
            ]

            # Peaks windowed
            c_w = c[i0:i1, j0:j1] * weights_w
            peaks_w = c_w > 0

            # Number of peaks inside window
            N[i, j] = np.sum(peaks_w)

            if N[i, j] > 0:
                # Disparity windowed
                dx_w = D[:, i0:i1, j0:j1]
                dy_w = D[1, i0:i1, j0:j1]

                # Mean disparity (bias): Eq. (3) (left)
                mu[0, i, j] = np.sum(c_w * dx_w) / np.sum(c_w)
                mu[1, i, j] = np.sum(c_w * dy_w) / np.sum(c_w)

                # Std. dev. disparity (rms): Eq. (3) (right)
                sigma[0, i, j] = np.sqrt(
                    np.sum(c_w * (dx_w - mu[0, i, j]) ** 2) / np.sum(c_w)
                )
                sigma[1, i, j] = np.sqrt(
                    np.sum(c_w * (dy_w - mu[1, i, j]) ** 2) / np.sum(c_w)
                )

                # Instantanous error estimation
                delta[0, i, j] = np.sqrt(
                    mu[0, i, j] ** 2 + (sigma[0, i, j] / np.sqrt(N[i, j])) ** 2
                )
                delta[1, i, j] = np.sqrt(
                    mu[1, i, j] ** 2 + (sigma[1, i, j] / np.sqrt(N[i, j])) ** 2
                )


def disparity_statistics(D, c, window_size=16, window="gaussian"):
    r"""Calculate disparity statistics inside a window.

    Parameters
    ----------
    D : np.ndarray
        Disparity map :math:`D` of size :math:`2 \times N \times M` defined by Eq. (2) [1]_.
    c : np.ndarray
        Disparity weight map :math:`c` of size :math:`N \times M` defined by Eq. (3) [1]_.
    window_size : int, default: 16
        Size of the window.
    window : {"gaussian", "tophat"}, default: "gaussian"
        Window type for the disparity statistics.

    Returns
    -------
    N : np.ndarray
        Number of peaks inside the window.
    mu : np.ndarray
        Mean disparity map of size :math:`2 \times N \times M` defined by Eq. (3) [1]_.
    sigma : np.ndarray
        Standard deviation disparity map of size :math:`2 \times N \times M` defined by Eq. (3) [1]_.
    delta : np.ndarray
        Instantaneous error map of size :math:`2 \times N \times M` defined by Eq. (3) [1]_.

    References
    ----------
    .. [1] Sciacchitano, A., Wieneke, B., & Scarano, F. (2013). PIV uncertainty quantification by image matching.
        Measurement Science and Technology, 24 (4). https://doi.org/10.1088/0957-0233/24/4/045302.
    """

    # Generate windowed
    n, m = D.shape[1:]

    # Gaussian windowing
    if window == "gaussian":
        coeff = 1.75
        wr = int(np.round(window_size / 2 * coeff))
        weights = scipy.signal.windows.gaussian(wr * 2, wr / 2)
    elif window == "tophat":
        coeff = 1
        wr = int(np.round(window_size / 2 * coeff))
        weights = np.ones(wr * 2)
    else:
        raise ValueError(f"Window type `{window}` not valid.")

    # Uncertainty statistics
    N = np.zeros((n, m))
    mu = np.zeros((2, n, m))
    sigma = np.zeros((2, n, m))
    delta = np.zeros((2, n, m))

    # Accumulate disparity statistics within the window (numba accelerated loop)
    accumulate_windowed_statistics(D, c, weights, wr, N, mu, sigma, delta)

    return N, mu, sigma, delta
