import numba
import numpy as np
import scipy.ndimage
import skimage


def construct_subpixel_position_map(im):
    """Generate 3-point Gaussian distribution function map based on neighbourhood intensities.

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


@numba.jit(nopython=True, cache=True)
def find_peak_position(im, ic, jc, radius=1):
    """Peak position finder around the radius of centroid.

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
        To perform gaussian smoothing of the image intensity product :math:`Pi`.

    Returns
    -------
    D : np.ndarray
        Disparity map of size :math:`2 \times N \times M` defined by Eq. (2) [1]_.
    c : np.ndarray
        Disparity weight map of size :math:`N \times M` defined by Eq. (3) [1]_.
    peaks : np.ndarray
        Binary image of size :math:`N \times M` containing the peak positions of the particle positions.

    References
    ----------
    .. [1] Sciacchitano, A., Wieneke, B., & Scarano, F. (2013). PIV uncertainty quantification by image matching.
        Measurement Science and Technology, 24 (4). https://doi.org/10.1088/0957-0233/24/4/045302
    """

    frame_a, frame_b = warped_image_pair

    # # Ensure images are float
    frame_a = frame_a.astype("float")
    frame_b = frame_b.astype("float")

    # Image intensity product (Eq. 1): :math:`\Pi = \hat{I}_1\hat{I}_2`
    imgPI = frame_a * frame_b

    # Smoothing to suppress noise
    if sigma:
        imgPI = scipy.ndimage.gaussian_filter(imgPI, sigma)

    # Generate binary image for thresholding
    thres = skimage.filters.threshold_otsu(imgPI)
    imgPI_b = imgPI > (thres * threshold_ratio)

    # # Calculate peaks values
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
