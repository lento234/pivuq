import scipy.ndimage


def sliding_avg_subtract(im, window_size):
    im_avg = scipy.ndimage.gaussian_filter(im, sigma=window_size // 2 + 1)
    return im - im_avg
