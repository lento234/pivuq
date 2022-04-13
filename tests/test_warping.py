import numpy as np
from numpy.testing import assert_array_equal

import pivuq


def test_warp_skimage():

    frame = np.pad(np.ones((3, 3)), 2)
    U = np.ones((2, *frame.shape))
    coords = np.meshgrid(
        np.arange(frame.shape[0]), np.arange(frame.shape[1]), indexing="ij"
    )

    assert_array_equal(pivuq.warping.warp_skimage(frame, 0 * U, coords), frame)

    assert_array_equal(
        pivuq.warping.warp_skimage(frame, U, coords), np.roll(frame, 1, axis=(0, 1))
    )

    assert_array_equal(
        pivuq.warping.warp_skimage(frame, -1 * U, coords),
        np.roll(frame, -1, axis=(0, 1)),
    )

    assert_array_equal(pivuq.warping.warp_skimage(frame, 10 * U, coords), frame * 0.0)

    assert_array_equal(
        pivuq.warping.warp_skimage(frame, 0.5 * U, coords),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.25, 0.5, 0.5, 0.25, 0.0],
                [0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0],
                [0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0],
                [0.0, 0.0, 0.25, 0.5, 0.5, 0.25, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
