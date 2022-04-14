import numpy as np
from numpy.testing import assert_almost_equal

import pivuq


def test_ilk():

    frame_a = np.pad(np.ones((3, 3)), 2)
    frame_b = np.roll(frame_a, 1, axis=(0, 1))

    X, Y, D = pivuq.disparity.ilk(
        (frame_a, frame_b),
        np.ones((2, 2, 2)),
        window_size=16,
        prefilter=True,
        window="gaussian",
        velocity_upsample_kind="linear",
        warp_direction="center",
        warp_order=1,
        warp_nsteps=1,
    )
    assert X.shape == frame_a.shape
    assert Y.shape == frame_a.shape
    assert D.shape == (2, *frame_a.shape)
    assert D.sum() == 0

    _, _, D = pivuq.disparity.ilk(
        (frame_a, frame_b),
        np.zeros((2, 2, 2)),
        window_size=16,
        prefilter=True,
        window="gaussian",
        velocity_upsample_kind="linear",
        warp_direction="center",
        warp_order=1,
        warp_nsteps=1,
    )
    assert_almost_equal(D, np.ones_like(D), decimal=6)

    # Test window sizes
    _, _, D = pivuq.disparity.ilk(
        (frame_a, frame_b),
        np.zeros((2, 2, 2)),
        window_size=8,
        prefilter=True,
        window="gaussian",
        velocity_upsample_kind="linear",
        warp_direction="center",
        warp_order=1,
        warp_nsteps=1,
    )
    assert_almost_equal(D, np.ones_like(D), decimal=6)

    _, _, D = pivuq.disparity.ilk(
        (frame_a, frame_b),
        np.zeros((2, 2, 2)),
        window_size=24,
        prefilter=True,
        window="gaussian",
        velocity_upsample_kind="linear",
        warp_direction="center",
        warp_order=1,
        warp_nsteps=1,
    )
    assert_almost_equal(D, np.ones_like(D), decimal=6)

    # Test pre-filtering
    _, _, D = pivuq.disparity.ilk(
        (frame_a, frame_b),
        np.zeros((2, 2, 2)),
        window_size=24,
        prefilter=False,
        window="gaussian",
        velocity_upsample_kind="linear",
        warp_direction="center",
        warp_order=1,
        warp_nsteps=1,
    )
    assert_almost_equal(D, np.ones_like(D), decimal=6)

    # Test window type
    _, _, D = pivuq.disparity.ilk(
        (frame_a, frame_b),
        np.zeros((2, 2, 2)),
        window_size=24,
        prefilter=True,
        window="tophat",
        velocity_upsample_kind="linear",
        warp_direction="center",
        warp_order=1,
        warp_nsteps=1,
    )
    assert_almost_equal(D, np.ones_like(D), decimal=6)
