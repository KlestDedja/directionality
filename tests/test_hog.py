from pathlib import Path

import numpy as np
import pytest
from skimage.feature import hog

from edgehog import HOGDescriptor


@pytest.mark.parametrize(
    "name,dtype,cell,block,bins",
    [
        ("uint8", np.uint8, (4, 6), (1, 1), 9),
        ("float32", np.float32, (5, 7), (2, 2), 9),
        ("float64", np.float64, (4, 4), (2, 3), 45),
    ],
)
@pytest.mark.parametrize("block_norm", [None, "None"])
@pytest.mark.parametrize("feature_vector", [False, True])
def test_raw_matches_original_fork(name, dtype, cell, block, bins, block_norm, feature_vector):
    with np.load(Path(__file__).parent / "data/raw_hog_reference.npz") as ref:
        image = ref["image"].astype(dtype)
        if name != "uint8":
            image = image / np.array(251, dtype=dtype)
        descriptor = HOGDescriptor(bins, cell, block)
        actual, vis = descriptor.compute_hog(
            image, block_norm=block_norm, feature_vector=feature_vector
        )
        expected = ref[name + "_descriptor"]
        if feature_vector:
            expected = expected.ravel()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(vis, ref[name + "_visualization"])
        assert actual.dtype == expected.dtype
        # Calling again without visualization must clear the previous image.
        actual2, vis2 = descriptor.compute_hog(
            image, block_norm=block_norm, feature_vector=feature_vector, visualize=False
        )
        np.testing.assert_array_equal(actual2, expected)
        assert vis2 is None and descriptor.hog_image is None


@pytest.mark.parametrize("norm", ["L1", "L1-sqrt", "L2", "L2-Hys"])
@pytest.mark.parametrize("visualize", [True, False])
def test_normalized_modes_stay_upstream(norm, visualize):
    image = np.random.default_rng(9).normal(size=(32, 40))
    actual, vis = HOGDescriptor(9, (8, 8), (2, 2)).compute_hog(
        image, visualize=visualize, block_norm=norm
    )
    expected = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm=norm,
        visualize=visualize,
        feature_vector=False,
    )
    if visualize:
        expected, expected_vis = expected
        np.testing.assert_array_equal(vis, expected_vis)
    else:
        assert vis is None
    np.testing.assert_array_equal(actual, expected)


def test_raw_preserves_signal_and_leaves_scikit_image_untouched():
    image = np.tile(np.arange(24, dtype=float), (24, 1))
    descriptor = HOGDescriptor(9, (8, 8), (1, 1))
    raw, _ = descriptor.compute_hog(image)
    doubled, _ = descriptor.compute_hog(2 * image)
    np.testing.assert_array_equal(doubled, 2 * raw)
    # A full interior cell has a horizontal derivative of 2 at every pixel.
    assert raw[1, 1, 0, 0, 0] == 2
    assert raw[1, 1, 0, 0, 1:].sum() == 0
    with pytest.raises(ValueError):
        hog(image, block_norm="None")


def test_blank_image_and_one_cell_axis():
    raw, vis = HOGDescriptor(9, (8, 8), (1, 1)).compute_hog(np.zeros((8, 24)))
    assert raw.shape == (1, 3, 1, 1, 9)
    assert not raw.any() and not vis.any()


@pytest.mark.parametrize("image", [np.zeros((7, 8)), np.zeros((16, 16, 3))])
def test_invalid_images(image):
    with pytest.raises(ValueError, match="too small|2D"):
        HOGDescriptor(9, (8, 8), (1, 1)).compute_hog(image)


def test_unknown_normalization_rejected():
    with pytest.raises(ValueError):
        HOGDescriptor(9, (8, 8), (1, 1)).compute_hog(np.zeros((24, 24)), block_norm="typo")
