"""Unnormalized, single-channel HOG using NumPy and public scikit-image APIs.

Gradient and visualization conventions follow scikit-image 0.23--0.25's HOG
implementation (BSD-3-Clause; see THIRD_PARTY_NOTICES.txt). Keeping the raw
histogram here avoids modifying scikit-image or importing its private helpers.
"""

from numbers import Integral

import numpy as np
from skimage.draw import line


def _positive_pair(value, name):
    if len(value) != 2 or any(
        not isinstance(v, Integral) or isinstance(v, bool) or v <= 0 for v in value
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return tuple(value)


def hog_without_normalization(
    image, orientations, pixels_per_cell, cells_per_block, visualize, feature_vector
):
    """Return raw HOG blocks, optionally with the usual HOG visualization.

    Only 2D images are accepted: EDGEHOG selects a channel before calling HOG.
    As in scikit-image, histograms contain mean gradient magnitudes per cell,
    incomplete edge cells are omitted, and blocks may overlap. Both histogram
    accumulation precision and the output dtype follow the legacy HOG path.
    """
    image = np.asarray(image)
    if image.ndim != 2 or np.iscomplexobj(image):
        raise ValueError("HOG requires a real 2D image; select a channel first")
    if (
        not isinstance(orientations, Integral)
        or isinstance(orientations, bool)
        or orientations <= 0
    ):
        raise ValueError("orientations must be a positive integer")
    cell_rows, cell_cols = _positive_pair(pixels_per_cell, "pixels_per_cell")
    block_rows, block_cols = _positive_pair(cells_per_block, "cells_per_block")
    nrows, ncols = image.shape
    cells_y, cells_x = nrows // cell_rows, ncols // cell_cols
    if cells_y < block_rows or cells_x < block_cols:
        raise ValueError(
            "The input image is too small for pixels_per_cell and cells_per_block; "
            f"it needs at least {cell_rows * block_rows} rows and "
            f"{cell_cols * block_cols} columns"
        )

    dtype = np.float32 if image.dtype in (np.float16, np.float32) else np.float64
    image = image.astype(dtype, copy=False)
    if not np.isfinite(image).all():
        raise ValueError("HOG requires finite image values")
    grad_row = np.zeros_like(image)
    grad_col = np.zeros_like(image)
    grad_row[1:-1, :] = image[2:, :] - image[:-2, :]
    grad_col[:, 1:-1] = image[:, 2:] - image[:, :-2]
    # scikit-image computes histogram angles and magnitudes in float64, even
    # when the image gradients were calculated in float32.
    grad_row = grad_row.astype(float, copy=False)
    grad_col = grad_col.astype(float, copy=False)
    magnitude = np.hypot(grad_col, grad_row)
    angles = np.rad2deg(np.arctan2(grad_row, grad_col)) % 180
    height, width = cells_y * cell_rows, cells_x * cell_cols
    magnitude = magnitude[:height, :width]
    angles = angles[:height, :width]
    edges = (180.0 / orientations) * np.arange(1, orientations + 1)
    bins = np.searchsorted(edges, angles, side="right")
    cell_ids = (
        np.arange(height)[:, None] // cell_rows * cells_x + np.arange(width)[None, :] // cell_cols
    )
    indices = (cell_ids * orientations + bins).ravel()
    # Unbuffered additions preserve pixel traversal order and float32 totals
    # used by scikit-image's original Cython histogram accumulator.
    totals = np.zeros(cells_y * cells_x * orientations, dtype=np.float32)
    valid = bins.ravel() < orientations
    np.add.at(totals, indices[valid], magnitude.ravel()[valid])
    hist = (totals / np.float32(cell_rows * cell_cols)).astype(float)
    hist = hist.reshape(cells_y, cells_x, orientations)

    blocks = (
        np.lib.stride_tricks.sliding_window_view(hist, (block_rows, block_cols), axis=(0, 1))
        .transpose(0, 1, 3, 4, 2)
        .astype(dtype, copy=True)
    )
    if feature_vector:
        blocks = blocks.ravel()
    if not visualize:
        return blocks

    hog_image = np.zeros(image.shape, dtype=dtype)
    radius = max(0, min(cell_rows, cell_cols) // 2 - 1)
    midpoints = np.pi * (np.arange(orientations) + 0.5) / orientations
    drs, dcs = radius * np.sin(midpoints), radius * np.cos(midpoints)
    for r in range(cells_y):
        for c in range(cells_x):
            cy, cx = r * cell_rows + cell_rows // 2, c * cell_cols + cell_cols // 2
            for o, (dr, dc) in enumerate(zip(drs, dcs)):
                rr, cc = line(int(cy - dc), int(cx + dr), int(cy + dc), int(cx - dr))
                hog_image[rr, cc] += hist[r, c, o]
    return blocks, hog_image
