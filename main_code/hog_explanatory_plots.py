import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

from main_code.pipeline_utils import (
    HOGDescriptor,
    cell_signal_strengths,
    correct_round_angles,
    plot_polar_histogram,
)
from main_code.utils_other import calculate_and_print_percentiles


def plot_zoomed_input_with_grid(image):
    """Zoom in and show image windowing with grid overlay."""
    height, width = image.shape[:2]
    zoomed = image[: height // 3, int(0.8 * width / 3) : int(1.8 * width / 3)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(zoomed)
    ax.set_title("Input Image (zoomed with grid)", fontsize=12)
    ax.axis("off")

    for x in range(0, zoomed.shape[1], 64):
        ax.axvline(x, color="red", linewidth=1.5)
    for y in range(0, zoomed.shape[0], 64):
        ax.axhline(y, color="red", linewidth=1.5)

    return fig


def plot_normalized_hog_zoom(image):
    """HOG image normalized per cell block (after masking background)."""
    zoomed = image[
        : image.shape[0] // 3,
        int(0.8 * image.shape[1] / 3) : int(1.8 * image.shape[1] / 3),
    ]

    hog_descriptor = HOGDescriptor(
        orientations=15,
        pixels_per_cell=(64, 64),
        cells_per_block=(1, 1),
        channel_axis=-1,
    )

    fd_raw, hog_image = hog_descriptor.compute_hog(
        zoomed, block_norm=None, feature_vector=False
    )
    fd = np.squeeze(fd_raw)
    strengths = cell_signal_strengths(fd, norm_ord=1)
    cells_to_keep = strengths > 3

    py, px = hog_descriptor.pixels_per_cell
    h_cells, w_cells = strengths.shape
    cell_imgs = hog_image.reshape(h_cells, py, w_cells, px)

    norm_cells = np.empty_like(cell_imgs)
    for i in range(h_cells):
        for j in range(w_cells):
            block = cell_imgs[i, :, j, :]
            norm_block = block / (strengths[i, j] + 1e-7)
            if not cells_to_keep[i, j]:
                norm_block[:] = 0
            norm_cells[i, :, j, :] = norm_block

    filtered_hog = norm_cells.reshape(hog_image.shape)
    hog_image_norm = exposure.rescale_intensity(filtered_hog, in_range=(0, 0.2))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(hog_image_norm)
    ax.set_title("Normalized HOG (after filtering)", fontsize=12)
    ax.axis("off")
    return fig


def plot_background_filter_strengths(image):
    """Show raw HOG image and strength heatmap, with background masking."""
    zoomed = image[
        : image.shape[0] // 3,
        int(0.8 * image.shape[1] / 3) : int(1.8 * image.shape[1] / 3),
    ]

    hog_descriptor = HOGDescriptor(
        orientations=45,
        pixels_per_window=(64, 64),
        windows_per_block=(1, 1),
        channel_axis=-1,
    )
    fd_raw, hog_image = hog_descriptor.compute_hog(
        zoomed, block_norm=None, feature_vector=False
    )
    fd = np.squeeze(fd_raw)
    strengths = cell_signal_strengths(fd, norm_ord=1)
    cells_to_keep = strengths > 2
    calculate_and_print_percentiles(strengths)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.imshow(exposure.rescale_intensity(hog_image, in_range=(0, 1.5)))
    ax1.set_title("Raw HOG", fontsize=12)
    ax1.axis("off")

    ax2.imshow(strengths, cmap="viridis")
    ax2.imshow(np.ma.masked_where(cells_to_keep, strengths), cmap="gray", alpha=1)
    ax2.set_title("Signal Strength (gray = masked)", fontsize=12)
    ax2.axis("off")

    return fig


def plot_explanatory_polar(image, threshold, correction_artifacts=True):
    """Show polar histogram from a zoomed-in image section."""
    zoomed = image[
        : image.shape[0] // 3,
        int(0.8 * image.shape[1] / 3) : int(1.8 * image.shape[1] / 3),
    ]
    hog_descriptor = HOGDescriptor(
        orientations=45,
        pixels_per_window=(64, 64),
        windows_per_block=(1, 1),
        channel_axis=-1,
    )
    fd_raw, _ = hog_descriptor.compute_hog(
        zoomed, block_norm="None", feature_vector=False
    )
    fd = np.squeeze(fd_raw)
    strengths = cell_signal_strengths(fd, norm_ord=1)
    cells_to_keep = strengths > threshold
    fd_norm = fd / (1e-7 + strengths[:, :, np.newaxis])
    fd_norm[~cells_to_keep] = 0

    gradient_hist_180 = fd_norm[cells_to_keep].mean(axis=0)
    orientations_360_deg = (
        360
        * (np.arange(2 * hog_descriptor.orientations) + 0.5)
        / (2 * hog_descriptor.orientations)
    )
    orientations_180_deg = orientations_360_deg[: len(gradient_hist_180)]

    gradient_hist = dict(zip(orientations_180_deg, gradient_hist_180))
    if correction_artifacts:
        gradient_hist = correct_round_angles(gradient_hist, corr90=True, corr45=True)

    gradient_hist_360 = np.tile(np.array(list(gradient_hist.values())), 2)
    orientations_polar_deg = np.mod(orientations_360_deg, 360)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    plot_polar_histogram(ax, gradient_hist_360, orientations_polar_deg, plot_mean=False)
    ax.set_title("Explanatory Polar Histogram", fontsize=14)

    return fig
