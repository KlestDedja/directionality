import matplotlib
import os
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from pipeline_utils import (
    HOGDescriptor,
    cell_signal_strengths,
    correction_of_round_angles,
    plot_polar_histogram,
)
from utils_other import calculate_and_print_percentiles


def external_plot_hog_analysis(
    image,
    hog_image,
    gradient_hist,
    cells_to_keep,
    strengths,
):

    gradient_hist_360 = np.tile(np.array(list(gradient_hist.values())), 2)

    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3, projection="polar")
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.axis("off")
    ax1.imshow(image)
    ax1.set_title("Original input image")

    ax2.axis("off")
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1.7))
    ax2.imshow(hog_image_rescaled)
    ax2.set_title("Histogram of Oriented Gradients")

    ROTATE_FOR_GRADIENT = 0
    orientations_360_deg = np.linspace(0, 360, 90, endpoint=False)
    orientations_polar_deg = np.mod(orientations_360_deg + ROTATE_FOR_GRADIENT, 360)
    estim_ymax = np.array(list(gradient_hist.values())).max()

    bars = plot_polar_histogram(
        ax3, gradient_hist_360, orientations_polar_deg, plot_mean=False
    )
    ymax_lim = max(estim_ymax, 1e-3)
    ax3.set_yticks(np.linspace(0, ymax_lim, num=4))
    ax3.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax3.yaxis.label.set_size(6)
    ax3.set_ylim(0, 1.1 * ymax_lim)
    ax3.set_title("Directionality plot")
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)

    # included_cells = cells_to_keep.reshape(strengths.shape)
    ax4.axis("off")
    heatmap = ax4.imshow(strengths, cmap="viridis", interpolation="nearest")
    cbar = fig.colorbar(heatmap, ax=ax4, shrink=0.6, pad=0.05, fraction=0.07)
    cbar.ax.tick_params(labelsize=8)
    ax4.set_title("Signal heatmap with mask in grey")

    rgb_color = (0.7, 0.7, 0.7)  # light gray
    cmap_gray = matplotlib.colors.ListedColormap([rgb_color])
    masked_im = ax4.imshow(
        np.ma.masked_where(cells_to_keep, strengths),
        cmap=cmap_gray,
        interpolation=None,
        alpha=1,
    )

    rows, cols = strengths.shape
    for i in range(rows):
        for j in range(cols):
            if not cells_to_keep[i, j]:
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    linewidth=0.2,
                    edgecolor="black",
                    facecolor="none",
                )
                ax4.add_patch(rect)
                if i % 3 == 0 and j % 3 == 0:
                    ax4.text(
                        j,
                        i,
                        f"({i},{j})",
                        ha="center",
                        va="center",
                        fontsize=2,
                        color="black",
                    )

    return plt


def explanatory_plot_intro(image):

    height, width = image.shape[:2]
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(9, 4))

    # --- Left plot: original image ---
    # ax1.set_title("Original Image")
    # ax1.imshow(image)
    # ax1.axis("off")

    # --- Middle plot: zoomed-in top-left corner ---
    # The coordinate system in Matplotlib for images typically has origin in top-left.

    zoomed_image = image[: height // 3, int(0.8 * width / 3) : int(1.8 * width / 3)]
    zoomed_height, zoomed_width = zoomed_image.shape[:2]

    # Limit the visible area to the zoomed part
    ax2.set_title("Example image (zoomed)")
    ax2.imshow(zoomed_image)
    ax2.axis("off")
    # Note: we invert the y-limits because images in pyplot have y increasing downward
    # ax2.set_xlim([zoom_width, 2 * zoom_width])
    # ax2.set_ylim([zoom_height, 0])

    # --- Right plot: highlight a grid of windows 64×64 ---
    ax3.imshow(zoomed_image)
    ax3.set_title("Image splitting")
    ax3.axis("off")

    # Draw grid lines at multiples of 64 pixels
    window_size = 64
    for x in range(0, zoomed_width, window_size):
        ax3.axvline(x, color="red", linewidth=1.5)
    for y in range(0, zoomed_height, window_size):
        ax3.axhline(y, color="red", linewidth=1.5)

    plt.tight_layout()
    # plt.show()
    return plt


from skimage import exposure  # , filters, feature


def explanatory_plot_hog(image):

    height, width = image.shape[:2]
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(9, 4))

    zoomed_image = image[: height // 3, int(0.8 * width / 3) : int(1.8 * width / 3)]
    zoomed_height, zoomed_width = zoomed_image.shape[:2]

    # --- Left plot: highlight a grid of windows 64×64 ---
    ax3.imshow(zoomed_image)
    ax3.set_title("Image splitting")
    ax3.axis("off")

    # Draw grid lines at multiples of 64 pixels
    window_size = 64
    for x in range(0, zoomed_width, window_size):
        ax3.axvline(x, color="red", linewidth=1.5)
    for y in range(0, zoomed_height, window_size):
        ax3.axhline(y, color="red", linewidth=1.5)

    # HOG image here (example)
    hog_descriptor = HOGDescriptor(
        orientations=15,
        pixels_per_cell=(64, 64),
        cells_per_block=(1, 1),
        channel_axis=-1,
    )

    _, hog_image = hog_descriptor.compute_hog(
        zoomed_image, block_norm=None, feature_vector=False
    )

    ax4.axis("off")
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1.5))
    ax4.imshow(hog_image_rescaled)
    ax4.set_title("Direction of the Oriented Gradients")

    plt.tight_layout()
    # plt.show()
    return plt


def explanatory_normalized_hog(image):

    height, width = image.shape[:2]
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(9, 4))

    zoomed_image = image[: height // 3, int(0.8 * width / 3) : int(1.8 * width / 3)]
    zoomed_height, zoomed_width = zoomed_image.shape[:2]

    # --- Left plot: highlight a grid of windows 64×64 ---
    ax3.imshow(zoomed_image)
    ax3.set_title("Image splitting")
    ax3.axis("off")

    # Draw grid lines at multiples of 64 pixels
    window_size = 64
    for x in range(0, zoomed_width, window_size):
        ax3.axvline(x, color="red", linewidth=1.5)
    for y in range(0, zoomed_height, window_size):
        ax3.axhline(y, color="red", linewidth=1.5)

    # HOG image here (example)
    hog_descriptor = HOGDescriptor(
        orientations=15,
        pixels_per_cell=(64, 64),
        cells_per_block=(1, 1),
        channel_axis=-1,
    )

    fd_raw_bg, hog_image = hog_descriptor.compute_hog(
        zoomed_image, block_norm=None, feature_vector=False
    )

    fd_bg = np.squeeze(fd_raw_bg)
    strengths = cell_signal_strengths(fd_bg, norm_ord=1)

    fd_norm = fd_bg / (1e-7 + strengths[:, :, np.newaxis])

    cells_to_keep = strengths > 3  # manual threshold for the example

    fd_norm[~cells_to_keep] = np.zeros_like(
        fd_bg.shape[-1]
    )  # exclude cells below threshold, fill with zeroes
    # fd_norm with shape (N, M, n_orientations), orientation for each cell block
    fd_norm[~cells_to_keep, :] = 0

    # now do the same normalization procedure to hog_image itself
    py, px = hog_descriptor.pixels_per_cell
    h_cells, w_cells = strengths.shape

    # reshape the visualization into (n_cells_y, py, n_cells_x, px)
    cell_imgs = hog_image.reshape(h_cells, py, w_cells, px)

    # normalize each cell by that same strengths map
    #    (so bright cells stay bright, weak cells get dimmed/masked)
    norm_cells = np.empty_like(cell_imgs)
    for i in range(h_cells):
        for j in range(w_cells):
            block = cell_imgs[i, :, j, :]
            # divide the entire block by its original strength, then mask
            norm_block = block / (strengths[i, j] + 1e-7)
            if not cells_to_keep[i, j]:
                norm_block[:] = 0
            norm_cells[i, :, j, :] = norm_block

    # re‑assemble back to full image
    filtered_hog_vis = norm_cells.reshape(hog_image.shape)

    # rescale & plot just like before
    hog_image_norm = exposure.rescale_intensity(filtered_hog_vis, in_range=(0, 0.2))

    ax4.axis("off")
    ax4.imshow(hog_image_norm)
    ax4.set_title("Direction of Oriented Gradients, after filtering + normalizzation")

    plt.tight_layout()
    # plt.show()
    return plt


def explanatory_plot_filter(image):

    height, width = image.shape[:2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    zoomed_image = image[: height // 3, int(0.8 * width / 3) : int(1.8 * width / 3)]
    zoomed_height, zoomed_width = zoomed_image.shape[:2]

    # HOG image here (example)
    hog_descriptor = HOGDescriptor(
        orientations=45,
        pixels_per_cell=(64, 64),
        cells_per_block=(1, 1),
        channel_axis=-1,
    )

    fd_raw, hog_image = hog_descriptor.compute_hog(
        zoomed_image, block_norm=None, feature_vector=False
    )
    fd = np.squeeze(fd_raw)  # fd has now shape (N, M, n_orientations)

    # computes strenghts of signal for each cell. Assumes block_norm is None
    strengths = cell_signal_strengths(
        fd, norm_ord=1
    )  # output shape is shape (N, M), we normalize by L1 norm.
    calculate_and_print_percentiles(strengths)

    cells_to_keep = strengths > 2  # boolean (N,M) array?

    ax1.axis("off")
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1.5))
    ax1.imshow(hog_image_rescaled)
    ax1.set_title("Direction of the Oriented Gradients")
    # Draw grid lines at multiples of 64 pixels
    window_size = 64
    for x in range(0, zoomed_width, window_size):
        ax1.axvline(x, color="red", linewidth=1.5)
    for y in range(0, zoomed_height, window_size):
        ax1.axhline(y, color="red", linewidth=1.5)

    included_cells = cells_to_keep.reshape(fd.shape[0], fd.shape[1])

    ax2.set_title("Heatmap of the magnitude of Oriented Gradients")
    ax2.axis("off")
    # im = ax2.imshow(included_cells)
    ax2.set_title("Visualisation of included blocks")
    heatmap = ax2.imshow(strengths, cmap="viridis", interpolation="nearest")
    cbar = fig.colorbar(heatmap, ax=ax2, shrink=0.6, pad=0.05, fraction=0.07)
    cbar.ax.tick_params(labelsize=8)
    # Overlay cells below threshold in red
    # cmap_red = matplotlib.colors.ListedColormap(['red'])
    rgb_color = (0.7, 0.7, 0.7)  # light gray
    cmap_gray = matplotlib.colors.ListedColormap([rgb_color])
    masked_im = ax2.imshow(
        np.ma.masked_where(cells_to_keep, strengths),
        cmap=cmap_gray,
        interpolation=None,
        alpha=1,
    )

    plt.tight_layout()
    # plt.show()
    return plt


def explanatory_plot_polar(image, plot_mean=False):

    fig = plt.figure(figsize=(13, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="polar")

    height, width = image.shape[:2]
    zoomed_image = image[: height // 3, int(0.8 * width / 3) : int(1.8 * width / 3)]
    zoomed_height, zoomed_width = zoomed_image.shape[:2]

    # if zoomed image, uncomment:
    image = zoomed_image

    ax1.imshow(image)
    ax1.set_title("Image splitting")
    ax1.axis("off")

    # HOG image here (example)
    hog_descriptor = HOGDescriptor(
        orientations=45,
        pixels_per_cell=(64, 64),
        cells_per_block=(1, 1),
        channel_axis=-1,
    )

    fd_raw, hog_image = hog_descriptor.compute_hog(
        image, block_norm=None, feature_vector=False
    )

    orient_arr = np.arange(2 * hog_descriptor.orientations)
    orientations_360_deg = (
        360 * (orient_arr + 0.5) / (2 * hog_descriptor.orientations)
    )  # from skimage hog source code
    #   ^--- this is how it is done in skimage, do not change!
    orientations_180_deg = orientations_360_deg[: len(orientations_360_deg) // 2]

    fd = np.squeeze(fd_raw)  # fd has now shape (N, M, n_orientations)

    # computes strenghts of signal for each cell. Assumes block_norm is None
    strengths = cell_signal_strengths(
        fd, norm_ord=1
    )  # output shape is shape (N, M), we normalize by L1 norm.
    calculate_and_print_percentiles(strengths)

    cells_to_keep = strengths > 2  # boolean (N,M) array?

    strengths = cell_signal_strengths(
        fd, norm_ord=1
    )  # output shape is shape (N, M), we normalize by L1 norm.
    fd_norm = fd / (1e-7 + strengths[:, :, np.newaxis])

    fd_norm[~cells_to_keep] = np.zeros_like(
        fd.shape[-1]
    )  # exclude cells below threshold, fill with zeroes
    # fd_norm with shape (N, M, n_orientations), orientation for each cell block

    # NOW LET'S TAKE THE AVERAGE OVER THE BLOCKS (some of them are filtered out as zeros already)
    # Now below we have a signal in [0, 180) for every cell-block.
    gradient_hist_180 = fd_norm[cells_to_keep].mean(axis=0)  # shape (n_orientations)
    assert len(gradient_hist_180) == hog_descriptor.orientations

    gradient_hist = {}
    for key, hist in zip(orientations_180_deg, gradient_hist_180):
        gradient_hist[key] = hist

    # correct strong signal at 0 and 90 degrees (happens a lot with constant black backgound)
    gradient_hist = correction_of_round_angles(gradient_hist, corr90=True, corr45=True)

    # mean_angle_deg, std_dev_deg, abs_dev_deg = compute_average_direction(
    #     gradient_hist, orientations_180_deg
    # )

    # prepare data and find argmax:
    angles = orientations_180_deg
    heights = list(gradient_hist.values())
    max_idx = np.argmax(heights)

    # build a color array: all (default) blue except the max one
    colors = ["C0"] * len(heights)
    colors[max_idx] = "red"

    # plot with per‐bar coloring
    ax2.bar(
        x=angles,
        height=heights,
        width=180 / len(angles),
        color=colors,
        edgecolor="black",
        linewidth=0.2,
    )
    # rest is the same…
    ax2.set_xticks(np.arange(min(angles) - 2, max(angles) + 3, 30))
    ax2.set_title("Histogram of Oriented Gradients")
    # ax2.set_xticks([])
    ax2.set_yticks([])
    # ax2.set_xlabel("")
    ax2.set_ylabel("")

    gradient_hist_360 = np.tile(
        np.array(list(gradient_hist.values())), 2
    )  # extend direction measurement to [0, 360) interval

    orientations_polar_deg = np.mod(orientations_360_deg, 360)

    estim_ymax = np.array(list(gradient_hist.values())).max()

    # max_y_tick = ceil(estim_ymax/0.5)*0.5 # round up to nearest half-integer
    bars = plot_polar_histogram(
        ax3, gradient_hist_360, orientations_polar_deg, plot_mean=plot_mean
    )
    ymax_lim = max(estim_ymax, 1e-3)
    # Do not set the ax3.axis('off') !

    ax3.set_yticks(np.linspace(0, ymax_lim, num=4))  # Adjust the number of ticks here
    ax3.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax3.yaxis.label.set_size(6)
    ax3.set_ylim(0, 1.1 * ymax_lim)
    ax3.set_title("Directionality plot")
    ax3.set_theta_zero_location("N")  # North,  West, or East?
    ax3.set_theta_direction(-1)  # -1 for Clockwise

    plt.tight_layout()

    return plt


def list_boo_boo() -> list:
    return [
        "20241115 bMyoB fkt21 P3_2h R2-Image Export-29_i2z005c1-2.tif",
        "20241115 bMyoB fkt21 P3_2h R2-Image Export-29_i2z105c1-2.tif",
        "20241115 bMyoB fkt21 P3_2h R2-Image Export-29_i2z095c1-2.tif",
        "20241115 bMyoB fkt21 P3_2h R2-Image Export-29_i2z085c1-2.tif",
        "20241115 bMyoB fkt21 P3_2h R2-Image Export-29_i2z075c1-2.tif",
        "20241115 bMyoB fkt21 P3_2h R2-Image Export-29_i2z065c1-2.tif",
        "20241115 bMyoB fkt21 P3_2h R2-Image Export-29_i2z055c1-2.tif",
        "20241115 bMyoB fkt21 P3_2h R2-Image Export-29_i2z035c1-2.tif",
        "20241115 bMyoB fkt21 P3_2h R2-Image Export-29_i2z025c1-2.tif",
        "20241115 bMyoB fkt21 P3_2h R2-Image Export-29_i2z015c1-2.tif",
        "20241115 bMyoB fkt21 P3_1_5'0_5' R2_3-Image Export-19_i2z20c1-2.tif",
        "20241115 bMyoB fkt21 P3_1_5'0_5' R2_3-Image Export-19_i1z10c1-2.tif",
        "20241115 bMyoB fkt21 P3_1_5'0_5' R2_3-Image Export-19_i2z10c1-2.tif",
    ]


def list_images_plot() -> list:
    return [
        "20241115 bMyoB fkt21 P3_2h R2_4-Image Export-28_i1z080c1-2.tif",
        "20241115 bMyoB fkt20 P3_1_5'0_5' R2-Image Export-05_i2z140c1-2.tif",
        "20241115 bMyoB fkt20 P3_ctr R1_2-Image Export-11_i1z040c1-2.tif",
        "20241115 bMyoB fkt20 P3_ctr R1_2-Image Export-11_i1z040c1-2.tif",
    ]
