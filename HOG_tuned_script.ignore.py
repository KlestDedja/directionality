import os
import time
import numpy as np
import pandas as pd
from math import ceil
import re
import warnings
from skimage import exposure  # , filters, feature
import matplotlib.colors as mcolors

# import opencv # STILL NOT WORKING DAMNNN
# print(opencv.__version__)

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pipeline_utils import load_and_prepare_image
from pipeline_utils import HOGDescriptor, plot_polar_histogram
from pipeline_utils import average_directions_over_cells, compute_average_direction
from pipeline_utils import correction_of_round_angles, cell_signal_strengths
from utils_other import calculate_and_print_percentiles

root_folder = os.getcwd()
print("current working directory:", root_folder)

group_folders = ["CTR", "MS"]
group_folders = ["ALL"]

df_statistics = pd.DataFrame()

# block_normalize = [None] # only possible normalization method (for now)
block_norm = None
# for block_norm in block_normalize:

for group in group_folders:

    image_folder = os.path.join(root_folder, "Images_confocal", group)
    image_folder = os.path.join(root_folder, "Images-fluorescence-2D", group)

    experiments_folder = os.path.dirname(image_folder)
    image_file_list = os.listdir(image_folder)
    # print("List of images: ", image_file_list)

    for img_idx, image_file in enumerate(image_file_list):
        is_last_pict = image_file == image_file_list[-1]

        image_filename = image_file.replace(" ", "_")
        image_filename = re.sub(
            r"^\d*_", "", image_filename
        )  # cut off initial digits (date and time)
        image_filename = image_filename.rsplit(".", 1)[0]
        t1 = time.time()

        # load image and select Green channel if the image is in RGB(A) format
        image = load_and_prepare_image(image_folder, image_file, channel=1)
        # output image is 2d, therefore need to set the channel_axis=None in HOGDescriptor:

        # image (width) size is 1920, possible (near) divisors: 32, 64, 120, 128, 160, 192, 240
        hog_descriptor = HOGDescriptor(
            orientations=45,
            pixels_per_cell=(64, 64),
            cells_per_block=(1, 1),
            channel_axis=-1,
        )

        n_pixels_cell = hog_descriptor.pixels_per_cell
        n_cells_block = (
            hog_descriptor.cells_per_block[0] * hog_descriptor.cells_per_block[1]
        )

        fd_raw, hog_image = hog_descriptor.compute_hog(
            image, block_norm=block_norm, feature_vector=False
        )
        # Note: block_norm ``L2-Hys`` is not any better with the artefact

        # we imitate skimage hog source code (based on 180 degrees range) and duplicate the range
        orient_arr = np.arange(2 * hog_descriptor.orientations)
        orientations_360_deg = (
            360 * (orient_arr + 0.5) / (2 * hog_descriptor.orientations)
        )  # from skimage hog source code
        #   ^--- this is how it is done in skimage, do not change!
        orientations_180_deg = orientations_360_deg[: len(orientations_360_deg) // 2]
        ## fd is of shape (cells_H_axis, cells_W_axis, Blocks_per_cell_H, Blocks_per_cell_W, n_orientations)

        # if n_cells_block == 1:
        fd = np.squeeze(fd_raw)  # fd has now shape (N, M, n_orientations)

        # computes strenghts of signal for each cell
        if block_norm is None:  # manually normalize:
            strengths = cell_signal_strengths(
                fd, norm_ord=1
            )  # output shape is shape (N, M), we normalize by L1 norm.
            calculate_and_print_percentiles(strengths)
            fd_norm = fd / (
                1e-7 + strengths[:, :, np.newaxis]
            )  # normalize everything to  |v| = 1
        # threshold = np.percentile(strengths.flatten(), 55) # for now, we set this as threshold, but can improve
        else:  # no normalization step is needed:
            strengths = np.ones(fd_norm.shape[0], fd_norm.shape[1])
            fd_norm = fd
            warnings.warn(
                f"There is no implemented solution for background elimination"
                f" in case block_norm is not None. Found {block_norm}"
            )
        threshold = 1.843  # old threshold = 1.568

        """ with cell blocks of size (32, 32)
        Decile analysis of the signal over the first 6 pictures: [
        min -> 0.58064974    10 -> 0.95527117      20 -> 1.09113596    30 -> 1.21609939
        40  -> 1.37751670    50 -> 1.56824218      60 -> 1.84369730    70 -> 2.19784849
        80  -> 2.79174948    90 -> 3.93258352     max -> 12.97092674
        """
        # if "z057c1" in image_filename:
        #     plt.title("Kept cell, at the edge")
        #     plt.bar(orientations_180_deg, fd_norm[0, 10, :], width=3)
        #     plt.show()

        cells_to_keep = strengths > threshold  # boolean (N,M) array?
        fd_norm[~cells_to_keep] = np.zeros_like(
            fd.shape[-1]
        )  # exclude cells below threshold, fill with zeroes
        # fd_norm with shape (N, M, n_orientations), orientation for each cell block

        # NOW LET'S TAKE THE AVERAGE OVER THE BLOCKS (come of them are filtered out as zeros already)
        # Now below we have a signal in [0, 180) for every cell-block.
        gradient_hist_180 = fd_norm[cells_to_keep].mean(
            axis=0
        )  # shape (n_orientations)
        assert len(gradient_hist_180) == hog_descriptor.orientations

        # correct strong signal at 0 and 90 degrees (happens a lot with constant black backgound)
        gradient_hist_180 = correction_of_round_angles(
            orientations_180_deg, gradient_hist_180, corr90=True, corr45=True
        )
        # ToDO consider adding correction at 45*, if necessary

        mean_angle_deg, std_dev_deg, abs_dev_deg = compute_average_direction(
            gradient_hist_180, orientations_180_deg
        )

        # We did not account for symmetries: we duplicate the signal vector length for a 360* view in polar plots

        print("image file:", image_file)
        print(f"average direction  : {mean_angle_deg:+.2f} degrees.".replace("+", " "))
        print(f"standard deviation : {std_dev_deg:.2f} degrees")
        print(f"mean abs. deviation: {abs_dev_deg:.2f} degrees")

        gradient_hist_360 = np.tile(
            gradient_hist_180, 2
        )  # extend direction measurement to [0, 360) interval
        # fd_final_grid = np.tile(fd_norm, (1, 1, 2)) #repeat values along the third (last) axis
        # fd_final_lin = fd_final_grid.reshape(-1, len(global_histogram))

        fig = plt.figure(figsize=(8, 10))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3, projection="polar")
        ax4 = fig.add_subplot(2, 2, 4)

        ax1.axis("off")
        ax1.imshow(image)  # 'image' should be defined previously
        ax1.set_title("Original input image")

        ax2.axis("off")
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1.7))
        ax2.imshow(hog_image_rescaled)
        ax2.set_title("Histogram of Oriented Gradients")

        # Pass the pre-defined axis object 'ax3' which was set to be a polar plot
        # traslate by 90 degrees as we nned the angle perpendicular to the steepest gradient

        # intuitively, this should be = 90. In practice, with HOG it's more = 0
        ROTATE_FOR_GRADIENT = 0
        orientations_polar_deg = np.mod(orientations_360_deg + ROTATE_FOR_GRADIENT, 360)

        estim_ymax = gradient_hist_180.max()

        # max_y_tick = ceil(estim_ymax/0.5)*0.5 # round up to nearest half-integer
        bars = plot_polar_histogram(ax3, gradient_hist_360, orientations_polar_deg)

        # Do not set the ax3.axis('off') !
        ax3.set_yticks(
            np.linspace(0, estim_ymax, num=4)
        )  # Adjust the number of ticks here
        ax3.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
        ax3.yaxis.label.set_size(6)
        ax3.set_ylim(0, 1.1 * estim_ymax)
        ax3.set_title("Directionality plot")

        included_cells = cells_to_keep.reshape(fd.shape[0], fd.shape[1])
        ax4.axis("off")
        im = ax4.imshow(included_cells)
        ax4.set_title("Visualisation of included blocks")
        heatmap = ax4.imshow(strengths, cmap="viridis", interpolation="nearest")
        cbar = fig.colorbar(heatmap, ax=ax4, shrink=0.6, pad=0.05, fraction=0.07)
        cbar.ax.tick_params(labelsize=8)
        ax4.set_title("Signal heatmap with mask in grey")
        # Overlay cells below threshold in red
        # cmap_red = matplotlib.colors.ListedColormap(['red'])
        rgb_color = (0.7, 0.7, 0.7)  # light gray
        cmap_gray = mcolors.ListedColormap([rgb_color])
        masked_im = ax4.imshow(
            np.ma.masked_where(cells_to_keep, strengths),
            cmap=cmap_gray,
            interpolation=None,
            alpha=1,
        )

        # Add black borders to the gray cells
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
                    if i % 3 == 0 and j % 3 == 0 and n_pixels_cell[0] > 30:
                        ax4.text(
                            j,
                            i,
                            f"({i},{j})",
                            ha="center",
                            va="center",
                            fontsize=2,
                            color="black",
                        )

        plt.tight_layout()
        plt.suptitle(f"Image n*{img_idx}, block_norm: {block_norm}")
        plt.savefig(
            os.path.join(
                experiments_folder,
                "HOG_plot_analysis",
                image_filename + f"_{block_norm}_{n_pixels_cell[0]}.png",
            ),
            dpi=450,
        )

        # if not is_last_pict:
        plt.close()  # plt.close(fig)

        t2 = time.time()

        elapsed_seconds = int(t2 - t1)
        mins = elapsed_seconds // 60
        secs = elapsed_seconds % 60
        elapsed_time_formatted = f"{mins}:{secs:02d}"

        image_stats = pd.DataFrame(
            {
                "group": group,
                "image size": str(image.shape),
                "pixels_per_cell": str(n_pixels_cell),
                "elapsed_time (min:sec)": elapsed_time_formatted,
                "avg. direction": round(mean_angle_deg, 3),
                "std. deviation": round(std_dev_deg, 3),
                "abs. deviation": round(abs_dev_deg, 3),
                "signal_threshold": threshold,
                "signal_stats": [
                    np.percentile(strengths.flatten(), 10 * np.arange(11))
                ],
            },
            index=[image_file.split(".tif", 1)[0]],
        )

        if len(group_folders) == 1:  # no grouping inthe file folder
            image_stats = image_stats.drop(group_folders[0], axis=1, errors="ignore")

            # extract donor and condition from filename using regex

            d_pattern = r"fkt\d{1,2}"
            match = re.search(d_pattern, image_filename)
            if not match:
                raise ValueError("Donor identifier not found")
            donor = match.group(0)

            image_stats.insert(2, "donor", donor)
            # done with donor pattern, now the condition pattern:

            c_pattern1 = r"\d{1,2}[hm']"
            match_pattern1 = re.search(c_pattern1, image_filename)
            if match_pattern1:
                condition = match_pattern1.group(0).replace("'", "m")

            c_pattern2 = r"\d{1}[-,.]?\d?'\d{1}[-,.]?\d?"  # 10'5' or 1,5'0,5'
            match_pattern2 = re.search(c_pattern2, image_filename)
            if match_pattern2:
                condition = match_pattern2.group(0).replace("'", "m")

            c_pattern_ctr = r"CTR"
            match_pattern_ctr = re.search(c_pattern_ctr, image_filename)
            if match_pattern_ctr:
                condition = match_pattern_ctr.group(0)

            image_stats.insert(2, "condition", condition)

            """
            let's see:
            """
            """
            wrong matches:
            - 140h_14h -> becomes 40h instead of 14h


            """

        df_statistics = pd.concat(
            [df_statistics, image_stats], axis=0
        )  # [brackets] necessary if passing dictionary instead

    # plt.ioff()  # Disable interactive mode to keep the last plot open
    # plt.show()  # Show the last plot
    plt.close("all")

# stacked_arrs = np.stack(df_statistics['signal_stats'].values)
# print("Decile analysis:", np.mean(stacked_arrs, axis=0))

filename = (
    "HOG_stats_"
    + str(block_norm)
    + "_"
    + str(hog_descriptor.pixels_per_cell[0])
    + "pixels"
)
if len(image_file_list) < 0.7 * len(os.listdir(image_folder)):
    filename = filename + "_draft"

df_statistics.to_csv(os.path.join(experiments_folder, filename + ".csv"))
