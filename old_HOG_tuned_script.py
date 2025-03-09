import os
import time
import winsound
import numpy as np
import pandas as pd
import re
import warnings
from skimage import exposure  # , filters, feature
import warnings
import matplotlib

# import opencv # STILL NOT WORKING DAMNNN
# print(opencv.__version__)

import matplotlib.pyplot as plt
from pipeline_utils import load_and_prepare_image
from pipeline_utils import HOGDescriptor, plot_polar_histogram
from pipeline_utils import average_directions_over_cells, compute_distribution_direction
from pipeline_utils import correction_of_round_angles, cell_signal_strengths
from plotting_utils import plot_hog_analysis
from utils_other import print_top_values
from utils_other import calculate_and_print_percentiles, get_folder_threshold

root_folder = os.getcwd()
print("current working directory:", root_folder)

group_folders = ["CTR", "MS"]
group_folders = ["ALL"]

df_statistics = pd.DataFrame()

# block_normalize = [None] # only possible normalization method (for now)
block_norm = None
DRAFT = True
# for block_norm in block_normalize:

image_folder_case = []

for group in group_folders:

    # image_folder = os.path.join(
    #     root_folder, "images-3D-lightsheet-20240928_BAM_fkt20_P3_fkt21_P3_PEMFS", group
    # )

    image_folder = os.path.join(
        root_folder,
        "images-3D-lightsheet-20241115_BAM_fkt20-P3-fkt21-P3-PEMFS-12w",
        group,
    )

    # image_folder = os.path.join(
    #     root_folder,
    #     "images-confocal-20241022-fusion-bMyoB-BAMS-TM-6w",
    #     group,
    # )

    # image_folder = os.path.join(
    #     root_folder,
    #     "images-confocal-20241116-fusion-bMyoB-PEMFS-TM-12w",
    #     group,
    # )

    threshold = get_folder_threshold(image_folder)
    print(f"Using threshold={threshold}")

    experiments_folder = os.path.dirname(image_folder)
    image_file_list = os.listdir(image_folder)
    print("Processing images in folder:", image_folder)
    print("Original amount of images:", len(image_file_list))

    # filter image takes (every 5 takes) for 3D-lightsheet data:
    if "3D-lightsheet" in image_folder:
        filtered_list = []

        for image_file in image_file_list:
            z_regex = re.search(
                r"z(\d{1,4})", image_file
            )  # search for substring z followed by 1-4 digits
            z = int(z_regex.group(1))  # extract the number after z
            if z % 5 == 0:
                filtered_list.append(image_file)
        image_file_list_run = filtered_list  # overwrite to image_file_list
    else:
        image_file_list_run = image_file_list

    # if DRAFT is True and len(image_file_list_run) > 15:
    #     # run only on ~10 images (approx. uniformly spread over entire list of images)
    #     js = max(1, (len(image_file_list_run) // 10))
    #     image_file_list_run = image_file_list_run[::js][:10]  # reduce size for Draft

    print("Amount of processed images:", len(image_file_list_run))

    for img_idx, image_file in enumerate(image_file_list_run):

        if img_idx not in [1004, 863, 1413]:
            continue

        is_last_pict = image_file == image_file_list_run[-1]

        image_filename = image_file.replace(" ", "_")
        # cut off initial digits (date and time)
        image_filename = re.sub(r"^\d*_", "", image_filename)
        image_filename = image_filename.rsplit(".", 1)[0]
        t1 = time.time()

        # load image and select Green channel if the image is in RGB(A) format
        image = load_and_prepare_image(image_folder, image_file, channel=1)
        # output image is 2d, therefore need to set the channel_axis=None in HOGDescriptor:

        # image (width) size is 1920, possible (quasi) divisors: 32, 64, 120, 128, 160, 192, 240
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

        gradient_hist_180 = None

        print(f"Final threshold: {threshold:.1e}")
        cells_to_keep = strengths > threshold  # boolean (N,M) array?

        if cells_to_keep.mean() < 0.02:
            # winsound.Beep(440, 500)
            warnings.warn(
                f"Less than 2% of the cells have signal above threshold = {threshold:.1e}."
                f"Is the image uncer-exposed? Skipping image. Consider lowering the threshold."
            )
            continue
        elif cells_to_keep.mean() > 0.98:
            # winsound.Beep(440, 500)
            warnings.warn(
                f"Less than 2% of the cells have been cut off by the threshold = {threshold:.1e}."
                f"Is the image over-exposed? Consider modifying the image or increasing the threshold."
            )

        fd_norm[~cells_to_keep] = np.zeros_like(
            fd.shape[-1]
        )  # exclude cells below threshold, fill with zeroes
        # fd_norm with shape (N, M, n_orientations), orientation for each cell block

        # NOW LET'S TAKE THE AVERAGE OVER THE BLOCKS (some of them are filtered out as zeros already)
        # Now below we have a signal in [0, 180) for every cell-block.

        gradient_hist_180 = fd_norm[cells_to_keep].mean(
            axis=0
        )  # shape (n_orientations)
        assert len(gradient_hist_180) == hog_descriptor.orientations

        gradient_hist = {}
        for key, hist in zip(orientations_180_deg, gradient_hist_180):
            gradient_hist[key] = hist

        # print("Before smoothing")
        # print_top_values(gradient_hist, top_n=3)

        # correct strong signal at 0 and 90 degrees (happens a lot with constant black backgound)
        # TODO: still issue around zero. Maybe problem is in the width of the bin?
        gradient_hist = correction_of_round_angles(
            gradient_hist, corr90=True, corr45=True
        )
        # print("After smoothing")
        # print_top_values(gradient_hist, top_n=3)

        gradient_hist_360 = np.tile(
            np.array(list(gradient_hist.values())), 2
        )  # extend direction measurement to [0, 360) interval

        mean_angle_deg, mode_angle_deg, std_dev_deg, abs_dev_deg = (
            compute_distribution_direction(gradient_hist, orientations_180_deg)
        )

        # We did not account for symmetries: we duplicate the signal vector length for a 360* view in polar plots

        print("image file:", image_file)
        print(f"average direction  : {mean_angle_deg:+.2f} degrees.".replace("+", " "))
        print(f"standard deviation : {std_dev_deg:.2f} degrees")
        print(f"mean abs. deviation: {abs_dev_deg:.2f} degrees")

        # if not is_last_pict:
        plt = plot_hog_analysis(
            image,
            hog_image,
            orientations_360_deg,
            gradient_hist,
            gradient_hist_360,
            cells_to_keep,
            strengths,
            n_pixels_cell,
            fd,
            img_idx,
            block_norm,
            image_filename,
            experiments_folder,
        )
        # plt.close()  # or: plt.close(fig)
        plt.show()
        t2 = time.time()

        elapsed_seconds = int(t2 - t1)
        mins = elapsed_seconds // 60
        secs = elapsed_seconds % 60
        elapsed_time_formatted = f"{mins}:{secs:02d}"

        from plotting_utils import (
            explanatory_plot_intro,
            explanatory_plot_hog,
            explanatory_plot_filter,
            explanatory_plot_polar,
        )

        if img_idx == 0 and DRAFT is True:

            print("First image processed. Time elapsed:", elapsed_time_formatted)

            plt5 = explanatory_plot_polar(image)
            plt5.savefig(
                os.path.join(
                    root_folder,
                    "Polar_histogram.png",
                ),
                dpi=500,
            )
            plt5.show()

            plt2 = explanatory_plot_intro(image)
            plt2.savefig(
                os.path.join(
                    root_folder,
                    "Illustration_windowing.png",
                ),
                dpi=500,
            )
            plt2.show()

            plt3 = explanatory_plot_hog(image)
            plt3.savefig(
                os.path.join(
                    root_folder,
                    "Illustration_HOG.png",
                ),
                dpi=500,
            )
            plt3.show()

            plt4 = explanatory_plot_filter(image)
            plt4.savefig(
                os.path.join(
                    root_folder,
                    "Illustration_filter.png",
                ),
                dpi=500,
            )
            plt4.show()

        image_stats = pd.DataFrame(
            {
                "group": group,
                "image size": str(image.shape),
                "pixels_per_cell": str(n_pixels_cell),
                "elapsed_time (min:sec)": elapsed_time_formatted,
                "avg. direction": round(mean_angle_deg, 3),
                "mode direction": round(mode_angle_deg, 3),
                "std. deviation": round(std_dev_deg, 3),
                "abs. deviation": round(abs_dev_deg, 3),
                "signal_threshold": threshold,
                # "signal_stats": [np.percentile(strengths.flatten(),
                #                             10*np.arange(11))]
            },
            index=[image_file.split(".tif", 1)[0]],
        )

        if len(group_folders) == 1:  # no grouping in the file folder
            image_stats = image_stats.drop(group_folders[0], axis=1, errors="ignore")

            # extract donor and condition from filename using regex

            donor_pattern = r"fkt\d{1,2}"
            match = re.search(donor_pattern, image_filename)
            if not match:
                raise ValueError("Donor identifier not found")
            donor = match.group(0)
            image_stats.insert(2, "donor", donor)

            # done with donor pattern, now the condition pattern:
            ## TODO: double check donor patterns together

            from utils_other import set_sample_condition

            condition = set_sample_condition(image_filename, suppress_warnings=True)

            image_stats.insert(1, "condition", condition)

        df_statistics = pd.concat(
            [df_statistics, image_stats], axis=0
        )  # [brackets] necessary if passing dictionary instead

    # plt.ioff()  # Disable interactive mode to keep the last plot open
    # plt.show()  # Show the last plot
    # plt.close('all')

# stacked_arrs = np.stack(df_statistics['signal_stats'].values)
# print("Decile analysis:", np.mean(stacked_arrs, axis=0))

filename = "HOG_stats_" + str(hog_descriptor.pixels_per_cell[0]) + "pixels"

pattern = r"-(\d{6,10})_"
match = re.search(pattern, filename)

if match:
    filename = filename + match.group(1)

if DRAFT is True:
    filename = filename + "_draft_debug"

df_statistics.to_csv(os.path.join(experiments_folder, filename + ".csv"))

# fill in manually for post-processing, or use same value as filename... ?
filename_in = filename
# filename_in = "HOG_stats_64pixels"

filename_out = filename_in

if "confocal-20241116" in experiments_folder:
    filename_out += "-confocal-20241116"
elif "lightsheet-20241115" in experiments_folder:
    filename_out += "-lsheet-20241115"
elif "lightsheet-20240928" in experiments_folder:
    filename_out += "-lsheet-20240928"
elif "confocal-20241022" in experiments_folder:
    filename_out += "-confocal-20241022"

input_csv_path = os.path.join(experiments_folder, filename_in + ".csv")
output_csv_path = os.path.join(experiments_folder, filename_out + "_clean.csv")


# post-processing: correct columns names
from utils_other import update_conditions_to_csv

update_conditions_to_csv(input_csv_path, output_csv_path, suppress_warnings=False)

# import winsound
# winsound.Beep(frequency=880, duration=2500)
