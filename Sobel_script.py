
import os
import time
import numpy as np
import pandas as pd
import math
from math import ceil
import re
import warnings
from skimage import exposure
from skimage import io, filters, color
import matplotlib
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
print('current working directory:', root_folder)

group_folders = ['CTR', 'MS']
# group_folders = ['ALL']

df_statistics = pd.DataFrame()

# block_normalize = [None] # only possible normalization method (for now)
block_norm = None
# for block_norm in block_normalize:

for group in group_folders:

    image_folder = os.path.join(root_folder, 'Images_confocal', group)
    image_folder = os.path.join(root_folder, 'Images-fluorescence-2D', group)
    image_folder = os.path.join(root_folder, 'images-data-trial', group)

    experiments_folder = os.path.dirname(image_folder)
    image_file_list = os.listdir(image_folder)[::3]
    # print("List of images: ", image_file_list)

    for img_idx, image_file in enumerate(image_file_list):
        is_last_pict = (image_file == image_file_list[-1])

        image_filename = image_file.replace(" ", "_")
        image_filename = re.sub(r'^\d*_', '', image_filename) #cut off initial digits (date and time)
        image_filename = image_filename.rsplit(".", 1)[0]
        t1 = time.time()

        # load image and select Green channel if the image is in RGB(A) format
        image = load_and_prepare_image(image_folder, image_file, channel=1)
        # output image is 2d, therefore need to set the channel_axis=None in HOGDescriptor:

        # image (width) size is 1920, possible (near) divisors: 32, 64, 120, 128, 160, 192, 240


        grad_x = filters.sobel_v(image)  # Vertical Sobel filter (for x gradient) Shape equal to image size
        grad_y = filters.sobel_h(image)  # Horizontal Sobel filter (for y gradient) Shape equal to image size

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi)  # Convert radians to degrees

        # output orientation seems to be in [-180, 180]

        # Divide orientations into 60 bins (6 degrees per bin)
        num_bins = 60
        bin_width = 360 / num_bins
        bins = np.arange(-180, 180, bin_width)

        # Initialize the histogram
        histogram = np.zeros(num_bins)

        # Compute the squared gradient magnitude
        magnitude_squared = magnitude ** 2

        # Bin the magnitudes based on their orientations
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                bin_idx = int(orientation[i, j] // bin_width)
                histogram[bin_idx] += magnitude_squared[i, j]

        # Plot the histogram of oriented gradients
        plt.bar(bins, histogram, width=bin_width)
        plt.xlabel('Orientation (degrees)')
        plt.ylabel('Squared Gradient Magnitude')
        plt.title('Histogram of Oriented Gradients (HOG) using Sobel Filters')
        plt.show()

        # Create polar plot
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

        # Plot the bars
        bars = ax.bar(bins, histogram, width=2*np.radians(bin_width), bottom=0.0, align='edge')

        # Set labels
        ax.set_theta_zero_location("N")  # Set 0 degrees to be at the top
        ax.set_theta_direction(-1)       # Set clockwise direction for angles
        ax.set_xticks(np.linspace(0, np.pi, 6))  # Set major ticks at 0, 30, 60, 90, 120, 150 degrees
        ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°'])

        plt.title('Polar Histogram of gradient magnitude using Sobel Filters')
        plt.show()

        # threshold = 1.843 # old threshold = 1.568


        # fig = plt.figure(figsize=(8, 10))
        # ax1 = fig.add_subplot(2, 2, 1)
        # ax2 = fig.add_subplot(2, 2, 2)
        # ax3 = fig.add_subplot(2, 2, 3, projection='polar')
        # ax4 = fig.add_subplot(2, 2, 4)

        # ax1.axis('off')
        # ax1.imshow(image)  # 'image' should be defined previously
        # ax1.set_title('Original input image')

        # ax2.axis('off')
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1.7))
        # ax2.imshow(hog_image_rescaled)
        # ax2.set_title('Histogram of Oriented Gradients')

        # # Pass the pre-defined axis object 'ax3' which was set to be a polar plot
        # # traslate by 90 degrees as we nned the angle perpendicular to the steepest gradient

        # # intuitively, this should be = 90. In practice, with HOG it's more = 0
        # ROTATE_FOR_GRADIENT = 0
        # orientations_polar_deg = np.mod(orientations_360_deg + ROTATE_FOR_GRADIENT , 360)

        # estim_ymax = gradient_hist_180.max()

        # # max_y_tick = ceil(estim_ymax/0.5)*0.5 # round up to nearest half-integer
        # bars = plot_polar_histogram(ax3, gradient_hist_360, orientations_polar_deg)

        # # Do not set the ax3.axis('off') !
        # ax3.set_yticks(np.linspace(0, estim_ymax, num=4))  # Adjust the number of ticks here
        # ax3.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
        # ax3.yaxis.label.set_size(6)
        # ax3.set_ylim(0, 1.1*estim_ymax)
        # ax3.set_title('Directionality plot')
        # ax3.set_theta_zero_location('N')  # North,  West, or East?
        # ax3.set_theta_direction(-1)  # -1 for Clockwise

        # included_cells = cells_to_keep.reshape(fd.shape[0], fd.shape[1])
        # ax4.axis('off')
        # im = ax4.imshow(included_cells)
        # ax4.set_title('Visualisation of included blocks')
        # heatmap = ax4.imshow(strengths, cmap='viridis', interpolation='nearest')
        # cbar = fig.colorbar(heatmap, ax=ax4, shrink=0.6, pad=0.05, fraction=0.07)
        # cbar.ax.tick_params(labelsize=8)
        # ax4.set_title('Signal heatmap with mask in grey')
        # # Overlay cells below threshold in red
        # # cmap_red = matplotlib.colors.ListedColormap(['red'])
        # rgb_color = (0.7, 0.7, 0.7) # light gray
        # cmap_gray = matplotlib.colors.ListedColormap([rgb_color])
        # masked_im = ax4.imshow(np.ma.masked_where(cells_to_keep, strengths), cmap=cmap_gray,
        #                     interpolation=None, alpha=1)

        # # Add black borders to the gray cells
        # rows, cols = strengths.shape
        # for i in range(rows):
        #     for j in range(cols):
        #         if not cells_to_keep[i, j]:
        #             rect = Rectangle((j-0.5, i-0.5), 1, 1, linewidth=0.2,
        #                              edgecolor='black', facecolor='none')
        #             ax4.add_patch(rect)
        #             if i % 3 == 0 and j % 3 == 0 and n_pixels_cell[0] > 30:
        #                 ax4.text(j, i, f'({i},{j})', ha='center', va='center',
        #                          fontsize=2, color='black')

        # plt.tight_layout()
        # plt.suptitle(f"Image n*{img_idx}, block_norm: {block_norm}")
        # plt.savefig(os.path.join(experiments_folder, "HOG_plot_analysis",
        #                             image_filename +\
        #                             f"_{block_norm}_{n_pixels_cell[0]}.png"), dpi=450)

        # #if not is_last_pict:
        # plt.close() #plt.close(fig)

        t2 = time.time()

        elapsed_seconds = int(t2 - t1)
        mins = elapsed_seconds // 60
        secs = elapsed_seconds % 60
        elapsed_time_formatted = f"{mins}:{secs:02d}"

        image_stats = pd.DataFrame({"group" : group,
                                    "image size" : str(image.shape),
                                    # "pixels_per_cell": str(n_pixels_cell),
                                    "elapsed_time (min:sec)": elapsed_time_formatted,
                                    "avg. direction" : round(mean_angle_deg, 3),
                                    "std. deviation" : round(std_dev_deg, 3),
                                    "abs. deviation" : round(abs_dev_deg, 3),
                                    "signal_threshold": threshold,
                                    "signal_stats": [np.percentile(strengths.flatten(),
                                                                10*np.arange(11))]
                                    },
                            index = [image_file.split(".tif", 1)[0]])

        if len(group_folders) == 1: # no grouping inthe file folder
            image_stats = image_stats.drop(group_folders[0], axis=1, errors='ignore')

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

            c_pattern2 = r"\d{1}[-,.]?\d?'\d{1}[-,.]?\d?" # 10'5' or 1,5'0,5'
            match_pattern2 = re.search(c_pattern2, image_filename)
            if match_pattern2:
                condition = match_pattern2.group(0).replace("'", "m")

            c_pattern_ctr = r"CTR"
            match_pattern_ctr = re.search(c_pattern_ctr, image_filename)
            if match_pattern_ctr:
                condition = match_pattern_ctr.group(0)


            image_stats.insert(2, "condition", condition)

            '''
            let's see:
            '''
            '''
            wrong matches:
            - 140h_14h -> becomes 40h instead of 14h


            '''

        df_statistics = pd.concat([df_statistics, image_stats], axis=0) # [brackets] necessary if passing dictionary instead


    # plt.ioff()  # Disable interactive mode to keep the last plot open
    # plt.show()  # Show the last plot
    plt.close('all')

# stacked_arrs = np.stack(df_statistics['signal_stats'].values)
# print("Decile analysis:", np.mean(stacked_arrs, axis=0))

filename = "Sobel_stats_"+ str(block_norm) + "_" + str(hog_descriptor.pixels_per_cell[0]) + "pixels"
if len(image_file_list) < 0.7*len(os.listdir(image_folder)):
    filename = filename + "_draft"

df_statistics.to_csv(os.path.join(experiments_folder, filename+'.csv'))
