
import os
import time
import numpy as np
import pandas as pd
import math
from math import ceil
import re

from skimage import exposure, filters#, feature
# from skimage import color
# from skimage.feature import hog
import matplotlib
# matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
# matplotlib.use('TkAgg')  # Use interactive plotting?

from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

from pipeline_utils import load_and_prepare_image
from pipeline_utils import HOGDescriptor, plot_polar_histogram
from pipeline_utils import average_directions_over_cells, compute_average_direction
from pipeline_utils import correction_of_round_angles, cell_signal_strengths
from utils_other import calculate_and_print_percentiles

root_folder = os.getcwd()
print('current working directory:', root_folder)

group_folders = ['CTR', 'MS']
df_statistics = pd.DataFrame()

# plt.ion()

block_norms = [None, "L1", "L2"]

for block_norm in block_norms:

    for group in group_folders:
        image_folder = os.path.join(root_folder, 'Images_confocal', group)
        image_folder = os.path.join(root_folder, 'images-data', group)

        experiments_folder = os.path.dirname(image_folder)
        image_file_list = os.listdir(image_folder)[:3]

        print("Analysing images in: ", image_folder)
        # print("List of images: ", image_file_list)
        print('Block norm:', block_norm)

        for img_idx, image_file in enumerate(image_file_list):

            is_last_pict = (image_file == image_file_list[-1])

            image_filename = image_file.replace(" ", "_")
            image_filename = re.sub(r'^\d*_', '', image_filename) #cut of initial digits (date and time)
            image_filename = image_filename.rsplit(".", 1)[0] + ".png"
            t1 = time.time()

            # load image and select Green channel if the image is in RGB(A) format
            image = load_and_prepare_image(image_folder, image_file, channel=1)

            # output image is 2d, therefore need to set the channel_axis=None in HOGDescriptor:
            hog_descriptor = HOGDescriptor(orientations=30, pixels_per_cell=(32, 32), cells_per_block=(1, 1),
                                        channel_axis=-1)
            fd_raw, hog_image = hog_descriptor.compute_hog(image, block_norm=block_norm, feature_vector=False)
            # Note: block_norm ``L2-Hys`` is not any better with the artefact

            # we imitate skimage hog source code ( based on 180 degrees range) and duplicate the range
            orient_arr = np.arange(2*hog_descriptor.orientations)
            orientations_360_deg = 360 * (orient_arr + 0.5) / (2*hog_descriptor.orientations) # from skimage hog source code
            orientations_180_deg = orientations_360_deg[:len(orientations_360_deg)//2]
            ## fd is of shape (cells_H_axis, cells_W_axis, Blocks_per_cell_H, Blocks_per_cell_W, n_orientations)

            fd  = np.squeeze(fd_raw) # fd has now shape (N, M, n_orientations)
            strengths = cell_signal_strengths(fd, norm_ord=1) # shape (N, M), default norm by L1 norm.
            calculate_and_print_percentiles(strengths)

            '''werid stuff happens whne nomr is not None. Probably for a good reason :)'''
            fd_norm = fd/(1e-7 + strengths[:, :, np.newaxis]) # normalize everything to  |v| = 1
            # threshold = np.percentile(strengths.flatten(), 55) # for now, we set this as threshold, but can improve
            threshold = 1.568
            ''' with cell blocks of size (32, 32)
            Decile analysis of the signal over the first 6 pictures: [
            min -> 0.58064974    10 -> 0.95527117      20 -> 1.09113596    30 -> 1.21609939
            40  -> 1.37751670    50 -> 1.56824218      60 -> 1.84369730    70 -> 2.19784849
            80  -> 2.79174948    90 -> 3.93258352     max -> 12.97092674
            '''


            # ToDo: select a global threshold instead of a image-based one.
            cells_to_keep = strengths > threshold # boolean (N,M) array?
            fd_norm[~cells_to_keep] = np.zeros_like(fd.shape[-1])   # exclude cells below threshold, fill with zeroes
            # fd_norm with shape (N, M, n_orientations), orientation for each cell block

            # NOW LET'S TAKE THE AVERAGE OVER THE BLOCKS (come of them are filtered out as zeros already)
            # Now below we have a signal in [0, 180) for every cell-block.
            gradient_hist_180 = fd_norm[cells_to_keep].mean(axis=0) # shape (n_orientations)
            assert len(gradient_hist_180) == hog_descriptor.orientations

            # correct strong signal at 0 and 90 degrees (happens a lot with constant black backgound)
            gradient_hist_180 = correction_of_round_angles(gradient_hist_180)
            # ToDO consider adding correction at 90*, if necessary

            mean_angle_deg, std_dev_deg, abs_dev_deg = compute_average_direction(gradient_hist_180,
                                                                                orientations_180_deg)

            # We did not account for symmetries: we duplicate the signal vector length for a 360* view in polar plots

            print('image file:', image_file)
            print(f"average direction  : {mean_angle_deg:+.2f} degrees.".replace('+', ' '))
            print(f"standard deviation : {std_dev_deg:+.2f} degrees".replace('+', ' '))
            print(f"mean abs. deviation: {abs_dev_deg:+.2f} degrees".replace('+', ' '))

            gradient_hist_360 = np.tile(gradient_hist_180, 2) # extend direction measurement to [0, 360) interval
            # fd_final_grid = np.tile(fd_norm, (1, 1, 2)) #repeat values along the third (last) axis
            # fd_final_lin = fd_final_grid.reshape(-1, len(global_histogram))

            fig = plt.figure(figsize=(8, 9))
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3, projection='polar')
            ax4 = fig.add_subplot(2, 2, 4)

            ax1.axis('off')
            ax1.imshow(image)  # 'image' should be defined previously
            ax1.set_title('Original input image')

            ax2.axis('off')
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 4))
            ax2.imshow(hog_image_rescaled)
            # ax2.imshow(hog_image)
            ax2.set_title('Histogram of Oriented Gradients')

            # Pass the pre-defined axis object 'ax3' which was set to be a polar plot
            # traslate by 90 degrees as we nned the angle perpendicular to the steepest gradient
            orientations_polar_deg = np.mod(orientations_360_deg+90, 360)

            estim_ymax = gradient_hist_180.max()

            max_y_tick = ceil(estim_ymax/0.2)*0.2
            bars = plot_polar_histogram(ax3, gradient_hist_360, orientations_polar_deg)

            # Do not set the ax3.axis('off') !
            ax3.set_yticks(np.linspace(0, max_y_tick, num=4))  # Adjust the number of ticks here
            ax3.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
            ax3.set_ylim(0, 1.1*estim_ymax)
            ax3.set_title('Directionality plot')
            ax3.set_theta_zero_location('E')  # North,  West, or East?
            # ax3.set_theta_direction(-1)  # -1 for Clockwise

            included_cells = cells_to_keep.reshape(fd.shape[0], fd.shape[1])
            ax4.axis('off')
            im = ax4.imshow(included_cells)
            ax4.set_title('Visualisation of included blocks')
            heatmap = ax4.imshow(strengths, cmap='viridis', interpolation='nearest')
            fig.colorbar(heatmap, ax=ax4)
            ax4.set_title('Signal heatmap with mask in grey')
            # Overlay cells below threshold in red
            # cmap_red = matplotlib.colors.ListedColormap(['red'])
            rgb_color = (0.7, 0.7, 0.7) # light gray
            cmap_gray = matplotlib.colors.ListedColormap([rgb_color])
            masked_im = ax4.imshow(np.ma.masked_where(cells_to_keep, strengths), cmap=cmap_gray,
                                interpolation=None, alpha=1)

            plt.tight_layout()
            plt.suptitle(f"Image n*{img_idx}, block_norm: {block_norm}")
            plt.savefig(os.path.join(experiments_folder, "HOG_plot_analysis",
                                     f"{block_norm}_" + image_filename), dpi=300)

            # plt.draw()
            # Pause to allow the GUI event loop to process events
            # plt.pause(0.1)
            #if not is_last_pict:
            plt.close() #plt.close(fig)

            t2 = time.time()

            elapsed_seconds = int(t2 - t1)
            mins = elapsed_seconds // 60
            secs = elapsed_seconds % 60
            elapsed_time_formatted = f"{mins}:{secs:02d}"

            image_stats = pd.DataFrame({"group" : group,
                                        "image size" : str(image.shape),
                                        "elapsed_time (min:sec)": elapsed_time_formatted,
                                        "avg. direction" : round(mean_angle_deg, 3),
                                        "std. deviation" : round(std_dev_deg, 3),
                                        "abs. deviation" : round(abs_dev_deg, 3),
                                        "signal_threshold": threshold,
                                        "signal_stats": [np.percentile(strengths.flatten(),
                                                                    10*np.arange(11))]
                                        },
                                index = [image_file.split(".tif", 1)[0]])

            df_statistics = pd.concat([df_statistics, image_stats], axis=0) # [brackets] necessary if passing dictionary instead


    # plt.ioff()  # Disable interactive mode to keep the last plot open
    # plt.show()  # Show the last plot

# stacked_arrs = np.stack(df_statistics['signal_stats'].values)
# print("Decile analysis:", np.mean(stacked_arrs, axis=0))

filename = "HOG_stats_"+ str(block_norm)
if len(image_file_list) < 7:
    filename = filename + "_draft"

df_statistics.to_csv(os.path.join(experiments_folder, filename+'.csv'))


'''
TODOs:
- double check the histogram computation (arctan and so on) debug all of it.
    Is it 360* based or 180* ? Still unclear.
- manually smoothing the 0 and 180 degrees. Can it be done any better?
'''