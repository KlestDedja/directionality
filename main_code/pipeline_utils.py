# from typing import Dict, Tuple
import os
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.filters import scharr_h, scharr_v


def load_and_prepare_image(path, name, to_grayscale=False, channel=None):
    """Load an image and proivde option to convert it to grayscale."""
    image = io.imread(os.path.join(path, name))

    # Check if the image has four channels (RGBA)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]  # ignore the alpha channel

    # Convert RGB to grayscale
    if image.ndim == 3 and to_grayscale == True:
        image = rgb2gray(image)
    if image.ndim == 3 and to_grayscale == False and channel is not None:
        image = image[:, :, channel]
    return image


class HOGDescriptor:
    def __init__(
        self,
        orientations=9,
        pixels_per_window=(8, 8),
        windows_per_block=(3, 3),
        channel_axis: int | None | str = -1,
    ):
        self.orientations = orientations
        self.pixels_per_window = pixels_per_window
        self.windows_per_block = windows_per_block
        self.channel_axis = channel_axis
        self.fd = None
        self.hog_image = None

    def get_orientations(self):
        return self.orientations

    def compute_hog(
        self,
        image,
        visualize=True,
        block_norm="None",
        feature_vector=False,
    ):
        """Compute Histogram of Oriented Gradients (HOG) RGB or grayscale image."""

        # Assumes image channel has been selected already, and that
        # the resulting image is therefore a 2D array (grayscale-style)
        # for this reason we set channel_axis=None

        ## BEWARE: what we call "window" is called "cell" in skimage library
        # TODO: fork skimage for block_norm = None support
        if visualize:
            self.fd, self.hog_image = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_window,
                cells_per_block=self.windows_per_block,
                visualize=True,
                block_norm=block_norm,
                feature_vector=feature_vector,
                channel_axis=None,
            )
        else:
            self.fd = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_window,
                cells_per_block=self.windows_per_block,
                visualize=False,
                block_norm=block_norm,
                feature_vector=feature_vector,
                channel_axis=None,
            )
            self.hog_image = None

        return self.fd, self.hog_image


def plot_polar_histogram(ax, global_histogram, orientations_deg, plot_mean=True):
    # bin centers in radians
    orientations_rad = np.deg2rad(orientations_deg)
    n_bins = len(orientations_deg)

    # draw bars
    bars = ax.bar(
        orientations_rad,
        global_histogram,
        width=(2 * np.pi / n_bins),
        bottom=0,
    )

    # highlight all max bins in red
    max_idxs = np.where(global_histogram == global_histogram.max())[0]
    for i in max_idxs:
        bars[i].set_color("red")

    if plot_mean:
        # compute mean direction (in deg → rad) and plot as green line
        mean_deg = compute_vector_mean(global_histogram, orientations_deg)
        mean_rad = np.deg2rad(mean_deg)
        r_max = ax.get_ylim()[1]  # current radial max

        for angle in (mean_rad, mean_rad + np.pi):
            ax.plot(
                [angle, angle],  # theta
                [0, r_max],  # radius
                linestyle="-",
                linewidth=3,
                color="green",
            )

    return bars


def correct_round_angles(histog_dict, corr90=True, corr45=False):

    keys_sorted = np.array(sorted(histog_dict.keys()))
    n = len(keys_sorted)

    # For a single minimal index: interpolate 5 indices (anchors set two indeces apart from minimal index)
    def linear_interpolate_neighbors(histog_dict, keys_sorted, idx):
        im2 = (idx - 2) % n
        im1 = (idx - 1) % n
        ip1 = (idx + 1) % n
        ip2 = (idx + 2) % n

        # Use the values at positions im2 and ip2 as anchors
        a = histog_dict[keys_sorted[im2]]
        b = histog_dict[keys_sorted[ip2]]
        # Interpolate at positions -1, 0, +1 (using positions 1/4, 2/4, 3/4 between anchors)
        new_im1 = a + (1 / 4) * (b - a)
        new_i = a + (2 / 4) * (b - a)
        new_ip1 = a + (3 / 4) * (b - a)

        histog_dict[keys_sorted[im1]] = new_im1
        histog_dict[keys_sorted[idx]] = new_i
        histog_dict[keys_sorted[ip1]] = new_ip1

    # For exactly two contiguous minimal indices: interpolate using 6 indices.
    # Let L be the left minimal and R the right minimal (with R == (L+1) mod n).
    # We define:
    #   left_anchor1 = index at L-2, left_anchor2 = L-1,
    #   right_anchor1 = index at R+1, right_anchor2 = R+2.
    # Then we linearly interpolate the inner four positions (positions 1,2,3,4 in a 0–5 scale)
    # between the anchors at positions 0 and 5.
    def linear_interpolate_neighbors_double(histog_dict, keys_sorted, L, R):

        assert R == (L + 1) % n

        left_anchor1 = (L - 2) % n
        left_anchor2 = (L - 1) % n
        right_anchor1 = (R + 1) % n  # note: since R = L+1 (mod n)
        right_anchor2 = (R + 2) % n

        v0 = histog_dict[keys_sorted[left_anchor1]]
        v5 = histog_dict[keys_sorted[right_anchor2]]

        # Positions in a 0-to-5 scale; we update positions 1, 2, 3, 4.
        new_val_1 = v0 + (1 / 5) * (v5 - v0)  # for left_anchor2 (position 1)
        new_val_2 = v0 + (2 / 5) * (v5 - v0)  # for left minimal (position 2)
        new_val_3 = v0 + (3 / 5) * (v5 - v0)  # for right minimal (position 3)
        new_val_4 = v0 + (4 / 5) * (v5 - v0)  # for right_anchor1 (position 4)

        histog_dict[keys_sorted[left_anchor2]] = new_val_1
        histog_dict[keys_sorted[L]] = new_val_2
        histog_dict[keys_sorted[R]] = new_val_3
        histog_dict[keys_sorted[right_anchor1]] = new_val_4

    # Process interpolation for a given target angle.
    def process_interpolation(target_angle):
        d_target = np.abs(keys_sorted - target_angle)
        indices = np.where(d_target == d_target.min())[0]
        if len(indices) > 2:
            raise ValueError(
                f"More than two minimal indices found for angle {target_angle}: {indices}."
            )
        elif len(indices) == 1:
            linear_interpolate_neighbors(histog_dict, keys_sorted, indices[0])
        elif len(indices) == 2:
            i1, i2 = sorted(indices)
            if not ((i2 - i1 == 1) or (i1 == 0 and i2 == n - 1)):
                raise ValueError(
                    f"Tied minimal indices for angle {target_angle} are not contiguous: {indices}."
                )
            linear_interpolate_neighbors_double(histog_dict, keys_sorted, i1, i2)

    # Correction at 0 degrees (always needed) used to be here, separately:
    # process_interpolation(0)

    if corr90 is True:
        process_interpolation(90)
        process_interpolation(0)

    if corr45 is True:
        process_interpolation(45)
        process_interpolation(135)

    return histog_dict


def cell_signal_strengths(fd_data, norm_ord=1):

    strengths = np.zeros_like(fd_data[:, :, 0])  # ignore last axis (n_orientations)

    for i in range(fd_data.shape[0]):
        for j in range(fd_data.shape[1]):
            strengths[i, j] = np.linalg.norm(fd_data[i, j], ord=norm_ord)

    return strengths


def compute_vector_mean(global_hist_vals, orientations_deg):
    """Compute the mean direction using vector summation."""
    bin_angles_rad = np.deg2rad(orientations_deg)
    x = np.sum(global_hist_vals * np.cos(bin_angles_rad))
    y = np.sum(global_hist_vals * np.sin(bin_angles_rad))

    # Calculate the resultant vector's angle (avg direction) in range (-90, 90)
    mean_angle_rad = np.arctan2(y, x)  # output in [-pi, pi]
    mean_angle_deg = np.rad2deg(mean_angle_rad)  # now output in [-90, 90]

    # Adjust the mean angle to be within the range (0, 180)
    if mean_angle_deg < 0:
        mean_angle_deg += 180

    return mean_angle_deg


def compute_deviations(global_hist_vals, orientations_deg, reference_angle_deg):
    """Compute standard deviation and absolute deviation w.r.t. a reference angle."""
    # Calculate angle residuals (in degs). Remember we are only using the angles 0-180 (right side of the polar plot)
    angle_diffs = np.abs(orientations_deg - reference_angle_deg)

    # angle_diffs = np.abs(np.minimum(angle_diffs, 90 - angle_diffs))
    # angle_diffs_real = np.minimum(
    #     np.abs(angle_diffs), np.abs(180 - angle_diffs), np.abs(360 - angle_diffs)
    # )
    angle_diffs_real = np.min(
        np.stack(
            [np.abs(angle_diffs), np.abs(180 - angle_diffs), np.abs(360 - angle_diffs)]
        ),
        axis=0,
    )
    assert np.all(angle_diffs_real <= 90), "Not all values are less than or equal to 90"

    # Calculate the standard deviation (sqrt of the average squared residuals)
    std_dev_deg = np.sqrt(
        np.sum(global_hist_vals * angle_diffs_real**2) / np.sum(global_hist_vals)
    )

    # Calculate average of absolute residuals
    abs_dev_deg = np.sum(
        np.abs(global_hist_vals * angle_diffs_real) / np.sum(global_hist_vals)
    )

    return std_dev_deg, abs_dev_deg


def compute_distribution_direction(
    global_histogram: dict | np.ndarray,
    orientations_deg: (
        np.ndarray | None
    ) = None,  # consider @overload for typing (2 cases)
) -> tuple[dict, dict]:
    """Compute mean, mode, and deviations (w.r.t. mean and mode)."""

    # INPUTS typing:
    #  - either a dict with angles as keys and histogram values as values (2nd input = None)
    #  - or a np.ndarray with histogram values, and orientations_deg as a separate array

    # Handle retro-compatibility and convert dict input to numpy arrays
    if isinstance(global_histogram, dict):
        global_hist_vals = np.array(list(global_histogram.values()))
        orientations_deg = np.array(list(global_histogram.keys()))
    else:
        global_hist_vals = global_histogram  # Assume already in numpy array form

    if global_hist_vals.ndim > 2:
        raise ValueError(
            f"Input should be 2D, got shape {global_hist_vals.shape} instead."
        )

    if orientations_deg is None:
        raise ValueError(
            "orientations_deg could not be computed and " "therefore must be provided."
        )

    # Compute mean direction and deviations
    mean_angle_deg = compute_vector_mean(global_hist_vals, orientations_deg)
    std_dev_mean, abs_dev_mean = compute_deviations(
        global_hist_vals, orientations_deg, mean_angle_deg
    )

    # Grouped dictionary for mean-related metrics
    mean_stats = {
        "angle": mean_angle_deg,
        "std_dev": std_dev_mean,
        "abs_dev": abs_dev_mean,
    }

    # Compute mode direction and deviations
    main_dir_idx = np.argmax(global_hist_vals)
    mode_angle_deg = orientations_deg[main_dir_idx]
    std_dev_mode, abs_dev_mode = compute_deviations(
        global_hist_vals, orientations_deg, mode_angle_deg
    )

    # Grouped dictionary for mode-related metrics
    mode_stats = {
        "angle": mode_angle_deg,
        "std_dev": std_dev_mode,
        "abs_dev": abs_dev_mode,
    }

    return mean_stats, mode_stats
