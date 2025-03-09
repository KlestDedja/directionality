import os
from typing import Dict, Tuple
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage import color
from skimage.feature import hog


def load_and_prepare_image(
    path, name, to_grayscale: bool = False, channel: int | None = None
):
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
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3),
        channel_axis=-1,
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.channel_axis = channel_axis
        self.fd = None
        self.hog_image = None

    def get_orientations(self):
        return self.orientations

    def compute_hog(
        self,
        image,
        visualize=True,
        block_norm=None,
        feature_vector=False,
        channel_axis=-1,
    ):
        """Compute Histogram of Oriented Gradients (HOG) RGB or grayscale image."""
        if channel_axis is None or str(channel_axis).lower() == "grayscale":
            image = color.rgb2gray(image)
            channel_axis = None

        if (
            image.ndim == 2
        ):  # make sure we do not search for a channel axis that does not exist
            channel_axis = None

        if visualize:
            self.fd, self.hog_image = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=visualize,
                block_norm=block_norm,
                feature_vector=feature_vector,
                channel_axis=channel_axis,
            )
        else:
            self.fd = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=False,
                block_norm=block_norm,
                feature_vector=feature_vector,
                channel_axis=channel_axis,
            )
            self.hog_image = None

        # return fd, hog_image
        return self.fd, self.hog_image


def reshape_hog_array(array, len_axis1):
    # Calculate the total number of elements in the array
    N = array.size
    # Ensure that m is a divisor of N
    if N % len_axis1 != 0:
        raise ValueError(
            f"The given len_axis1 ({len_axis1}) is not a divisor of the total number of elements ({N})."
        )

    new_shape = (N // len_axis1, len_axis1)
    # Reshape the array
    reshaped_array = array.reshape(new_shape)
    return reshaped_array


def plot_polar_histogram(ax, global_histogram, orientations_deg):

    orientations_rad = np.deg2rad(orientations_deg)
    n_bins = len(orientations_deg)

    # Plot the histogram on the given axis
    bars = ax.bar(
        orientations_rad, global_histogram, width=(2 * np.pi / n_bins), bottom=0
    )

    max_idx = np.argwhere(global_histogram == global_histogram.max()).flatten()

    for idx in max_idx:  # it's always at least 2 maxima:
        bars[idx].set_color("red")

    return bars


def average_directions_over_cells(
    fd, orientations, N=None, M=None, fd_data_as_array=True
):
    if N is None:
        N = fd.shape[0]
    if M is None:
        M = fd.shape[1]

    fd_data = {"grid_size": (N, M)}
    key = 0

    if fd.ndim > 3:
        fd = fd.squeeze()

    assert fd.ndim == 3
    assert len(orientations) == fd.shape[-1]

    for i in range(N):
        for j in range(M):
            plot_fd = fd[
                i, j, :
            ].mean()  # Average over sub-cells (cells_per_block_x, cells_per_block_y)
            key += 1
            fd_data[key] = plot_fd

    if fd_data_as_array:
        vectors = []
        for key, value in fd_data.items():
            if key == "grid_size":
                continue  # Skip the 'grid_size' key
            # Append the vector to the list
            vectors.append(value)
        # Convert the list of vectors to a 2D numpy array
        fd_data = np.array(vectors)
    return fd_data


def correction_of_round_angles(histog_dict, corr90=True, corr45=False):

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

    # Correction at 0 degrees
    process_interpolation(0)

    if corr90:
        process_interpolation(90)

    if corr45:
        process_interpolation(45)
        process_interpolation(135)

    return histog_dict


def correction_of_round_angles_old(histog_dict, corr90=True, corr45=False):
    # Sort the keys (angles) for proper indexing (in case they are not sorted)
    keys_sorted = np.array(sorted(histog_dict.keys()))

    # correction at 0 degrees
    index_0 = np.argmin(np.abs(keys_sorted))
    histog_dict[keys_sorted[index_0]] = 0.5 * (
        histog_dict[keys_sorted[index_0 - 1]] + histog_dict[keys_sorted[index_0 + 1]]
    )

    if corr90:  # correction at 90 degrees (smoothing)
        index_90 = np.argmin(np.abs(keys_sorted - 90))
        histog_dict[keys_sorted[index_90]] = 0.5 * (
            histog_dict[keys_sorted[index_90 - 1]]
            + histog_dict[keys_sorted[index_90 + 1]]
        )

    if corr45:  # smoothing at 45 and 135 degrees
        index_45 = np.argmin(np.abs(keys_sorted - 45))
        index_135 = np.argmin(np.abs(keys_sorted - 135))

        histog_dict[keys_sorted[index_45]] = 0.5 * (
            histog_dict[keys_sorted[index_45 - 1]]
            + histog_dict[keys_sorted[index_45 + 1]]
        )
        histog_dict[keys_sorted[index_135]] = 0.5 * (
            histog_dict[keys_sorted[index_135 - 1]]
            + histog_dict[keys_sorted[index_135 + 1]]
        )

    return histog_dict


def correction_of_round_angles_older(
    orientations_180_deg, histog_data, corr90=True, corr45=False
):

    # correction at 0*: normally speaking the index should be always = 0 but
    # we compute the argmin here just in case there were previous manipulations
    index_0 = np.argmin(np.abs(orientations_180_deg))
    histog_data[index_0] = 0.5 * (histog_data[index_0 + 1] + histog_data[index_0 - 1])

    if corr90:  # make correction at 90 degrees:
        index_90 = np.argmin(np.abs(orientations_180_deg - 90))
        # smooth with neighbours
        histog_data[index_90] = 0.5 * (
            histog_data[index_90 + 1] + histog_data[index_90 - 1]
        )
    # # correction at 45 and 135 degrees:
    if corr45:
        index_45 = np.argmin(np.abs(orientations_180_deg - 45))
        index_135 = np.argmin(np.abs(orientations_180_deg - 135))

        histog_data[index_45] = 0.5 * (
            histog_data[index_45 + 1] + histog_data[index_45 - 1]
        )
        histog_data[index_135] = 0.5 * (
            histog_data[index_135 + 1] + histog_data[index_135 - 1]
        )

    return histog_data


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
    # Calculate angle residuals (in degs)
    angle_diffs = np.abs(orientations_deg - reference_angle_deg)
    angle_diffs = np.abs(np.minimum(angle_diffs, 90 - angle_diffs))

    # Calculate the standard deviation (sqrt of the average squared residuals)
    std_dev_deg = np.sqrt(
        np.sum(global_hist_vals * angle_diffs**2) / np.sum(global_hist_vals)
    )

    # Calculate average of absolute residuals
    abs_dev_deg = np.sum(
        np.abs(global_hist_vals * angle_diffs) / np.sum(global_hist_vals)
    )

    return std_dev_deg, abs_dev_deg


def compute_distribution_direction(
    global_histogram: Dict | np.ndarray,
    orientations_deg: (
        np.ndarray | None
    ) = None,  # consider @overload for typing (2 cases)
) -> Tuple[Dict, Dict]:
    """Compute mean, mode, and deviations (w.r.t. mean and mode)."""

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


# def compute_distribution_direction(
#     global_histogram, orientations_deg
# ):  # can be improved now that we input dict!

#     if isinstance(global_histogram, dict):
#         global_hist_vals = np.array(
#             list(global_histogram.values())
#         )  # retrocompatibility
#         orientations_deg = np.array(list(global_histogram.keys()))  # retrocompatibility

#     if global_hist_vals.ndim > 2:
#         raise ValueError(
#             f"Input should be 2D, got shape {global_hist_vals.shape} instead."
#         )

#     # Convert histogram values to vectors
#     bin_angles_rad = np.deg2rad(orientations_deg)

#     x = np.sum(global_hist_vals * np.cos(bin_angles_rad))
#     y = np.sum(global_hist_vals * np.sin(bin_angles_rad))

#     # Calculate the resultant vector's angle (avg direction) in range (-90, 90)
#     mean_angle_rad = np.arctan2(y, x)  # output in [-pi, pi]
#     mean_angle_deg = np.rad2deg(mean_angle_rad)  # now output in [-90, 90]
#     # Compuite `main` direction in the sense of most frequent one (mode)
#     main_dir_idx = np.argmax(global_hist_vals)
#     mode_angle_deg = orientations_deg[main_dir_idx]

#     # Adjust the mean angle to be within the range (0, 180)
#     if mean_angle_deg < 0:
#         mean_angle_deg += 180
#         mean_angle_rad += np.pi

#     # Calculate angle residuals (in degs)
#     # take minimum abolute difference between diff angle and its complementary angle
#     angle_diffs = np.abs(orientations_deg - mean_angle_deg)
#     angle_diffs = np.abs(np.minimum(angle_diffs, 90 - angle_diffs))

#     # Calculate the standard deviation (sqrt of the average squared residuals)
#     # weighted by histogram height, and normalised.
#     std_dev_deg = np.sqrt(
#         np.sum(global_hist_vals * angle_diffs**2) / np.sum(global_hist_vals)
#     )

#     # Calculate average of absolute residuals)
#     # weighted by histogram height, and normalised.
#     abs_dev_deg = np.sum(
#         np.abs(global_hist_vals * angle_diffs) / np.sum(global_hist_vals)
#     )

#     return mean_angle_deg, mode_angle_deg, std_dev_deg, abs_dev_deg
