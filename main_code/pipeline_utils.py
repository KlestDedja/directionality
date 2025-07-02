# from typing import Dict, Tuple
import numpy as np
from skimage.feature import hog


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


def plot_polar_histogram(
    ax,
    global_histogram,
    orientations_deg,
    plot_mean=True,
    n_max_directions=1,
    min_direction_gap=20.0,
):
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

    # # highlight all max bins in red
    # max_idxs = np.where(global_histogram == global_histogram.max())[0]
    # for i in max_idxs:
    #     bars[i].set_color("red")

    global_hist_180 = global_histogram[: n_bins // 2]
    orientations_deg_180 = orientations_deg[: n_bins // 2]

    main_peaks_180 = compute_peaks_from_histogram(
        global_hist_180,
        orientations_deg_180,
        n_max_directions,  # histogram is in [0, 180) and contains two copies per direction
        min_direction_gap,
    )

    main_peaks = main_peaks_180 + [
        main_peaks_180[i] + 180 for i in range(len(main_peaks_180))
    ]

    # given the angle of the peaks, map them to indeces:
    peak_idxs = []
    for peak in main_peaks:
        # find the closest index to the peak angle
        idx = np.argmin(np.abs(orientations_deg - peak))
        peak_idxs.append(idx)

    # now that all peaks have been mapped to indeces, color them
    for i in peak_idxs:
        bars[i].set_color("red")

    if plot_mean:
        # compute mean direction (in deg -> rad) and plot as green line
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

    # Correction at 0 degrees (always needed)
    process_interpolation(0)

    if corr90:
        process_interpolation(90)

    if corr45:
        process_interpolation(45)
        process_interpolation(135)

    return histog_dict


def compute_polar_direction_diff(
    angle1: float | np.ndarray, angle2: float | np.ndarray, in_degs: bool = True
) -> np.ndarray | float:
    """
    This function computes the angular difference between two angles in polar coordinates,
    where directions 180 degrees apart are to be considered identical.
    This implies that the resulting difference is always less than or equal to 90 degrees.

    This function assumes the angles are in the [-90, 360] range (in degrees by default).
    """
    if in_degs is False:
        angle1 = np.rad2deg(angle1)
        angle2 = np.rad2deg(angle2)

    angle_diff = np.abs(angle1 - angle2)

    angle_diff_real = np.min(
        np.stack(
            [np.abs(angle_diff), np.abs(180 - angle_diff), np.abs(360 - angle_diff)]
        ),
        axis=0,
    )
    assert np.all(angle_diff_real <= 90), "Not all values are <= 90, smth went wrong"

    return angle_diff_real


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
    """Compute standard deviation and absolute deviation w.r.t. a (list of) reference angle(s)."""
    # Calculate angle residuals (in degrees). It should work for any angle in 0-360,
    #  but we only need this to work for angles in 0-180
    angle_diffs_real = compute_polar_direction_diff(
        orientations_deg, reference_angle_deg
    )

    # Calculate the standard deviation (sqrt of the average squared residuals)
    std_dev_deg = np.sqrt(
        np.sum(global_hist_vals * angle_diffs_real**2) / np.sum(global_hist_vals)
    )

    # Calculate average of absolute residuals
    abs_dev_deg = np.sum(
        np.abs(global_hist_vals * angle_diffs_real) / np.sum(global_hist_vals)
    )

    return std_dev_deg, abs_dev_deg


def compute_peaks_from_histogram(
    global_histogram: np.ndarray,
    orientations_deg: np.ndarray,
    n_max_directions: int = 1,
    min_direction_gap: float = 20.0,
    enforce_local_maxima: bool = True,
) -> list[float]:

    hist_copy = global_histogram.copy()
    peaks_found = []

    if enforce_local_maxima:
        # suppress non-local maxima. Precompute mask:
        left_neighbors = np.roll(global_histogram, 1)
        right_neighbors = np.roll(global_histogram, -1)
        is_local_max = (global_histogram >= left_neighbors) & (
            global_histogram >= right_neighbors
        )
        # Only local maxima remain
        hist_copy[~is_local_max] = 0

    for _ in range(n_max_directions):
        idx = np.argmax(hist_copy)
        peak_angle = orientations_deg[idx]
        peaks_found.append(peak_angle)

        # suppress neighbors to enforce distance of at least min_direction_gap
        # between peaks. Also, suppress the peak itself so it is not selected again
        angle_diff = compute_polar_direction_diff(orientations_deg, peak_angle)
        suppression_mask = angle_diff < min_direction_gap
        hist_copy[suppression_mask] = 0

        if np.all(hist_copy == 0):
            break

    return peaks_found


def compute_distribution_directions(
    global_histogram: dict | np.ndarray,
    orientations_deg: np.ndarray | None = None,
    n_max_directions: int = 1,
    min_direction_gap: float = 20.0,
    enforce_local_maxima: bool = True,
) -> tuple[dict, dict]:
    """
    Find (possibly multiple) dominant directions, spaced by at least min_direction_gap,
    and compute a global deviation relative to the closest dominant direction.

    Parameters
    ----------
    global_histogram : dict or np.ndarray
        Histogram of directions.
    orientations_deg : np.ndarray, optional
        Angles corresponding to the histogram values.
    n_max_directions : int
        Number of dominant peaks to extract.
    min_direction_gap : float
        Minimum distance (degrees) to keep between peaks.

    Returns
    -------
    mean_stats : list of dict
        List of peak statistics with mean angle, std_dev, abs_dev w.r.t. to the mean.
    mode_stats : list of dict
        List of peak statistics with most frequent angle, std_dev, abs_dev w.r.t. to the mode.


    deviation_stats : dict
        Overall deviation relative to the closest dominant direction.
    """
    if isinstance(global_histogram, dict):
        global_hist_vals = np.array(list(global_histogram.values()))
        orientations_deg = np.array(list(global_histogram.keys()))
    else:
        global_hist_vals = global_histogram

    if orientations_deg is None:
        raise ValueError("orientations_deg must be provided if histogram is ndarray.")

    if global_hist_vals.ndim > 2:
        raise ValueError(
            f"Input should be 1D or 2D, got shape {global_hist_vals.shape} instead."
        )

    if n_max_directions < 1:
        raise ValueError(
            f"n_max_directions must be at least 1, got {n_max_directions} instead."
        )

    if min_direction_gap < 0:
        raise ValueError(
            f"min_direction_gap must be non-negative, got {min_direction_gap} instead."
        )

    mean_stats = []
    mode_stats = []

    peaks_found = compute_peaks_from_histogram(
        global_hist_vals,
        orientations_deg,
        n_max_directions,
        min_direction_gap,
        enforce_local_maxima,
    )

    # mean stats is computed over a single direction
    mean_angle_deg = compute_vector_mean(global_hist_vals, orientations_deg)
    std_dev_mean, abs_dev_mean = compute_deviations(
        global_hist_vals, orientations_deg, mean_angle_deg
    )
    mean_stats = {
        "angle": mean_angle_deg,
        "std_dev": std_dev_mean,
        "abs_dev": abs_dev_mean,
    }

    # now compute deviations w.r.t. the closest peak
    deltas_per_peak = []
    for peak in peaks_found:
        deltas = compute_polar_direction_diff(orientations_deg, peak, in_degs=True)
        deltas_per_peak.append(deltas)

    # stack arrays and take the minimum value across the peaks (axis=0)
    stacked_deltas = np.stack(deltas_per_peak, axis=0)
    closest_delta = np.min(stacked_deltas, axis=0)

    std_dev_mode = np.sqrt(
        np.sum(global_hist_vals * closest_delta**2) / np.sum(global_hist_vals)
    )
    abs_dev_mode = np.sum(global_hist_vals * closest_delta) / np.sum(global_hist_vals)

    angle_str = "angle" if n_max_directions == 1 else "angles"

    mode_stats = {
        angle_str: peaks_found,
        "std_dev": std_dev_mode,
        "abs_dev": abs_dev_mode,
    }

    return mean_stats, mode_stats


def old_compute_distribution_directions(
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
