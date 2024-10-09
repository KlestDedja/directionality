
import os
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage import color

from skimage.feature import hog


def load_and_prepare_image(path, name, to_grayscale=False, channel=None):
    """Load an image and proivde option to convert it to grayscale."""
    image = io.imread(os.path.join(path, name))

    # Check if the image has four channels (RGBA)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3] # ignore the alpha channel

    # Convert RGB to grayscale
    if image.ndim == 3 and to_grayscale==True:
        image = rgb2gray(image)
    if image.ndim == 3 and to_grayscale==False and channel is not None:
        image = image[:,:, channel]
    return image


class HOGDescriptor:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), channel_axis=-1):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.channel_axis = channel_axis
        self.fd = None
        self.hog_image = None

    def get_orientations(self):
        return self.orientations


    def compute_hog(self, image, visualize=True,
                block_norm=None, feature_vector=False, channel_axis=-1):

        """Compute Histogram of Oriented Gradients (HOG) RGB or grayscale image."""
        if channel_axis is None or str(channel_axis).lower() == 'grayscale':
            image = color.rgb2gray(image)
            channel_axis = None

        if image.ndim == 2: # make sure we do not search for a channel axis that does not exist
            channel_axis = None

        if visualize:
            self.fd, self.hog_image = hog(image, orientations=self.orientations,
                                          pixels_per_cell=self.pixels_per_cell,
                                          cells_per_block=self.cells_per_block,
                                          visualize=visualize,
                                          block_norm=block_norm,
                                          feature_vector=feature_vector,
                                          channel_axis=channel_axis)
        else:
            self.fd  = hog(image, orientations=self.orientations,
                           pixels_per_cell=self.pixels_per_cell,
                           cells_per_block=self.cells_per_block, visualize=False,
                           block_norm=block_norm, feature_vector=feature_vector,
                           channel_axis=channel_axis)
            self.hog_image = None

        # return fd, hog_image
        return self.fd, self.hog_image


def reshape_hog_array(array, len_axis1):
    # Calculate the total number of elements in the array
    N = array.size
    # Ensure that m is a divisor of N
    if N % len_axis1 != 0:
        raise ValueError(f"The given len_axis1 ({len_axis1}) is not a divisor of the total number of elements ({N}).")

    new_shape = (N // len_axis1, len_axis1)
    # Reshape the array
    reshaped_array = array.reshape(new_shape)
    return reshaped_array

# def polar_histogram(fd, ax, n_orientations=30, ytick_format="%.2g", y_lim=None):

#     # Sum the histograms across all cells/blocks
#     global_histogram = np.sum(fd, axis=0)

#     bin_angles = np.linspace(0, 360, n_orientations, endpoint=False)
#     bin_angles_rad = np.deg2rad(bin_angles)

#     # Find the bin with the maximum value
#     max_index = np.argmax(global_histogram)
#     max_value = global_histogram[max_index]

#     n_bins = len(global_histogram)
#     # Generate bin angles
#     bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)  # +1 to close the circle
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Middle of each bin


def plot_polar_histogram(ax, global_histogram, orientations_deg):

    orientations_rad = np.deg2rad(orientations_deg)
    n_bins = len(orientations_deg)

    # Plot the histogram on the given axis
    bars = ax.bar(orientations_rad, global_histogram, width=(2 * np.pi / n_bins), bottom=0)

    max_idx = np.argwhere(global_histogram == global_histogram.max()).flatten()

    for idx in max_idx: # it's always at least 2 maxima:
        bars[idx].set_color('red')

    return bars


def average_directions_over_cells(fd, orientations, N=None, M=None, fd_data_as_array=True):
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
            plot_fd = fd[i, j, :].mean()  # Average over sub-cells (cells_per_block_x, cells_per_block_y)
            key += 1
            fd_data[key] = plot_fd


    if fd_data_as_array:
        vectors = []
        for key, value in fd_data.items():
            if key == 'grid_size':
                continue  # Skip the 'grid_size' key
            # Append the vector to the list
            vectors.append(value)
        # Convert the list of vectors to a 2D numpy array
        fd_data = np.array(vectors)
    return fd_data



def correction_of_round_angles(orientations_180_deg, histog_data, corr90=True, corr45=False):

        # correction at 0*: normally speaking the index should be always = 0 but
        # we compute the argmin here just in case there were previous manipulations
        index_0 = np.argmin(np.abs(orientations_180_deg))
        histog_data[index_0] = 0.5*(histog_data[index_0+1] + histog_data[index_0-1])

        if corr90:         # make correction at 90 degrees:
            index_90 = np.argmin(np.abs(orientations_180_deg-90))
            # smooth with neighbours
            histog_data[index_90] = 0.5*(histog_data[index_90+1] + histog_data[index_90-1])
        # # correction at 45 and 135 degrees:
        if corr45:
            index_45 =  np.argmin(np.abs(orientations_180_deg-45 ))
            index_135 = np.argmin(np.abs(orientations_180_deg-135))

            histog_data[index_45 ] = 0.5*(histog_data[index_45 +1] + histog_data[index_45 -1])
            histog_data[index_135] = 0.5*(histog_data[index_135+1] + histog_data[index_135-1])

        return histog_data


def cell_signal_strengths(fd_data, norm_ord=1):

    strengths = np.zeros_like(fd_data[:,:,0]) # ignore last axis (n_orientations)

    for i in range(fd_data.shape[0]):
        for j in range(fd_data.shape[1]):
            strengths[i,j] = np.linalg.norm(fd_data[i,j], ord=norm_ord)

    return strengths



def compute_average_direction(global_histogram, orientations_deg):

    if global_histogram.ndim > 2:
        raise ValueError(f'Input should be 2D, got shape {global_histogram.shape} instead ')


    # Convert histogram values to vectors
    bin_angles_rad = np.deg2rad(orientations_deg)

    x = np.sum(global_histogram * np.cos(bin_angles_rad))
    y = np.sum(global_histogram * np.sin(bin_angles_rad))

    # Calculate the resultant vector's angle (avg direction) in range (-90, 90)
    mean_angle_rad = np.arctan2(y, x) # output in [-pi, pi]
    mean_angle_deg = np.rad2deg(mean_angle_rad) # now output in [-90, 90]

    # Adjust the mean angle to be within the range (0, 180)
    if mean_angle_deg < 0:
        mean_angle_deg += 180
        mean_angle_rad += np.pi

    # Calculate angle residuals (in degs)
    # take minimum abolute difference between diff angle and its complementary angle
    angle_diffs = np.abs(orientations_deg - mean_angle_deg)
    angle_diffs = np.abs(np.minimum(angle_diffs, 90 - angle_diffs))

    # Calculate the standard deviation (sqrt of the average squared residuals)
    # weighted by histogram height, and normalised.
    std_dev_deg = np.sqrt(np.sum(global_histogram * angle_diffs**2) / np.sum(global_histogram))

    # Calculate average of absolute residuals)
    # weighted by histogram height, and normalised.
    abs_dev_deg = np.sum(np.abs(global_histogram * angle_diffs) / np.sum(global_histogram))

    return mean_angle_deg, std_dev_deg, abs_dev_deg
