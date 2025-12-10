import os
import warnings
import re
import time
import numpy as np
from scipy.ndimage import convolve
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import scharr_h, scharr_v


from main_code.pipeline_utils import (
    HOGDescriptor,
    compute_distribution_direction,
    correct_round_angles,
    cell_signal_strengths,
)
from main_code.utils_other import (
    clean_filename,
)
from main_code.plotting_utils import external_plot_analysis


class HOGAnalysis:
    """This class does the heavy lifting: includes all the  necessarysteps, from
    loading images, to computing HOG or Scharr-based histograms, to normalizing,
     to excluding the backgrond under a certain signal intensity, to plotting and storing.

     Note that the name HOGAnalysis is kept for compatibility, even if the 'scharr' method
     and it should be changed to something more generic in the future.
    """

    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        block_norm: str = "None",
        pixels_per_window: tuple[int, int] = (64, 64),
        channel_image: int | str = 1,  # default: green channel for RGBA images
        background_range=(0.10, 0.90),
        draft: bool = False,
        show_plots: bool = False,
        post_normalization: bool = True,
        correct_edge_angles: tuple[str, str] = ("interpolate", "interpolate"),
        num_bins: int = 45,
        method: str = "scharr",
    ):
        self.input_folder = input_folder  # default: ./input_images
        self.output_folder = output_folder  # default: ./output_analysis
        self.block_norm = block_norm
        self.pixels_per_window = pixels_per_window
        self.channel_image = channel_image
        self.background_range = background_range
        self.draft = draft
        self.show_plots = show_plots
        self.post_normalization = post_normalization
        self.correct_edge_angles = correct_edge_angles
        self.num_bins = num_bins
        self.method = method.lower()
        self.df_statistics = pd.DataFrame()
        self.saved_stats_path = None

        # If the user wants to see plots, turn on interactive mode
        if self.show_plots:
            plt.ion()  # hopefully works both for .py and for .ipynb

        # keep a HOGDescriptor instance for compatibility (stores pixel/window info)
        self.hog_descriptor = HOGDescriptor(
            orientations=num_bins,
            pixels_per_window=self.pixels_per_window,
            windows_per_block=(1, 1),
            channel_axis=-1,  # default value for RGB images
        )

    def process_folder(
        self,
        image_folder: str | bytes,
        output_filename: str | bytes,
        threshold: float | int,
        save_stats: bool = True,
        save_plots: bool = True,
        verbose: int = 0,
    ):

        image_files = [
            f
            for f in os.listdir(image_folder)
            if f.lower().endswith((".tif", ".png", ".jpg"))  # .endswith(".tif")
        ]

        if "images-lightsheet" in image_folder:
            image_files = [
                f
                for f in os.listdir(image_folder)
                if not "z" in f or int(re.search(r"z(\d+)", f).group(1)) % 10 == 0
            ]

        if self.draft and len(image_files) > 15:
            step = max(1, len(image_files) // 10)
            image_files = image_files[::step][:10]
            print(
                f"Draft mode: processing {len(image_files)} images, picked one every {step}."
            )

        for idx, image_file in enumerate(image_files):
            if len(image_file) > 100:  # output message every 20 images
                if idx % 20 == 0 or (idx % 5 == 0 and verbose > 0):
                    print(f"Processing image {idx + 1} out of {len(image_files)}")
            else:  # output message every 20% progress
                chunk20 = len(image_files) // 5
                chunk05 = len(image_files) // 20
                if idx % chunk20 == 0 or (idx % chunk05 == 0 and verbose > 0):
                    print(f"Processing image {idx + 1} out of {len(image_files)}")

            self.process_image(image_folder, image_file, threshold, save_plots)

        if save_stats:

            self.saved_stats_path = self.save_results_to_file(
                self.output_folder, output_filename
            )

    def clean_and_select_channel(self, image, channel_image: int | str = -1):

        # Images with four channels (RGBA) are problematic, ignore alpha channel
        if image.ndim == 3 and image.shape[2] == 4:
            warnings.warn("RGBA image detected, ignoring alpha channel.")
            image = image[:, :, :3]  # ignore the alpha channel

        # Convert RGB to grayscale if requested:
        if channel_image in ["greyscale", "grayscale"]:
            if image.ndim == 3:
                image = color.rgb2gray(image)
        elif isinstance(channel_image, int):
            # Select a specific channel (RED is 0, GREEN is 1, BLUE is 2)
            # -1 is the last channel, which is often BLUE
            if (
                image.ndim == 3
                and -1 <= channel_image < image.shape[2]
                and isinstance(channel_image, int)
            ):
                image = image[:, :, channel_image]
            else:
                raise ValueError(
                    f"Invalid channel index {channel_image} for image with shape {image.shape}."
                )
        return image

    def process_image(self, folder, filename, threshold, save_plots):
        t1 = time.time()

        image = io.imread(os.path.join(folder, filename))

        image = self.clean_and_select_channel(image, self.channel_image)

        # Choose computation method: 'hog', 'scharr', or 'sobel' (5x5)
        if self.method == "hog":
            fd_raw_bg, hog_image_bg = self.hog_descriptor.compute_hog(
                image, block_norm="None", feature_vector=False
            )
            fd_bg = np.squeeze(fd_raw_bg)
            strengths = cell_signal_strengths(fd_bg, norm_ord=1)
        elif self.method == "scharr" or self.method == "sobel":
            # Compute gradients with Scharr or Sobel filters and build orientation histograms per window
            if image.ndim == 3:
                image_proc = color.rgb2gray(image)
            else:
                image_proc = image

            if self.method == "scharr":
                grad_y = scharr_v(image_proc)
                grad_x = scharr_h(image_proc)
            elif self.method == "sobel":
                # 5x5 Sobel kernels (Fiji style)
                sobel5_x = np.array(
                    [
                        [2, 1, 0, -1, -2],
                        [3, 2, 0, -2, -3],
                        [4, 3, 0, -3, -4],
                        [3, 2, 0, -2, -3],
                        [2, 1, 0, -1, -2],
                    ],
                    dtype=float,
                )
                sobel5_y = sobel5_x.T
                grad_x = convolve(image_proc, sobel5_x, mode="reflect")
                grad_y = convolve(image_proc, sobel5_y, mode="reflect")

            magnitude = np.hypot(grad_x, grad_y)
            orientation = np.rad2deg(np.arctan2(grad_y, grad_x))
            orientation = (
                np.where(orientation < 0, orientation + 180, orientation) % 180
            )

            win_h, win_w = self.pixels_per_window
            h, w = image_proc.shape[:2]
            n_windows_y = h // win_h
            n_windows_x = w // win_w

            # histogram bins for orientations 0..180
            bin_edges = np.linspace(0, 180, self.num_bins + 1)

            fd_bg = np.zeros((n_windows_y, n_windows_x, self.num_bins))
            for i in range(n_windows_y):
                y0, y1 = i * win_h, (i + 1) * win_h
                for j in range(n_windows_x):
                    x0, x1 = j * win_w, (j + 1) * win_w
                    w_mag = magnitude[y0:y1, x0:x1].ravel()
                    w_ori = orientation[y0:y1, x0:x1].ravel()
                    if w_mag.size == 0:
                        continue
                    hist, _ = np.histogram(w_ori, bins=bin_edges, weights=w_mag)
                    fd_bg[i, j, :] = hist

            hog_image_bg = magnitude  # reuse name for compatibility with plotting
            strengths = cell_signal_strengths(fd_bg, norm_ord=1)
        else:
            raise ValueError(
                f"Method {self.method} is not recognized. Currently only 'hog', 'scharr', and 'sobel' are supported."
            )

        skip, threshold, cells_to_keep = self.adjust_threshold(
            strengths, threshold, background_range=self.background_range
        )

        if skip:  # if skip > skip_tol, we were unable to adjust threshold properly
            self.save_nan_stats(filename, image, threshold)
            return

        # Normalization: if using original 'hog' method, defer to HOGDescriptor
        # behavior; otherwise apply per-cell normalization for Scharr histograms.
        if self.method == "hog":
            if self.block_norm != "None":
                fd_norm, hog_image = self.hog_descriptor.compute_hog(
                    image, block_norm=self.block_norm, feature_vector=False
                )
                fd_norm = np.squeeze(fd_norm)
            else:
                hog_image = hog_image_bg
                if self.post_normalization is True:
                    fd_norm = fd_bg / (1e-7 + strengths[:, :, np.newaxis])
                else:
                    fd_norm = fd_bg

        elif self.method == "scharr":
            # per-cell normalization for Scharr descriptor
            # using same normalization schemes as HOG for consistency
            if self.block_norm != "None":
                fd_norm = np.copy(fd_bg)
                # simple per-cell normalization
                for iy in range(fd_norm.shape[0]):
                    for ix in range(fd_norm.shape[1]):
                        cell = fd_norm[iy, ix, :]
                        if self.block_norm == "L1":
                            nrm = np.sum(np.abs(cell)) + 1e-7
                            fd_norm[iy, ix, :] = cell / nrm
                        elif self.block_norm == "L2":
                            nrm = np.sqrt(np.sum(cell**2)) + 1e-7
                            fd_norm[iy, ix, :] = cell / nrm
                        elif self.block_norm == "L2-Hys":
                            nrm = np.sqrt(np.sum(cell**2)) + 1e-7
                            celln = cell / nrm
                            celln = np.minimum(celln, 0.2)
                            nrm2 = np.sqrt(np.sum(celln**2)) + 1e-7
                            fd_norm[iy, ix, :] = celln / nrm2
                        else:
                            fd_norm[iy, ix, :] = cell
                hog_image = hog_image_bg
            else:
                hog_image = hog_image_bg
                if self.post_normalization is True:
                    fd_norm = fd_bg / (1e-7 + strengths[:, :, np.newaxis])
                else:
                    fd_norm = fd_bg
        else:
            raise ValueError(
                f"Method {self.method} is not recognized. Currently only 'hog' and 'scharr' are supported."
            )

        # Drop cells below threshold, the rest is normalised if self.block_norm
        # is "None", so that all cells have equal weight in the histogram
        fd_norm[~cells_to_keep] = 0

        gradient_hist_180 = fd_norm[cells_to_keep].mean(axis=0)
        gradient_hist = dict(
            zip(
                np.linspace(0, 180, len(gradient_hist_180), endpoint=False),
                gradient_hist_180,
            )
        )

        # gradient_hist = correct_round_angles(
        #     gradient_hist, corr90=True, corr45=("20240928" not in folder)
        # )

        corr90_bool = self.correct_edge_angles[0].lower() == "interpolate"
        corr45_bool = self.correct_edge_angles[1].lower() == "interpolate"

        gradient_hist = correct_round_angles(
            gradient_hist,
            corr90=corr90_bool,
            corr45=corr45_bool,
        )

        mean_stats, mode_stats = compute_distribution_direction(gradient_hist)

        self.save_stats(filename, image, threshold, mean_stats, mode_stats, t1)

        if save_plots:
            self.save_plot(
                image, hog_image, gradient_hist, cells_to_keep, strengths, filename
            )

    def save_plot(
        self, image, hog_image, gradient_hist, cells_to_keep, strengths, filename
    ):

        fig = external_plot_analysis(
            image,
            hog_image,
            gradient_hist,
            cells_to_keep,
            strengths,
            method=self.method,
        )
        fig.tight_layout()

        filename_info = (
            f"_{self.block_norm}_{self.hog_descriptor.pixels_per_window[0]}p"
        )
        filename_png = clean_filename(filename).replace(".tif", filename_info + ".png")

        os.makedirs(self.output_folder, exist_ok=True)
        fig.savefig(os.path.join(self.output_folder, filename_png), dpi=300)
        if self.show_plots:
            plt.show(block=False)
            input("Close the plot window, or press [Enter] to continue:")
        plt.close(fig)

    def adjust_threshold(
        self,
        strengths: np.ndarray,
        threshold: float,
        background_range: tuple[float, float],
        skip_tol=1e-3,
    ):
        cells_to_keep = strengths > threshold

        if cells_to_keep.mean() <= background_range[0]:
            while cells_to_keep.mean() <= background_range[0] and threshold > skip_tol:
                threshold *= 0.75
                cells_to_keep = strengths > threshold
            return threshold <= skip_tol, threshold, cells_to_keep

        if cells_to_keep.mean() >= background_range[1]:
            while (
                cells_to_keep.mean() >= background_range[1] and threshold < 1 / skip_tol
            ):
                threshold *= 1.25
                cells_to_keep = strengths > threshold
            return threshold > 1 / skip_tol, threshold, cells_to_keep

        return False, threshold, cells_to_keep

    def save_nan_stats(self, filename, image, threshold):
        stats = {
            "avg. direction": np.nan,
            "std. deviation (mean)": np.nan,
            "abs. deviation (mean)": np.nan,
            "mode direction": np.nan,
            "std. deviation (mode)": np.nan,
            "abs. deviation (mode)": np.nan,
            "image size": str(image.shape),
            "signal_threshold": threshold,
            "elapsed time (mm:ss)": "nan",
        }
        self.df_statistics = pd.concat(
            [self.df_statistics, pd.DataFrame(stats, index=[filename])]
        )

    def save_stats(self, filename, image, threshold, mean_stats, mode_stats, t_start):
        elapsed = round(time.time() - t_start)
        time_fmt = f"{elapsed // 60}:{elapsed % 60:02d}"

        stats = {
            "image size": str(image.shape),
            "avg. direction": round(mean_stats["angle"], 3),
            "std. deviation (mean)": round(mean_stats["std_dev"], 3),
            "abs. deviation (mean)": round(mean_stats["abs_dev"], 3),
            "mode direction": round(mode_stats["angle"], 3),
            "std. deviation (mode)": round(mode_stats["std_dev"], 3),
            "abs. deviation (mode)": round(mode_stats["abs_dev"], 3),
            "signal_threshold": threshold,
            "elapsed time (mm:ss)": time_fmt,
        }

        df_row = pd.DataFrame(stats, index=[filename])

        self.df_statistics = pd.concat([self.df_statistics, df_row])

    def save_results_to_file(self, save_folder, filename):

        if self.draft:
            filename += "_draft"
        saved_stats_path = os.path.join(save_folder, filename + ".csv")
        self.df_statistics.to_csv(saved_stats_path, index=True)
        print(f"Saved results to {saved_stats_path}")

        return saved_stats_path
