import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color
import warnings
import re


from main_code.pipeline_utils import (
    HOGDescriptor,
    compute_distribution_direction,
    correct_round_angles,
    cell_signal_strengths,
)
from main_code.utils_other import (
    clean_filename,
)
from main_code.plotting_utils import external_plot_hog_analysis


class HOGAnalysis:
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
        self.df_statistics = pd.DataFrame()
        self.saved_stats_path = None

        # If the user wants to see plots, turn on interactive mode
        if self.show_plots:
            plt.ion()  # hopefully works both for .py and for .ipynb

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
                if idx % 20 == 0:
                    print(f"Processing image {idx + 1} out of {len(image_files)}")
            else:  # output message every 20% progress
                chunk20 = len(image_files) // 5
                if idx % chunk20 == 0:
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

        fd_raw_bg, hog_image_bg = self.hog_descriptor.compute_hog(
            image, block_norm="None", feature_vector=False
        )
        fd_bg = np.squeeze(fd_raw_bg)
        strengths = cell_signal_strengths(fd_bg, norm_ord=1)

        skip, threshold, cells_to_keep = self.adjust_threshold(
            strengths, threshold, background_range=self.background_range
        )
        if skip:
            self.save_nan_stats(filename, image, threshold)
            return

        if self.block_norm != "None":
            # (re)compute HOG features with the specified block normalization
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
        fig = external_plot_hog_analysis(
            image, hog_image, gradient_hist, cells_to_keep, strengths
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
