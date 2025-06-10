# hog_analysis.py

import os
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main_code.pipeline_utils import (
    load_and_prepare_image,
    HOGDescriptor,
    compute_distribution_direction,
    correct_round_angles,
    cell_signal_strengths,
)
from main_code.utils_other import (
    clean_filename,
    get_folder_threshold,
)
from main_code.plotting_utils import external_plot_hog_analysis


class HOGAnalysis:
    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        block_norm: str | None = None,
        pixels_per_cell: tuple[int, int] = (64, 64),
        channel_axis: int | None | str = -1,
        draft: bool = False,
        show_plots: bool = False,
    ):
        self.input_folder = input_folder  # default: ./input_images
        self.output_folder = output_folder  # default: ./output_analysis
        self.block_norm = block_norm
        self.pixels_per_cell = pixels_per_cell
        self.channel_axis = channel_axis
        self.draft = draft
        self.show_plots = show_plots
        self.df_statistics = pd.DataFrame()

        # If the user wants to see plots, turn on interactive mode
        if self.show_plots:
            plt.ion()  # opefully works both for .py and for .ipynb

        self.hog_descriptor = HOGDescriptor(
            orientations=45,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=(1, 1),
            channel_axis=self.channel_axis,
        )

    def process_folder(
        self,
        image_folder: str,
        threshold: float | int,
        save_stats: bool = True,
        save_plots: bool = True,
    ):

        image_files = [
            f for f in os.listdir(image_folder) if f.lower().endswith(".tif")
        ]

        if self.draft and len(image_files) > 15:
            step = max(1, len(image_files) // 10)
            image_files = image_files[::step][:10]
            print(f"Draft mode: Processing {len(image_files)} uniformly spread images")

        for idx, image_file in enumerate(image_files):
            if idx % 20 == 0:
                print(f"Processing image {idx + 1} out of {len(image_files)}")
            self.process_image(image_folder, image_file, threshold, save_plots)

        if save_stats:
            self.save_results(self.output_folder)

    def process_image(self, folder, filename, threshold, save_plots):
        t1 = time.time()
        grayscale = "good-bad" in folder
        image = load_and_prepare_image(
            folder, filename, channel=1, to_grayscale=grayscale
        )

        fd_raw_bg, hog_image_bg = self.hog_descriptor.compute_hog(
            image, block_norm=None, feature_vector=False
        )
        fd_bg = np.squeeze(fd_raw_bg)
        strengths = cell_signal_strengths(fd_bg, norm_ord=1)

        skip, threshold, cells_to_keep = self.adjust_threshold(
            strengths, threshold, filename
        )
        if skip:
            self.save_nan_stats(filename, image, threshold)
            return

        if self.block_norm:
            fd_norm, hog_image = self.hog_descriptor.compute_hog(
                image, block_norm=self.block_norm, feature_vector=False
            )
            fd_norm = np.squeeze(fd_norm)
        else:
            hog_image = hog_image_bg
            fd_norm = fd_bg / (1e-7 + strengths[:, :, np.newaxis])

        fd_norm[~cells_to_keep] = 0
        gradient_hist_180 = fd_norm[cells_to_keep].mean(axis=0)
        gradient_hist = dict(
            zip(
                np.linspace(0, 180, len(gradient_hist_180), endpoint=False),
                gradient_hist_180,
            )
        )

        gradient_hist = correct_round_angles(
            gradient_hist, corr90=True, corr45=("20240928" not in folder)
        )

        mean_stats, mode_stats = compute_distribution_direction(
            gradient_hist, list(gradient_hist.keys())
        )

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

        filename_info = f"_{self.block_norm}_{self.hog_descriptor.pixels_per_cell[0]}p"
        filename_png = clean_filename(filename).replace(".tif", filename_info + ".png")

        os.makedirs(self.output_folder, exist_ok=True)
        fig.savefig(os.path.join(self.output_folder, filename_png), dpi=300)
        if self.show_plots:
            plt.show()
        plt.close(fig)

    def adjust_threshold(self, strengths, threshold, filename):
        cells_to_keep = strengths > threshold

        if cells_to_keep.mean() <= 0.05:
            while cells_to_keep.mean() <= 0.05 and threshold > 0.05:
                threshold *= 0.75
                cells_to_keep = strengths > threshold
            return threshold <= 0.05, threshold, cells_to_keep

        if cells_to_keep.mean() >= 0.995:
            while cells_to_keep.mean() >= 0.995 and threshold < 50:
                threshold *= 1.25
                cells_to_keep = strengths > threshold
            return threshold > 50, threshold, cells_to_keep

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

    def save_stats(self, filename, image, threshold, mean_stats, mode_stats, t1):
        elapsed = round(time.time() - t1)
        time_fmt = f"{elapsed // 60}:{elapsed % 60:02d}"

        stats = {
            "avg. direction": round(mean_stats["angle"], 3),
            "std. deviation (mean)": round(mean_stats["std_dev"], 3),
            "abs. deviation (mean)": round(mean_stats["abs_dev"], 3),
            "mode direction": round(mode_stats["angle"], 3),
            "std. deviation (mode)": round(mode_stats["std_dev"], 3),
            "abs. deviation (mode)": round(mode_stats["abs_dev"], 3),
            "image size": str(image.shape),
            "signal_threshold": threshold,
            "elapsed time (mm:ss)": time_fmt,
        }

        df_row = pd.DataFrame(stats, index=[filename])

        # df_row.insert(
        #     1, "condition", set_sample_condition(filename, suppress_warnings=True)
        # )
        # df_row.insert(2, "donor", self.extract_donor(filename))
        # df_row.insert(
        #     3, "replicate", set_sample_replicate(filename, suppress_warnings=True)
        # )

        self.df_statistics = pd.concat([self.df_statistics, df_row])

    def extract_donor(self, filename):
        match = re.search(r"fkt\d{1,2}", filename)
        return match.group(0) if match else "Unknown"

    def save_results(self, save_folder):
        fname = f"HOG_stats_{self.block_norm}_{self.hog_descriptor.pixels_per_cell[0]}pixels"
        if self.draft:
            fname += "_draft"
        final_path = os.path.join(save_folder, fname + ".csv")
        self.df_statistics.to_csv(final_path, index=True)
        print(f"Saved results to {final_path}")
