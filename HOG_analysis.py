import os
import time
import re
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import pandas as pd

from pipeline_utils import (
    load_and_prepare_image,
    HOGDescriptor,
    plot_polar_histogram,
    compute_distribution_direction,
    correction_of_round_angles,
    cell_signal_strengths,
)
from plotting_utils import external_plot_hog_analysis
from utils_other import (
    print_top_values,
    set_sample_condition,
    get_folder_threshold,
    update_conditions_to_csv,
    clean_csv_rows,
    clean_filename,
)


root_folder = os.getcwd()

DRAFT = False
SHOW_PLOTS = False
SAVE_PLOTS = False
SAVE_STATS = True


class HOGAnalysis:
    def __init__(
        self, root_folder, group_folders=["ALL"], block_norm=None, draft=True
    ):  # TODO: test block_norm = "L2-Hys" (change approach with threshold, as norm is always = 1)
        self.root_folder = root_folder
        self.group_folders = group_folders
        self.image_results_dir = "HOG_plot_analysis"
        self.block_norm = block_norm
        self.draft = draft
        self.df_statistics = pd.DataFrame()
        self.hog_descriptor = HOGDescriptor(
            orientations=45,
            pixels_per_cell=(64, 64),
            cells_per_block=(1, 1),
            channel_axis=-1,
        )

    def process_and_save(self):
        self.image_folder = os.path.join(
            self.root_folder,
            # "images-3D-lightsheet-20241115_BAM_fkt20-P3-fkt21-P3-PEMFS-12w",
            # "images-3D-lightsheet-20240928_BAM_fkt20_P3_fkt21_P3_PEMFS",
            "images-confocal-20241022-fusion-bMyoB-BAMS-TM-6w",
            # "images-confocal-20241116-fusion-bMyoB-PEMFS-TM-12w",
        )
        for group in self.group_folders:
            self.process_group(self.image_folder, group)

    def process_group(self, image_folder, group):
        sub_group_folder = os.path.join(image_folder, group)
        threshold = get_folder_threshold(sub_group_folder)
        print(f"Processing {sub_group_folder} with threshold {threshold}")
        if "lightsheet" in os.path.basename(os.path.normpath(image_folder)):
            image_files = [
                f
                for f in os.listdir(sub_group_folder)
                if not "z" in f or int(re.search(r"z(\d+)", f).group(1)) % 5 == 0
            ]
        else:
            image_files = os.listdir(sub_group_folder)
        print(f"Images after filtering: {len(image_files)}")

        if self.draft:
            if "lightsheet-20241115" in sub_group_folder:
                # Only process specific indices for lightsheet-20241115 in Draft mode
                selected_indices = [1004, 863, 1413]
                print(f"Draft mode: Processing only indices {selected_indices}")
                for img_idx, image_file in enumerate(image_files):
                    if img_idx in selected_indices:
                        self.process_image(sub_group_folder, image_file, threshold)
            else:
                # For other folders in Draft mode, process ~10 images uniformly distributed
                if len(image_files) > 15:
                    js = max(1, (len(image_files) // 10))
                    image_files = image_files[::js][
                        :10
                    ]  # Approx. 10 uniformly spread images
                    print(f"Draft mode: Processing approx. 10 uniformly spread images.")
                for img_idx, image_file in enumerate(image_files):
                    self.process_image(sub_group_folder, image_file, threshold)
        else:
            # If not in Draft mode, process all images
            for img_idx, image_file in enumerate(image_files):

                if img_idx % 50 == 0:
                    print("Processing img n*", img_idx)
                self.process_image(sub_group_folder, image_file, threshold)

        # Store (csv) results outside the group folder, in the condition folder
        subgroup_folder_results = os.path.dirname(sub_group_folder)
        if SAVE_STATS:
            self.save_results(subgroup_folder_results, group)

    def process_image(self, folder, filename, threshold: float):
        t1 = time.time()
        image = load_and_prepare_image(folder, filename, channel=1)
        fd_raw, hog_image = self.hog_descriptor.compute_hog(
            image, block_norm=self.block_norm, feature_vector=False
        )
        fd = np.squeeze(fd_raw)
        strengths = (
            cell_signal_strengths(fd, norm_ord=1)
            if self.block_norm is None
            else np.ones(fd.shape[:2])
        )
        fd_norm = (
            fd / (1e-7 + strengths[:, :, np.newaxis]) if self.block_norm is None else fd
        )

        # Adjust threshold and check if computation should be skipped
        skip_computation, threshold, cells_to_keep = self.adjust_threshold(
            strengths, threshold, filename
        )
        if skip_computation:
            self.save_nan_stats(filename, image, threshold, t1)
            return

        # Check signal coverage
        self.check_signal_coverage(cells_to_keep, threshold)

        fd_norm[~cells_to_keep] = 0
        gradient_hist_180 = fd_norm[cells_to_keep].mean(axis=0)
        gradient_hist = dict(
            zip(
                np.linspace(0, 180, len(gradient_hist_180), endpoint=False),
                gradient_hist_180,
            )  # bin centered around 0, etc.. ok?
        )

        gradient_hist = correction_of_round_angles(
            gradient_hist, corr90=True, corr45=True
        )
        # print("Top values after smoothing:")
        # print(print_top_values(gradient_hist, n=3))

        # Compute distribution directions and grouped statistics
        mean_stats, mode_stats = compute_distribution_direction(
            gradient_hist, list(gradient_hist.keys())
        )

        # Save statistics
        self.save_stats(filename, image, threshold, mean_stats, mode_stats, t1)

        if SHOW_PLOTS or SAVE_PLOTS:
            plt = self.plot_results(
                image,
                hog_image,
                gradient_hist,
                cells_to_keep,
                strengths,
            )
            if SAVE_PLOTS:
                filename_info_str = "_" + (
                    str(self.block_norm)
                    + "_"
                    + str(self.hog_descriptor.pixels_per_cell[0])
                    + "p"
                )

                filename = clean_filename(filename)

                filename_png = filename.replace(".tif", filename_info_str + ".png")
                plt.savefig(
                    os.path.join(
                        self.image_folder, self.image_results_dir, filename_png
                    ),
                    dpi=450,
                )
                # f"{image_filename}_{self.block_norm}_{self.n_pixels}.png",

            if SHOW_PLOTS:
                plt.show()
            plt.close()  # Clean up just in case

    def adjust_threshold(self, strengths, threshold, filename):
        """Dynamically adjusts the threshold until signal is found or limit reached."""
        cells_to_keep = strengths > threshold

        if cells_to_keep.mean() <= 0.05:
            while cells_to_keep.mean() <= 0.05:

                threshold *= 0.75
                warnings.warn(
                    f"No signal found in image {filename}. Decreasing threshold to {threshold:.2f}"
                )
                cells_to_keep = strengths > threshold
                if threshold < 1e-4:
                    warnings.warn(
                        f"Threshold became too low (< 1e-4) for image {filename}. Skipping computation."
                    )
                    return True, threshold, cells_to_keep  # Skip computation

            return False, threshold, cells_to_keep

        if cells_to_keep.mean() >= 0.95:
            while cells_to_keep.mean() >= 0.95:

                threshold *= 1.25
                warnings.warn(
                    f"No background found in image {filename}. Increasing threshold to {threshold:.2f}"
                )
                cells_to_keep = strengths > threshold
                if threshold > 1e2:
                    warnings.warn(
                        f"Threshold became too high (> 1e2) for image {filename}. Skipping computation."
                    )
                    return True, threshold, cells_to_keep  # Skip computation
        # else: is the normal case where the threshold is good enough
        return False, threshold, cells_to_keep

    def save_nan_stats(self, filename, image, threshold, t1):
        """Save NaN values for image statistics when computation is skipped."""
        mean_stats = {
            "angle": np.nan,
            "std_dev": np.nan,
            "abs_dev": np.nan,
        }
        mode_stats = {
            "angle": np.nan,
            "std_dev": np.nan,
            "abs_dev": np.nan,
        }
        metrics_to_include = {
            "mean": True,
            "mode": True,
            "std_dev": True,
            "abs_dev": True,
        }
        elapsed_time_frmt = "nan"

        self.save_image_stats(
            filename,
            image,
            threshold,
            mean_stats,
            mode_stats,
            metrics_to_include,
            elapsed_time_frmt,
        )
        print(f"Image {filename}: Computation skipped. NaN values saved.")

    def check_signal_coverage(self, cells_to_keep, threshold):
        """Checks the signal coverage and issues warnings for extreme cases."""
        if cells_to_keep.mean() > 0.95:
            warnings.warn(
                f"Little/no background identified ({1-cells_to_keep.mean():.1%} of the image). Consider increasing the threshold (currently={threshold:.2f})."
            )

        if cells_to_keep.mean() < 0.05:
            warnings.warn(
                f"Too many window partitions are classified as part of the background ({1-cells_to_keep.mean():.1%} of the image). Consider decreasing the threshold (currently={threshold:.2f})."
            )

    def save_stats(self, filename, image, threshold, mean_stats, mode_stats, t1):
        """Save image statistics after computation."""
        elapsed_seconds = round(time.time() - t1)
        mins, secs = divmod(elapsed_seconds, 60)
        elapsed_time_frmt = f"{mins}:{secs:02d}"

        metrics_to_include = {
            "mean": True,
            "mode": True,
            "std_dev": True,
            "abs_dev": True,
        }

        self.save_image_stats(
            filename,
            image,
            threshold,
            mean_stats,
            mode_stats,
            metrics_to_include,
            elapsed_time_frmt,
        )

    def save_image_stats(
        self,
        filename,
        image,
        threshold,
        mean_stats,
        mode_stats,
        metrics_to_include,
        elapsed_time_frmt,
    ):
        stats_dict = {
            "image size": str(image.shape),
            "elapsed time (mm:ss)": elapsed_time_frmt,
            "signal_threshold": threshold,
        }

        # Add Mean Metrics
        if metrics_to_include.get("mean", False):
            stats_dict["avg. direction"] = round(mean_stats["angle"], 3)
            if metrics_to_include.get("std_dev", False):
                stats_dict["std. deviation (mean)"] = round(mean_stats["std_dev"], 3)
            if metrics_to_include.get("abs_dev", False):
                stats_dict["abs. deviation (mean)"] = round(mean_stats["abs_dev"], 3)

        # Add Mode Metrics
        if metrics_to_include.get("mode", False):
            stats_dict["mode direction"] = round(mode_stats["angle"], 3)
            if metrics_to_include.get("std_dev", False):
                stats_dict["std. deviation (mode)"] = round(mode_stats["std_dev"], 3)
            if metrics_to_include.get("abs_dev", False):
                stats_dict["abs. deviation (mode)"] = round(mode_stats["abs_dev"], 3)

        stats_df = pd.DataFrame(stats_dict, index=[filename])

        # If only one group folder (i.e. no grouping), add donor and condition
        if self.group_folders is not None and len(self.group_folders) == 1:

            # Extract donor from filename using regex
            donor_pattern = r"fkt\d{1,2}"
            match = re.search(donor_pattern, filename)
            if not match:
                raise ValueError("Donor identifier not found")
            donor = match.group(0)
            stats_df.insert(2, "donor", donor)

            # Extract condition using the helper function
            condition = set_sample_condition(filename, suppress_warnings=True)
            stats_df.insert(1, "condition", condition)

        self.df_statistics = pd.concat([self.df_statistics, stats_df], axis=0)

    def plot_results(
        self,
        image,
        hog_image,
        gradient_hist,
        cells_to_keep,
        strengths,
    ):
        plt = external_plot_hog_analysis(
            image, hog_image, gradient_hist, cells_to_keep, strengths
        )
        plt.tight_layout()

        return plt

    def save_results(self, save_folder, group):
        if group == "ALL":
            filename = f"HOG_stats_{self.block_norm}_{self.hog_descriptor.pixels_per_cell[0]}pixels"
        else:
            filename = f"HOG_stats_{group}_{self.block_norm}_{self.hog_descriptor.pixels_per_cell[0]}pixels"
        if self.draft:
            filename += "_draft2"
        # Create the directory if it doesn't exist
        output_dir = os.path.join(save_folder)
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, filename + ".csv")
        self.df_statistics.to_csv(csv_path)
        csv_path_out = os.path.join(output_dir, filename + "_clean.csv")
        verbose_condition = not self.draft

        update_conditions_to_csv(
            csv_path, csv_path_out, print_process=verbose_condition
        )
        clean_csv_rows(csv_path, csv_path_out, missing_threshold=2)

        print("Output file:\n", csv_path_out)


if __name__ == "__main__":
    root_folder = os.getcwd()
    hog_analysis = HOGAnalysis(root_folder, draft=DRAFT)
    hog_analysis.process_and_save()
