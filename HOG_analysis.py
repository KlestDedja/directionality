import os
import time
import re
import warnings
import matplotlib.pyplot as plt

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
from plotting_utils import (
    explanatory_normalized_hog,
    explanatory_plot_intro,
    explanatory_plot_hog,
    explanatory_plot_polar,
    explanatory_plot_filter,
    external_plot_hog_analysis,
    list_boo_boo,
    list_images_plot,
)
from utils_other import (
    print_top_values,
    set_sample_replicate,
    set_sample_condition,
    get_folder_threshold,
    update_conditions_to_csv,
    update_replicates_to_csv,
    clean_csv_rows,
    clean_filename,
)

root_folder = os.getcwd()

DRAFT = True
SHOW_PLOTS = False
SAVE_PLOTS = True
SAVE_STATS = True
CORRECT_ARTEFACTS = True
EXTRA_PLOTS = False


class HOGAnalysis:
    def __init__(
        self, root_folder, group_folders=["ALL-old"], block_norm="L2-Hys", draft=True
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

        self.expl_idx = 0

    def process_and_save(self):
        self.image_folder = os.path.join(
            self.root_folder,
            "images-lightsheet-20241115_BAM_fkt20-P3-fkt21-P3-PEMFS-12w",
            # "images-lightsheet-20240928_BAM_fkt20_P3_fkt21_P3_PEMFS",
            # "images-confocal-20241022-fusion-bMyoB-BAMS-TM-6w",
            # "images-confocal-20241116-fusion-bMyoB-PEMFS-TM-12w",
            # "images-selection-HOG_fkt_19_P2_MS",
            # "images-selection-HOG_fkt21_P3_MS",
            # "20230208 TM bMyoB fkt16 P2 fusion assay PEMS",
            # "20230228 TM bMyoB fkt11 P2 fusion assay PEMS",
            # "20230628 TM bMyoB fkt8 P2 fusion assay PEMS",
            # "20230901 TM bMyoB fkt20-fkt21 P2 fusion assay PEMS",
            # "20240131 TM bMyoB fkt3 P4 PEMS fusie",
            # "20240206 TM bMyoB fkt4 P4 PEMS fusie",
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
            # if "lightsheet-20241115" in sub_group_folder:
            # Only process specific indices for lightsheet-20241115 in Draft mode
            # selected_indices = [1004, 863, 1413]
            # list_run = list_boo_boo()
            # list_run = list_images_plot()
            # name_matches = [image_file in list_run for image_file in image_files]
            # selected_indices = list(np.where(name_matches)[0])
            # print(f"Draft mode: Processing only indices {selected_indices}")
            # for img_idx, image_file in enumerate(image_files):
            #     if img_idx in selected_indices:
            #         # if image_file in list_run:
            #         self.process_image(sub_group_folder, image_file, threshold)
            #         print("Processed image in the list:", image_file)
            #         print("indexed", img_idx)

            # else:
            # For other folders in Draft mode, process ~10 images uniformly distributed
            if len(image_files) > 15:
                js = max(1, (len(image_files) // 10))
                image_files = image_files[::js][
                    :10
                ]  # Approx. 10 uniformly spread images
                print(
                    f"Draft mode: Processing {len(image_files)}, uniformly spread images."
                )
            for img_idx, image_file in enumerate(image_files):
                self.process_image(sub_group_folder, image_file, threshold)
        else:
            # If not in Draft mode, process all images
            for img_idx, image_file in enumerate(image_files):

                if img_idx % 20 == 0:
                    print("Processing img n*", img_idx)
                self.process_image(sub_group_folder, image_file, threshold)

        # Store (csv) results outside the group folder, in the condition folder
        subgroup_folder_results = os.path.dirname(sub_group_folder)
        if SAVE_STATS:
            self.save_results(subgroup_folder_results, group)

    def process_image(self, folder, filename, threshold: float):
        t1 = time.time()
        image = load_and_prepare_image(folder, filename, channel=1)

        #### Background processing: Compute HOG and signal strength
        fd_raw_bg, hog_image_bg = self.hog_descriptor.compute_hog(
            image, block_norm=None, feature_vector=False
        )
        fd_bg = np.squeeze(fd_raw_bg)
        strengths = cell_signal_strengths(fd_bg, norm_ord=1)

        # Given distribution of signal strenght, adjust threshold
        # and check if computation should be skipped
        skip_computation, threshold, cells_to_keep = self.adjust_threshold(
            strengths, threshold, filename
        )
        if skip_computation:
            self.save_nan_stats(filename, image, threshold, t1)
            return

        #### Run normalized HOG on image

        if self.block_norm != None:
            fd_norm, hog_image = self.hog_descriptor.compute_hog(
                image, block_norm=self.block_norm, feature_vector=False
            )
            fd_norm = np.squeeze(fd_norm)
        else:  # no need to re-run the HOG descriptor, as it was already done above
            hog_image = hog_image_bg
            fd_norm = fd_bg / (1e-7 + strengths[:, :, np.newaxis])

        # cells to keep has been computed with the raw HOG
        fd_norm[~cells_to_keep] = 0
        gradient_hist_180 = fd_norm[cells_to_keep].mean(axis=0)
        # TODO: check bin centers, what does _compute_HOG assume?
        gradient_hist = dict(
            zip(
                np.linspace(0, 180, len(gradient_hist_180), endpoint=False),
                gradient_hist_180,
            )  # bin centered around 0, etc.. ok?
        )

        import pickle

        root_folder = os.getcwd()
        file_path = os.path.join(
            root_folder, "testing", filename.strip(".tif") + "dict_pre.pkl"
        )

        # Serialize (pickle) the dictionary to a file in the project root
        with open(file_path, "wb") as f:
            pickle.dump(gradient_hist, f)

        if CORRECT_ARTEFACTS:
            correct_45deg = False if "20240928" in folder else True
            # correct_45deg = True
            gradient_hist = correction_of_round_angles(
                gradient_hist, corr90=True, corr45=correct_45deg
            )

        file_path = os.path.join(
            root_folder, "testing", filename.strip(".tif") + "dict_post.pkl"
        )

        # Serialize (pickle) the dictionary to a file in the project root
        with open(file_path, "wb") as f:
            pickle.dump(gradient_hist, f)

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

                filename_clean = clean_filename(filename)

                filename_png = filename_clean.replace(
                    ".tif", filename_info_str + ".png"
                )
                plt.savefig(
                    os.path.join(
                        self.image_folder, self.image_results_dir, filename_png
                    ),
                    dpi=300,
                )
                # f"{image_filename}_{self.block_norm}_{self.n_pixels}.png",

            if SHOW_PLOTS:
                plt.show()
                # Show main plot + build and show extra plot for explanations in the manuscript:
                plt = explanatory_plot_polar(image)
                self.expl_idx += 1
                plt.savefig(
                    os.path.join(
                        os.path.dirname(self.image_folder),
                        f"illustration-polar-histogram-explain-{self.expl_idx}.png",
                    ),
                    dpi=200,
                )
                plt.show()

            plt.close()  # Clean up just in case

            if EXTRA_PLOTS and DRAFT and self.expl_idx == 1:

                plt5 = explanatory_plot_polar(image)
                plt5.savefig(
                    os.path.join(
                        root_folder,
                        "Polar_histogram-new.png",
                    ),
                    dpi=300,
                )
                plt5.show()

                plt2 = explanatory_plot_intro(image)
                plt2.savefig(
                    os.path.join(
                        root_folder,
                        "Illustration_windowing-new.png",
                    ),
                    dpi=300,
                )
                plt2.show()

                plt3 = explanatory_plot_hog(image)
                plt3.savefig(
                    os.path.join(
                        root_folder,
                        "Illustration_HOG-new.png",
                    ),
                    dpi=300,
                )
                plt3.show()

                plt4 = explanatory_plot_filter(image)
                plt4.savefig(
                    os.path.join(
                        root_folder,
                        "Illustration_filter-new.png",
                    ),
                    dpi=300,
                )
                plt4.show()

                plt5 = explanatory_normalized_hog(image)
                plt5.savefig(
                    os.path.join(
                        root_folder,
                        "Illustration_norm_HOG-new.png",
                    ),
                    dpi=300,
                )
                plt5.show()

    def adjust_threshold(self, strengths, threshold, filename):
        """Dynamically adjusts the threshold until signal is found or limit reached."""
        cells_to_keep = strengths > threshold

        if cells_to_keep.mean() <= 0.05:
            while cells_to_keep.mean() <= 0.05:

                threshold *= 0.75
                # warnings.warn(
                #     f"No signal found in image {filename}. Decreasing threshold to {threshold:.2f}"
                # )
                cells_to_keep = strengths > threshold
                if threshold < 0.05:
                    warnings.warn(
                        f"Threshold became too low (< 0.05) for image {filename}. Skipping computation."
                    )
                    return True, threshold, cells_to_keep  # Skip computation = True

            return False, threshold, cells_to_keep  # Skip computation = False

        if cells_to_keep.mean() >= 0.90:
            while cells_to_keep.mean() >= 0.90:

                threshold *= 1.25
                # warnings.warn(
                #     f"No background found in image {filename}. Increasing threshold to {threshold:.2f}"
                # )
                cells_to_keep = strengths > threshold
                if threshold > 50:
                    warnings.warn(
                        f"Threshold became too high (> 50) for image {filename}. Skipping computation."
                    )
                    return True, threshold, cells_to_keep  # Skip computation = True
        # else: is the normal case where the threshold is good enough
        return False, threshold, cells_to_keep  # Skip computation = False

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
                stats_df.insert(2, "donor", "Unknown")
            else:
                donor = match.group(0)
                stats_df.insert(2, "donor", donor)

            # Extract condition using the helper function
            condition = set_sample_condition(filename, suppress_warnings=True)
            stats_df.insert(1, "condition", condition)
            replicate = set_sample_replicate(filename, suppress_warnings=True)
            stats_df.insert(3, "replicate", replicate)

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
            filename += "_draft"
        # Create the directory if it doesn't exist
        output_dir = os.path.join(save_folder)
        os.makedirs(output_dir, exist_ok=True)
        # add prefix with experiment type:
        experiment_type = os.path.basename(os.path.normpath(self.image_folder))
        experiment_type = experiment_type.replace("images-", "")
        # Find a sequence of 8 to 10 digits and crop everything after it
        match = re.search(r"(\d{7,10})", experiment_type)
        if match:
            experiment_type = experiment_type[: match.end()]

        filename = f"{experiment_type}-{filename}"

        if CORRECT_ARTEFACTS is False:
            filename += "_no_correction"

        csv_path_in = os.path.join(output_dir, filename + ".csv")
        self.df_statistics.to_csv(csv_path_in)
        csv_path_out = os.path.join(output_dir, filename + ".csv")
        verbose_condition = not self.draft

        update_conditions_to_csv(
            csv_path_in, csv_path_out, print_process=verbose_condition
        )
        update_replicates_to_csv(
            csv_path_out, csv_path_out, print_process=verbose_condition
        )

        clean_csv_rows(csv_path_in, csv_path_out, missing_threshold=2)

        print("Output file:\n", csv_path_out)


if __name__ == "__main__":
    root_folder = os.getcwd()
    hog_analysis = HOGAnalysis(root_folder, draft=DRAFT, block_norm=None)  # "L2-Hys"
    # hog_analysis = HOGAnalysis(root_folder, draft=DRAFT, block_norm="L2-Hys") # "L2-Hys"
    hog_analysis.process_and_save()
