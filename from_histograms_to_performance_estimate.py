import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from main_code.evaluation_utils import (
    circular_wasserstein_1,
    direction_gaussian_peak_mean,
    parse_distribution_csv,
    build_distribution_from_data,
    directions_to_hist,
)

# ========= SETTINGS ==========
# needed inof since at runtime we need to know num_bins, method and interpolation type
N_BINS = 45
CORRECT_EDGES = ("gaussian", "1")
# CORRECT_EDGES = ("none", "none")
METHODS = ["hog", "scharr", "sobel"]

SMOOTH_MODE = False

suffix_filename = "smooth" if SMOOTH_MODE else "raw"
DRAFT_MODE = False

# ========== FOLDERS FOR  COMPARISON ==========

ROOT_FOLDER = os.getcwd()
# change accordingly if your structure differs from the demo
DATA_FOLDER_NAME = os.path.join("data", "synthetic-golden-standard")
# DATA_FOLDER_NAME = os.path.join("data", "test-golden")
INPUT_FOLDER = "input-images"
INPUT_METADATA = "input-metadata"

OUTPUT_FOLDER = "output-analysis"

FOLDERS = [
    f"{method}_{N_BINS}bins_{CORRECT_EDGES[0]}_{CORRECT_EDGES[1]}" for method in METHODS
]


# ======== FILE PATHS ==========

edgehog_directories_results = [
    os.path.join(ROOT_FOLDER, DATA_FOLDER_NAME, OUTPUT_FOLDER, folder)
    for folder in FOLDERS
]

hog_file = os.path.join(
    edgehog_directories_results[0], "distribution_stats_vertical_format.csv"
)
scharr_file = os.path.join(
    edgehog_directories_results[1], "distribution_stats_vertical_format.csv"
)
sobel_file = os.path.join(
    edgehog_directories_results[2], "distribution_stats_vertical_format.csv"
)

fiji_fft = os.path.join(
    ROOT_FOLDER, DATA_FOLDER_NAME, OUTPUT_FOLDER, "fiji", "20251215_Fiji.csv"
)

fiji_gradient = os.path.join(
    ROOT_FOLDER, DATA_FOLDER_NAME, OUTPUT_FOLDER, "fiji", "20251215_FijiGradient.csv"
)


# Load distributions from EDGEHOG and FIJI
hog_distrib_dict = parse_distribution_csv(hog_file)
scharr_distrib_dict = parse_distribution_csv(scharr_file)
sobel_distrib_dict = parse_distribution_csv(sobel_file)

fiji_fft_distrib_dict = parse_distribution_csv(fiji_fft)
fiji_gradient_distrib_dict = parse_distribution_csv(fiji_gradient)

ground_truth_files = os.listdir(
    os.path.join(ROOT_FOLDER, DATA_FOLDER_NAME, INPUT_METADATA)
)

ground_truth_files = [f for f in ground_truth_files if f.endswith(".csv")]


perf_results = pd.DataFrame(
    columns=[
        "Filename",
        "Main direction HOG (deg)",
        "Main direction SCHARR (deg)",
        "Main direction SOBEL (deg)",
        "Main direction FIJI FFT (deg)",
        "Main direction FIJI Gradient (deg)",
        "Main direction Ground Truth (deg)",
    ]
)

file_list_run = list(hog_distrib_dict.keys())
if DRAFT_MODE is True:
    file_list_run = file_list_run[::5][:4]  # take every 5th, max 4 files


bin_column_to_read = "Smoothed value" if SMOOTH_MODE else "Binned value"
# depending on which column you want to use for the distribution values

# Plot HOG, FIJI, and Ground Truth histograms in the same loop for each image
for fname in file_list_run:
    # HOG, SCHARR and SOBEL (EDGEHOG) distribution
    data_hog = hog_distrib_dict[fname]
    bins_hog, values_hog = build_distribution_from_data(
        data_hog, direction_colname="Direction (deg)", value_colname=bin_column_to_read
    )

    data_scharr = scharr_distrib_dict[fname]
    bins_scharr, values_scharr = build_distribution_from_data(
        data_scharr,
        direction_colname="Direction (deg)",
        value_colname=bin_column_to_read,
    )

    data_sobel = sobel_distrib_dict[fname]
    bins_sobel, values_sobel = build_distribution_from_data(
        data_sobel,
        direction_colname="Direction (deg)",
        value_colname=bin_column_to_read,
    )

    # FIJI distribution (if available)
    data_fiji_fft = fiji_fft_distrib_dict.get(fname, None)
    data_fiji_gradient = fiji_gradient_distrib_dict.get(fname, None)

    fiji_fft_colname = "Fourier component" if SMOOTH_MODE else "Fourier fit"
    fiji_gradient_colname = (
        "Gradient" if SMOOTH_MODE else "Gradient fit"
    )  # same name in both cases

    bins_fiji_fft, values_fiji_fft = build_distribution_from_data(
        data_fiji_fft,
        direction_colname="Direction °",
        value_colname=fiji_fft_colname,
    )

    # despite the naming, the column of interest has the same name
    # in both FIJI files,
    bins_fiji_gradient, values_fiji_gradient = build_distribution_from_data(
        data_fiji_gradient,
        direction_colname="Direction °",
        value_colname=fiji_gradient_colname,
    )

    # Fiji contains both endpoints, remove the last one after averaging out:
    values_fiji_fft[0] = 0.5 * (values_fiji_fft[0] + values_fiji_fft[-1])  # wrap-around
    bins_fiji_fft = bins_fiji_fft[:-1]
    values_fiji_fft = values_fiji_fft[:-1]

    values_fiji_gradient[0] = 0.5 * (
        values_fiji_gradient[0] + values_fiji_gradient[-1]
    )  # wrap-around
    bins_fiji_gradient = bins_fiji_gradient[:-1]
    values_fiji_gradient = values_fiji_gradient[:-1]

    # Translate direction values by 90 degrees to map [-90, 90) -> [0, 180)
    bins_fiji_fft = bins_fiji_fft % 180
    bins_fiji_gradient = bins_fiji_gradient % 180
    # Sort bins and values for consistency
    sort_idx_fft = np.argsort(bins_fiji_fft)
    bins_fiji_fft = bins_fiji_fft[sort_idx_fft]
    values_fiji_fft = values_fiji_fft[sort_idx_fft]

    sort_idx_gradient = np.argsort(bins_fiji_gradient)
    bins_fiji_gradient = bins_fiji_gradient[sort_idx_gradient]
    values_fiji_gradient = values_fiji_gradient[sort_idx_gradient]

    # Find corresponding ground truth file by matching fname (strip extension if needed)
    # Assume ground truth file contains fname or its stem
    fname_stem = os.path.splitext(fname)[0]
    gt_file_match = None
    for gt_file in ground_truth_files:
        if fname_stem in gt_file:
            gt_file_match = gt_file
            break

    if gt_file_match:
        gt_file_path = os.path.join(
            ROOT_FOLDER, DATA_FOLDER_NAME, INPUT_METADATA, gt_file_match
        )
        df_gt = pd.read_csv(gt_file_path)
        bins_gt, values_gt = directions_to_hist(
            df_gt,
            direction_col="angle_deg",
            frequency_col="visible_segment_length",
            n_bins=44,
        )
    else:
        raise ValueError(f"No matching ground truth file found for image {fname}")

    # Make sure binning is consistent across methods
    assert np.allclose(bins_hog, bins_scharr)
    assert np.allclose(bins_scharr, bins_sobel)
    assert np.allclose(bins_sobel, bins_fiji_fft)
    assert np.allclose(bins_fiji_fft, bins_fiji_gradient)
    assert np.allclose(bins_fiji_gradient, bins_gt)

    bins_common = bins_gt

    wasser_hog_gt = circular_wasserstein_1(values_hog, values_gt, L=180)
    wasser_scharr_gt = circular_wasserstein_1(values_scharr, values_gt, L=180)
    wasser_sobel_gt = circular_wasserstein_1(values_sobel, values_gt, L=180)

    if np.any(values_fiji_fft < 0):
        print(
            f"Warning: Negative values detected in FIJI FFT distribution for image {fname}."
        )
        print("Clipping negative values to zero for Wasserstein distance calculation.")
        values_fiji_fft = np.clip(values_fiji_fft, a_min=0, a_max=None)

    values_fiji_fft = np.clip(values_fiji_fft, a_min=0, a_max=None)

    if np.any(values_fiji_gradient < 0):
        print(
            f"Warning: Negative values detected in FIJI Gradient distribution for image {fname}."
        )
        print("Clipping negative values to zero for Wasserstein distance calculation.")
        values_fiji_gradient = np.clip(values_fiji_gradient, a_min=0, a_max=None)

    wasser_fiji_fft_gt = circular_wasserstein_1(values_fiji_fft, values_gt, L=180)
    wasser_fiji_gradient_gt = circular_wasserstein_1(
        values_fiji_gradient, values_gt, L=180
    )

    main_dir_hog = direction_gaussian_peak_mean(
        bins_hog, values_hog, period_deg=180.0, sigma_bins=0.1
    )

    main_dir_scharr = direction_gaussian_peak_mean(
        bins_scharr, values_scharr, period_deg=180.0, sigma_bins=0.1
    )

    main_dir_sobel = direction_gaussian_peak_mean(
        bins_sobel, values_sobel, period_deg=180.0, sigma_bins=0.1
    )

    main_dir_fiji_fft = direction_gaussian_peak_mean(
        bins_fiji_fft, values_fiji_fft, period_deg=180.0, sigma_bins=0.1
    )
    main_dir_fiji_gradient = direction_gaussian_peak_mean(
        bins_fiji_gradient, values_fiji_gradient, period_deg=180.0, sigma_bins=0.1
    )
    main_dir_truth = direction_gaussian_peak_mean(
        bins_gt, values_gt, period_deg=180.0, sigma_bins=0.1
    )

    perf_results = pd.concat(
        [
            perf_results,
            pd.DataFrame(
                {
                    "Filename": [fname],
                    "Main direction HOG (deg)": [main_dir_hog],
                    "Main direction SCHARR (deg)": [main_dir_scharr],
                    "Main direction SOBEL (deg)": [main_dir_sobel],
                    "Main direction FIJI FFT (deg)": [main_dir_fiji_fft],
                    "Main direction FIJI Gradient (deg)": [main_dir_fiji_gradient],
                    "Main direction Ground Truth (deg)": [main_dir_truth],
                    "Wasserstein HOG vs Ground Truth": [wasser_hog_gt],
                    "Wasserstein SCHARR vs Ground Truth": [wasser_scharr_gt],
                    "Wasserstein SOBEL vs Ground Truth": [wasser_sobel_gt],
                    "Wasserstein FIJI FFT vs Ground Truth": [wasser_fiji_fft_gt],
                    "Wasserstein FIJI Gradient vs Ground Truth": [
                        wasser_fiji_gradient_gt
                    ],
                }
            ),
        ],
        ignore_index=True,
    )

    if DRAFT_MODE is True:
        # Plot HOG
        plt.figure(figsize=(7, 4))
        plt.bar(
            bins_hog,
            values_hog,
            width=(bins_hog[1] - bins_hog[0]) if len(bins_hog) > 1 else 1,
            align="center",
            alpha=0.7,
        )
        plt.title(f"HOG Distribution for {fname}")
        plt.xlabel("Direction (deg)")
        plt.xticks(np.arange(0, 181, 15))
        plt.ylabel("Binned value")
        plt.tight_layout()
        plt.show()

        # Plot SCHARR
        plt.figure(figsize=(7, 4))
        plt.bar(
            bins_scharr,
            values_scharr,
            width=(bins_scharr[1] - bins_scharr[0]) if len(bins_scharr) > 1 else 1,
            align="center",
            alpha=0.7,
        )
        plt.title(f"SCHARR Distribution for {fname}")
        plt.xlabel("Direction (deg)")
        plt.xticks(np.arange(0, 181, 15))
        plt.ylabel("Binned value")
        plt.tight_layout()
        plt.show()

        # Plot SOBEL
        plt.figure(figsize=(7, 4))
        plt.bar(
            bins_sobel,
            values_sobel,
            width=(bins_sobel[1] - bins_sobel[0]) if len(bins_sobel) > 1 else 1,
            align="center",
            alpha=0.7,
        )
        plt.title(f"SOBEL Distribution for {fname}")
        plt.xlabel("Direction (deg)")
        plt.xticks(np.arange(0, 181, 15))
        plt.ylabel("Binned value")
        plt.tight_layout()
        plt.show()

        # Plot FIJI FFT if available
        plt.figure(figsize=(7, 4))
        plt.bar(
            bins_fiji_fft,
            values_fiji_fft,
            width=(
                (bins_fiji_fft[1] - bins_fiji_fft[0]) if len(bins_fiji_fft) > 1 else 1
            ),
            align="center",
            alpha=0.7,
        )
        plt.title(f"FIJI FFT Distribution for {fname}")
        plt.xlabel("Direction (deg)")
        plt.ylabel("Binned value")
        plt.tight_layout()
        plt.show()

        # Plot FIJI Gradient if available
        plt.figure(figsize=(7, 4))
        plt.bar(
            bins_fiji_gradient,
            values_fiji_gradient,
            width=(
                (bins_fiji_gradient[1] - bins_fiji_gradient[0])
                if len(bins_fiji_gradient) > 1
                else 1
            ),
            align="center",
            alpha=0.7,
        )
        plt.title(f"FIJI Gradient Distribution for {fname}")
        plt.xlabel("Direction (deg)")
        plt.ylabel("Binned value")
        plt.tight_layout()
        plt.show()


# Append average row for Wasserstein columns
wasser_cols = [
    "Wasserstein HOG vs Ground Truth",
    "Wasserstein SCHARR vs Ground Truth",
    "Wasserstein SOBEL vs Ground Truth",
    "Wasserstein FIJI FFT vs Ground Truth",
    "Wasserstein FIJI Gradient vs Ground Truth",
]
avg_row = {col: perf_results[col].mean() for col in wasser_cols if col in perf_results}
avg_row.update(
    {
        col: ""
        for col in perf_results.columns
        if col not in wasser_cols and col != "Filename"
    }  # type: ignore
)
avg_row["Filename"] = "average"  # type: ignore
perf_results.loc[len(perf_results)] = avg_row  # type: ignore

# Save performance results to CSV
output_dir = os.path.join(ROOT_FOLDER, DATA_FOLDER_NAME, OUTPUT_FOLDER)

perf_filename = (
    f"performance_comparison_{suffix_filename}_{CORRECT_EDGES[0]}_{CORRECT_EDGES[1]}.csv"
    if DRAFT_MODE is False
    else f"performance_comparison_draft_{suffix_filename}_{CORRECT_EDGES[0]}_{CORRECT_EDGES[1]}.csv"
)

perf_results.to_csv(os.path.join(output_dir, perf_filename), index=False)
