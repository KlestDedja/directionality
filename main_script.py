"""
This is the main script for running EDGEHOG
It is assumed that the script is located in the project root folder
"""

import os
import time
from main_code.hog_analysis_script import HOGAnalysis
from main_code.postprocess_csv_files import postprocess_hog_csv

from main_code.get_default_params import (
    get_folder_winsize,
    get_folder_threshold,
    get_folder_channel,
    get_background_range,
)

# ===== DEVELOPER SETTINGS ==========
DRAFT_MODE = False
BLOCK_NORM = "None"  # or "L2-Hys"
VERBOSE = 1  # higher value -> printing more debug messages

# ========== USER SETTINGS ==========
# WINDOW_SIZE = 32  # pixels per window for HOG descriptor
CHANNEL = 1  # channel color for HOG descriptor
POST_NORMALIZATION = True  # normalize color brightness across windows
N_BINS = 45
# two entries, each 'interpolate' or 'None'
# first for 90-degree correction, second for 45-degree correction, correction at 0 = correction at 90
CORRECT_EDGES = ("none", "none")
# CORRECT_EDGES = ("interpolate", "interpolate")
CORRECT_EDGES = ("gaussian", "1")

# Choose method: 'scharr' or 'hog'
# METHOD = "scharr"
METHOD = "hog"
METHOD = "sobel"

SAVE_STATS = True  # save statistics to CSV
SAVE_PLOTS = True  # save ouctcomes of directionality analysis
SHOW_PLOTS = False  # show plots interactively

# ========== FOLDER STRUCTURE ==========
ROOT_FOLDER = os.getcwd()

# change accordingly if your structure differs from the demo
# DATA_FOLDER_NAME = os.path.join("data", "synthetic-golden-standard")
DATA_FOLDER_NAME = os.path.join("data", "test-golden")

DATA_FOLDER_NAME = os.path.join("data", "test-fibers")
# DATA_FOLDER_NAME = os.path.join("data", "images-good-bad-validation")
# DATA_FOLDER_NAME = os.path.join("data", "images-photoalignment-tool3")
# DATA_FOLDER_NAME = os.path.join(
#     "data", "images-lightsheet-20241115_BAM_fkt20-P3-fkt21-P3-PEMFS-12w"
# )
DATA_FOLDER_NAME = os.path.join(
    "data", "images-lightsheet-20240928_BAM_fkt20_P3_fkt21_P3_PEMFS"
)
# DATA_FOLDER_NAME = os.path.join(
#     "data", "images-confocal-20241022-fusion-bMyoB-BAMS-TM-6w"
# )
# DATA_FOLDER_NAME = os.path.join(
#     "data", "images-confocal-20241116-fusion-bMyoB-PEMFS-TM-12w"
# )

INPUT_FOLDER = "input-images"
# OUTPUT_FOLDER will be constructed at runtime to include num_bins, method and interpolation type
OUTPUT_FOLDER = "output-analysis"

# ========== RUN ANALYSIS ==========

t0 = time.time()

if __name__ == "__main__":

    data_folder_path = os.path.join(ROOT_FOLDER, DATA_FOLDER_NAME)

    image_folder_path = os.path.join(data_folder_path, INPUT_FOLDER)

    # create a method-specific subfolder inside the base output folder
    INTERPOLATION_STR = f"{CORRECT_EDGES[0][:5]}_{str(CORRECT_EDGES[1])[:5]}"
    SUBFOLDER_NAME = f"{METHOD}_{N_BINS}bins_{INTERPOLATION_STR}"

    output_folder_path = os.path.join(data_folder_path, OUTPUT_FOLDER, SUBFOLDER_NAME)

    # fetch per-folder defaults
    THRESHOLD = get_folder_threshold(image_folder_path)
    WINDOW_SIZE = get_folder_winsize(image_folder_path)
    CHANNEL = get_folder_channel(image_folder_path)
    BG_RANGE = get_background_range(image_folder_path)

    if VERBOSE > 0:
        print(f"Input folder:\n{image_folder_path}")
        print(f"Method: {METHOD}")
        print(f"Threshold={THRESHOLD}")
        print(f"Window size={WINDOW_SIZE}")
        print(f"Channel={CHANNEL}")
        print(f"Background range=({100*BG_RANGE[0]}%, {100*BG_RANGE[1]}%)")
        print(f"Staining normalization: {POST_NORMALIZATION}")
        print(f"Correction edge angles: {CORRECT_EDGES}")

    hog_runner = HOGAnalysis(
        input_folder=image_folder_path,
        output_folder=output_folder_path,
        block_norm=BLOCK_NORM,
        pixels_per_window=(WINDOW_SIZE, WINDOW_SIZE),
        channel_image=CHANNEL,
        background_range=BG_RANGE,
        draft=DRAFT_MODE,
        show_plots=SHOW_PLOTS,
        post_normalization=POST_NORMALIZATION,
        correct_edge_angles=CORRECT_EDGES,
        num_bins=N_BINS,
        method=METHOD,
    )

    FILENAME = f"HOG_stats_{BLOCK_NORM}_{WINDOW_SIZE}pixels"

    hog_runner.process_folder(
        image_folder=image_folder_path,
        output_filename=FILENAME,
        threshold=THRESHOLD,
        save_stats=SAVE_STATS,
        save_plots=SAVE_PLOTS,
        verbose=VERBOSE,
    )

    t1 = time.time()
    print(
        f"Total processing time for {DATA_FOLDER_NAME} with {METHOD}:\n{t1 - t0:.3f} seconds"
    )
    runtime_message = f"Total processing time for {DATA_FOLDER_NAME} with {METHOD}:\n{t1 - t0:.3f} seconds"
    print(runtime_message)

    # Save runtime to file
    runtime_file = os.path.join(output_folder_path, f"runtime_{METHOD}.txt")
    with open(runtime_file, "w", encoding="utf-8") as f:
        f.write(runtime_message)

    # ========== POSTPROCESS RESULTS FILE ==========

    stats_file = hog_runner.saved_stats_path
    df, clean_csv_path = postprocess_hog_csv(stats_file, "_clean", verbose=0)
    df.to_csv(clean_csv_path, index=False)
    if VERBOSE > 0:
        print(f"Stored final results in {clean_csv_path}")
