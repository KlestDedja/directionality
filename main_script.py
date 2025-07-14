"""
This is the main script for running EDGEHOG
It is assumed that the script is localted in the project root folder
"""

import os

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
# WINDOW_SIZE = 64  # pixels per window for HOG descriptor
# CHANNEL = 1  # channel color for HOG descriptor
POST_NORMALIZATION = False  # normalize color brightness across windows
N_BINS = 45

MAX_MAIN_DIRECTIONS = 1
MIN_DIRECTION_GAP = 20
ENFORCE_LOCAL_MAXIMA = True

SAVE_STATS = True  # save statistics to CSV
SAVE_PLOTS = True  # save ouctcomes of directionality analysis
SHOW_PLOTS = False  # show plots interactively

# ========== FOLDER STRUCTURE ==========
ROOT_FOLDER = os.getcwd()

# change accordingly if your structure differs from the demo
DATA_FOLDER_NAME = "data"
# SUBFOLDER_NAME = "images-longitudinal"
SUBFOLDER_NAME = "images-fotosalignement-tool3"

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_analysis_notnorm"

# ========== RUN ANALYSIS ==========
if __name__ == "__main__":

    data_folder = os.path.join(ROOT_FOLDER, DATA_FOLDER_NAME, SUBFOLDER_NAME)

    image_folder_path = os.path.join(data_folder, INPUT_FOLDER)
    output_folder_path = os.path.join(data_folder, OUTPUT_FOLDER)

    # fetch per-folder defaults
    THRESHOLD = get_folder_threshold(image_folder_path)
    WINDOW_SIZE = get_folder_winsize(image_folder_path)
    CHANNEL = get_folder_channel(image_folder_path)
    BG_RANGE = get_background_range(image_folder_path)

    if VERBOSE > 0:
        print(f"Input folder:\n{image_folder_path}")
        print(f"Using threshold={THRESHOLD}")
        print(f"Using window size={WINDOW_SIZE}")
        print(f"Using channel={CHANNEL}")
        print(f"Using background range=({100*BG_RANGE[0]}%, {100*BG_RANGE[1]}%)")
        print(f"Staining normalization: {POST_NORMALIZATION}")
        print(f"Maximum main directions: {MAX_MAIN_DIRECTIONS}")
        print(f"Minimum gap between main directions: {MIN_DIRECTION_GAP} degrees")
        print(f"Enforcing local maxima: {ENFORCE_LOCAL_MAXIMA}")

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
        num_bins=N_BINS,
        max_main_directions=MAX_MAIN_DIRECTIONS,
        min_direction_gap=MIN_DIRECTION_GAP,
        enforce_local_maxima=ENFORCE_LOCAL_MAXIMA,
    )

    filename = f"HOG_stats_{BLOCK_NORM}_{WINDOW_SIZE}pixels"

    hog_runner.process_folder(
        image_folder=image_folder_path,
        output_filename=filename,
        threshold=THRESHOLD,
        save_stats=SAVE_STATS,
        save_plots=SAVE_PLOTS,
    )

    # ========== POSTPROCESS RESULTS FILE ==========

    stats_file = hog_runner.saved_stats_path
    if VERBOSE > 0:
        print(f"Storing final cvs results in {stats_file}")
    df, clean_csv_path = postprocess_hog_csv(stats_file, "_clean")
    df.to_csv(clean_csv_path, index=False)
