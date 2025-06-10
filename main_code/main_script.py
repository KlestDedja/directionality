# main_script.py

import os
from main_code.hog_analysis_script import HOGAnalysis
from main_code.postprocess_csv_files import postprocess_hog_csv

from main_code.default_params import (
    get_folder_cellsize,
    get_folder_threshold,
    get_folder_channel,
)

# ========== USER SETTINGS ==========

DRAFT_MODE = False
BLOCK_NORM = None  # or "L2-Hys"
CELL_SIZE = 64  # pixels per cell for HOG descriptor
CHANNEL = -1  # channel axis for HOG descriptor

SAVE_STATS = True  # save statistics to CSV
SAVE_PLOTS = True  # save ouctcomes of directionality analysis
SHOW_PLOTS = False  # show plots interactively

# ========== FOLDER STRUCTURE ==========
ROOT_FOLDER = os.getcwd()

DATA_FOLDER_NAME = "images-confocal-20241022-fusion-bMyoB-BAMS-TM-6w"
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_analysis"

# ========== RUN ANALYSIS ==========
if __name__ == "__main__":

    data_folder = os.path.join(ROOT_FOLDER, "data", DATA_FOLDER_NAME)

    image_folder_path = os.path.join(data_folder, INPUT_FOLDER)
    output_folder_path = os.path.join(data_folder, OUTPUT_FOLDER)

    # fetch per-folder defaults
    THRESHOLD = get_folder_threshold(image_folder_path)
    CELL_SIZE = get_folder_cellsize(image_folder_path)
    CHANNEL = get_folder_channel(image_folder_path)

    hog_runner = HOGAnalysis(
        input_folder=image_folder_path,
        output_folder=output_folder_path,
        block_norm=BLOCK_NORM,
        pixels_per_cell=(CELL_SIZE, CELL_SIZE),
        channel_axis=CHANNEL,
        draft=DRAFT_MODE,
        show_plots=SHOW_PLOTS,
    )

    hog_runner.process_folder(
        image_folder=image_folder_path,
        threshold=THRESHOLD,
        save_stats=SAVE_STATS,
        save_plots=SAVE_PLOTS,
    )

    # ========== POSTPROCESS RESULTS FILE ==========

    filename = (
        f"HOG_stats_{BLOCK_NORM}_{CELL_SIZE}pixels"  # manual? Can we rather retrieve?
    )
    if DRAFT_MODE:
        filename += "_draft"
    csv_path = os.path.join(output_folder_path, f"{filename}.csv")

    df, clean_csv_path = postprocess_hog_csv(csv_path)
    df.to_csv(clean_csv_path, index=False)
