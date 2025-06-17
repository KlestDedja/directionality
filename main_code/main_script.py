import os
from main_code.hog_analysis_script import HOGAnalysis
from main_code.postprocess_csv_files import postprocess_hog_csv

from main_code.get_default_params import (
    get_folder_cellsize,
    get_folder_threshold,
    get_folder_channel,
    get_background_range,
)

# ===== DEVELOPER SETTINGS ==========
DRAFT_MODE = True
BLOCK_NORM = None  # or "L2-Hys"
VERBOSE = 1  # higher value -> printing more debug messages

# ========== USER SETTINGS ==========
CELL_SIZE = 64  # pixels per cell for HOG descriptor
CHANNEL = 1  # channel color for HOG descriptor

SAVE_STATS = True  # save statistics to CSV
SAVE_PLOTS = False  # save ouctcomes of directionality analysis
SHOW_PLOTS = False  # show plots interactively

# ========== FOLDER STRUCTURE ==========
ROOT_FOLDER = os.getcwd()

# DATA_FOLDER_NAME = "images-confocal-20241022-fusion-bMyoB-BAMS-TM-6w"
# DATA_FOLDER_NAME = "images-good-bad-validation"
# DATA_FOLDER_NAME = "images-confocal-20241116-fusion-bMyoB-PEMFS-TM-12w"
DATA_FOLDER_NAME = "images-lightsheet-20241115_BAM_fkt20-P3-fkt21-P3-PEMFS-12w"

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
    BG_RANGE = get_background_range(image_folder_path)

    if VERBOSE > 0:
        print(f"Input folder:\n{image_folder_path}")
        print(f"Using threshold={THRESHOLD}")
        print(f"Using cell size={CELL_SIZE}")
        print(f"Using channel={CHANNEL}")
        print(f"Using background range=({100*BG_RANGE[0]}%, {100*BG_RANGE[1]}%)")

    hog_runner = HOGAnalysis(
        input_folder=image_folder_path,
        output_folder=output_folder_path,
        block_norm=BLOCK_NORM,
        pixels_per_cell=(CELL_SIZE, CELL_SIZE),
        channel_image=CHANNEL,
        background_range=BG_RANGE,
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

    # filename = (
    #     f"HOG_stats_{BLOCK_NORM}_{CELL_SIZE}pixels"  # manual? Can we rather retrieve?
    # )

    stats_file = hog_runner.saved_stats_path
    df, clean_csv_path = postprocess_hog_csv(stats_file)
    df.to_csv(clean_csv_path, index=False)

    # if DRAFT_MODE:
    #     filename += "_draft2"
    # csv_path = os.path.join(output_folder_path, f"{filename}.csv")

    # df, clean_csv_path = postprocess_hog_csv(csv_path)
    # df.to_csv(clean_csv_path, index=False)
