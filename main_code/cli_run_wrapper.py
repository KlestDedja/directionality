import argparse
import os
from main_code.hog_analysis_script import HOGAnalysis
from main_code.postprocess_csv_files import postprocess_hog_csv

### TODO: STILL DRAFTY


def run_pipeline(data_dir, block_norm, window_size, channel_axis, draft):
    input_dir = os.path.join(data_dir, "input_images")
    output_dir = os.path.join(data_dir, "output_analysis")

    hog_runner = HOGAnalysis(
        input_folder=input_dir,
        output_folder=output_dir,
        block_norm=block_norm,
        pixels_window=(window_size, window_size),
        channel_image=channel_axis,
        draft=draft,
    )

    filename = f"HOG_stats_{BLOCK_NORM}_{WINDOW_SIZE}pixels"

    hog_runner.process_folder(
        image_folder=input_dir,
        output_filename=filename,
        threshold=THRESHOLD,
        save_stats=SAVE_STATS,
        save_plots=SAVE_PLOTS,
    )

    # Compose CSV filename
    fname = f"HOG_stats_input_images_{block_norm}_{window_size}pixels"
    if draft:
        fname += "_draft"
    raw_csv = os.path.join(output_dir, fname + ".csv")

    # Postprocess CSV
    postprocess_hog_csv(raw_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HOG directionality analysis.")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the folder containing 'input_images' and 'output_analysis'",
    )
    parser.add_argument(
        "--block_norm",
        type=str,
        default=None,
        help="Block normalization type for HOG (e.g. 'L2-Hys', or leave blank for None)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="Pixels per cell in HOG descriptor (default: 64)",
    )
    parser.add_argument(
        "--channel_image",
        type=int,
        default=1,
        help="Channel axis for HOG (default: 1)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=3,
        help="Threshold for HOG processing (default: 3)",
    )

    parser.add_argument(
        "--save_stats",
        action="store_true",
        help="If set, statistics will be saved.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="If set, plots will be saved.",
    )

    parser.add_argument(
        "--draft",
        type=bool,
        default=False,
        action="store_true",
        help="If set, only a subset of images will be processed.",
    )

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        block_norm=args.block_norm,
        window_size=args.window_size,
        channel_axis=args.channel_axis,
        draft=args.draft,
    )
