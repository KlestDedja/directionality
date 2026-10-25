"""Command-line interface for folder-based EDGEHOG analysis."""

import argparse
from pathlib import Path

from main_code.get_default_params import (
    get_background_range,
    get_folder_channel,
    get_folder_threshold,
    get_folder_winsize,
)
from main_code.hog_analysis_script import HOGAnalysis
from main_code.postprocess_csv_files import postprocess_hog_csv


def run_pipeline(
    data_dir: str | Path | None = None,
    block_norm: str | None = "None",
    window_size: int | None = None,
    channel_axis: int | str | None = None,
    draft: bool = False,
    *,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    threshold: float | None = None,
    method: str = "hog",
    num_bins: int = 45,
    save_plots: bool = True,
    post_normalization: bool = True,
    smoothing_sigma: float = 1.0,
) -> Path:
    """Analyze a folder and return the cleaned statistics CSV path.

    ``data_dir`` retains the old wrapper's input_images/output_analysis layout.
    New callers can supply explicit input_dir and output_dir paths instead.
    Folder-specific defaults are read from the packaged default_params.csv.
    """
    if data_dir is not None:
        if input_dir is not None or output_dir is not None:
            raise ValueError("Use data_dir or explicit input_dir/output_dir, not both")
        input_dir = Path(data_dir) / "input_images"
        output_dir = Path(data_dir) / "output_analysis"
    if input_dir is None or output_dir is None:
        raise ValueError("Provide input_dir and output_dir (or data_dir)")
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not any(
        p.suffix.lower() in {".tif", ".png", ".jpg"} for p in input_dir.iterdir() if p.is_file()
    ):
        raise ValueError(f"No .tif, .png or .jpg images found in {input_dir}")
    # Distribution exports append rows, so each CLI run needs a fresh folder.
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory must be empty; choose a new folder: {output_dir}")
    folder = str(input_dir)
    window_size = get_folder_winsize(folder) if window_size is None else window_size
    channel = get_folder_channel(folder) if channel_axis is None else channel_axis
    channel = _channel(str(channel))
    threshold = get_folder_threshold(folder) if threshold is None else threshold
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    # The adaptive threshold loop scales this value multiplicatively, so zero
    # cannot converge when every cell initially contains signal.
    if not 0 < threshold <= 100:
        raise ValueError("threshold must be greater than 0 and at most 100")
    if not 0 <= smoothing_sigma < float("inf"):
        raise ValueError("smoothing_sigma must be finite and nonnegative")
    if method not in {"hog", "scharr", "sobel"}:
        raise ValueError("method must be hog, scharr or sobel")
    if block_norm is None:
        block_norm = "None"
    if block_norm not in {"None", "L1", "L1-sqrt", "L2", "L2-Hys"}:
        raise ValueError("Unsupported block normalization")
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = HOGAnalysis(
        input_folder=folder,
        output_folder=str(output_dir),
        block_norm=block_norm,
        pixels_per_window=(window_size, window_size),
        channel_image=channel,
        background_range=get_background_range(folder),
        draft=draft,
        post_normalization=post_normalization,
        correct_edge_angles=("gaussian", str(smoothing_sigma)),
        num_bins=num_bins,
        method=method,
    )
    runner.process_folder(
        image_folder=folder,
        output_filename=f"HOG_stats_{block_norm}_{window_size}pixels",
        threshold=threshold,
        save_stats=True,
        save_plots=save_plots,
    )
    df, clean_csv = postprocess_hog_csv(runner.saved_stats_path, verbose=0)
    df.to_csv(clean_csv, index=False)
    return Path(clean_csv)


def _channel(value: str) -> int | str:
    if value.lower() in {"grayscale", "greyscale"}:
        return "grayscale"
    try:
        channel = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use grayscale or a channel index (0, 1, 2, -1)") from exc
    if channel not in {-1, 0, 1, 2}:
        raise argparse.ArgumentTypeError("Channel index must be 0, 1, 2 or -1")
    return channel


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze image directionality with EDGEHOG.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path, help="Folder of .tif, .png and .jpg images")
    source.add_argument(
        "--data-dir",
        "--data_dir",
        type=Path,
        help="Dataset folder containing input_images; saves to output_analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New or empty output folder (required with --input-dir)",
    )
    parser.add_argument("--method", choices=["hog", "scharr", "sobel"], default="hog")
    parser.add_argument(
        "--window-size",
        "--window_size",
        type=int,
        help="Window side in pixels (uses folder default)",
    )
    parser.add_argument(
        "--channel",
        "--channel_image",
        type=_channel,
        help="grayscale or 0=red, 1=green, 2=blue, -1=last",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Signal threshold greater than 0 and at most 100 (uses folder default)",
    )
    parser.add_argument("--bins", type=int, default=45)
    parser.add_argument(
        "--block-norm",
        "--block_norm",
        choices=["None", "L1", "L1-sqrt", "L2", "L2-Hys"],
        default="None",
    )
    parser.add_argument("--smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--no-plots", action="store_true", help="Save CSV results only")
    parser.add_argument("--no-post-normalization", action="store_true")
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Process at most 10 images when the folder contains more than 15",
    )
    args = parser.parse_args(argv)
    if args.input_dir is not None and args.output_dir is None:
        parser.error("--output-dir is required with --input-dir")
    if args.data_dir is not None and args.output_dir is not None:
        parser.error(
            "--data-dir already chooses output_analysis; use --input-dir with --output-dir"
        )
    try:
        result = run_pipeline(
            data_dir=args.data_dir,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            method=args.method,
            window_size=args.window_size,
            channel_axis=args.channel,
            threshold=args.threshold,
            num_bins=args.bins,
            block_norm=args.block_norm,
            draft=args.draft,
            save_plots=not args.no_plots,
            post_normalization=not args.no_post_normalization,
            smoothing_sigma=args.smoothing_sigma,
        )
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    print(f"Saved cleaned statistics to {result}")


if __name__ == "__main__":
    main()
