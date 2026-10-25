import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from skimage import io

from main_code.cli_run_wrapper import run_pipeline
from main_code.get_default_params import get_folder_winsize


@pytest.fixture
def input_dir(tmp_path):
    folder = tmp_path / "inputs"
    folder.mkdir()
    y, x = np.indices((96, 128))
    image = ((np.sin((x + y) / 3) + 1) * 60 + x / 2).astype(np.uint8)
    image[:48, :64] = 0
    io.imsave(folder / "sample.png", image, check_contrast=False)
    return folder


@pytest.mark.parametrize("method", ["hog", "scharr", "sobel"])
def test_installed_cli_batch_from_another_directory(input_dir, tmp_path, method):
    output = tmp_path / method
    args = [
        sys.executable,
        "-m",
        "edgehog",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output),
        "--window-size",
        "16",
        "--method",
        method,
    ]
    if method != "hog":
        args.append("--no-plots")
    result = subprocess.run(
        args,
        cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    raw = pd.read_csv(output / "HOG_stats_None_16pixels.csv")
    assert len(raw) == 1
    assert np.isfinite(raw["avg. direction"]).all()
    assert np.isfinite(raw["std. deviation (mean)"]).all()
    assert (output / "HOG_stats_None_16pixels_clean.csv").is_file()
    assert (output / "distribution_statistics.csv").is_file()
    assert (output / "distribution_stats_vertical_format.csv").is_file()
    assert bool(list(output.glob("*.png"))) == (method == "hog")


def test_old_data_dir_layout_and_output_protection(input_dir, tmp_path):
    input_dir.rename(tmp_path / "input_images")
    result = run_pipeline(data_dir=tmp_path, window_size=16, save_plots=False)
    assert result.is_file()
    before = result.read_bytes()
    with pytest.raises(ValueError, match="must be empty"):
        run_pipeline(data_dir=tmp_path, window_size=16, save_plots=False)
    assert result.read_bytes() == before


def test_packaged_defaults():
    assert get_folder_winsize("anything") == 64
    assert get_folder_winsize("synthetic-golden-standard") == 32


def test_empty_input_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="No .tif"):
        run_pipeline(input_dir=tmp_path, output_dir=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_zero_threshold_is_rejected_before_processing(input_dir, tmp_path):
    with pytest.raises(ValueError, match="greater than 0"):
        run_pipeline(input_dir=input_dir, output_dir=tmp_path / "out", threshold=0)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("shape", [(16, 64), (64, 16)])
def test_hog_pipeline_preserves_single_cell_axis(tmp_path, shape):
    folder = tmp_path / "input"
    folder.mkdir()
    y, x = np.indices(shape)
    image = ((x + y) % 16 * 10).astype(np.uint8)
    if shape[0] == 16:
        image[:, :32] = 0
    else:
        image[:32, :] = 0
    io.imsave(folder / "small.png", image, check_contrast=False)
    result = run_pipeline(
        input_dir=folder,
        output_dir=tmp_path / "output",
        window_size=16,
        save_plots=False,
    )
    assert result.is_file()
    stats = pd.read_csv(result)
    assert len(stats) == 1
    assert np.isfinite(stats["avg. direction"]).all()
