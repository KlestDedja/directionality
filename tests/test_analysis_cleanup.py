"""Regression checks for histogram exports and the analysis normalization contract."""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from main_code.hog_analysis_script import HOGAnalysis
from main_code.utils_other import clean_filename, set_sample_condition, set_sample_replicate

EXPECTED = json.loads(Path(__file__).with_name("analysis_expected.json").read_text())


def make_analysis(tmp_path, **kwargs):
    return HOGAnalysis(
        input_folder=str(tmp_path),
        output_folder=str(tmp_path / "results"),
        pixels_per_window=(8, 8),
        channel_image="grayscale",
        background_range=(0, 1),
        correct_edge_angles=("gaussian", "1"),
        num_bins=9,
        **kwargs,
    )


def test_histogram_csv_append_and_replace(tmp_path):
    analysis = make_analysis(tmp_path)
    output = tmp_path / "results"
    histogram = {0.0: 2.5, 90.0: 4.5}
    analysis.distribution_stats_to_temp_csv("first.tif", histogram)
    analysis.distribution_stats_to_temp_csv("second.tif", {0.0: 3.5, 90.0: 5.5})
    expected = pd.DataFrame(
        {"filename": ["first.tif", "second.tif"], "0.0": [2.5, 3.5], "90.0": [4.5, 5.5]}
    )
    temporary = output / "distribution_statistics.temp"
    pd.testing.assert_frame_equal(pd.read_csv(temporary), expected)
    final = output / "distribution_statistics.csv"
    final.write_text("previous results\n")
    analysis.save_distribution_stats_to_csv()
    assert not temporary.exists()
    pd.testing.assert_frame_equal(pd.read_csv(final), expected)
    analysis.save_distribution_stats_to_csv()  # No new histograms: keep the final CSV.
    pd.testing.assert_frame_equal(pd.read_csv(final), expected)


@pytest.mark.parametrize("histogram", [{0.0: 1.0}, {0.0: 1.0, 45.0: 2.0}, {90.0: 2.0, 0.0: 1.0}])
def test_incompatible_bin_columns_do_not_append(tmp_path, histogram):
    analysis = make_analysis(tmp_path)
    analysis.distribution_stats_to_temp_csv("first.tif", {0.0: 1.0, 90.0: 2.0})
    path = tmp_path / "results" / "distribution_statistics.temp"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Bin columns for image second.tif"):
        analysis.distribution_stats_to_temp_csv("second.tif", histogram)
    assert path.read_bytes() == before


def test_failed_finalization_preserves_previous_results(tmp_path, monkeypatch):
    analysis = make_analysis(tmp_path)
    analysis.distribution_stats_to_temp_csv("first.tif", {0.0: 1.0})
    final = tmp_path / "results" / "distribution_statistics.csv"
    final.write_text("previous results\n")

    def deny_replace(*args):
        raise PermissionError("File is open")

    monkeypatch.setattr("main_code.hog_analysis_script.os.replace", deny_replace)
    with pytest.raises(PermissionError):
        analysis.save_distribution_stats_to_csv()
    assert final.read_text() == "previous results\n"
    assert final.with_suffix(".temp").exists()


def test_folder_accepts_byte_paths_and_selects_lightsheet_slices(tmp_path):
    folder = tmp_path / "images-lightsheet"
    folder.mkdir()
    for name in ("image_z01.tif", "image_z10.tif", "image.tif"):
        (folder / name).touch()
    analysis = make_analysis(tmp_path)
    processed = []
    analysis.process_image = lambda folder, filename, *args: processed.append(filename)
    analysis.process_folder(os.fsencode(folder), b"stats", 3, save_stats=False, save_plots=False)
    assert set(processed) == {"image_z10.tif", "image.tif"}


def test_lightsheet_slice_without_number_reports_filename(tmp_path):
    folder = tmp_path / "images-lightsheet"
    folder.mkdir()
    (folder / "image_z.tif").touch()
    analysis = make_analysis(tmp_path)
    with pytest.raises(ValueError, match="Missing z slice number.*image_z.tif"):
        analysis.process_folder(str(folder), "stats", 3, save_stats=False, save_plots=False)


@pytest.mark.parametrize("method", ["scharr", "sobel", "hog"])
@pytest.mark.parametrize("post_normalization", [True, False])
@pytest.mark.parametrize("block_norm", ["L1", "L1-sqrt", "L2", "L2-Hys"])
def test_statistics_follow_post_normalization(
    tmp_path, monkeypatch, method, post_normalization, block_norm
):
    """Block normalization was overwritten before the cleanup; preserve those outputs."""
    image = np.random.default_rng(42).random((32, 32))
    monkeypatch.setattr("main_code.hog_analysis_script.io.imread", lambda _: image)
    # The installability PR owns support for unnormalized HOG. A deterministic
    # descriptor here exercises this pipeline without requiring the custom fork.
    descriptor = np.arange(1, 145, dtype=float).reshape(4, 4, 1, 1, 9)

    def fake_hog(self, image, **kwargs):
        self.fd = descriptor.copy()
        self.hog_image = image.copy()
        return self.fd, self.hog_image

    monkeypatch.setattr("main_code.pipeline_utils.HOGDescriptor.compute_hog", fake_hog)
    captured = []
    for index, normalization in enumerate(("None", block_norm)):
        analysis = make_analysis(
            tmp_path / str(index),
            method=method,
            post_normalization=post_normalization,
            block_norm=normalization,
        )
        plots = []
        analysis.save_plot = lambda *args: plots.append(args)
        analysis.process_image(str(tmp_path), "image.tif", threshold=50, save_plots=True)
        assert len(plots) == 1
        stats = analysis.df_statistics.drop(columns="elapsed time (mm:ss)")
        # Captured from main at ed2a148 before the cleanup (NumPy 2.5.4,
        # SciPy 1.18.1, scikit-image 0.25.2). Allow numeric roundoff across versions.
        expected = EXPECTED[f"{method}_{post_normalization}"]
        pd.testing.assert_frame_equal(
            stats,
            pd.DataFrame([expected["statistics"]], index=["image.tif"]),
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            list(plots[0][2].values()),
            expected["histogram"],
            rtol=1e-10,
            atol=1e-12,
        )
        histogram_path = tmp_path / str(index) / "results" / "distribution_statistics.temp"
        captured.append((stats, pd.read_csv(histogram_path), plots[0]))
    pd.testing.assert_frame_equal(captured[0][0], captured[1][0])
    pd.testing.assert_frame_equal(captured[0][1], captured[1][1])
    for first, second in zip(captured[0][2][0:2], captured[1][2][0:2]):
        np.testing.assert_array_equal(first, second)


def test_filename_metadata_keeps_specific_match_precedence():
    filename = "20241022  CTR_fkt21_R2_10'Stim-5'Break.tif"
    assert clean_filename(filename) == "CTR_fkt21_R2_10'Stim-5'Break.tif"
    assert set_sample_condition(filename) == "10m-5m"
    assert set_sample_replicate(filename) == "R2"
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert set_sample_condition("unknown.tif", verbose=0) == "Unknown"
        assert set_sample_replicate("unknown.tif", verbose=0) == "Unknown"
    assert not recorded
    with pytest.warns(UserWarning, match="Condition unknown"):
        set_sample_condition("unknown.tif", verbose=1)
    with pytest.warns(UserWarning, match="Unknown replicate"):
        set_sample_replicate("unknown.tif", verbose=1)
