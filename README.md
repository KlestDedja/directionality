<a name="logo-anchor"></a>
<p align="center">
<img src="directionality-edgehog-logo.png?raw=true" alt="Directionality Logo" width="25%"/>
</p>

# Directionality with EDGEHOG

**EDGEHOG** analyzes fiber directionality in images using Histograms of Oriented
Gradients (HOG), Scharr or Sobel gradients. It produces orientation distributions,
mean and modal directions, dispersion statistics, CSV exports and optional plots.
A manuscript describing the method and biomedical applications is in preparation.

## Installation

Use Python 3.10 or newer. Python 3.12 is recommended for reproducing the demo.
Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install directly from GitHub (requires Git):

```bash
python -m pip install "git+https://github.com/KlestDedja/directionality.git"
```

Or clone the repository to get the demo images, then install from its root:

```bash
git clone https://github.com/KlestDedja/directionality.git
cd directionality
python -m pip install .
```

The package installs standard PyPI dependencies, including scikit-image
0.23–0.25. **No custom fork, compiler, or manual changes to scikit-image are
required.** The distribution name is `edgehog-directionality`; the Python
import and command are `edgehog`. It is not currently published on PyPI.

Existing `python -m pip install -r requirements.txt` also installs the package.
For conda, run `conda env create -f environment.yaml` from the repository root,
then `conda activate directionality`. Notebook users can add
`python -m pip install ".[notebook]"`.

## Run the demo

From the cloned repository:

```bash
edgehog --input-dir demo-data/synthetic-golden-standard/input_images --output-dir results/demo-hog --method hog --draft
```

`--draft` samples up to 10 images when the input folder has more than 15.
Omit it to process all images. Use `--no-plots` for CSV output only.
Choose a new or empty output folder for each run: distribution exports append
rows, so reusing a populated folder could mix results.

The installed command works from any directory. Pass full paths for your own data:

```bash
edgehog --input-dir /path/to/images --output-dir /path/to/new-results --method hog --channel grayscale --window-size 32
```

Use `--channel 0`, `1`, or `2` for an RGB channel; the default uses grayscale
unless a dataset override applies. Supported images are `.tif`, `.png`, and
`.jpg`. Folder-specific defaults are bundled in `main_code/default_params.csv`.
Run `edgehog --help` for threshold, bin, normalization, and smoothing options.

`python -m edgehog` provides the same interface. The editable
`python main_script.py` example also runs the bundled synthetic dataset.
The legacy wrapper accepts `--data-dir` / `--data_dir` for a dataset containing
`input_images/`, writing to `output_analysis/`.

Outputs include raw and cleaned `HOG_stats_*.csv` summary statistics,
`distribution_statistics.csv`, a vertical distribution CSV, and optional
per-image plots. Images without sufficient signal retain NaN statistics in the
raw CSV and are excluded from the cleaned summary.

## Python API and raw HOG

```python
from edgehog import HOGAnalysis, HOGDescriptor

descriptor = HOGDescriptor(
    orientations=45,
    pixels_per_window=(32, 32),
    windows_per_block=(1, 1),
)
# image is a 2D NumPy array after selecting a channel.
raw_histograms, visualization = descriptor.compute_hog(image, block_norm="None")
```

Existing imports from `main_code` remain supported. Both Python `None` and
the legacy string `"None"` request raw magnitudes. EDGEHOG computes these locally
using NumPy and public scikit-image APIs, preserving the fork's histogram and
visualization conventions. `L1`, `L1-sqrt`, `L2`, and `L2-Hys` continue to use
upstream scikit-image. No installed dependency is modified or monkey-patched.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
```

Tests compare raw descriptors and visualizations with fixed outputs from the
original fork, exercise standard normalization modes, and run the installed CLI
on synthetic data. CI builds and installs a wheel on Windows and Linux, including
its default-parameter CSV, and checks supported scikit-image versions.
See `tests/data/README.md` for reference provenance and
`THIRD_PARTY_NOTICES.txt` for attribution.


