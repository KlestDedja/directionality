<a name="logo-anchor"></a>
<p align="center">
<img src="directionality-edgehog-logo.png?raw=true" alt="Directionality Logo" width="25%"/>
</p>

# Directionality with EDGEHOG

**EDGEHOG** provides a reproducible framework to analyze images using Histograms of Oriented Gradients (HOGs) to derive orientation statistics.  
This tool automates the extraction of directions and their distirbutions, by producing both visual and statistical outputs for images. A paper is underway explaining how EDGEHOG works and how it can be employed in a several biomedical applications.

## 🚀 Installation

The repository includes a self-contained demo dataset, making it easy to get started. Please ensure you have [Git](https://git-scm.com/) (or [GitHub Desktop](https://docs.github.com/en/desktop/installing-and-authenticating-to-github-desktop/installing-github-desktop)) and [Python 3.10+](https://www.python.org/downloads/) installed on your system
> *In the future, we want to provide a fully fledged package, installable via `pip`.*

---

### 🔁 1. Clone the repository

Clone the repository to the desired local path either with Git:
```bash
git clone https://github.com/Klest94/directionality.git ~/your/local/path/directionality-demo
```
where `directionality-demo` is the name of the new repository.

Or, you can clone with GitHub Desktop: click on `Add -> Clone repository -> url -> https://github.com/klest94/directionality.git` Name the repo as e.g. `directionality-demo` in your local path.

### 🐍 2. Set up the environment

#### Option A: Python virtual environment.

Navigate to the new folder with your favourite terminal (**Powershell**, **GitBash**, **Command Prompt**,...) and create a Python environment:
```
cd ~/your/local/path/directionality-demo
python -m venv .venv
```
This creates a virtual environment `.venv` in `directionality-demo/.venv/`.
Activate the virtual environment, if you are using **Windows**
```
.venv\Scripts\activate
```

if you are using **macOS / Linux / WSL**:
```
source .venv/bin/activate 
```

And install the required packages
```
pip install -r requirements.txt
```

⚠️ These instructions have been tested with Python 3.12. All version above `3.10` should work, but compatibility is not guaranteed.

#### Option B: if you are familiar with `conda`

Assuming you have `anaconda` installed, then:
```
conda create -n directionality-demo python=3.12
conda activate directionality-demo
```
`cd` your way to the repository location (default location is `GitHub/directionality-demo`) and install required packages:
```
pip install -r requirements.txt
```
### 🐛 3. Fix scikit-image code (temporary)

As we haven't managed to make build the scikit-image fork correctly, best current workaround is to navigate to the installation of `scikit-image` within the `directionality-demo` environment, search for the `_hog_normalize_block` function under `skimage/feature/_hog.py` and replace the lines:
```
def _hog_normalize_block(block, method, eps=1e-5):
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block**2) + eps**2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block**2) + eps**2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out**2) + eps**2)
```

with the following ones:

```
def _hog_normalize_block(block, method, eps=1e-5):
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block**2) + eps**2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block**2) + eps**2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out**2) + eps**2)
    elif method == 'None':
        out = block
``` 
In practice, we added an extra case to the normalization options, namely *no* normalization.

You should be good to go!

## 🚀 Try It Out: Run the example demo
To see the Directionality tool in action, you can use the example dataset in the `demo-data` folder.

You simply have to run  `main_script.py` (in the `main_code` folder) from your favorite IDE or through a bash command. Make sure you are pointing to the location of this repository.  

## 📂 What happens when you run it?

The demo script:

- Loads `.tif`, `.png` and `.jpg` images from `demo-data/input_images/`,

-  runs the directionality estimator EDGEOG with default or user provided parameters, computes signal directionality statistics.

- saves results into `demo-data/output_analysis/`, including:

    - 📊 Cleaned CSV file: `HOG_stats_<params>_clean.csv` summary of average and modal signal directions, with deviations (filtered and enriched with metadata)

    - 🖼️ Optionally, a PNG plot per image, containing:original image, HOG visualization, signal strength map, and a histogram of the distribution of directioanlity, in poalr coordinates. 


