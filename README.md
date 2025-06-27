<a name="logo-anchor"></a>
<p align="center">
<img src="directionality-logo.png?raw=true" alt="Directionality Logo" width="20%"/>
</p>

# Directionality

This repository analyzes image  by computing Histograms of Oriented Gradients (HOGs) and deriving orientation statistics. 
This tool automates the extraction of dominant gradient directions and produces visual and statistical outputs to aid biological interpretation.

## 🧠 Purpose

The aim is to quantify and visualize how directional structures (like fibers) are distributed in fluorescence and brightfield images. This is particularly useful in studies where the biological samples exhibit alignment or anisotropy.

## 🚀 Installation

The repository includes a self-contained demo dataset to help you create a folder named `directionality-demo` to get started quickly. You first need to make sure you have [Git](https://git-scm.com/) or [GitHub](https://docs.github.com/en/desktop/installing-and-authenticating-to-github-desktop/installing-github-desktop) and [Python 3.10+](https://www.python.org/downloads/) installed.


### 🔁 1. Clone the repository

Clone the repository either with Git:
```bash
git clone https://github.com/Klest94/directionality.git directionality-demo
cd directionality-demo
```
where `directionality-demo` can be replaced by the name you want to give to the newly formed repository.

Alternatively, you can clone with GitHub Desktop: click on `Add -> Clone repository -> url -> https://github.com/klest94/directionality.git` Name the repo as `directionality-demo` or anything of your choice in your local path.

### 🐍 2. Set up the environment

#### Option A: Python virtual environment.

Navigate to the newly created folder:
```
cd directionality-demo
```
Create a Python environment:
```
python3.12 -m venv .venv
```
This creates a virtual environment inside `directionality-demo/.venv/`.
Now we can activate the virtual environment and install the required packages:
```
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

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

Loads `.tif`, `.png` and `.jpg` images from `demo-data/input_images/`

Selects sensible defaults:

Computes signal directionality and visualizations

Saves results into `demo-data/output_analysis/`, including:

- 🖼️ One PNG plot per image: overlays original image, HOG visualization, signal strength map, and polar histogram

- 📊 Cleaned CSV file: `HOG_stats_<params>_clean.csv` summary of average and modal signal directions, with deviations (filtered and enriched with metadata)



