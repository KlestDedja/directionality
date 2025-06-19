<a name="logo-anchor"></a>
<p align="center">
<img src="directionality-logo.png?raw=true" alt="Directionality Logo" width="20%"/>
</p>

# Directionality

This repository analyzes image  by computing Histograms of Oriented Gradients (HOGs) and deriving orientation statistics. 
This tool automates the extraction of dominant gradient directions and produces visual and statistical outputs to aid biological interpretation.

## 🧠 Purpose

The aim is to quantify and visualize how directional structures (like fibers or gradients) are distributed in fluorescence and brightfield images. This is particularly useful in studies where the biological samples exhibit alignment or anisotropy.

## 🚀 Installation

The repository includes a self-contained demo dataset to help you get started quickly.

### 🔁 1. Clone the repository

Make sure you have [Git](https://git-scm.com/) and [Python 3.10+](https://www.python.org/downloads/) installed. 
You can clone the repository through GitHub

```bash
git clone https://github.com/klest94/directionality.git
cd directionality
```

### 🐍 2. Set up the environment

```
python -m venv directionality
source directionality/bin/activate  # On Windows: venv\Scripts\activate ?
pip install -r requirements.txt
```
Alternatively, if you are familiar with `conda`:
```
conda create -n directionality python=3.12
conda activate directionality
pip install -r requirements.txt
```
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



