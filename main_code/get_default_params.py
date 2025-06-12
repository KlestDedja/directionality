import os
import pandas as pd

# assumes the 'default_params' file is in the same folder as this script
PATH = os.path.join(os.path.dirname(__file__), "default_params.csv")
df = pd.read_csv(PATH, index_col=0)

# ---- extract default and overriding values ----
defaults_df = df.loc["default"]
DEFAULT_CELL_SIZE = int(defaults_df["cell_size"])
DEFAULT_THRESHOLD = float(defaults_df["threshold"])
DEFAULT_CHANNEL = int(defaults_df["channel"])
DEFAULT_BACKGROUND_MIN = float(defaults_df["background_min"])
DEFAULT_BACKGROUND_MAX = float(defaults_df["background_max"])

overrides_df = df.drop(index="default")


def _find_override(folder: str | bytes, col: str, cast):
    """
    Look for the first override whose 'dataset_name' (index) appears in folder,
    and return the non-NaN value for column `col`:
      - if it’s a string, return it (stripped) [useful for cahnnel = 'grayscale']
      - otherwise, cast it via `cast` (e.g. int, float)
    Returns None if no override applies.
    """
    for name, row in overrides_df.iterrows():
        key = str(name)  # ensure we’re working with a string
        if key in folder and not pd.isna(row[col]):
            val = row[col]
            if isinstance(val, str):
                return val.strip()
            try:
                return cast(val)
            except (ValueError, TypeError):
                return str(val).strip()
    return None


def get_folder_threshold(folder: str | bytes) -> float:
    """
    Return the threshold for this folder path.g
    Falls back to DEFAULT_THRESHOLD if no override applies.
    """
    thresh = _find_override(folder, "threshold", float)
    if thresh is None:
        thresh = DEFAULT_THRESHOLD

    return float(thresh)


def get_folder_channel(folder: str | bytes) -> int | str:
    """
    Return the channel axis for this folder path.
    Falls back to DEFAULT_CHANNEL if no override applies.
    """
    channel = _find_override(folder, "channel", int)

    if channel is None:
        channel = DEFAULT_CHANNEL

    return channel


def get_folder_cellsize(folder: str | bytes) -> int:
    """
    Return the cell size (n,n) for this folder path.
    Falls back to DEFAULT_CELL_SIZE if no override applies.
    """
    size = _find_override(folder, "cell_size", int)

    if size is None:
        size = DEFAULT_CELL_SIZE

    return int(size)


def get_background_range(folder: str | bytes, verbose: int = 0) -> tuple[float, float]:
    """
    Return (background_min, background_max) for this folder path.
    Falls back to DEFAULT_BACKGROUND_MIN and DEFAULT_BACKGROUND_MAX if no
    overrides apply.
    """
    bmin = _find_override(folder, "background_min", float)
    if bmin is None:
        bmin = DEFAULT_BACKGROUND_MIN

    bmax = _find_override(folder, "background_max", float)
    if bmax is None:
        bmax = DEFAULT_BACKGROUND_MAX

    return float(bmin), float(bmax)
