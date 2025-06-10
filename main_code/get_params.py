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

overrides_df = df.drop(index="default")


def _find_override(folder: str, col: str, cast):
    """
    Look for the first override whose 'dataset_name' appears in folder,
    and return the non-NaN value for column `col` cast to `cast`,
    otherwise return None (default value will be used).
    """
    for name, row in overrides_df.iterrows():
        if name in folder and not pd.isna(row[col]):
            return cast(row[col])
    return None


def get_folder_threshold(folder: str) -> float:
    """
    Return the threshold for this folder path.g
    Falls back to DEFAULT_THRESHOLD if no override applies.
    """
    thresh = _find_override(folder, "threshold", float)
    return thresh if thresh is not None else DEFAULT_THRESHOLD


def get_folder_channel(folder: str) -> int:
    """
    Return the channel axis for this folder path.
    Falls back to DEFAULT_CHANNEL if no override applies.
    """
    channel = _find_override(folder, "channel", int)
    return channel if channel is not None else DEFAULT_CHANNEL


def get_folder_cellsize(folder: str) -> int:
    """
    Return the cell size (n,n) for this folder path.
    Falls back to DEFAULT_CELL_SIZE if no override applies.
    """
    size = _find_override(folder, "cell_size", int)
    return size if size is not None else DEFAULT_CELL_SIZE
