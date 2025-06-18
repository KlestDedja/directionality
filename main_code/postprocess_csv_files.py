import os
import pandas as pd
import re
from main_code.utils_other import (
    clean_filename,
    set_sample_condition,
    set_sample_replicate,
)


def clean_csv_df(df: pd.DataFrame, missing_threshold: int = 2) -> pd.DataFrame:
    """
    Drop rows in `df` that have more than `missing_threshold` missing values.
    """
    mask = df.isnull().sum(axis=1) <= missing_threshold
    return df.loc[mask].copy()


def add_conditions_to_df(
    df: pd.DataFrame, verbose: bool = False, suppress_warnings: bool = False
) -> pd.DataFrame:

    df = df.copy()
    filename_column = df.columns[0]

    df["condition"] = df[filename_column].apply(
        lambda x: set_sample_condition(str(x), suppress_warnings)
    )

    return df


def add_replicates_to_df(
    df: pd.DataFrame,
    verbose: bool = False,
    suppress_warnings: bool = False,
) -> pd.DataFrame:

    df = df.copy()
    filename_column = df.columns[0]

    # 1) Ensure there's a 'donor' column
    if "donor" not in df.columns:
        df["donor"] = df[filename_column].apply(lambda x: extract_donor(str(x)))

    # 2) Compute the raw replicate (as before)
    df["replicate"] = df[filename_column].apply(
        lambda x: set_sample_replicate(str(x), suppress_warnings)
    )

    # 3) Combine donor + replicate into a single string
    df["replicate"] = df.apply(
        lambda row: f"{row['donor']}__{row['replicate']}", axis=1
    )

    return df


def extract_donor(filename):
    """Extracts donor name from string using regex (e.g. 'fkt21')."""
    filename = clean_filename(filename)
    match = re.search(r"fkt\d{1,2}", filename)
    return match.group(0) if match else "Unknown"


def postprocess_hog_csv(
    csv_path, filename_extra_tail="_clean", verbose=True, missing_threshold=2
):
    """
    Postprocessign steps: add 'condition' and 'replicate' column.
    Finally, drop rows that exceed the missing‐value threshold.
    """

    cleaned_csv_path = csv_path.replace(".csv", filename_extra_tail + ".csv")

    if verbose:
        print(f"Reading raw CSV from:\n{csv_path}")

    df = pd.read_csv(csv_path)

    df = add_conditions_to_df(df, suppress_warnings=False, verbose=verbose)

    df = add_replicates_to_df(df, suppress_warnings=False, verbose=verbose)

    if verbose:
        print(f"Dropping rows with more than {missing_threshold} missing values")
    df = clean_csv_df(df, missing_threshold=missing_threshold)

    os.makedirs(os.path.dirname(cleaned_csv_path), exist_ok=True)

    return df, cleaned_csv_path
