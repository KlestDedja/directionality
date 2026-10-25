import re
import warnings

import numpy as np


def clean_filename(filename: str) -> str:
    """Shorten the filename by removing initial date"""
    pattern = r"(\d{6,10} {1,2})"  # match for 6-10 consecutive digits followed by one or two spaces
    match = re.search(pattern, filename[:12])  # search only among the first 12 chars
    if match:
        filename = filename.replace(match.group(0), "")

    filename = filename.replace(" ", "_")
    return filename


def set_sample_condition(filename: str, verbose: int = 0) -> str:
    """
    Assuming the Image Fluorescence 2D images are given as an input, this function extracts the
    condition from the filename string. The checks are very manual due to inconsistent naming (different people)
    output: condition (str)
    """
    # remove initial date and replace spaces with underscores
    filename = clean_filename(filename)

    condition = "Unknown"  # default case, this is kept if no condition is found

    if "CTR" in filename.upper() or "CRT" in filename or "NO" in filename[:10]:
        condition = "CTR"

    if "14h_" in filename or "14hStim_" in filename or "140h_14h" in filename:
        condition = "14h"

    if "2h_" in filename or "2hStim_" in filename or "_2h_" in filename:
        condition = "2h"

    if "10'_" in filename or "10'Stim_" in filename or "10m_" in filename:
        condition = "10m"

    # use if instead of elif and go more and mpore specific to update the matches in case

    if (
        "10'-0.5'" in filename
        or "10'0.5'" in filename
        or "10'-0,5'" in filename
        or "10'Stim-30''Break" in filename
        or "10m0.5m" in filename
    ):
        condition = "10m-30s"

    if (
        "10'-5'" in filename
        or "10'5'" in filename
        or "10'Stim-5'Break" in filename
        or "10m5m" in filename
    ):
        condition = "10m-5m"

    if (
        "1.5'-0.5'" in filename
        or "1.5_" in filename
        or "1,5'0.5'" in filename
        or "1'30''Stim-30''Break" in filename
        or "1.5m0.5m" in filename
        or "1_5'0_5'" in filename
        or "1_5_0_5_" in filename
    ):
        condition = "90s-30s"

    if condition == "Unknown" and verbose >= 1:
        warnings.warn(f"Condition unknown. Double check file:\n{filename}")

    return condition


def set_sample_replicate(filename: str, verbose: int = 0) -> str:
    """
    Assuming the Image Fluorescence 2D images are given as an input, this function extracts the
    condition from the filename string. The checks are very manual due to inconsistent naming (different people)
    output: condition (str)
    """
    # remove initial date and replace spaces with underscores
    filename = clean_filename(filename)

    replicate = "Unknown"  # default case, this is kept if no replicate is found

    if "R1" in filename or "MS_1" in filename or "CTR_1" in filename:
        replicate = "R1"

    if "R2" in filename or "MS_2" in filename or "CTR_2" in filename:
        replicate = "R2"

    if "R3" in filename or "MS_3" in filename or "CTR_3" in filename:
        replicate = "R3"

    if replicate == "Unknown":
        pattern = r"(\d)(?:_\d+)?(?=(?:_new-Image_Export|-Image_Export|-Image Export|_z\d{2,3}))"
        # match a digit followed by optional underscore and digits, followed by specific patterns
        m = re.search(pattern, filename)
        if m:
            replicate = m.group(1)

    if replicate == "Unknown" and verbose >= 1:
        warnings.warn(f"Unknown replicate. Double check file:\n{filename}")

    return replicate


def calculate_and_print_percentiles(arr, percentiles=(0, 25, 50, 75, 100), format_str="{:.2f}"):
    # Calculate the percentiles
    percentile_values = np.percentile(arr, percentiles)

    # Create a dictionary with percentiles as keys and their values
    percentile_dict = dict(zip(percentiles, percentile_values))

    # Print the dictionary in a nice format
    print("Percentile Values:")
    for p, v in percentile_dict.items():
        print(f"{p:3}th percentile: {format_str.format(v)}")

    return percentile_dict


# useful for debugging:
def print_top_values(my_dict, top_n=3):
    # Sort dictionary by value descending
    sorted_items = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)
    # Print the top_n pairs
    for k, v in sorted_items[:top_n]:
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    arr = np.array([1, 2.11, 3.141, 4.43333, 5, 6.81, 7, 8, 9, 120 / 13])
    percentiles = [0, 25, 50, 75, 100]

    percentile_dict = calculate_and_print_percentiles(arr, percentiles, format_str="{:.3f}")
