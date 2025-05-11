import numpy as np
import warnings
import os
import pandas as pd
import re


def clean_filename(filename):
    """Shorten the filename by removing initial date"""
    pattern = r"(\d{6,10} {1,2})"  # match for 6-10 consecutive digits followed by one or two spaces
    match = re.search(pattern, filename[:12])  # search only among the first 12 chars
    if match:
        filename = filename.replace(match.group(0), "")

    filename = filename.replace(" ", "_")
    return filename


def get_folder_threshold(image_folder) -> float:

    if "images-3D-lightsheet-20240928_BAM_fkt20_P3_fkt21_P3_PEMFS" in image_folder:
        threshold = 1.5  # approved
    if "images-3D-lighsheet-20241115_BAM_fkt20-P3-fkt21-P3-PEMFS-12w" in image_folder:
        threshold = 4  # seems goood. Directionality is very pronounced as the signal is mostly near the walls

    # elif "confocal-20241022-fusion" in image_folder:
    #     threshold = 2  # probably doesnt work anyway
    # elif "confocal-20241116-fusion":
    #     threshold = 2
    elif "20241116-fusion-bMyoB-PEMFS-TM-12w" in image_folder:
        threshold = 6  # Some images have no signal to show, but no worries: We drop them later in the pipeline.
    elif "confocal-20241022-fusion-bMyoB-BAMS" in image_folder:
        threshold = 6  # doesn't really work anyway, contrast needs to be enhanced
        warnings.warn(
            "As of now, the method does not work with this folder. Consider enhancing contrast."
        )
    elif "images-longitudinal" in image_folder:
        threshold = 10

    else:
        threshold = 3  # random value, in between

    return threshold


def set_sample_condition(filename, suppress_warnings=False, verbose=False):
    """
    Assuming the Image Fluorescence 2D images are given as an input, this function extracts the
    condition from the filename string. The checks are very manual due to inconsistent naming (different people)
    output: condition (str)
    """
    if verbose is True:
        print(str(filename))
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

    if condition == "Unknown":  # if still Unknown, raise a warning
        if suppress_warnings is False:
            warnings.warn(f"Condition unknown. Double check file:\n{filename}")

    if verbose is True:
        print("condition:", condition)
    return condition


def update_conditions_to_csv(
    input_csv, output_csv, suppress_warnings=False, print_process=False
) -> None:
    """
    Reads a CSV file, extracts conditions from the filenames in the first column,
    and adds a 'condition' column with the extracted conditions.

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path where the output CSV with conditions will be saved.
        suppress_warnings (bool): If False, warnings are raised for unknown conditions.
    """
    df = pd.read_csv(input_csv)

    # Check if the DataFrame has at least one column
    if df.shape[1] < 1:
        raise ValueError("Input CSV must have at least one column with filenames.")

    # Assume the first column contains the filenames
    filename_column = df.columns[0]

    # Apply the set_sample_condition function to each filename
    df["condition"] = df[filename_column].apply(
        lambda x: set_sample_condition(str(x), suppress_warnings, verbose=print_process)
    )

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    if print_process is True:
        print(f"Conditions added and saved to {output_csv}")


def set_sample_replicate(filename, suppress_warnings=False, verbose=False):
    """
    Assuming the Image Fluorescence 2D images are given as an input, this function extracts the
    condition from the filename string. The checks are very manual due to inconsistent naming (different people)
    output: condition (str)
    """
    if verbose is True:
        print(str(filename))
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

    if replicate == "Unknown":  # if still Unknown, raise a warning
        if suppress_warnings is False:
            warnings.warn(f"Unknown replicate. Double check file:\n{filename}")

    if verbose is True:
        print("replicate:", replicate)
    return replicate


def update_replicates_to_csv(
    input_csv, output_csv, suppress_warnings=False, print_process=False
) -> None:
    """
    Reads a CSV file, extracts replicates from the filenames in the first column,
    and adds a 'condition' column with the extracted replicates.

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path where the output CSV with replicates will be saved.
        suppress_warnings (bool): If False, warnings are raised for unknown replicates.
    """
    df = pd.read_csv(input_csv)

    # Check if the DataFrame has at least one column
    if df.shape[1] < 1:
        raise ValueError("Input CSV must have at least one column with filenames.")

    # Assume the first column contains the filenames
    filename_column = df.columns[0]

    # Apply the set_sample_replicate function to each filename, if df has some data:
    if df.shape[0] > 0:
        df["replicate"] = df[filename_column].apply(
            lambda x: set_sample_replicate(
                str(x), suppress_warnings, verbose=print_process
            )
        )

        df["replicate"] = df.apply(
            lambda row: str(row["donor"]) + "__" + str(row["replicate"]), axis=1
        )

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    if print_process is True:
        print(f"Replicates added and saved to {output_csv}")


def clean_csv_rows(input_csv, output_csv, missing_threshold=2) -> None:
    """
    Removes rows with more than 'missing_threshold' missing values,
    Parameters:
        input_csv (str): Path to the input Excel file.
        output_csv (str): Path where the cleaned Excel file will be saved.
        missing_threshold (int): Maximum allowed missing values per row.
    """
    df = pd.read_csv(input_csv)

    # Keep only rows where the number of missing values is less than or equal to the threshold
    df_clean = df[df.isnull().sum(axis=1) <= missing_threshold]

    df_clean.to_csv(output_csv, index=False)


def calculate_and_print_percentiles(
    arr, percentiles=[0, 25, 50, 75, 100], format_str="{:.2f}"
):
    # Calculate the percentiles
    percentile_values = np.percentile(arr, percentiles)

    # Create a dictionary with percentiles as keys and their values
    percentile_dict = {p: v for p, v in zip(percentiles, percentile_values)}

    # Print the dictionary in a nice format
    print("Percentile Values:")
    for p, v in percentile_dict.items():
        print(f"{p:3}th percentile: {format_str.format(v)}")

    return percentile_dict


def print_top_values(my_dict, top_n=3):
    # Sort dictionary by value descending
    sorted_items = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)
    # Print the top_n pairs
    for k, v in sorted_items[:top_n]:
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    arr = np.array([1, 2.11, 3.141, 4.43333, 5, 6.81, 7, 8, 9, 120 / 13])
    percentiles = [0, 25, 50, 75, 100]

    percentile_dict = calculate_and_print_percentiles(
        arr, percentiles, format_str="{:.3f}"
    )
