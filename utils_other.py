import numpy as np
import warnings
import os
import pandas as pd


def get_folder_threshold(image_folder):

    if "images-3D-lightsheet-20240928_BAM_fkt20_P3_fkt21_P3_PEMFS" in image_folder:
        threshold = 1.5  # approved
    if "images-3D-lighsheet-20241115_BAM_fkt20-P3-fkt21-P3-PEMFS-12w" in image_folder:
        threshold = 4  # seems goood. Directionality is very pronounced as the signal is mostly near the walls

    elif "confocal-20241022-fusion" in image_folder:
        threshold = 2  # probably doesnt work anyway
    elif "confocal-20241116-fusion":
        threshold = 2
    elif "20241116-fusion-bMyoB-PEMFS-TM-12w" in image_folder:
        threshold = 6  # Some images have no signal to show, but no worries: We drop them later in the pipeline.
    elif "confocal-20241022-fusion-bMyoB" in image_folder:
        threshold = 2  # doesn't really work anyway, contrast needs to be enhanced
        warnings.warn(
            "As of now, the method does not work with this folder. Consider enhancing contrast."
        )
    else:
        threshold = 3  # random value, in between

    return threshold


# IF WE WANT IMAGES TO FORCELY GENERATE A SIGNAL, UNCOMMENT RUN THE FOLLOWING:
# while True: # iterate until suitable threshold is found
#     cells_to_keep = strengths > running_threshold
#     subset = fd_norm[cells_to_keep]
#     # if subset.size == 0: # If no elements match, halve the threshold and try again
#     if cells_to_keep.mean() < 0.05: # less than 1% of the picture cells are kept
#         running_threshold /= 2
#         if running_threshold < 1e-6:
#             raise ValueError("Threshold value became too low, unable to find non-empty slice.")
#     else:
#         gradient_hist_180 = subset.mean(axis=0)
#         break

# if "fluorescence-2D" in image_folder:
#     threshold = 1.843 #for 2D images
# elif "3D-lightsheet" in image_folder:
#     threshold = 1 # for 3D images
# elif "confocal-3D" in image_folder:
#     threshold = 5 # for confocal 3D images
# elif "test-params" in image_folder:
#     threshold = 10 # for testing images
# elif "20241115_BAM-fkt20" in image_folder:
#     threshold = 5
# # elif "lightsheet-20240928_BAM" in image_folder:
# #     threshold = 1.5
# elif "confocal-20241022-fusion" in image_folder:
#     threshold = 2 # probably doesnt work anyway
# elif "confocal-20241116-fusion":
#     threshold = 2
# elif "20241116-fusion-bMyoB-PEMFS-TM-12w" in image_folder:
#     threshold = 1
# elif "20241115_BAM_fkt20-P3-fkt21-P3-PEMFS-12w" in image_folder:
#     threshold = 1
# else:
#     threshold = 3 # random value, in between

# if "z057c1" in image_filename:
#     plt.title("Kept cell, at the edge")
#     plt.bar(orientations_180_deg, fd_norm[0, 10, :], width=3)
#     plt.show()


def set_sample_condition(filename, suppress_warnings=False, verbose=False):
    """
    Assuming the Image Fluorescence 2D images are given as an input, this function extracts the
    condition from the filename string. The checks are very manual due to inconsistent naming (different people)
    output: condition (str)
    """
    if verbose is True:
        print(str(filename))

    condition = "Unknown"  # default case, this is kept if no condition is found

    if "CTR" in filename or "CRT" in filename or "NO" in filename[:8]:
        condition = "CTR"

    if "14h_" in filename or "14hStim_" in filename or "140h_14h" in filename:
        condition = "14h"

    if "2h_" in filename or "2hStim_" in filename or " 2h " in filename:
        condition = "2h"

    if "10'_" in filename or "10'Stim_" in filename or "10m_" in filename:
        condition = "10m"

    # use if instead of elif and go more and mpore specific to update the mathces in case

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
        or "1.5 " in filename
        or "1,5'0.5'" in filename
        or "1'30''Stim-30''Break" in filename
        or "1.5m0.5m" in filename
        or "1_5 0_5" in filename
        or "1_5'0_5'" in filename
    ):
        condition = "90s-30s"

    if condition == "Unknown":  # if still Unknown, raise a warning
        if suppress_warnings is False:
            warnings.warn(f"Condition unknown. Double check filename:\n{filename}")

    if verbose is True:
        print("condition:", condition)
    return condition


def update_conditions_to_csv(input_csv, output_csv, suppress_warnings=False):
    """
    Reads a CSV file, extracts conditions from the filenames in the first column,
    and adds a 'condition' column with the extracted conditions.

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path where the output CSV with conditions will be saved.
        suppress_warnings (bool): If False, warnings are raised for unknown conditions.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)

    io_dir = os.path.dirname(input_csv)

    # Check if the DataFrame has at least one column
    if df.shape[1] < 1:
        raise ValueError("Input CSV must have at least one column with filenames.")

    # Assume the first column contains the filenames
    filename_column = df.columns[0]

    # Apply the set_sample_condition function to each filename
    df["condition"] = df[filename_column].apply(
        lambda x: set_sample_condition(str(x), suppress_warnings, verbose=True)
    )

    # Save the updated DataFrame to a new CSV file
    df.to_csv(os.path.join(io_dir, output_csv), index=False)
    print(f"Conditions added and saved to {output_csv}")


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


if __name__ == "__main__":
    arr = np.array([1, 2.11, 3.141, 4.43333, 5, 6.81, 7, 8, 9, 120 / 13])
    percentiles = [0, 25, 50, 75, 100]

    percentile_dict = calculate_and_print_percentiles(
        arr, percentiles, format_str="{:.3f}"
    )
