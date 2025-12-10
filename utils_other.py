import numpy as np


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
