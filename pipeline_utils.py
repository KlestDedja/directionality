
from skimage import io, transform
from skimage.color import rgb2gray

import os

def load_and_prepare_image(path, name, to_grayscale=False):
    """Load an image and proivde option to convert it to grayscale."""
    image = io.imread(os.path.join(path, name))


    # Check if the image has four channels (RGBA)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3] # ignore the alpha channel

    # Convert RGB to grayscale
    if image.ndim == 3 and to_grayscale==True:
        image = rgb2gray(image)

    return image