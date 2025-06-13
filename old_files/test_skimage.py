import os
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, io
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage import color

img = data.astronaut()

plt.figure()
plt.imshow(img)  # 'image' should be defined previously
plt.title("Original image")
plt.show(block=False)



img_test = color.rgb2gray(img).astype(dtype=np.float32, copy=False)

fd, hog_image_l2 = hog(img_test, visualize=True, block_norm="L2")

plt.figure()
plt.imshow(hog_image_l2)  # 'image' should be defined previously
plt.title("HOG image, L2")
plt.show(block=False)


fd, hog_image_none = hog(img_test, visualize=True, block_norm=None)

plt.figure()
plt.imshow(hog_image_none)  # 'image' should be defined previously
plt.title("HOG image, None")
plt.show()
