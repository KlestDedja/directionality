raw_hog_reference.npz contains outputs generated from KlestDedja/scikit-image-fork's
skimage/feature/_hog.py (blob SHA 321bdedc930c9fb60e0a5d679b40447371973034),
with stock scikit-image 0.25.2 providing its compiled histogram implementation.
The reference function was executed in an isolated namespace; no installed
scikit-image file or global function was modified. The fork is not required to
run tests. Its only normalization change is the identity branch for 'None'.

The archive includes the uint8 source image and descriptors/visualizations for:
- uint8: 9 bins, 4x6 pixels per cell, 1x1 cells per block;
- float32 (image / 251): 9 bins, 5x7 pixels per cell, 2x2 cells per block;
- float64 (image / 251): 45 bins, 4x4 pixels per cell, 2x3 cells per block.

These cover rectangular and odd cell sizes, overlapping blocks, incomplete edge
cells, dtype preservation, bin conventions and raw signal magnitudes.
