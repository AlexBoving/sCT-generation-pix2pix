# Metrics Evaluation

main.py processes the result images from the pix2pix model and proceeds with the metrics evaluation. Declare carefully in the main.py script if the images are from the RGB, the 8bit or the 16bit dataset.

## Contents

- `main.py`: The main script to execute all the functions.
- `prediction_preparation.py`: Converts the result images into nifti file.
- `evaluation_3D.py`: Proceeds with the evaluation metrics on a voxel-wise basis.
- `evaluation_2D.py`: Proceeds with the evaluation metrics on a pixel-wise basis.
- `clipping.py`: Plot the clipped images.

## Prerequisites

The `evaluation_3D.py` and `evaluation_2D.py` script requires the following Python packages:

- `skimage.metrics` (for metrics)

Install these dependencies by running:

```bash
pip install -U scikit-image
```
