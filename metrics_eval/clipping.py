import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

"""
# Paths to the 6 images.
images = ['1BC049_090_8bit.png', '1BC049_090_rgb_wo.png', '1BC049_090_rgb_w.png', '1BC049_090_16bit.png', '1BC049_090_3stacked.png', '1BC049_090_5stacked.png']
images = [np.array(Image.open(image)) for image in images]

# Normalize the images to the range [0, 1] and convert them to float32. 
images = [(image - np.min(image)) / (np.max(image) - np.min(image)) for image in images]
images = [image.astype(np.float32) for image in images]

# Transform the images to Hounsfield Units (HU) [-1024, 3000]
images = [image * 3000 - 1024 for image in images]

# Clip the values between -1000 and 1000
images = [np.clip(image, -1000, 1000) for image in images]

fig, axes = plt.subplots(1, 6, figsize=(18, 3), gridspec_kw={'wspace': 0, 'hspace': 0})

for i, ax in enumerate(axes):
    im = ax.imshow(images[i], cmap='gray', vmin=-1000, vmax=1000)
    ax.axis('off')

# Add a single colorbar on the right
# Decrease the gap between the colorbar and the images
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
cbar.set_label('Housefield Units (HU)')

plt.show()
"""

def convert_rgb_to_grayscale(rgb_image):
    """
    Convert an 8-bit RGB image back to a 16-bit grayscale image
    """

    R = (rgb_image[:, :, 0] >> 4).astype(np.uint16)  # Extract the 4 most significant bits
    G = (rgb_image[:, :, 1] >> 4).astype(np.uint16)  # Extract the 4 most significant bits
    B = (rgb_image[:, :, 2] >> 4).astype(np.uint16)  # Extract the 4 most significant bits

    # Reconstruct the original 16-bit grayscale image
    # Shift the red channel to its original position (12th to 9th bits)
    # Shift the green channel to its original position (8th to 5th bits)
    # The blue channel is already in the correct position (4th to 1st bits)
    grayscale_image = (R << 8) | (G << 4) | B

    return grayscale_image

image = "rgb_with_bit_placement.png"

# Normalize the images to the range [0, 1] and convert them to float32.
image = np.array(Image.open(image).convert('RGB'))
image = convert_rgb_to_grayscale(image)
image = (image - np.min(image)) / (np.max(image) - np.min(image))
image = image.astype(np.float32)

# Transform the images to Hounsfield Units (HU) [-1024, 3000]
image = image * 3000 - 1024

# Clip the values between -1000 and 1000
image = np.clip(image, -1000, 1000)

plt.imshow(image, cmap='gray', vmin=-1000, vmax=1000)
plt.axis('off')
plt.colorbar
plt.show()
