from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# image path and name
image = '1BC049_090_fake_B.png'
image = np.array(Image.open(image))

# Normalize the image to the range [0, 1] and convert it to float32
image = (image - np.min(image)) / (np.max(image) - np.min(image))
image = image.astype(np.float32)

# Convert it to [-1024, 3000] Hounsfield Units (HU) and clip the values between -1000 and 1000
image = image * 3000 - 1024
image = np.clip(image, -1000, 1000)

# Normalize again the image to the range [0, 1]
image = (image - np.min(image)) / (np.max(image) - np.min(image))
# Convert it to [0, 255] and cast it to uint8
image = (image * 255).astype(np.uint8)

# Plot the image
plt.imshow(image, cmap='gray', vmin=-1000, vmax=1000)
plt.axis('off')
plt.colorbar(label='Housefield Units (HU)')
plt.show()

image = Image.fromarray(image)
image.save('1BC049_090_8bit_clipped.png')