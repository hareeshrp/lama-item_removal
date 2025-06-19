from PIL import Image
import os

img_path = 'input_images/input2.png'
mask_path = 'input_images/input2_mask.png'

img = Image.open(img_path)
mask = Image.open(mask_path)

# Resize mask to image size
mask_resized = mask.resize(img.size, Image.NEAREST)  # NEAREST to preserve binary mask edges
mask_resized.save('input_images/input2_mask.png')  # Overwrite or save to a new file
