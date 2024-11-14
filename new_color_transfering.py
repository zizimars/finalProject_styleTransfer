import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to load and prepare an image
def load_image(filepath, size=(256, 256)):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img / 255.0  # Normalize to [0, 1] range for processing

def bgr2lab(image):
    # Convert from BGR to LAB using OpenCV
    lab_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2Lab)
    return lab_image.astype(np.float32) / 255.0  # Normalize LAB values

def lab2bgr(lab_image):
    # Convert from LAB to BGR using OpenCV
    lab_image = (lab_image * 255).astype(np.uint8)
    return cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB) / 255.0  # Normalize back to [0, 1]

def color_transfer_np(source, target):
    # Convert source and target to LAB color space
    source_lab = bgr2lab(source)
    target_lab = bgr2lab(target)

    # Calculate mean and standard deviation for each channel in LAB
    def lab_stats(lab_image):
        l, a, b = cv2.split(lab_image)
        return (np.mean(l), np.std(l), np.mean(a), np.std(a), np.mean(b), np.std(b))

    src_mean_std = lab_stats(source_lab)
    tgt_mean_std = lab_stats(target_lab)

    # Transfer color by adjusting the L, A, and B channels
    l, a, b = cv2.split(source_lab)
    l = (l - src_mean_std[0]) * (tgt_mean_std[1] / src_mean_std[1]) + tgt_mean_std[0]
    a = (a - src_mean_std[2]) * (tgt_mean_std[3] / src_mean_std[3]) + tgt_mean_std[2]
    b = (b - src_mean_std[4]) * (tgt_mean_std[5] / src_mean_std[5]) + tgt_mean_std[4]

    # Merge channels and clip values to maintain valid range
    transfer_lab = cv2.merge([np.clip(l, 0, 1), np.clip(a, 0, 1), np.clip(b, 0, 1)])
    return lab2bgr(transfer_lab)

# Load images and mask
style = load_image("./picasso.jpg")
foreground_image = load_image("./me.jpg")
background_image = load_image("./bgrd_style.jpg")
mask_image = load_image("./mask.jpg")  # Assume mask is binary: 1 for foreground, 0 for background

foreground = cv2.imread('/home/zhuldyz/Downloads/Telegram Desktop/photo_2024-11-07_21-17-34.jpg')
print("Original shape foregraound:", foreground.shape)

# Crop to a specific height if needed
foreground = foreground[:720, :, :]

# Resize to desired dimensions
img = cv2.resize(foreground, (256, 256))
print("Resized shape:", img.shape)

# Save the image in BGR format for correct colors
cv2.imwrite("./original.jpg", img)
print("Image saved successfully.")
# Perform color transfer only on the foreground
foreground_transferred = color_transfer_np(foreground_image, style)

# Blend the transferred foreground with the background using the mask
result = mask_image * foreground_transferred + (1 - mask_image) * background_image
result = np.clip(result, 0, 1) * 255  # Scale back to [0, 255] for display

# Convert result to uint8 for saving
result = result.astype(np.uint8)
# cv2.imwrite("/home/zhuldyz/Documents/Data Science 2 year/Deep Learning/result_full.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print("Color transfer completed. Result saved as 'result_full.png'")

# Display the result
plt.imshow(result)
plt.axis('off')
plt.show()
