import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

# Load the TFLite model and allocate tensors.
im = tf.io.read_file("./original.jpg")
style = tf.io.read_file("./picasso.jpg")

# Decode JPEG images and convert them to float32 tensors
im = tf.image.decode_jpeg(im, channels=3)
style = tf.image.decode_jpeg(style, channels=3)

# Resize the image to 257x257
res_im = tf.image.resize(im, [257, 257])


interpreter = tf.lite.Interpreter(model_path="lite-model_deeplabv3_1_metadata_2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

res_im = res_im / 255.0  # Scale to [0, 1] range for float32
res_im = tf.cast(res_im, dtype=tf.float32)  # Ensure the data type is float32

# Add batch dimension if necessary
if len(res_im.shape) == 3:
    res_im = tf.expand_dims(res_im, 0)

# Test the model on input data
input_data = tf.constant(res_im)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.tensor(output_details[0]['index'])()  # Get a pointer to the tensor

# Define the labels array as a TensorFlow constant (optional, if you need it later)
labelsArrays = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                "car", "cat", "chair", "cow", "dining table", "dog", "horse",
                "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv"]

# Find the class with the maximum score for each pixel
# `output_data` is assumed to be of shape (1, 257, 257, 21)
mSegmentBits = tf.argmax(output_data[0], axis=-1, output_type=tf.int32)  # Shape: (257, 257)

# Create outputbitmap by checking if the predicted class is 'person' (class index 15)
outputbitmap = tf.cast(tf.equal(mSegmentBits, 15), tf.int32)

mask3 = tf.stack([outputbitmap] * 3, axis=-1)  # Shape: (257, 257, 3)

# Apply mask for the foreground (res_im where outputbitmap == 1, otherwise 0)
foreground_mask = tf.where(tf.equal(mask3, 1), res_im, tf.zeros_like(res_im))

# Apply mask for the background (res_im where outputbitmap == 0, otherwise 1) and remove any extra dimensions
background_mask = tf.where(tf.equal(mask3, 0), res_im, tf.ones_like(res_im))

style_pred_model_path = './style_pred_model.tflite'
style_transfer_model_path = './style_transfer_model_20epochs.tflite'

content_img = tf.image.convert_image_dtype(background_mask, tf.float32)
style_img = tf.expand_dims(tf.image.convert_image_dtype(style, tf.float32), axis=0)
# print(content_img.shape, style_img.shape)


def preprocess_image(image, target_size):
    img_shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    min_dim = min(img_shape)
    scale_ratio = target_size / min_dim
    resized_shape = tf.cast(img_shape * scale_ratio, tf.int32)
    image = tf.image.resize(image, resized_shape)
    image = tf.image.resize_with_crop_or_pad(image, target_size, target_size)
    return image


# Assuming content_img and style_img are TensorFlow tensors with shape (height, width, channels)
processed_content_img = preprocess_image(content_img, 384)
processed_style_img = preprocess_image(style_img, 256)
# def display_image(image, title=None):
#     if len(image.shape) > 3:
#         image = tf.squeeze(image, axis=0)
#         # tf.keras.utils.save_img('./styled_image.jpg', image)
#
#     plt.imshow(image)
#     if title:
#         plt.title(title)
#
# display_image(content_img, 'stylized content image')
# plt.show()


def predict_style(processed_style_img):
    style_interpreter = tf.lite.Interpreter(model_path=style_pred_model_path)
    style_interpreter.allocate_tensors()
    input_info = style_interpreter.get_input_details()
    style_interpreter.set_tensor(input_info[0]["index"], processed_style_img)
    style_interpreter.invoke()
    style_descriptor = style_interpreter.tensor(
        style_interpreter.get_output_details()[0]["index"]
    )()
    return style_descriptor

style_descriptor = predict_style(processed_style_img)

def apply_style_transform(style_descriptor, processed_content_img):
    transform_interpreter = tf.lite.Interpreter(model_path=style_transfer_model_path)
    transform_interpreter.allocate_tensors()
    input_info = transform_interpreter.get_input_details()
    transform_interpreter.set_tensor(input_info[0]["index"], processed_content_img)
    transform_interpreter.set_tensor(input_info[1]["index"], style_descriptor)
    transform_interpreter.invoke()
    styled_img = transform_interpreter.tensor(
        transform_interpreter.get_output_details()[0]["index"]
    )()
    return styled_img

# stylized content image
styled_content_img = apply_style_transform(style_descriptor, processed_content_img)

def bgr2lab(image):
    # Convert from BGR to LAB using OpenCV and TensorFlow
    lab_image = cv2.cvtColor(tf.cast(image * 255, tf.uint8).numpy(), cv2.COLOR_RGB2Lab)
    return tf.cast(lab_image, tf.float32) / 255.0  # Normalize LAB values to [0, 1]

def lab2bgr(lab_image):
    # Convert from LAB to BGR using OpenCV and TensorFlow
    lab_image = tf.cast(lab_image * 255, tf.uint8).numpy()
    return tf.cast(cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB), tf.float32) / 255.0  # Normalize back to [0, 1]

def color_transfer_tf(source, target):
    # Convert source and target to LAB color space
    source_lab = bgr2lab(source)
    target_lab = bgr2lab(target)

    # Calculate mean and standard deviation for each channel in LAB
    def lab_stats(lab_image):
        l, a, b = tf.split(lab_image, num_or_size_splits=3, axis=-1)
        return (tf.reduce_mean(l), tf.math.reduce_std(l),
                tf.reduce_mean(a), tf.math.reduce_std(a),
                tf.reduce_mean(b), tf.math.reduce_std(b))

    src_mean_std = lab_stats(source_lab)
    tgt_mean_std = lab_stats(target_lab)

    # Transfer color by adjusting the L, A, and B channels
    l, a, b = tf.split(source_lab, num_or_size_splits=3, axis=-1)
    l = (l - src_mean_std[0]) * (tgt_mean_std[1] / src_mean_std[1]) + tgt_mean_std[0]
    a = (a - src_mean_std[2]) * (tgt_mean_std[3] / src_mean_std[3]) + tgt_mean_std[2]
    b = (b - src_mean_std[4]) * (tgt_mean_std[5] / src_mean_std[5]) + tgt_mean_std[4]

    # Merge channels and clip values to maintain valid range
    transfer_lab = tf.clip_by_value(tf.concat([l, a, b], axis=-1), 0.0, 1.0)
    return lab2bgr(transfer_lab)


styled_content_img = styled_content_img.squeeze()
styled_content_img = cv2.resize(styled_content_img, (257, 257))

foreground_transferred = tf.squeeze(color_transfer_tf(tf.squeeze(foreground_mask), tf.squeeze(styled_content_img)))

# Blend the transferred foreground with the background using the mask
mask3 = tf.cast(mask3, tf.float32)
mask3 = mask3 / 255.0 if tf.reduce_max(mask3) > 1 else mask3
# Blend the transferred foreground with the background using the mask
result = mask3 * foreground_transferred + (1 - mask3) * styled_content_img
# print(result.shape)
# display_image(result, 'stylized content image')
# plt.show()# Convert result to uint8 for saving

# cv2.imwrite("/home/zhuldyz/Documents/Data Science 2 year/Deep Learning/result_full.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
# print("Color transfer completed. Result saved as 'result_full.png'")

# Display the result
# plt.imshow(result)
# plt.axis('off')
# plt.show()