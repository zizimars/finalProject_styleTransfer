import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the TFLite model and allocate tensors.
im = Image.open("./original.jpg")
style = Image.open("./picasso.jpg")
res_im = im.resize((257, 257))


interpreter = tf.lite.Interpreter(model_path="lite-model_deeplabv3_1_metadata_2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

np_res_im = np.array(res_im)
np_res_im = (np_res_im/255).astype('float32')

if len(np_res_im.shape) == 3:
    np_res_im = np.expand_dims(np_res_im, 0)
# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np_res_im
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
labelsArrays = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
      "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
      "person", "potted plant", "sheep", "sofa", "train", "tv"]
mSegmentBits = np.zeros((257, 257)).astype(int)
outputbitmap = np.zeros((257, 257)).astype(int)
for y in range(257):
    for x in range(257):
        maxval = 0
        mSegmentBits[x][y] = 0

        for c in range(21):
            value = output_data[0][y][x][c]
            if c == 0 or value > maxVal:
                maxVal = value
                mSegmentBits[y][x] = c
        #         print(mSegmentBits[x][y])
        label = labelsArrays[mSegmentBits[x][y]]
        #         print(label)
        if (mSegmentBits[y][x] == 15):
            outputbitmap[y][x] = 1
        else:
            outputbitmap[y][x] = 0

foreground = np.dstack((outputbitmap, outputbitmap, outputbitmap))
foreground_mask = np.where(foreground == 1, np_res_im, 0).squeeze()

background = np.dstack((outputbitmap, outputbitmap, outputbitmap))
background_mask = np.where(background == 0, np_res_im, 1).squeeze()
print(background_mask.shape)

style_pred_model_path = './style_pred_model.tflite'
style_transfer_model_path = './style_transfer_model_20epochs.tflite'

def load_image_tensor(image):
    # image = tf.io.read_file(img_path)
    # image = tf.io.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image[tf.newaxis, :]

content_img = load_image_tensor(background_mask)
style = np.array(style)
style_img = load_image_tensor(style)
print(content_img.shape, style_img.shape)

def load_image(filepath, size=(256, 256)):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img / 255.0  # Normalize to [0, 1] range for processing

def preprocess_image(image, target_size):
    img_shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    min_dim = min(img_shape)
    scale_ratio = target_size / min_dim
    resized_shape = tf.cast(img_shape * scale_ratio, tf.int32)
    image = tf.image.resize(image, resized_shape)
    image = tf.image.resize_with_crop_or_pad(image, target_size, target_size)
    return image

processed_content_img = preprocess_image(content_img, 384)
processed_style_img = preprocess_image(style_img, 256)

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
def display_image(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
        # tf.keras.utils.save_img('./styled_image.jpg', image)

    plt.imshow(image)
    if title:
        plt.title(title)

display_image(styled_content_img, 'stylized content image')
plt.show()


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

print(styled_content_img.shape)

display_image(styled_content_img, 'stylized content image')
plt.show()

styled_content_img = styled_content_img.squeeze()
styled_content_img = cv2.resize(styled_content_img, (257, 257))

foreground_transferred = color_transfer_np(foreground_mask, styled_content_img)

# Blend the transferred foreground with the background using the mask
result = foreground * foreground_transferred + (1 - foreground) * styled_content_img
result = np.clip(result, 0, 1) * 255  # Scale back to [0, 255] for display

# Convert result to uint8 for saving
result = result.astype(np.uint8)
# cv2.imwrite("/home/zhuldyz/Documents/Data Science 2 year/Deep Learning/result_full.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
# print("Color transfer completed. Result saved as 'result_full.png'")

# Display the result
plt.imshow(result)
plt.axis('off')
plt.show()