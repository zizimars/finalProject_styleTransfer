import tensorflow as tf
import numpy as np
import time
import functools
import matplotlib.pyplot as plt
import matplotlib as mpl

# set plot size and disable grid
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

# paths for images and models
content_img_path = './original.jpg'
style_img_path = './picasso.jpg'
style_pred_model_path = './style_pred_model.tflite'
style_transfer_model_path = './style_transfer_model_20epochs.tflite'
def load_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image[tf.newaxis, :]

content_img = load_image(content_img_path)
style_img = load_image(style_img_path)


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
print('style image shape:', processed_style_img.shape)
print('content image shape:', processed_content_img.shape)

def display_image(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
        tf.keras.utils.save_img(
            './styled_image.jpg', image)

    plt.imshow(image)
    if title:
        plt.title(title)

plt.subplot(1, 2, 1)
display_image(processed_content_img, 'content image')
plt.subplot(1, 2, 2)
display_image(processed_style_img, 'style image')
plt.show()


# run style prediction model
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
print('style descriptor shape:', style_descriptor.shape)

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
display_image(styled_content_img, 'stylized content image')
plt.show()

# style descriptor for content image
content_style_descriptor = predict_style(preprocess_image(content_img, 256))


blend_ratio = 0.2  # 0% style from content, 100% from style image
# blend
blended_style_descriptor = blend_ratio * content_style_descriptor \
                           + (1 - blend_ratio) * style_descriptor
blended_styled_img = apply_style_transform(blended_style_descriptor, processed_content_img)
display_image(blended_styled_img, 'blended stylized image')
plt.show()
