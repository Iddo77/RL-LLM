import io

import numpy as np
from PIL import Image
import base64


def preprocess_frame(frame, crop_values):
    gray_frame = convert_to_grayscale(frame)
    resized_frame = resize_frame(gray_frame, 84, 110)
    # Crop to 84x84 (roughly the playing area)
    crop_start, crop_end = crop_values
    cropped_frame = resized_frame[crop_start:crop_end, :]
    return cropped_frame


def convert_to_grayscale(frame):
    grayscale = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
    return grayscale.astype(np.uint8)


def resize_frame(frame, x, y):
    return np.array(Image.fromarray(frame).resize((x, y)))


def merge_images_with_bars(images, has_color=False):
    """
    Merges a set of images into a single image, separated by white vertical bars.

    Parameters:
    - images: A numpy array of shape (N, H, W) where N is the number of images,
              H is the height, and W is the width of each image.
    - has_color: flag indicating whether the images have color-channels (default False)

    Returns:
    - A new image as a numpy array with white bars included.
    """

    # Create the white vertical bar of 2 pixels wide
    height = images.shape[1]
    if has_color:
        white_bar = np.full((height, 2, 3), 255, dtype=np.uint8)
    else:
        white_bar = np.full((height, 2), 255, dtype=np.uint8)

    # Create a list with the images and the vertical bars
    image_list = []
    for i in range(images.shape[0]):
        image_list.append(images[i])
        if i < images.shape[0] - 1:  # Don't add a bar after the last image
            image_list.append(white_bar)

    # Concatenate all the images and bars horizontally
    merged_image = np.hstack(image_list)

    return merged_image


def save_image_to_file(image_array, file_path):
    image = Image.fromarray(image_array)
    image.save(file_path)


def convert_image_to_base64(image_array):
    image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()
