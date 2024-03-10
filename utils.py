from enum import Enum

import numpy as np
from PIL import Image


class CropValues(Enum):
    BOXING = (13, 97)
    BREAKOUT = (18, 102)
    RIVERRAID = (2, 86)


def preprocess_frame(frame, game: CropValues):
    gray_frame = convert_to_grayscale(frame)
    resized_frame = resize_frame(gray_frame, 84, 110)
    # Crop to 84x84 (roughly the playing area)
    crop_start, crop_end = game.value
    cropped_frame = resized_frame[crop_start:crop_end, :]
    return cropped_frame


def convert_to_grayscale(frame):
    # simply take the mean of the channels
    return np.mean(frame, axis=2).astype(np.uint8)


def resize_frame(frame, x, y):
    return np.array(Image.fromarray(frame).resize((x, y)))
