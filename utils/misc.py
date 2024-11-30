import os
import argparse
import time
import logging
from typing import Callable

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

from utils.config import *
from processing import *

log = logging.getLogger("Misc")

# decorator for calculating the time taken to execute a function
def timer(func: Callable) -> Callable:
    """Decorator to calculate the time taken to execute a function.

    Parameters
    ----------
    func : function
        The function to be executed

    Returns
    -------
    function
        The wrapper function
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed_time = (end - start) * 1000
        log = logging.getLogger(
            func.__module__.lower().replace('processing.', '').capitalize()
        )
        log.info(f"{func.__name__} execution time: {elapsed_time:.2f} ms")
        return result
    return wrapper

def clear_output_data() -> None:
    """Clear the output data.

    Clear the output data from the previous run.
    """
    for root, _, files in os.walk('outputs'):
        for file in files:
            if file.endswith('.jpg'):
                os.remove(os.path.join(root, file))

def compare_images(
    image1: cv.typing.MatLike, 
    image2: cv.typing.MatLike, 
    image1_name: str ="Image 1", 
    image2_name: str ="Image 2"
) -> None:
    """Compare two images side by side.

    Parameters
    ----------
    image1: cv.typing.MatLike
        The first image to compare.
    image2: cv.typing.MatLike
        The second image to compare.
    image1_name: str
        The description of the first image.
    image2_name: str
        The description of the second image.
    """

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_name, fontsize=50)
    ax2.imshow(image2)
    ax2.set_title(image2_name, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def draw_points(
    image: cv.typing.MatLike, 
    points: np.ndarray, 
    color: tuple = (0, 255, 0)
) -> cv.typing.MatLike:
    """Draw points on an image.

    Draw points on an image using the specified color. The points are drawn
    as circles with a radius of 8 pixels.

    Parameters
    ----------
    image : cv.typing.MatLike
        The image to draw the points on.

    points : np.ndarray
        The points to draw on the image.

    color : tuple, optional
        The color of the points, by default (0, 255, 0).

    Returns
    -------
    cv.typing.MatLike
        The image with the points drawn on it.
    """
    # for point in points:
    #     cv.circle(image, (int(point[0]), int(point[1])), 8, color, -1)
    top_right, bottom_right, bottom_left, top_left = points
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (0, 255, 255)
    cv.circle(image, (int(top_right[0]), int(top_right[1])), 8, red, -1)
    cv.circle(image, (int(bottom_right[0]), int(bottom_right[1])), 8, green, -1)
    cv.circle(image, (int(bottom_left[0]), int(bottom_left[1])), 8, blue, -1)
    cv.circle(image, (int(top_left[0]), int(top_left[1])), 8, yellow, -1)

    return image
    

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns
    -------
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Road Lane Detection")
    parser.add_argument(
        "-c",
        "--calibrate", 
        action="store_true", 
        help="Perform camera calibration"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear output images"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=BASE_NAME,
        help="The name of the image to process"
    )
    parser.add_argument(
        "-s",
        "--store",
        action="store_true",
        help="Store the images after processing"
    )
    return parser.parse_args()

def validate_base_name(name: str) -> bool:
    """Validate the base name of the image.

    Check if the base name of the image is valid.

    Parameters
    ----------
    name : str
        The base name of the image to process.

    Raises
    ------
    ValueError
        If the base name of the image is invalid.
    FileNotFoundError
        If the image file is not found.
    """
    if not name or name == '':
        raise ValueError("Base name of the image is invalid.")
    # Check if the image file exists
    if not os.path.exists(f'test_images/{name}.jpg'):
        raise FileNotFoundError(f"Image file not found: {name}.jpg")
    return True
