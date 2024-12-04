"""This module contains miscellaneous utility functions."""

from __future__ import annotations
import os
import argparse
import time
import logging
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

from utils.config import *
from processing import *

if TYPE_CHECKING:
    from cv2.typing import MatLike

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
    image1: MatLike, 
    image2: MatLike, 
    image1_name: str ="Image 1", 
    image2_name: str ="Image 2"
) -> None:
    """Compare two images side by side.

    Parameters
    ----------
    image1: MatLike
        The first image to compare.
    image2: MatLike
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
    image: MatLike, 
    points: np.ndarray, 
    color: tuple = (0, 255, 0)
) -> MatLike:
    """Draw points on an image.

    Draw points on an image using the specified color. The points are drawn
    as circles with a radius of 8 pixels.

    Parameters
    ----------
    image : MatLike
        The image to draw the points on.

    points : np.ndarray
        The points to draw on the image.

    color : tuple, optional
        The color of the points, by default (0, 255, 0).

    Returns
    -------
    MatLike
        The image with the points drawn on it.
    """
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
        "-i",
        "--image",
        type=str,
        help="The name of the image to process"
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        help="The name of the video to process"
    )
    parser.add_argument(
        "-s",
        "--store",
        action="store_true",
        help="Store the images after processing"
    )
    return parser.parse_args()

def validate_base_name(image_name: str | None, video_name: str | None) -> bool:
    """Validate the base name of the image or video.

    Validate the base name of the image or video to ensure that the file exists.

    Parameters
    ----------
    image_name : str
        The name of the image.
    video_name : str
        The name of the video.

    Returns
    -------
    bool
        True if the file exists, False otherwise.

    Raises
    ------
    ValueError
        If both image and video names are provided.
    FileNotFoundError
        If the image or video file is not found.
    """
    if image_name and video_name:
        raise ValueError("Both image and video names cannot be provided")
    
    if image_name:
        if not os.path.isfile(f"test_images/{image_name}.jpg"):
            raise FileNotFoundError(f"Image file not found: {image_name}")
        return True
    
    if video_name:
        if not os.path.isfile(f"test_videos/{video_name}.mp4"):
            raise FileNotFoundError(f"Video file not found: {video_name}")
        return True
    
    return False
