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

log_string = ""


def timer(**attrs) -> Callable:
    """Decorator to measure the execution time of a function.
    
    The decorator measures the execution time of a function and logs the
    elapsed time in milliseconds. The decorator can be used with the following
    optional arguments:
    
    - start: bool
        If True, log the start of the function.
    - end: bool
        If True, log the end of the function.
    - name: str
        The name of the function to log.
        
    Parameters
    ----------
    **attrs
        The optional arguments to the decorator.
        
    Returns
    -------
    Callable
        The decorated function.
    """
    def decorator(func: Callable) -> Callable:
        valid_args = ["start", "end", "name"]
        # Raise an error if an invalid argument is provided
        for key in attrs:
            if key not in valid_args:
                raise ValueError(f"Invalid argument: {key}")
        start: bool = attrs.get("start", False)
        name: str = attrs.get("name", func.__name__)
        name = name.strip()
        end: bool = attrs.get("end", False)
        # Raise an error if an invalid argument is provided
        if not isinstance(start, bool) or not isinstance(end, bool):
            raise TypeError("Invalid argument type. Expected bool.")
        if not isinstance(name, str):
            raise TypeError("Invalid argument type. Expected str.")

        def wrapper(*args, **kwargs):
            global log_string
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            log = logging.getLogger(
                func.__module__.lower().replace('processing.', '').capitalize()
            )

            if start:
                log_string = f"[ {name}: {elapsed_time:.2f} ms "
            else:
                log_string += f"{name}: {elapsed_time:.2f} ms "
            if end:
                log_string += "]"
                log.info(log_string)
                log_string = ""
            else:
                log_string += "| "
            return result
        return wrapper
    return decorator


def clear_output_data() -> None:
    """Clear the output data.

    Clear output images from the outputs directory.
    """
    for root, _, files in os.walk('outputs'):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.mp4'):
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
    ax1.set_title(image1_name, fontsize=40)
    ax2.imshow(image2)
    ax2.set_title(image2_name, fontsize=40)
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
    cv.circle(image, (int(top_right[0]), int(top_right[1])), 8, green, -1)
    cv.circle(image, (int(bottom_right[0]), int(bottom_right[1])), 8, red, -1)
    cv.circle(image, (int(bottom_left[0]), int(bottom_left[1])), 8, blue, -1)
    cv.circle(image, (int(top_left[0]), int(top_left[1])), 8, yellow, -1)

    return image


def get_parser() -> argparse.ArgumentParser:
    """Parse command-line arguments.

    Returns
    -------
    argparse.ArgumentParser: 
        The argument parser object.
    """
    parser = argparse.ArgumentParser(description="Road Lane Detection")
    parser.add_argument(
        "-c",
        "--calibrate",
        action="store_true",
        help="Perform camera calibration before processing"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear output images and videos before processing"
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="The name of the image to process. Use 'test_images' directory."
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        help="The name of the video to process. Use 'test_videos' directory."
    )
    parser.add_argument(
        "-s",
        "--store",
        action="store_true",
        help="Store the output images and videos in the 'outputs' directory"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode. Display additional information."
    )
    return parser


def validate_base_name(image_name: str | None, video_name: str | None) -> bool:
    """Validate the base name of the image or video.

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
    validate_output_directories()
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


def validate_output_directories() -> None:
    """Validate the output directories.

    Create the output directories if they do not exist.
    """
    directories = [
        'outputs',
        'outputs/undistorted',
        'outputs/thresholded',
        'outputs/warped',
        'outputs/final',
        'outputs/videos'
    ]
    for directory in directories:
        if not os.path.exists(directory):
            # Create the directory
            try:
                os.mkdir(directory)
                print(f"Directory '{directory}' created successfully.")
            except FileExistsError:
                print(f"Directory '{directory}' already exists.")
            except PermissionError:
                print(f"Permission denied: Unable to create '{directory}'.")
            except Exception as e:
                print(f"An error occurred: {e}")
