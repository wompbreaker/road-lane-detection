import contextlib
import os
import sys
import logging
from logging.handlers import RotatingFileHandler 
import argparse

import cv2 as cv

import config
from processing import *

@contextlib.contextmanager
def setup_logging():
    """Set up logging configuration.

    Set up logging configuration for the application. The log file is named
    'output.log' and is saved in the current working directory. The log level
    is set to INFO and the log file is rotated when it reaches 5 MB.
    """
    log = logging.getLogger()

    try:
        log.setLevel(logging.INFO)
        max_bytes = 5 * 1024 * 1024  # 5 MB
        handler = RotatingFileHandler(
            filename='output.log',
            encoding='utf-8',
            mode='w',
            maxBytes=max_bytes,
            backupCount=3
        )
        dt_fmt = "%d-%m-%Y %H:%M:%S"
        fmt = logging.Formatter(
            '[{asctime}] [{levelname:<7}] {name}: {message}', 
            dt_fmt,
            style='{'
        )
        handler.setFormatter(fmt)
        log.addHandler(handler)
        # Log to console
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        log.addHandler(console)

        yield
    finally:
        handlers = log.handlers[:]
        for handler in handlers:
            handler.close()
            log.removeHandler(handler)

def perspective_transform(*args, **kwargs):
    """Apply a perspective transform to an image.

    Transform an image to a bird's-eye view perspective.
    """
    ...

def detect_lane_lines(*args, **kwargs):
    """Detect lane pixels in an image.

    Identify lane lines in an image using sliding window search.
    """
    ...

def calculate_curvature(*args, **kwargs):
    """Calculate the radius of curvature of the lane.
    
    Determine the curvature of the lane and vehicle position with respect to
    the center of the lane.
    """
    ...

def warp_lane_boundaries(*args, **kwargs):
    """Warp lane boundaries back to the original image.

    Transform the detected lane boundaries back to the original image.
    """
    ...

def output_data(*args, **kwargs):
    """Output data to the image.

    Output visual display of the lane boundaries and numerical estimation 
    of lane curvature and vehicle position.
    """
    ...


def clear_output_data(clear: bool = False) -> None:
    """Clear the output data.

    Clear the output data from the previous run.
    """
    if clear:
        for file in os.listdir('outputs'):
            if file.endswith('.jpg'):
                os.remove(os.path.join('outputs', file))
    

def parse_args():
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
        help="Clear the calibration data"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=config.BASE_NAME,
        help="The name of the image to process"
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
    

def main():
    log = logging.getLogger(__name__.replace('__', ''))
    log.info(f"Python version: {sys.version}")
    log.info(f"OpenCV version: {cv.__version__}")
    args = parse_args()
    check_calibration: bool = args.calibrate
    clear: bool = args.clear
    image_name = args.name if args.name else config.BASE_NAME
    try:
        if validate_base_name(image_name):
            log.info(f"Processing image: {image_name}.jpg")
    except (ValueError, FileNotFoundError) as e:
        log.error(e)
        return
    
    clear_output_data(clear)
    if validate_base_name(image_name):
        log.info(f"Processing image: {image_name}.jpg")
    clear_output_data(clear)
    try:
        camera_calibration(check_calibration, clear)
    except ValueError as e:
        log.error(e)
    try:
        undistort_image(image_name)
    except FileNotFoundError as e:
        log.error(e)
    threshold_image(image_name)

if __name__ == '__main__':
    with setup_logging():
        main()
