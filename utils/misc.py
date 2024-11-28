import os
import argparse
import time
import logging
from typing import Callable

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
