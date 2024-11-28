import os
import argparse

from utils.config import *
from processing import *

def clear_output_data(clear: bool = False) -> None:
    """Clear the output data.

    Clear the output data from the previous run.
    """
    if clear:
        # Clear the output images from the outputs directory and all the subdirectories
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
