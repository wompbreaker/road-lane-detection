import os
import logging

import cv2 as cv
import numpy as np

import utils

log = logging.getLogger("Undistort")

# Load the calibration data
with np.load(utils.CALIBRATION_DATA_PATH) as data:
    matrix = data['mtx']
    dist_coeffs = data['dist']
    
@utils.timer
def undistort_image(image_name: str = utils.BASE_NAME) -> cv.typing.MatLike:
    """Undistort an image using camera calibration parameters.

    Apply camera calibration parameters to undistort an image. The camera
    calibration data is loaded from the calibration data file located in the
    config module. The undistorted image is returned.

    Parameters
    -----------
    image_name : str
        The base name of the image to undistort.

    Returns
    --------
    cv.typing.MatLike
        The undistorted image.

    Raises:
        FileNotFoundError: If the camera calibration data is not found.
    """
    log.info('Undistorting the image...')
    if not os.path.exists(utils.CALIBRATION_DATA_PATH):
        raise FileNotFoundError("Camera calibration data not found.")

    # Read the image to be undistorted
    image = cv.imread(utils.IMAGE_TO_UNDISTORT.format(name=image_name))
    height = image.shape[0]
    width = image.shape[1]
    if height != 720 or width != 1280:
        image = cv.resize(image, (1280, 720))

    # Get the optimal camera matrix for better undistortion
    # ROI: Region of Interest
    # matrix, roi = cv.getOptimalNewCameraMatrix(
    #     matrix, dist_coeffs, (width, height), 1, (width, height)
    # )

    # Undistort the image
    undistorted_image = cv.undistort(image, matrix, dist_coeffs, None, matrix)

    # Crop the image based on the region of interest
    # x, y, width, height = roi
    # undistorted_image = undistorted_image[y:y+height, x:x+width]

    # Return the undistorted image
    return undistorted_image
