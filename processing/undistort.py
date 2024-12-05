"""Module for undistorting images using camera calibration parameters."""

import os
import logging
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np

import utils

log = logging.getLogger("Undistort")

if TYPE_CHECKING:
    from cv2.typing import MatLike
    
# Load the calibration data
try:
    with np.load(utils.CALIBRATION_DATA_PATH) as data:
        matrix = data['mtx']
        dist_coeffs = data['dist']
except FileNotFoundError:
    log.error("Camera calibration data not found.")
    from .calibration import camera_calibration
    try:
        utils.validate_output_directories()
        camera_calibration(calibrate=True)
    except ValueError as e:
        log.error(e)


@utils.timer(name="undistort", start=True)
def undistort_image(image: 'MatLike') -> 'MatLike':
    """Undistort an image using camera calibration parameters.

    Apply camera calibration parameters to undistort an image. The camera
    calibration data is loaded from the calibration data file located in the
    config module. The undistorted image is returned.

    Parameters
    -----------
    image : MatLike
        The image to undistort.

    Returns
    --------
    MatLike
        The undistorted image.

    Raises:
        FileNotFoundError: If the camera calibration data is not found.
    """
    if utils.DEBUG:
        log.info('Undistorting the image...')
    if not os.path.exists(utils.CALIBRATION_DATA_PATH):
        raise FileNotFoundError("Camera calibration data not found.")

    height = image.shape[0]
    width = image.shape[1]
    if height != 720 or width != 1280:
        image = cv.resize(image, (1280, 720), interpolation=cv.INTER_LINEAR)

    # Get the optimal camera matrix for better undistortion
    # ROI: Region of Interest
    # matrix, roi = cv.getOptimalNewCameraMatrix(
    #     matrix, dist_coeffs, (width, height), 1, (width, height)
    # )

    # Undistort the image
    undistorted_image = cv.undistort(image, matrix, dist_coeffs, None, matrix)
    
    if utils.DEBUG:
        log.info('Image undistorted.')

    # Crop the image based on the region of interest
    # x, y, width, height = roi
    # undistorted_image = undistorted_image[y:y+height, x:x+width]

    # Return the undistorted image
    return undistorted_image
