import os
import logging

import cv2 as cv
import numpy as np

import config

log = logging.getLogger(__name__)

def undistort_image() -> None:
    """Undistort an image using camera calibration parameters.

    Apply camera calibration parameters to undistort an image.

    Raises:
        FileNotFoundError: If the camera calibration data is not found.
    """
    if not os.path.exists(config.CALIBRATION_DATA_PATH):
        raise FileNotFoundError("Camera calibration data not found.")
    # Load the calibration data
    with np.load(config.CALIBRATION_DATA_PATH) as data:
        matrix = data['mtx']
        dist_coeffs = data['dist']

    # Read the image to be undistorted
    img = cv.imread(config.IMAGE_TO_UNDISTORT)
    height, width = img.shape[:2]

    # Get the optimal camera matrix for better undistortion
    # ROI: Region of Interest
    new_matrix, roi = cv.getOptimalNewCameraMatrix(
        matrix, dist_coeffs, (width, height), 1, (width, height)
    )

    # Undistort the image
    undistorted_image = cv.undistort(img, matrix, dist_coeffs, None, new_matrix)

    # Crop the image based on the region of interest
    x, y, width, height = roi
    undistorted_image = undistorted_image[y:y+height, x:x+width]

    # Save the undistorted image
    cv.imwrite(config.UNDISTORTED_IMAGE_PATH, undistorted_image)
    log.info(f"Undistorted image saved to {config.UNDISTORTED_IMAGE_PATH}.")
