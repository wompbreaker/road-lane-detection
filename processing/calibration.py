"""This module contains functions to perform camera calibration."""

import glob
import logging
import os

import cv2 as cv
import numpy as np

import utils

log = logging.getLogger("Calibration")


@utils.timer(name="calibration", start=True, end=True)
def camera_calibration(calibrate: bool = False) -> None:
    """Perform camera calibration using chessboard images.

    Compute the camera calibration matrix and distortion coefficients given
    a set of chessboard images. Calibration images path is located in the
    config module.
    """
    if utils.DEBUG:
        log.info("Calibrating the camera...")

    if not os.path.exists(utils.CALIBRATION_DATA_PATH):
        utils.validate_output_directories()

    if calibrate:
        _clear_calibration_data()

    if (
        not calibrate
        and os.path.exists(utils.CALIBRATION_DATA_PATH)
        and utils.DEBUG
    ):
        log.info("Camera calibration data already exists.")
        return
    rows = utils.ROWS
    cols = utils.COLS
    images_path = utils.CALIBRATION_IMAGES

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    # Prepare object points
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    for fname in glob.glob(images_path):
        image = cv.imread(fname)
        if image.shape[0] != 720 or image.shape[1] != 1280:
            image = cv.resize(image, (1280, 720), interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (rows, cols), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners = cv.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners)

    # Calibrate the camera
    ret, mtx, dist, _, _ = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    if ret > 1.5:
        raise ValueError(f"Calibration error too high: {ret}")
    # Save the calibration data
    np.savez(utils.CALIBRATION_DATA_PATH, mtx=mtx, dist=dist)

    if utils.DEBUG:
        log.info(f"Calibration data saved to: {utils.CALIBRATION_DATA_PATH}")


def _clear_calibration_data() -> None:
    """Clear the camera calibration data."""
    try:
        os.remove(utils.CALIBRATION_DATA_PATH)
        if utils.DEBUG:
            log.info("Removed camera calibration data")
    except FileNotFoundError:
        if utils.DEBUG:
            log.error("Camera calibration data not found.")
