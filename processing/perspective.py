"""This module contains functions to turn an image to birds-eye view."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np

import utils

if TYPE_CHECKING:
    from cv2.typing import MatLike

log = logging.getLogger("Perspective")
h_inversed = None


def _get_homography_matrix(src: MatLike, dst: MatLike) -> MatLike:
    """Get the homography matrix.

    Get the homography matrix using the source and destination points. The
    homography matrix is used to warp the image to a bird's-eye view.

    Parameters
    ----------
    src : MatLike
        The source points to calculate the homography matrix.
    dst : MatLike
        The destination points to calculate the homography matrix.

    Returns
    -------
    MatLike
        The homography matrix.
    """
    if utils.DEBUG:
        log.info("Calculating homography matrix...")
    src = np.float32([src])
    dst = np.float32([dst])

    h, _ = cv.findHomography(src, dst, cv.RANSAC)
    return h


def get_inverse_perspective_matrix() -> MatLike:
    """Get the inverse perspective matrix.

    Get the inverse perspective matrix using the warp matrix. The inverse
    perspective matrix is used to transform the points back to the original
    perspective.

    Returns
    -------
    MatLike
        The inverse perspective matrix.
    """
    if utils.DEBUG:
        log.info("Calculating inverse perspective matrix...")
    src = utils.SRC_POINTS
    dst = utils.DST_POINTS

    h = _get_homography_matrix(src, dst)
    return np.linalg.inv(h)


def _warp_image(
    image: MatLike,
) -> MatLike:
    """Warp an image to a bird's-eye view.

    Warp an image to a bird's-eye view using the source and destination points.
    The source points are the region of interest (ROI) of the image and the
    destination points are the warped image.

    Parameters
    ----------
    image : MatLike
        The image to warp.

    src : MatLike
        The source points to warp the image.

    dst : MatLike
        The destination points to warp the image.

    Returns
    -------
    MatLike
        A bird's-eye view perspective of the image.
    """
    if utils.DEBUG:
        log.info("Warping image...")

    # Source points
    src = utils.SRC_POINTS

    # Destination points
    dst = utils.DST_POINTS

    h = _get_homography_matrix(src, dst)
    height = image.shape[0]
    width = image.shape[1]

    if utils.DEBUG:
        log.info("Warping image complete.")
    return cv.warpPerspective(image, h, (width, height))


@utils.timer(name="warp")
def perspective_transform(binary_image: MatLike) -> MatLike:
    """Apply a perspective transform to an image.

    Apply a perspective transform to an image using the source and destination
    points. The source points are the region of interest (ROI) of the image
    and the destination points are the warped image. The source points are
    hardcoded and the destination points are calculated based on the source
    points.

    Parameters
    ----------
    binary_image : MatLike
        A thresholded binary image to apply the perspective transform.

    Returns
    -------
    MatLike
        A bird's-eye view perspective of the binary image.
    """
    if utils.DEBUG:
        log.info("Applying perspective transform...")
    image = cv.cvtColor(binary_image, cv.COLOR_BGR2RGB)

    image = _warp_image(image)
    if utils.DEBUG:
        log.info("Perspective transform applied.")

    return cv.cvtColor(image, cv.COLOR_RGB2BGR)
