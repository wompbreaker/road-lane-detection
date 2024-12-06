"""This module contains functions to threshold an image."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np

import utils

log = logging.getLogger("Thresholding")

__all__ = ['threshold_image']

if TYPE_CHECKING:
    from cv2.typing import MatLike


def __remove_noise(image: MatLike) -> MatLike:
    """Remove noise from an image.

    Apply GaussianBlur to remove noise from an image. This
    function is used to remove noise from the image before applying color
    thresholding.

    Parameters
    ----------
    image : MatLike
        The image to remove noise from.

    Returns
    -------
    MatLike
        The image with noise removed.
    """
    if utils.DEBUG:
        log.info("Removing noise from the image")
    return cv.GaussianBlur(image, (5, 5), 0)


def _filter_yellow_white(image: MatLike) -> MatLike:
    """Filter yellow and white colors from an image.

    Apply color thresholding to the image to filter yellow and white colors.
    This helps in identifying lane lines. The function uses the HLS color space
    to filter yellow and white colors.

    Parameters
    ----------
    image : MatLike
        The image to filter yellow and white colors from.

    Returns
    -------
    MatLike
        The image after filtering yellow and white colors.
    """
    if utils.DEBUG:
        log.info("Filtering yellow and white colors from the image")
    image = __remove_noise(image)  # Apply noise removal to the image
    hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)

    # Define the yellow color range
    min_yellow = np.array([25 / 360 * 255, 100, 150])
    max_yellow = np.array([50 / 360 * 255, 255, 255])

    # Define the white color range
    min_white = np.array([0, 220, 0])
    max_white = np.array([150, 255, 255])

    # Create masks for the yellow and white colors
    yellow_mask = cv.inRange(hls, min_yellow, max_yellow)
    white_mask = cv.inRange(hls, min_white, max_white)

    # Combine the masks
    mask = cv.bitwise_or(yellow_mask, white_mask)

    # Apply the mask to the image
    result = cv.bitwise_and(image, image, mask=mask)

    if utils.DEBUG:
        log.info("Finished filtering yellow and white colors")
    return result


def _color_threshold(image: MatLike) -> MatLike:
    """Ignore all colors except yellow and white.

    Apply color thresholding to the image to ignore all colors except yellow
    and white. This helps in identifying lane lines. The function uses the L
    and S channels from the HLS color space. To ignore all colors except yellow
    and white, the function applies Sobel x to the L channel and thresholds it.

    Parameters
    ----------
    image : MatLike
        The image to apply color thresholding to.

    Returns
    -------
    MatLike
        The binary image after applying color thresholding.
    """
    if utils.DEBUG:
        log.info("Applying color thresholding to the image")
    # Convert the image to HLS color space
    hls = cv.cvtColor(image, cv.COLOR_RGB2HLS)
    # Separate the L and S channels
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # binary matrices for the thresholds
    sobel_binary = np.zeros(l_channel.shape, np.float32)
    s_binary = sobel_binary.copy()
    combined_binary = s_binary

    # Apply Sobel x to the L channel
    if utils.DEBUG:
        log.info("Applying Sobel x to the L channel")
    sobel_x = cv.Sobel(l_channel, cv.CV_64F, 1, 0, ksize=9)
    scaled_sobel = np.uint8(255 * np.abs(sobel_x) / np.max(np.abs(sobel_x)))

    # Apply thresholds to the Sobel x gradient
    min_magnitude = 20
    max_magnitude = 255
    sobel_mask = (
        (scaled_sobel > min_magnitude) 
        & (scaled_sobel <= max_magnitude)
    )
    sobel_binary[sobel_mask] = 1

    # Apply thresholds to the S channel
    min_s = 100
    max_s = 255
    s_mask = (s_channel > min_s) & (s_channel <= max_s)
    s_binary[s_mask] = 1

    # Combine the binary matrices
    combined_mask = (s_binary == 1) | (sobel_binary == 1)
    combined_binary[combined_mask] = 1
    combined_binary = np.uint8(255 * combined_binary / np.max(combined_binary))
    
    if utils.DEBUG:
        log.info("Finished applying color thresholding")

    return combined_binary


def _mask_image(binary_image: np.uint8) -> MatLike:
    """Mask the region of interest in the image.

    Apply a mask to the image to focus on the region of interest. The function
    creates a polygon that represents the region of interest and fills it with
    the ignore mask color. The function then applies the mask to the binary
    image.

    Parameters
    ----------
    binary_image : np.uint8
        The binary image to mask.

    Returns
    -------
    MatLike
        The masked image that focuses on the region of interest.
    """
    if utils.DEBUG:
        log.info("Masking the region of interest")
    offset = 100
    # Define the polygon for the mask
    height = binary_image.shape[0]
    width = binary_image.shape[1]
    mask_polyg = np.array(
        [[
            (offset, height),  # Bottom left
            (width // 2 - 45, height // 2 + 60),  # Top left
            (width // 2 + 45, height // 2 + 60),  # Top right
            (width - offset, height)  # Bottom right
        ]],
        dtype=np.int32
    )

    # Create a mask image
    mask_image = np.zeros_like(binary_image)

    # Fill the mask with the ignore mask color
    ignore_mask_color = (255, 255, 255)

    # Fill the mask with the polygon
    mask_image = cv.fillPoly(mask_image, mask_polyg, ignore_mask_color)
    # Apply the mask to the thresholded image
    masked_image = cv.bitwise_and(binary_image, mask_image)
    if utils.DEBUG:
        log.info("Masking the region of interest complete")

    # Turn the masked image lines from blue to white
    # white_masked_image = cv.inRange(masked_image, 1, 255)

    return masked_image


def _fill_lines(image: MatLike) -> MatLike:
    """Fill the lane lines in the image.

    By dilating and eroding the image, the function fills the lane lines in the
    image. This helps in identifying the lane lines more accurately.

    Parameters
    ----------
    image : MatLike
        The image to fill the lane lines in.

    Returns
    -------
    MatLike
        The image with the lane lines filled.
    """
    if utils.DEBUG:
        log.info("Filling the lane lines in the image")
    kernel = np.ones((5, 5), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    image = cv.erode(image, kernel, iterations=1)

    if utils.DEBUG:
        log.info("Lane lines filled in the image")
    return image


@utils.timer(name="threshold")
def threshold_image(undistorted_image: MatLike) -> MatLike:
    """Threshold an image to identify lane lines.

    Apply a combination of color and gradient thresholds to identify lane
    lines in an image. It needs to be called before perspective transform. It
    also needs to remove noise from the image.

    Parameters
    ----------
    undistorted_image : MatLike
        The undistorted image that needs to be thresholded.

    Returns
    -------
    MatLike
        A binary image after applying color and gradient thresholds.
    """
    if utils.DEBUG:
        log.info("Thresholding the image")
    # Apply color thresholding
    filtered_image = _filter_yellow_white(undistorted_image)

    # Apply color and gradient thresholding
    binary_image = _color_threshold(filtered_image)

    # Fill the lane lines in the image
    filled_binary_image = _fill_lines(binary_image)

    # Mask the region of interest
    masked_image = _mask_image(filled_binary_image)

    if utils.DEBUG:
        log.info("Thresholding complete")

    return masked_image
