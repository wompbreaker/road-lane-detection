"""This module contains functions to find the lane lines in an image."""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
import logging

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import utils
from .perspective import get_inverse_perspective_matrix

if TYPE_CHECKING:
    from cv2.typing import MatLike

log = logging.getLogger("LineFinding")


def _get_histogram(image: MatLike) -> MatLike:
    """Get the histogram of an image.

    Get the histogram of an image. The histogram is calculated along the
    x-axis of the image.

    Parameters
    ----------
    image : MatLike
        The image to calculate the histogram.

    Returns
    -------
    MatLike
        The histogram of the image.
    """
    if utils.DEBUG:
        log.info("Calculating histogram of the image")
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    return histogram


def histogram_peaks(histogram: MatLike) -> Tuple[int, int]:
    """Get the peaks of a histogram.

    Get the peaks of a histogram. The histogram is divided in two halves
    and the peaks are calculated for each half. The peaks are the maximum
    values of the histogram.

    Parameters
    ----------
    histogram : MatLike
        The histogram to calculate the peaks.

    Returns
    -------
    Tuple
        A tuple containing the left and right peaks of the histogram.
    """
    if utils.DEBUG:
        log.info("Calculating histogram peaks")
    midpoint = np.int32(histogram.shape[0] / 2)
    left_peak = np.argmax(histogram[:midpoint])
    right_peak = np.argmax(histogram[midpoint:]) + midpoint

    return left_peak, right_peak


def slide_window(
    binary_warped: MatLike,
    plot: bool = False
) -> Tuple[MatLike, MatLike]:
    """Get the left and right lane lines using sliding window.

    The sliding window is used to find the lane lines in the image. The image
    is divided in nwindows number of windows and the lane lines are found in
    each window.

    Parameters
    ----------
    binary_warped : MatLike
        The binary warped image to find the lane lines.

    Returns
    -------
    Tuple
        A tuple containing the left and right lane lines.
    """
    if utils.DEBUG:
        log.info("Finding lane lines using sliding window")

    # Check if the image is a color image
    if len(binary_warped.shape) == 3 and binary_warped.shape[2] == 3:
        binary_warped = cv.cvtColor(binary_warped, cv.COLOR_BGR2GRAY)

    # Take a histogram of the bottom half of the image
    histogram = _get_histogram(binary_warped)

    # Create an output image to draw on and visualize the result
    image = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Set the height of sliding windows
    nwindows = 9
    window_height = np.int32(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    left_lane_inds = []
    right_lane_inds = []

    # Find the peak of the left and right halves of the histogram
    leftx_base, rightx_base = histogram_peaks(histogram)
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = utils.MARGIN
    minpix = utils.MINIMUM_PIXELS

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv.rectangle(
            image,
            (win_xleft_low, win_y_low),
            (win_xleft_high, win_y_high),
            (0, 255, 0),
            2
        )
        cv.rectangle(
            image,
            (win_xright_low, win_y_low),
            (win_xright_high, win_y_high),
            (0, 255, 0),
            2
        )

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        ).nonzero()[0]

        # Append the good indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If the number of pixels found > minpix, recenter next window
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    left_x = nonzerox[left_lane_inds]
    left_y = nonzeroy[left_lane_inds]
    right_x = nonzerox[right_lane_inds]
    right_y = nonzeroy[right_lane_inds]

    # Check if there are enough points to fit a polynomial
    if (
        len(left_x) < 3
        or len(left_y) < 3
        or len(right_x) < 3
        or len(right_y) < 3
    ):
        raise ValueError("Not enough points to fit a polynomial")

    # Fit a second order polynomial to each lane
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    if plot is True:
        a_left = left_fit[0]
        b_left = left_fit[1]
        c_left = left_fit[2]

        a_right = right_fit[0]
        b_right = right_fit[1]
        c_right = right_fit[2]

        binary_height = binary_warped.shape[0]
        # Generate x and y values for plotting
        plot_y = np.linspace(0, binary_height - 1, binary_height)
        left_fitx = a_left * plot_y**2 + b_left * plot_y + c_left
        right_fitx = a_right * plot_y**2 + b_right * plot_y + c_right

        # Create an image to draw on and an image to show the selection window
        image[left_y, left_x] = [255, 0, 0]
        image[right_y, right_x] = [0, 0, 255]

        # Plot the polynomial lines on the image
        plt.imshow(image)
        plt.title('Sliding Window')
        plt.plot(left_fitx, plot_y, color='yellow')
        plt.plot(right_fitx, plot_y, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    if utils.DEBUG:
        log.info("Lane lines found using sliding window")

    return left_fit, right_fit


def previous_window(
    warped_image: MatLike,
    left_fit: np.ndarray,
    right_fit: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """Get the left and right lane lines using previous window.

    The previous window is used to find the lane lines in the image. The
    lane lines are found using the previous lane lines as a reference.

    Parameters
    ----------
    left_fit : np.ndarray
        The left lane line from the previous frame.
    right_fit : np.ndarray
        The right lane line from the previous frame.
    warped_image : MatLike
        The binary warped image to find the lane lines.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Tuple]
        A tuple containing the left and right lane lines and pixel points.
    """
    if utils.DEBUG:
        log.info("Finding lane lines using previous window")

    margin = utils.MARGIN

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_image.nonzero()
    y = np.array(nonzero[0])
    x = np.array(nonzero[1])

    # The coefficients of the polynomial for the left lane
    a_left = left_fit[0]
    b_left = left_fit[1]
    c_left = left_fit[2]

    # The coefficients of the polynomial for the right lane
    a_right = right_fit[0]
    b_right = right_fit[1]
    c_right = right_fit[2]

    # Set the area of search based on activated x-values
    left_lane_inds = (
        (x > (a_left * y**2 + b_left * y + c_left - margin))
        & (x < (a_left * y**2 + b_left * y + c_left + margin))
    )
    right_lane_inds = (
        (x > (a_right * y**2 + b_right * y + c_right - margin))
        & (x < (a_right * y**2 + b_right * y + c_right + margin))
    )

    left_x = x[left_lane_inds]
    left_y = y[left_lane_inds]
    right_x = x[right_lane_inds]
    right_y = y[right_lane_inds]

    pixel_points = (left_x, left_y, right_x, right_y)

    # Fit a second order polynomial to each lane
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    if utils.DEBUG:
        log.info("Finished finding lane lines using previous window")

    return left_fit, right_fit, pixel_points


@utils.timer(name="plot")
def create_ploty(
    warped_image: MatLike,
    left_fit: np.ndarray,
    right_fit: np.ndarray
) -> np.ndarray:
    """Create the y values for plotting the lane lines.

    Create the y values for plotting the lane lines. The y values are
    generated using the height of the image.

    Parameters
    ----------
    warped_image : MatLike
        The binary warped image to generate the y values.

    Returns
    -------
    np.ndarray
        The y values for plotting the lane lines.
    """
    if utils.DEBUG:
        log.info("Creating y values for plotting")

    height = warped_image.shape[0]
    plot_y = np.linspace(0, height - 1, height)

    a_left = left_fit[0]
    b_left = left_fit[1]
    c_left = left_fit[2]

    a_right = right_fit[0]
    b_right = right_fit[1]
    c_right = right_fit[2]

    left_fitx = a_left * plot_y**2 + b_left * plot_y + c_left
    right_fitx = a_right * plot_y**2 + b_right * plot_y + c_right

    if utils.DEBUG:
        log.info("Finished creating y values for plotting")

    return plot_y, left_fitx, right_fitx


@utils.timer(name="draw", end=True)
def draw_lines(
    image: MatLike,
    warped_image: MatLike,
    plot_y: np.ndarray,
    left_fitx: np.ndarray,
    right_fitx: np.ndarray,
    plot: bool = False
):
    """Draw the lane lines on the image.

    Draw the lane lines on the image using the left and right lane lines
    and the y values for plotting. The lane lines are drawn on the image
    using the red color.

    Parameters
    ----------
    image : MatLike
        The original image to draw the lane lines.

    warped_image : MatLike
        The binary warped image to draw the lane lines.

    plot_y : np.ndarray
        The y values for plotting the lane lines.

    left_fitx : np.ndarray
        The x values for the left lane line.

    right_fitx : np.ndarray
        The x values for the right lane line.

    plot : bool, optional
        A flag to plot the lane lines, by default False.

    Returns
    -------
    MatLike
        The image with the lane lines drawn on it.
    """
    if utils.DEBUG:
        log.info("Drawing lane lines on the image")

    # Create an image to draw the lines on
    color_warp = np.zeros_like(warped_image).astype(np.uint8)

    if len(color_warp.shape) == 2:
        color_warp = np.dstack((color_warp, color_warp, color_warp))

    # Recast the x and y points into usable format for cv.fillPoly()
    points_left = np.array([np.transpose(np.vstack([left_fitx, plot_y]))])
    points_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, plot_y])))]
    )
    points = np.hstack((points_left, points_right))

    # Draw the lane onto the warped blank image
    cv.fillPoly(color_warp, np.int_([points]), (230, 45, 30))

    # Get the inverse perspective matrix
    inverse_matrix = get_inverse_perspective_matrix()

    # Warp the blank back to original image space
    new_warp = cv.warpPerspective(
        color_warp, inverse_matrix, (image.shape[1], image.shape[0])
    )

    # Combine the result with the original image
    result = cv.addWeighted(image, 1, new_warp, 0.5, 0)

    if utils.DEBUG:
        log.info("Finished drawing lane lines on the image")

    return result
