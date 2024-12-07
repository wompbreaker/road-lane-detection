"""This module contains functions for calculating the curvature of the road and
the position of the car relative to the center of the lane."""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
import logging

import cv2 as cv
import numpy as np

import utils

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from numpy import ndarray

log = logging.getLogger("Calculations")

def _calculate_curvature(
    plot_y: ndarray,
    left_y: ndarray,
    left_x: ndarray,
    right_y: ndarray,
    right_x: ndarray
) -> Tuple[float, float]:
    """Calculate the curvature of the road.

    The function calculates the curvature of the road using the polynomial
    coefficients of the left and right lane lines. The curvature is calculated
    in meters.

    Parameters
    ----------
    plot_y : ndarray
        The y-values of the points on the road.

    left_y : ndarray
        The y-values of the left lane line.

    left_x : ndarray
        The x-values of the left lane line.

    right_y : ndarray
        The y-values of the right lane line.

    right_x : ndarray
        The x-values of the right lane line.

    Returns
    -------
    Tuple[float, float]
        The curvature of the left and right lane lines.
    """
    y_coeff = utils.YM_PER_PIX
    x_coeff = utils.XM_PER_PIX
    y_eval = np.max(plot_y)

    # Fit polynomial curves to the real world environment
    left_fit_curve = np.polyfit(left_y * y_coeff, left_x * (x_coeff), 2)
    right_fit_curve = np.polyfit(right_y * y_coeff, right_x * (x_coeff), 2)

    a_left = left_fit_curve[0]
    b_left = left_fit_curve[1]

    a_right = right_fit_curve[0]
    b_right = right_fit_curve[1]

    # Formula for the curvature of the road: R = (1 + (2Ay + B)^2)^(3/2) / |2A|
    num_left = (1 + (2 * a_left * y_eval * y_coeff + b_left)**2)**1.5
    den_left = np.absolute(2 * a_left)

    num_right = (1 + (2 * a_right * y_eval * y_coeff + b_right)**2)**1.5
    den_right = np.absolute(2 * a_right)

    # Calculate the radii of curvature
    left_curve = num_left / den_left
    right_curve = num_right / den_right

    print(left_curve, right_curve)

    return left_curve, right_curve


def _calculate_car_position(
    frame: ndarray,
    left_fit: ndarray,
    right_fit: ndarray
) -> float:
    """Calculate the position of the car relative to the center of the lane.

    The function calculates the position of the car relative to the center of
    the lane. The position is calculated in meters.

    Parameters
    ----------
    frame : ndarray
        The frame of the video.

    left_fit : ndarray
        The polynomial coefficients of the left lane line.

    right_fit : ndarray
        The polynomial coefficients of the right lane line.

    Returns
    -------
    float
        The position of the car relative to the center of the lane.
    """
    height = frame.shape[0]
    width = frame.shape[1]
    car_location = width / 2

    a_left = left_fit[0]
    b_left = left_fit[1]
    c_left = left_fit[2]

    a_right = right_fit[0]
    b_right = right_fit[1]
    c_right = right_fit[2]

    # Fine the x coordinate of the lane line bottom
    bottom_left = a_left * height**2 + b_left * height + c_left
    bottom_right = a_right * height**2 + b_right * height + c_right

    center_lane = (bottom_right - bottom_left) / 2 + bottom_left
    center_offset = (np.abs(car_location) - np.abs(center_lane))  # in pixels
    center_offset = center_offset * utils.XM_PER_PIX * 100  # in cm

    return center_offset


def display_curvature_offset(
    frame: MatLike,
    plot_y: ndarray,
    left_fit: float,
    right_fit: float,
    left_y: ndarray,
    left_x: ndarray,
    right_y: ndarray,
    right_x: ndarray
) -> MatLike:
    """Display the curvature and offset on the frame.

    The function displays the curvature and offset on the frame.

    Parameters
    ----------
    frame : MatLike
        The frame of the video.
    plot_y : ndarray
        The y-values of the points on the road.
    left_fit : float
        The polynomial coefficients of the left lane line.
    right_fit : float
        The polynomial coefficients of the right lane line.
    left_y : ndarray
        The y-values of the left lane line.
    left_x : ndarray
        The x-values of the left lane line.
    right_y : ndarray
        The y-values of the right lane line.
    right_x : ndarray
        The x-values of the right lane line.

    - **TODO**: Fix calculation of curvature after getting pixel points from
    the sliding window search.

    Returns
    -------
    MatLike
        The frame with the curvature and offset displayed.
    """
    if utils.DEBUG:
        log.info("Displaying curvature and offset...")
    frame_copy = frame.copy()
    center_offset = _calculate_car_position(frame, left_fit, right_fit)

    # left_curve, right_curve = _calculate_curvature(
    #     plot_y,
    #     left_y,
    #     left_x,
    #     right_y,
    #     right_x
    # )

    # cv.putText(
    #     frame_copy,
    #     f"Curve Radius: {int((left_curve + right_curve) / 2)} m",
    #     (10, 30),
    #     cv.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (255, 255, 255),
    #     2,
    #     cv.LINE_AA
    # )
    cv.putText(
        frame_copy,
        f"Center Offset: {center_offset:.2f} cm",
        (10, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv.LINE_AA
    )

    if utils.DEBUG:
        log.info(f"Center Offset: {center_offset:.2f} cm")

    return frame_copy
