from typing import TYPE_CHECKING, Tuple

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import utils

if TYPE_CHECKING:
    from cv2.typing import MatLike

def get_histogram(image: 'MatLike') -> 'MatLike':
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
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    return histogram


def histogram_peaks(histogram: 'MatLike') -> Tuple[int, int]:
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
    midpoint = np.int32(histogram.shape[0] / 2)
    left_peak = np.argmax(histogram[:midpoint])
    right_peak = np.argmax(histogram[midpoint:]) + midpoint
    
    return left_peak, right_peak


def slide_window(binary_warped: 'MatLike', plot: bool = False) -> Tuple['MatLike', 'MatLike']:
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
    if len(binary_warped.shape) == 3 and binary_warped.shape[2] == 3:
        binary_warped = cv.cvtColor(binary_warped, cv.COLOR_BGR2GRAY)
    
    # Take a histogram of the bottom half of the image
    histogram = get_histogram(binary_warped)
    
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
            &  (nonzerox < win_xright_high)
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
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each lane
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if plot is True:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        image[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        image[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Plot the polynomial lines on the image
        plt.imshow(image)
        plt.title('Sliding Window')
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    
    return left_fit, right_fit

def previous_window(left_fit: 'MatLike', right_fit: 'MatLike', warped_image: 'MatLike'):
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
    (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
    (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def draw_lines(image, warped_image, left_fit, right_fit, perspective):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv.warpPerspective(color_warp, perspective, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv.addWeighted(image, 1, newwarp, 0.3, 0)

    return result

def radius_of_curvature(left_fit, right_fit, warped_image):
    ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return left_curverad, right_curverad
