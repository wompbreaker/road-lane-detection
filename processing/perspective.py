import logging

import cv2 as cv
import numpy as np

import utils

log = logging.getLogger("Perspective")

def _get_homography_matrix(src: np.ndarray, dst: np.ndarray) -> cv.typing.MatLike:
    """Get the homography matrix.
    
    Get the homography matrix using the source and destination points. The
    homography matrix is used to warp the image to a bird's-eye view.
    
    Parameters
    ----------
    src : np.ndarray
        The source points to calculate the homography matrix.
    dst : np.ndarray
        The destination points to calculate the homography matrix.
        
    Returns
    -------
    cv.typing.MatLike
        The homography matrix.
    """
    src = np.float32([src])
    dst = np.float32([dst])
    
    h, _ = cv.findHomography(src, dst, cv.RANSAC)
    return h

def _warp_image(
    image: cv.typing.MatLike, 
    src: np.ndarray, 
    dst: np.ndarray
) -> cv.typing.MatLike:
    """Warp an image to a bird's-eye view.

    Warp an image to a bird's-eye view using the source and destination points.
    The source points are the region of interest (ROI) of the image and the
    destination points are the warped image.

    Parameters
    ----------
    image : np.ndarray
        The image to warp.

    src : np.ndarray
        The source points to warp the image.

    dst : np.ndarray
        The destination points to warp the image.

    Returns
    -------
    cv.typing.MatLike
        A bird's-eye view perspective of the image.
    """
    h = _get_homography_matrix(src, dst)
    
    return cv.warpPerspective(image, h, (image.shape[1], image.shape[0]))

def _draw_points(
    image: cv.typing.MatLike, 
    points: np.ndarray, 
    color: tuple = (0, 255, 0)
) -> cv.typing.MatLike:
    """Draw points on an image.

    Draw points on an image using the specified color. The points are drawn
    as circles with a radius of 8 pixels.

    Parameters
    ----------
    image : cv.typing.MatLike
        The image to draw the points on.

    points : np.ndarray
        The points to draw on the image.

    color : tuple, optional
        The color of the points, by default (0, 255, 0).

    Returns
    -------
    cv.typing.MatLike
        The image with the points drawn on it.
    """
    for point in points:
        cv.circle(image, (int(point[0]), int(point[1])), 8, color, -1)

    return image

@utils.timer
def perspective_transform(binary_image: cv.typing.MatLike) -> cv.typing.MatLike:
    """Apply a perspective transform to an image.
    
    Apply a perspective transform to an image using the source and destination
    points. The source points are the region of interest (ROI) of the image
    and the destination points are the warped image. The source points are
    hardcoded and the destination points are calculated based on the source
    points.

    Parameters
    ----------
    binary_image : cv.typing.MatLike
        A thresholded binary image to apply the perspective transform.

    Returns
    -------
    cv.typing.MatLike
        A bird's-eye view perspective of the binary image.
    """
    log.info("Applying perspective transform...")
    image = cv.cvtColor(binary_image, cv.COLOR_BGR2RGB)
    image_width = image.shape[1]
    image_height = image.shape[0]
    line_dst_offset = 180

    # Source points
    top_left = (image_width * 0.44, image_height * 0.65)
    top_right = (image_width * 0.58, image_height * 0.65)
    bottom_right = (image_width * 0.9, image_height)
    bottom_left = (image_width * 0.25, image_height)
    src = [top_left, top_right, bottom_right, bottom_left]

    # Destination points
    dst = [
        [src[3][0] + line_dst_offset, 0],
        [src[2][0] - line_dst_offset, 0],
        [src[2][0] - line_dst_offset, image_height],
        [src[3][0] + line_dst_offset, image_height]
    ]

    # image = _draw_points(image.copy(), src)
    image = _warp_image(image, src, dst)
    log.info("Perspective transform applied.")

    return cv.cvtColor(image, cv.COLOR_RGB2BGR)
