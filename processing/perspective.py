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
    height = image.shape[0]
    width = image.shape[1]
    return cv.warpPerspective(image, h, (width, height))

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

    # Source points
    top_right = (731, 477)
    bottom_right = (1056, 689)
    bottom_left = (260, 689)
    top_left = (556, 477)
    src = [top_right, bottom_right, bottom_left, top_left]

    # Destination points
    top_right = (900, 0)
    bottom_right = (900, 689)
    bottom_left = (250, 689)
    top_left = (250, 0)
    dst = [top_right, bottom_right, bottom_left, top_left]

    image = _warp_image(image, src, dst)
    log.info("Perspective transform applied.")

    return cv.cvtColor(image, cv.COLOR_RGB2BGR)
