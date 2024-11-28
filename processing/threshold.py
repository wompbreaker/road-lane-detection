import logging

import cv2 as cv
import numpy as np	

import config

log = logging.getLogger("Thresholding")

__all__ = ['threshold_image']

def __remove_noise(image: cv.typing.MatLike) -> cv.typing.MatLike:
    """Remove noise from an image.

    Apply fastNlMeansDenoisingColored to remove noise from an image. This
    function is used to remove noise from the image before applying color
    thresholding.

    Parameters
    ----------
    image : cv.typing.MatLike
        The image to remove noise from.

    Returns
    -------
    cv.typing.MatLike
        The image with noise removed.
    """
    log.info("Removing noise from the image")
    return cv.fastNlMeansDenoisingColored(image, 6, 6, 7, 21)

def _filter_yellow_white(image: cv.typing.MatLike) -> cv.typing.MatLike:
    """Filter yellow and white colors from an image.
    
    Apply color thresholding to the image to filter yellow and white colors.
    This helps in identifying lane lines. The function uses the HLS color space
    to filter yellow and white colors.
    
    Parameters
    ----------
    image : cv.typing.MatLike
        The image to filter yellow and white colors from.
        
    Returns
    -------
    cv.typing.MatLike
        The image after filtering yellow and white colors.
    """
    image = __remove_noise(image)  # Apply noise removal to the image
    hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)

    # Define the yellow color range
    min_yellow = np.array([25 / 360 * 255, 120, 120])
    max_yellow = np.array([70 / 360 * 255, 255, 255])

    # Define the white color range
    min_white = np.array([0, 200, 0])
    max_white = np.array([255, 255, 255])

    # Create masks for the yellow and white colors
    yellow_mask = cv.inRange(hls, min_yellow, max_yellow)
    white_mask = cv.inRange(hls, min_white, max_white)

    # Combine the masks
    mask = cv.bitwise_or(yellow_mask, white_mask)

    # Apply morphological operations to the mask to remove noise and fill in gaps
    kernel = np.ones((2, 2), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=1)  # Fill in gaps 
    mask = cv.erode(mask, kernel, iterations=1)  # Remove noise

    # Apply the mask to the image
    result = cv.bitwise_and(image, image, mask=mask)

    return result

def _color_threshold(image: cv.typing.MatLike) -> cv.typing.MatLike:
    """Ignore all colors except yellow and white.
    
    Apply color thresholding to the image to ignore all colors except yellow 
    and white. This helps in identifying lane lines. The function uses the L
    and S channels from the HLS color space. To ignore all colors except yellow
    and white, the function applies Sobel x to the L channel and thresholds it.
    
    Parameters
    ----------
    image : cv.typing.MatLike
        The image to apply color thresholding to.
        
    Returns
    -------
    cv.typing.MatLike
        The binary image after applying color thresholding.
    """
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
    log.info("Applying Sobel x to the L channel")
    sobel_x = cv.Sobel(l_channel, cv.CV_64F, 1, 0, ksize=5)
    abs_sobel = np.abs(sobel_x)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Apply thresholds to the Sobel x gradient
    min_magnitude = 20
    max_magnitude = 255
    sobel_mask = (scaled_sobel > min_magnitude) & (scaled_sobel <= max_magnitude)
    sobel_binary[sobel_mask] = 1

    # Apply thresholds to the S channel
    min_s = 190
    max_s = 255
    s_mask = (s_channel > min_s) & (s_channel <= max_s)
    s_binary[s_mask] = 1

    # Combine the binary matrices
    combined_mask = (s_binary == 1) | (sobel_binary == 1)
    combined_binary[combined_mask] = 1
    combined_binary = np.uint8(255 * combined_binary / np.max(combined_binary))

    return combined_binary

def _mask_image(binary_image: np.uint8) -> cv.typing.MatLike:
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
    cv.typing.MatLike
        The masked image that focuses on the region of interest.
    """
    offset = 100
    # Define the polygon for the mask
    mask_polyg = np.array(
        [[
            (offset, binary_image.shape[0]), 
            (binary_image.shape[1] / 2.5, binary_image.shape[0] / 1.65), 
            (binary_image.shape[1] / 1.8, binary_image.shape[0] / 1.65), 
            (binary_image.shape[1], binary_image.shape[0])
        ]], 
        np.int32
    )
    
    # Create a mask image
    mask_image = np.zeros_like(binary_image)

    # Fill the mask with the ignore mask color
    ignore_mask_color = 255

    # Fill the mask with the polygon
    cv.fillPoly(mask_image, mask_polyg, ignore_mask_color)
    # Apply the mask to the thresholded image
    masked_image = cv.bitwise_and(binary_image, mask_image)

    return masked_image


def threshold_image(image_name: str = config.BASE_NAME) -> None:
    """Threshold an image to identify lane lines.

    Apply a combination of color and gradient thresholds to identify lane 
    lines in an image. It needs to be called before perspective transform. It
    also needs to remove noise from the image.

    Parameters
    ----------
    image_name : str
        The name of the image to threshold, by default config.BASE_NAME
    """
    log.info("Thresholding the image")
    # Load the image
    image = cv.imread(config.UNDISTORTED_IMAGE_PATH.format(name=image_name))
    # Apply color thresholding
    filtered_image = _filter_yellow_white(image)
    # Apply color and gradient thresholding
    binary_image = _color_threshold(filtered_image)
    # Mask the region of interest
    masked_image = _mask_image(binary_image)
    # Save the thresholded image
    cv.imwrite(
        config.THRESHOLDED_IMAGE_PATH.format(name=image_name), 
        masked_image
    )
    log.info(
        f"Thresholded image saved to "
        f"{config.THRESHOLDED_IMAGE_PATH.format(name=image_name)}"
    )