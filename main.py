"""
This module serves as the entry point for the application.

It initializes logging, parses command-line arguments, and performs
image or video processing based on the provided arguments.
"""

from __future__ import annotations
import sys
import logging
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np
import moviepy as mp

import processing
import utils

if TYPE_CHECKING:
    from cv2.typing import MatLike

def main():
    """Main function for image processing.

    This function initializes logging, parses command-line arguments,
    and performs image or video processing based on the provided arguments.
    """
    log = logging.getLogger(__name__.replace('__', ''))
    log.info(f"Python version: {sys.version}")
    log.info(f"OpenCV version: {cv.__version__}")
    parser = utils.get_parser()
    args = parser.parse_args()
    calibrate: bool = args.calibrate if args.calibrate else False
    clear: bool = args.clear if args.clear else False
    image_name = args.image
    video_name = args.video
    utils.DEBUG = args.debug if args.debug else False
    utils.STORE = args.store if args.store else False
    try:
        if utils.validate_base_name(image_name, video_name):
            log.info(f"Processing: {image_name if image_name else video_name}")
        else:
            parser.print_help()
            return
    except (ValueError, FileNotFoundError) as e:
        log.error(e)
        return

    if calibrate:
        try:
            processing.camera_calibration(calibrate)
        except ValueError as e:
            log.error(e)
            return

    if clear:
        utils.clear_output_data()
        
    if video_name:
        display_video(video_name)
        return

    image = cv.imread(utils.IMAGE_TO_UNDISTORT.format(name=image_name))
    if image is None:
        log.error(f"Image not found: {image_name}")
        return
    result = processing_pipeline(image)
    if result is None:
        log.error(f"Unable to process image: {image_name}")
        return

    undistorted_image, thresholded_image, warped_image, output = result
    if utils.STORE:
        cv.imwrite(
            utils.UNDISTORTED_IMAGE_PATH.format(name=image_name),
            undistorted_image
        )
        cv.imwrite(
            utils.THRESHOLDED_IMAGE_PATH.format(name=image_name),
            thresholded_image
        )
        cv.imwrite(
            utils.PERSPECTIVE_IMAGE_PATH.format(name=image_name),
            warped_image
        )
        cv.imwrite(
            utils.FINAL_IMAGE_PATH.format(name=image_name),
            output
        )


def processing_pipeline(image: MatLike) -> MatLike:
    """Process an image using the lane detection pipeline.

    Parameters
    ----------
    image : MatLike
        The image to process.

    Returns
    -------
    MatLike
        The processed image.
    """
    log = logging.getLogger(__name__.replace('__', ''))
    try:
        # Undistort the image to remove lens distortion
        undistorted_image = processing.undistort_image(image)
    except FileNotFoundError as e:
        log.error(e)
        return
    
    # Threshold the image to highlight lane lines
    thresholded_image = processing.threshold_image(undistorted_image)
    
    # Apply a perspective transform to the image
    warped_image = processing.perspective_transform(thresholded_image)

    try:
        # Fill in the lane lines
        left_fit, right_fit, _ = processing.previous_window(
            warped_image,
            left_fit,
            right_fit
        )

        # left_y, left_x = pixel_points[0], pixel_points[1]
        # right_y, right_x = pixel_points[2], pixel_points[3]

        mov_avg_left = np.append(mov_avg_left, np.array([left_fit]), axis=0)
        mov_avg_right = np.append(mov_avg_right, np.array([right_fit]), axis=0)

    except:
        # skip the frame if there are not enough pixels to fit the line
        try:
            left_fit, right_fit = processing.slide_window(warped_image)
            mov_avg_left = np.array([left_fit])
            mov_avg_right = np.array([right_fit])
        except:
            return None
        
    left_fit = np.array([
        np.mean(mov_avg_left[::-1][:, 0][0:10]),
        np.mean(mov_avg_left[::-1][:, 1][0:10]),
        np.mean(mov_avg_left[::-1][:, 2][0:10])
    ])
    right_fit = np.array([
        np.mean(mov_avg_right[::-1][:, 0][0:10]),
        np.mean(mov_avg_right[::-1][:, 1][0:10]),
        np.mean(mov_avg_right[::-1][:, 2][0:10])
    ])

    # Generate the plot points
    plot_y, left_fitx, right_fitx = processing.create_ploty(
        warped_image,
        left_fit,
        right_fit
    )

    # Draw the lane lines on the image
    frame_with_lines = processing.draw_lines(
        undistorted_image,
        warped_image,
        plot_y,
        left_fitx,
        right_fitx,
        False
    )

    left_y, left_x, right_y, right_x = None, None, None, None

    output = processing.display_curvature_offset(
        frame_with_lines,
        plot_y,
        left_fit,
        right_fit,
        left_y,
        left_x,
        right_y,
        right_x
    )

    return undistorted_image, thresholded_image, warped_image, frame_with_lines, output


def display_video(video_name: str):
    """Display the processed video.

    Display the processed video frame by frame. The video is processed
    by undistorting, thresholding, and applying a perspective transform
    to each frame. The processed frame is displayed alongside the
    original frame.

    Parameters
    -----------
    video_name : str
        The name of the video file to process.
    """
    video_capture = cv.VideoCapture(
        utils.PROJECT_VIDEO_PATH.format(name=video_name)
    )
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_name}")

    if utils.STORE:
        frames = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            new_width, new_height = 1280, 720
            frame = cv.resize(
                frame,
                (new_width, new_height),
                interpolation=cv.INTER_LINEAR
            )

        output = processing_pipeline(frame)
        if output is None:
            continue
        final = output[-1]
        if utils.STORE:
            final_rgb = cv.cvtColor(final, cv.COLOR_BGR2RGB)
            frames.append(final_rgb)

        cv.imshow('Video', final)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()

    if utils.STORE:
        clip = mp.ImageSequenceClip(frames, fps=20)
        clip.write_videofile(
            utils.VIDEO_OUTPUT_PATH.format(name=video_name),
            codec='libx264',
            audio=False
        )
if __name__ == '__main__':
    with utils.setup_logging():
        main()
