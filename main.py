"""
This module serves as the entry point for the application.

It initializes logging, parses command-line arguments, and performs
image or video processing based on the provided arguments.
"""

import sys
import logging

import cv2 as cv
import numpy as np

import processing
import utils


def main():
    """Main function for image processing.

    This function initializes logging, parses command-line arguments,
    and performs image or video processing based on the provided arguments.
    """
    log = logging.getLogger(__name__.replace('__', ''))
    log.info(f"Python version: {sys.version}")
    log.info(f"OpenCV version: {cv.__version__}")
    args = utils.parse_args()
    calibrate: bool = args.calibrate
    clear: bool = args.clear
    image_name = args.image
    video_name = args.video
    store_images: bool = args.store if args.store else utils.STORE_IMAGES
    try:
        if utils.validate_base_name(image_name, video_name):
            log.info(f"Processing image: {image_name}")
    except (ValueError, FileNotFoundError) as e:
        log.error(e)
        return

    if utils.validate_base_name(image_name, video_name):
        log.info(f"Processing: {image_name if image_name else video_name}")
    if clear:
        utils.clear_output_data()
    try:
        processing.camera_calibration(calibrate)
    except ValueError as e:
        log.error(e)

    # display_video(video_name)
    # exit(0)
    image = cv.imread(utils.IMAGE_TO_UNDISTORT.format(name=image_name))
    try:
        undistorted_image = processing.undistort_image(image)
    except FileNotFoundError as e:
        log.error(e)
    thresholded_image = processing.threshold_image(undistorted_image)
    # compare_images(undistorted_image, thresholded_image)
    birds_eye_image = processing.perspective_transform(thresholded_image)
    processing.slide_window(birds_eye_image)

    # Store the images after processing
    if store_images:
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
            birds_eye_image
        )

    # Display the final output
    # cv.imshow('Output', birds_eye_image)
    # cv.waitKey(0)


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

    out = cv.VideoWriter(
        "final_video.mp4",
        cv.VideoWriter_fourcc(*'mp4v'),
        30,
        (1280 * 3, 720)
    )

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

        undistorted_image = processing.undistort_image(frame)
        warped_image = processing.perspective_transform(undistorted_image)
        binary_image = processing.threshold_image(warped_image)

        display_image = cv.cvtColor(warped_image, cv.COLOR_BGR2RGB)
        cv.imshow("Processed Frame", np.hstack((frame, display_image)))

        # Exit the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    with utils.setup_logging():
        main()
