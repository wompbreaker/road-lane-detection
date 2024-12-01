import sys
import logging

import cv2 as cv
import numpy as np

from processing import *
from utils import *

def main():
    log = logging.getLogger(__name__.replace('__', ''))
    log.info(f"Python version: {sys.version}")
    log.info(f"OpenCV version: {cv.__version__}")
    args = parse_args()
    calibrate: bool = args.calibrate
    clear: bool = args.clear
    image_name = args.image
    video_name = args.video
    store_images: bool = args.store if args.store else STORE_IMAGES
    try:
        if validate_base_name(image_name, video_name):
            log.info(f"Processing image: {image_name}")
    except (ValueError, FileNotFoundError) as e:
        log.error(e)
        return
    
    if validate_base_name(image_name, video_name):
        log.info(f"Processing file: {image_name}")
    if clear:
        clear_output_data()
    try:
        camera_calibration(calibrate)
    except ValueError as e:
        log.error(e)

    # display_video(video_name)
    # exit(0)
    image = cv.imread(IMAGE_TO_UNDISTORT.format(name=image_name))
    try:
        undistorted_image = undistort_image(image)
    except FileNotFoundError as e:
        log.error(e)
    thresholded_image = threshold_image(undistorted_image)
    # compare_images(undistorted_image, thresholded_image)
    birds_eye_image = perspective_transform(thresholded_image)
    
    
    
    # Store the images after processing
    if store_images:
        cv.imwrite(
            UNDISTORTED_IMAGE_PATH.format(name=image_name), 
            undistorted_image
        )
        cv.imwrite(
            THRESHOLDED_IMAGE_PATH.format(name=image_name), 
            thresholded_image
        )
        cv.imwrite(
            PERSPECTIVE_IMAGE_PATH.format(name=image_name), 
            birds_eye_image
        )

    # Display the final output
    # cv.imshow('Output', birds_eye_image)
    # cv.waitKey(0)

def display_video(video_name: str):
    video_capture = cv.VideoCapture(PROJECT_VIDEO_PATH.format(name=video_name))
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_name}")
    
    out = cv.VideoWriter("final_video.mp4", cv.VideoWriter_fourcc(*'mp4v'), 30, (1280 * 3, 720))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            new_width, new_height = 1280, 720
            frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_LINEAR)
        
        undistorted_image = undistort_image(frame)
        warped_image = perspective_transform(undistorted_image)
        binary_image = threshold_image(warped_image)
        smth = slide_window(binary_image)
        return
        lane_coordinates = get_histogram(binary_image)
        left_lane, right_lane = 0, 0

        for point in lane_coordinates:
            if point <= 650:
                left_lane = point
            else:
                right_lane = point
                break
        
        vehicle_position = ((left_lane + right_lane) / 2 - 665) * 0.006
        filtered_lanes = detect_vertical_lines(binary_image, left_lane, right_lane)
        for x1, y1, x2, y2 in filtered_lanes:
            cv.line(warped_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        display_image = cv.cvtColor(warped_image, cv.COLOR_BGR2RGB)
        cv.putText(
            display_image, 
            f"Vehicle offset from center: {vehicle_position:.3f}m", 
            (5, 35), 
            cv.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2,
            cv.LINE_AA
        )
        # result_image = draw_lane_lines(frame, warped_image, display_image, filtered_lanes)
        cv.imshow("Processed Frame", np.hstack((frame, display_image)))
        # out.write(result_image)

        # Exit the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    with setup_logging():
        main()
