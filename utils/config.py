"""Configuration file for the project"""

DEBUG = False
STORE = False

# Camera calibration parameters
ROWS = 6
COLS = 9
CALIBRATION_IMAGES = 'camera_cal/calibration*.jpg'
CALIBRATION_DATA_PATH = 'outputs/calibration_data.npz'
CALIBRATED = False

BASE_IMAGE_NAME = 'test1'
BASE_VIDEO_NAME = 'project_video01'

# Undistorted image output path
IMAGE_TO_UNDISTORT = 'test_images/{name}.jpg'

# Output path for the undistorted image
UNDISTORTED_IMAGE_PATH = 'outputs/undistorted/{name}_undistorted.jpg'

# Output path for the thresholded image
THRESHOLDED_IMAGE_PATH = 'outputs/thresholded/{name}_thresholded.jpg'

# Output path for the perspective transformed image
PERSPECTIVE_IMAGE_PATH = 'outputs/warped/{name}_warped.jpg'

# Output path for the final image
FINAL_IMAGE_PATH = 'outputs/final/{name}_final.jpg'

# Source points
_SRC_TOP_RIGHT = (731, 477)
_SRC_BOTTOM_RIGHT = (1056, 689)
_SRC_BOTTOM_LEFT = (260, 689)
_SRC_TOP_LEFT = (556, 477)
SRC_POINTS = [
    _SRC_TOP_RIGHT,
    _SRC_BOTTOM_RIGHT,
    _SRC_BOTTOM_LEFT,
    _SRC_TOP_LEFT
]

# Destination points
_DST_TOP_RIGHT = (900, 0)
_DST_BOTTOM_RIGHT = (900, 689)
_DST_BOTTOM_LEFT = (250, 689)
_DST_TOP_LEFT = (250, 0)
DST_POINTS = [
    _DST_TOP_RIGHT,
    _DST_BOTTOM_RIGHT,
    _DST_BOTTOM_LEFT,
    _DST_TOP_LEFT
]

# Project video path
PROJECT_VIDEO_PATH = 'test_videos/{name}.mp4'

# Output path for the project video
VIDEO_OUTPUT_PATH = 'outputs/videos/{name}_output.mp4'

# Line finding parameters
MARGIN = 100
MINIMUM_PIXELS = 50
