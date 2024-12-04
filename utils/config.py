# Camera calibration parameters
ROWS = 6
COLS = 9
CALIBRATION_IMAGES = 'camera_cal/calibration*.jpg'
CALIBRATION_DATA_PATH = 'outputs/calibration_data.npz'

BASE_IMAGE_NAME = 'test1'
BASE_VIDEO_NAME = 'project_video01'
STORE_IMAGES = False

# Undistorted image output path
IMAGE_TO_UNDISTORT = 'test_images/{name}.jpg'

# Output path for the undistorted image
UNDISTORTED_IMAGE_PATH = 'outputs/undistorted/{name}_undistorted.jpg'

# Output path for the thresholded image
THRESHOLDED_IMAGE_PATH = 'outputs/thresholded/{name}_thresholded.jpg'

# Output path for the perspective transformed image
PERSPECTIVE_IMAGE_PATH = 'outputs/warped/{name}_perspective.jpg'

# Project video path
PROJECT_VIDEO_PATH = 'test_videos/{name}.mp4'
