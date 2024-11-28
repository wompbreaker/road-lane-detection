# Camera calibration parameters
ROWS = 6  
COLS = 9
CALIBRATION_IMAGES = 'camera_cal/calibration*.jpg'
CALIBRATION_DATA_PATH = 'outputs/calibration_data.npz'

BASE_NAME = 'test2'
# Undistorted image output path
IMAGE_TO_UNDISTORT = 'test_images/{name}.jpg'

# Output path for the undistorted image
UNDISTORTED_IMAGE_PATH = 'outputs/{name}_undistorted.jpg'

# Output path for the thresholded image
THRESHOLDED_IMAGE_PATH = 'outputs/{name}_thresholded.jpg'

# Project video path
PROJECT_VIDEO_PATH = 'test_videos/project_video03.mp4'