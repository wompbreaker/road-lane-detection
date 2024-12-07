# Road Lane Detection

## Goals

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

<!-- Calibration section -->
[image20]: ./markdown_images/calibration_comparison.jpg "Calibration Comparison"

<!-- Undistort section -->
[image30]: ./markdown_images/original.jpg "Original"
[image31]: ./markdown_images/undistorted.jpg "Undistorted"

<!-- Threshold section -->
[image40]: ./markdown_images/threshold_yellow_white.jpg "Filtered yellow and white lines"
[image41]: ./markdown_images/threshold_color.jpg "Sobel X transform"
[image42]: ./markdown_images/threshold_masked.jpg "Threshold masked"

<!-- Perspective transform section -->
[image50]: ./markdown_images/warped.jpg "Warped Image"

<!-- Line finding section -->
[image60]: ./markdown_images/final.jpg "Final Image"
[image61]: ./markdown_images/sliding_window.jpg "Sliding Window"

<!-- Calculation section -->
[image70]: ./markdown_images/final_with_info.jpg "Final Image With Info"

<!-- Video section -->
[video1]: ./markdown_images/project_video01_output.mp4 "Final Video"
---

### 1. Setup

#### 1.1 Make sure to have python and pip installed

#### 1.2 Set up virtual environment
Change directory to the root of the project directory with

`cd path/to/your/directory/road-lane-detection`. 

After this run `python -m venv .venv` to create a virtual environment. 

#### 1.3 Install dependencies
Activating the environment: First, run

`. .venv/Scripts/activate` 

to activate the virtual environment.

To install dependencies, run

`pip install -r requirements.txt`

#### 1.4 Starting the program
To start the program you can just run `python main.py` and the help menu will be displayed. The most common usage is processing a video and saving it after it's processed. Simply pick a video from the `test_videos` directory and run:

`python main.py --video project_video01 --store`

If you set up everything correctly, this will process and save the video.

### 2. Camera Calibration

#### The camera calibration is designed to perform camera calibration using a set of chessboard images. The code for this step is located in "./processing/calibration.py".  

This process computes the camera calibration matrix and distortion coefficients, which are needed for correcting lens distortion in images. The number of rows and columns of the chessboard, as well as the path to the calibration images, are retrieved from a configuration file (`utils/config.py`). Object points, representing 3D points in real-world space, are prepared using a NumPy array and a meshgrid. Two lists, `objpoints` and `imgpoints`, are initialized to store 3D object points and 2D image points from all images. The function iterates over each image file, reads the image with `imread`, converts it to grayscale with `cvtColor`, and finds the chessboard corners with `findChessboardCorners`. If corners are found, their positions are refined using `cornerSubPix` and added to the lists. The camera is then calibrated using these points with `calibrateCamera`, and if the calibration error exceeds a threshold, an error is logged and a `ValueError` is raised. The output of this function is a `.npz` file which contains calibration data that will be used in the next step.

After a camera is calibrated, the image output of it is:

![Calibration Comparison][image20]

### 3. Undistorting

#### The `undistort_image` function is designed to undistort an image using previously computed camera calibration parameters. The code for this step is located in "./processing/undistort.py".  

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Original][image30]

It begins by checking if the calibration data file exists at the specified path `(utils.CALIBRATION_DATA_PATH)`. The calibration data, including the camera matrix (`matrix`) and distortion coefficients (`dist_coeffs`), are loaded using `np.load`. An image (or a video frame) is read with (`cv.imread`) and passed to the `undistort_image` function. This function then checks if the image size is 1280x720 and resizes it if it's not. It returns the undistorted image after `cv.undistort()` is called with the provided calibration data. The output of this step is:

![Undistorted Image][image31]

### 4. Thresholding

#### The `threshold_image` function applies a combination of color and gradient threshold to turn an undistorted image into a binary version of it. The code for this step is located in "./processing/threshold.py". 

Processing for this function is divided into 3 steps. First step is filtering the image with `_filter_yellow_white` to only show yellow and white lines (these lines are the only ones relevant on the road). Before masking the undistorted image, I applied Gaussian blur to remove any noise for easier and more accurate results. First I convert the image from BGR colorspace to HLS. 

```py
image = __remove_noise(image)  # Apply noise removal to the image
hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
```

After that I defined thresholds for yellow and white colors and created masks.

```py
# Define the yellow color range
min_yellow = np.array([25 / 360 * 255, 100, 150])
max_yellow = np.array([50 / 360 * 255, 255, 255])

# Define the white color range
min_white = np.array([0, 220, 0])
max_white = np.array([150, 255, 255])

# Create masks for the yellow and white colors
yellow_mask = cv.inRange(hls, min_yellow, max_yellow)
white_mask = cv.inRange(hls, min_white, max_white)
```

After creating masks, I applied the bitwise OR to combine the masks and then bitwise AND to apply the mask to the original image.

```py
# Combine the masks
mask = cv.bitwise_or(yellow_mask, white_mask)

# Apply the mask to the image
result = cv.bitwise_and(image, image, mask=mask)

return result
```

The result after this step looks something like this:

![Filtered yellow and white lines][image40]

After filtering out only relevant colors, the image is then turned to HLS colorspace so we can separate lightness `(L)` and saturation `(S)` channels.  Sobel gradient transform on X axis is applied to the image. This is because we're only interested in vertical lines.

```py
sobel_x = cv.Sobel(l_channel, cv.CV_64F, 1, 0, ksize=9)
scaled_sobel = np.uint8(255 * np.abs(sobel_x)/np.max(np.abs(sobel_x)))
```

The `scaled_sobel` variable contains the absolute values of these gradients, scaled to an 8-bit range (0-255) for easier processing and visualization. The `sobel_mask` variable holds binary values for gradients that are between values 20 and 255. Gradients below 20 are considered too weak to be significant edges. 

```py
sobel_mask = (scaled_sobel > min_magnitude) & (scaled_sobel <= max_magnitude)
sobel_binary[sobel_mask] = 1
```
This code sets the corresponding pixels in the `sobel_binary` image to 1 (white) where the mask is `True`. This effectively highlights the edges detected by the Sobel operator within the specified magnitude range.

Similiar thing is done with the saturation channel - a binary mask is made to hold significant values for pixels with saturation values withing range of 100 - 255. Masks are then combined and applied to the original binary image. 
```py
s_mask = (s_channel > min_s) & (s_channel <= max_s)
s_binary[s_mask] = 1
```
The output of this step is:

![Sobel X transform][image41]

In some cases, turning images into their binary form also causes lane lines to appear "hollow". To fill these gaps in lines, we can dilate and erode an image by calling the `_fill_lines` function.

To adjust this image and turn it into a binary image, showing only Region of Interest with white lines, the `_mask_image` function is applied. It creates a polygon that represents the region of interest and fills it with white color. 

```py
mask_polyg = np.array(
    [[
        (offset, height),  # Bottom left
        (width // 2 - 45, height // 2 + 60),  # Top left
        (width // 2 + 45, height // 2 + 60),  # Top right
        (width - offset, height)  # Bottom right
    ]],
    dtype=np.int32
)
```

This function then applies the mask to the image that was filtered with `_color_threshold` function.

```py
# Fill the mask with the polygon
mask_image = cv.fillPoly(mask_image, mask_polyg, ignore_mask_color)
# Apply the mask to the thresholded image
masked_image = cv.bitwise_and(binary_image, mask_image)
```

And the final output of this processing module is:

![Color threshold][image42]

### 5. Perspective transform
#### The `perspective_transform` function applies a "bird's eye view" perspective transform to an image using the source and destination points. The code for this step is located in "./processing/perspective.py". 

This function applies a perspective transform to an image using the source and destination points. The source points are the region of interest (ROI) of the image and the destination points are the warped image. The source and destination points are hardcoded in the configuration file. These points are then used in the `_get_homography_matrix` function which returns homography matrix. This matrix is then used in the `_warp_image` function when calling the `cv.warpPerspective` function which returns a bird's-eye view perspective of the binary image.

The output of this step is:

![Warped Image][image50]

### 6. Lane-line pixel identification
#### Identifying lane-line pixels and fitting their positions with a polynomial is done in code that is located in "./processing/line_finding.py". 

The identification of lane-line pixels and fitting their positions with a polynomial is performed using the `slide_window` and `previous_window` functions. The `slide_window` function uses the sliding window technique to locate lane lines in the binary warped image. The image is divided into a specified number of windows, and the lane lines are found in each window by searching for nonzero pixels. To identify the base points of the lane lines we compute the histogram of the bottom half of the binary warped image by calling the `_get_histogram` function. After this, we identify the x and y positions of all nonzero pixels in the image:

```py
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
```

To find peaks of the left and right halves of the histogram we call the `histogram_peaks` function. After getting the base points of the left and right lane lines from the histogram, we initialize the current positions for the sliding windows. In the sliding windows loop (`./processing/line_finding.py:121`), for each window, we identify the boundaries and draw rectangles on the output image. After that we identify the nonzero pixels within the window and append their indices to the left and right indices list. If the number of pixels found is greater than `minpix` threshold, recent the next window based on the mean position of the pixels. 

After the loop, we combine the lists of indices into single arrays for the left and right lanes.

```py
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)
```

Extract left and right line pixels positions:

```py
left_x = nonzerox[left_lane_inds]
left_y = nonzeroy[left_lane_inds]
right_x = nonzerox[right_lane_inds]
right_y = nonzeroy[right_lane_inds]
```

Fitting a second order polynomial to each lane:

```py
left_fit = np.polyfit(left_y, left_x, 2)
right_fit = np.polyfit(right_y, right_x, 2)
```

The function fits a second-order polynomial to the detected lane pixels and optionally plots the results. It returns polynomial coefficients for the left and right lane lines.

![Sliding Window][image61]

The `previous_window` function is used to find the lane lines in the image. The lane lines are found using the previous lane lines as a reference. This function uses the polynomial coefficients from a previous frame to narrow the search area for lane lines in the current frame.

```py
# Identify the x and y positions of all nonzero pixels in the image
nonzero = warped_image.nonzero()
y = np.array(nonzero[0])
x = np.array(nonzero[1])

# The coefficients of the polynomial for the left lane
a_left = left_fit[0]
b_left = left_fit[1]
c_left = left_fit[2]

# The coefficients of the polynomial for the right lane
a_right = right_fit[0]
b_right = right_fit[1]
c_right = right_fit[2]
```

The search area is defined by a margin around the previous polynomial fit. The margin parameter is set in the configuration file.

```py
# Set the area of search based on activated x-values
left_lane_inds = (
    (x > (a_left * y**2 + b_left * y + c_left - margin))
    & (x < (a_left * y**2 + b_left * y + c_left + margin))
)
right_lane_inds = (
    (x > (a_right * y**2 + b_right * y + c_right - margin))
    & (x < (a_right * y**2 + b_right * y + c_right + margin))
)

left_x = x[left_lane_inds]
left_y = y[left_lane_inds]
right_x = x[right_lane_inds]
right_y = y[right_lane_inds]
```

The function identifies the nonzero pixels within the search area and fits a second-order polynomial to the detected lane pixels.

```py
# Fit a second order polynomial to each lane
left_fit = np.polyfit(left_y, left_x, 2)
right_fit = np.polyfit(right_y, right_x, 2)
```

The polynomial coefficients for the left and right lane lines are returned.

Output of this step will be similar to output of the final image. The only difference is that the final image will have some additional information written in the corner of the image. Output for this step is:
![Final image without info][image60]

### 7. Curvature radius and vehicle center offset
#### Calculation of radius of a curvature and vehicle center offset is done through functions `_calculate_curvature` and `_calculate_car_position`. The code for this step is located in "./processing/calculations.py". 

This module contains functions for calculating the curvature of the road and the position of the car relative to the center of the lane. It uses polynomial fitting to approximate the lane lines and then computes the radius of curvature and the car's offset from the lane center.

The purpose of the `_calculate_curvature` function is to calculate the curbature of the road in meters using the polynomial coefficients of the left and right lane lines. It converts pixel coordinates to real-world coordinates using conversion factors `XM_PER_PIX` and `YM_PER_PIX` which are located in the `config.py` file.

```py
# Pixel to meter conversion
YM_PER_PIX = 30 / 720
XM_PER_PIX = 3.7 / 700
```

It next fits quadratic polynomials to the lane line points. The formula for calculating the radius of a curvature is:

```py
# Formula for the curvature of the road: R = (1 + (2Ay + B)^2)^(3/2) / |2A|
num_left = (1 + (2 * a_left * y_eval * y_coeff + b_left)**2)**1.5
den_left = np.absolute(2 * a_left)
```

The purpose of the `_calculate_car_position` function is to calculate the position of the car relative to the center of the lane in meters. It determines the x-coordinates of the bottom of the left and right lane lines.

```py
height = frame.shape[0]
width = frame.shape[1]
car_location = width / 2

a_left = left_fit[0]
b_left = left_fit[1]
c_left = left_fit[2]

a_right = right_fit[0]
b_right = right_fit[1]
c_right = right_fit[2]

# Fine the x coordinate of the lane line bottom
bottom_left = a_left * height**2 + b_left * height + c_left
bottom_right = a_right * height**2 + b_right * height + c_right
```

It calculates the center of the lane and the car's offset from this center and then converts it from pixels to centimeters.

```py
center_lane = (bottom_right - bottom_left) / 2 + bottom_left
center_offset = (np.abs(car_location) - np.abs(center_lane))  # in pixels
center_offset = center_offset * utils.XM_PER_PIX * 100  # in cm
```

After all this is done, we display this information with the `display_curvature_offset` function

#### 8. Plotted image

![Final image with info][image70]

### Video output

https://github.com/user-attachments/assets/70b1f4df-4656-48b1-a5e8-0220a57288de

### Discussion

#### Current issues

Right now, as you can see in the video, it appears there are some wobbly lines plotted when there are dashed road lines detected. Also, there is some strange behaviour when the code tries to process lines in the shadow. This is most likely a problem with how the image is thresholded.

Another issue is with calculating the radius of a curvature. Most likely not a problem with the function itself, but with the parameters that are being passed to it.
