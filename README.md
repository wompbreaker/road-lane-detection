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

[image0]: ./markdown_images/original.jpg "Original"
[image1]: ./markdown_images/undistorted_no_roi.jpg "Undistorted Without ROI"
[image2]: ./markdown_images/undistorted_with_roi.jpg "Undistorted With ROI"
[image3]: ./markdown_images/threshold_yellow_white.jpg "Filtered yellow and white lines"
[image4]: ./markdown_images/threshold_color.jpg "Sobel X transform"
[image5]: ./markdown_images/threshold_masked.jpg "Threshold masked"

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

### 2. Camera Calibration

#### The camera calibration is designed to perform camera calibration using a set of chessboard images. The code for this step is located in "./processing/calibration.py".  

This process computes the camera calibration matrix and distortion coefficients, which are needed for correcting lens distortion in images. The number of rows and columns of the chessboard, as well as the path to the calibration images, are retrieved from a configuration file. Object points, representing 3D points in real-world space, are prepared using a NumPy array and a meshgrid. Two lists, `objpoints` and `imgpoints`, are initialized to store 3D object points and 2D image points from all images. The function iterates over each image file, reads the image with `imread`, converts it to grayscale with `cvtColor`, and finds the chessboard corners with `findChessboardCorners`. If corners are found, their positions are refined using `cornerSubPix` and added to the lists. The camera is then calibrated using these points with `calibrateCamera`, and if the calibration error exceeds a threshold, an error is logged and a `ValueError` is raised. The output of this function is a `.npz` file which contains calibration data that will be used in the next step.

### 3. Undistorting

#### The `undistort_image` function is designed to undistort an image using previously computed camera calibration parameters. The code for this step is located in "./processing/undistort.py".  

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Original][image0]

It begins by checking if the calibration data file exists at the specified path `(utils.CALIBRATION_DATA_PATH)`. The calibration data, including the camera matrix (`matrix`) and distortion coefficients (`dist_coeffs`), is loaded using `np.load`. The function then reads (`imread`) the image to be undistorted from the path specified in `utils.IMAGE_TO_UNDISTORT` and retrieves its dimensions. To achieve better undistortion, it computes an optimal new camera matrix and the region of interest (ROI) using `getOptimalNewCameraMatrix`. Before taking the Region of Interest (`roi`), the picture looks like this:

![Undistorted Without ROI][image1]

To avoid having these black corners, it adds 2 lines of code which will represent RoI.

```py
# Crop the image based on the region of interest
x, y, width, height = roi
undistorted_image = undistorted_image[y:y+height, x:x+width]
# Save the undistorted image
cv.imwrite(utils.UNDISTORTED_IMAGE_PATH, undistorted_image)
```
The final output of step 2 is returned as an undistorted image with its Region of Interest:

![Undistorted With ROI][image2]


### 4. Thresholding

#### The `threshold_image` function applies a combination of color and gradient threshold to turn an undistorted image into a binary version of it. The code for this step is located in "./processing/threshold.py". 

Processing for this function is divided into 3 steps. First step is filtering the image with `_filter_yellow_white` to only show yellow and white lines (these lines are the only ones relevant on the road). Before masking the original image, I applied Gaussian blur to remove any noise for easier and more accurate results. First I convert the image from BGR colorspace to HLS. 
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

![Filtered yellow and white lines][image3]

After filtering out only relevant colors, the image is then turned to HLS colorspace so we can separate lightness `(L)` and saturation `(S)` channels.  Sobel gradient transform on X axis is applied to the image. This is because we're only interested in vertical lines.

```py
sobel_x = cv.Sobel(l_channel, cv.CV_64F, 1, 0, ksize=9)
scaled_sobel = np.uint8(255 * np.abs(sobel_x))
```
The `scaled_sobel` variable contains the absolute values of these gradients, scaled to an 8-bit range (0-255) for easier processing and visualization. The `sobel_mask` variable holds binary values for gradients that are between values 20 and 255. Gradients below 20 are considered too weak to be significant edges. 

```py
sobel_mask = (scaled_sobel > min_magnitude) & (scaled_sobel <= max_magnitude)
sobel_binary[sobel_mask] = 1
```
This code sets the corresponding pixels in the `sobel_binary` image to 1 (white) where the mask is `True`. This effectively highlights the edges detected by the Sobel operator withing the specified magnitude range.

Similiar thing is done with the saturation channel - a binary mask is made to hold significant values for pixels with saturation values withing range of 100 - 255. Masks are then combined and applied to the original binary image. 
```py
s_mask = (s_channel > min_s) & (s_channel <= max_s)
s_binary[s_mask] = 1
```
The output of this step is:

![Sobel X transform][image4]

To adjust this image and turn it into a binary image, showing only Region of Interest with white lines, the `_mask_image` function is applied. It creates a polygon that represents the region of interest and fills it with white color. 
```py
mask_polyg = np.array(
    [[
        (offset, binary_image.shape[0]),  # Bottom left
        (binary_image.shape[1] / 2.5, binary_image.shape[0] / 1.65),  # Top left
        (binary_image.shape[1] / 1.8, binary_image.shape[0] / 1.65),  # Top right
        (binary_image.shape[1], binary_image.shape[0])  # Bottom right
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

![Color threshold][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
