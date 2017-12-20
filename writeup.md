**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

The project is here: [https://github.com/balintfodor/CarND-Advanced-Lane-Lines](https://github.com/balintfodor/CarND-Advanced-Lane-Lines)

The writup is here: [https://github.com/balintfodor/CarND-Advanced-Lane-Lines/blob/master/writeup.md](https://github.com/balintfodor/CarND-Advanced-Lane-Lines/blob/master/writeup.md)

The video showing the final results can be found here: [https://youtu.be/bhCGGA5ezXY](https://youtu.be/bhCGGA5ezXY)

The video breaks down to the steps of the algorithm can be found here: [https://youtu.be/durShBo2A7A](https://youtu.be/durShBo2A7A)

The results for the test images can be found here: [https://github.com/balintfodor/CarND-Advanced-Lane-Lines/tree/master/output_images](https://github.com/balintfodor/CarND-Advanced-Lane-Lines/tree/master/output_images)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

`calibrate.py` holds the code for the calibration. It takes a folder to search for images, tries to detect chessboards on the images, calibrates and saves the output to a numpy matrix file. I used `cv2.cornerSubPix` to refine the corner positions. The following images show the original image with chessboard detection (Fig1), the original image with reprojected chessboard coordinates (after calibration, using the calibration matrix) (Fig2) and the undistorted image (Fig3).

![](writeup_images/calib_orig.png)
Fig1. Original

![](writeup_images/calib_repro.png)
Fig2. Reprojected

![](writeup_images/calib_undist.png)
Fig3. Undistorted

An example of the output of `calibrate.py`:

	camera matrix
	[[  1.15694047e+03   0.00000000e+00   6.65948820e+02]
	 [  0.00000000e+00   1.15213880e+03   3.88784788e+02]
	 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
	distortion
	[[ -2.37638062e-01  -8.54041488e-02  -7.90999658e-04  -1.15882238e-04
	    1.05725943e-01]]
	reprojection error for 17 images with size (1280, 720) with (9, 6) chessboard size is 0.8457746481608099
	calibration saved to debug/calibration.npz

The reprojection error 0.8458 compared to the size of the images seems ok.

### Pipeline (single images)

I used the python package `consecution` for assembling and configuring the pipeline. It it is a simple, single threaded, abstract pipeline implementation. This way I only had to implement the processing nodes. For every step of the algorithm I implemented a class that represents one pipeline node. The code is in file `pipeline.py`.

The final assembly looks like:

	Undistort (1)
	| \
	|  LineScoreMap (2)
	|  PerspectiveWarp (3)
	|  LaneDetector (4)
	|  LanePainter (5)
	|  PerspectiveWarp (6)
	|  InfoPainter (7)
	| /
	Add (8)
	ImageLog (9)

I will discuss the node details by demostrating example processing outputs. The original image I will show the steps on can be seen in Fig4.

![](writeup_images/p00.png)
Fig4. Original

#### 1. Provide an example of a distortion-corrected image.

Undistort node (1) loads the camera matrix and the distrortion parameters and undistorts the input image using `cv2.remap` feedef with the precomputed mapping parameters with `cv2.initUndistortRectifyMap`.

![](writeup_images/p01.png)
Fig5. Undistorted

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Instead of producing a binary map, node LineScoreMap (2) generates a grayscale image that combines informations about the color and the gradients. LineScoreMap gives a score to every pixel. Higher scores should correspond to pixels part of a lane line.

First the RGB image is converted to the HSV color space. The gradient magnitude and direction maps are calculated both for the S and the V channels.

I chose a target number for the direction value empirically. A direction score map is generated by measuring the pixel value distance from the target direction. This map is normalized to the range [0, 1] and multiplied by the corresponding gradient magnitude map (which is also normalized to [0, 1]) (`pipeline.py:72-81`).

I chose three HSV color values from the test images that matches the color of the yellow line, a darker yellow line and the white line. For every selected HSV color a similarity map is generated (member function `color_score`). The L2 distance (in HSV space) of the pixel and the target color is cut at a maximal value, the image is normalized to [0, 1] and inversed. The color score map shows brighter values at colors close to the target color.

Finally, all the generated score maps are summed up, normalized to [0, 255] and converted to uint8 images.

![](writeup_images/p02.png)
Fig6. Score map

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I picked 4 points laying on the lane lines forming a trapezoid manually from the images. These are the source points for the perspective transformation (`pipeline.py:326`). I map this trapezoid to a 256x256 image with the node PerspectiveWarp (3). I chose the dst point parameters so the mapping cuts the hood of the car.

![](writeup_images/p03.png)
Fig7. Perspective transformation

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Class LaneDetector (4) is responsible for fitting a lane to the top-down viewed score map.

First lane line candidate points are selected by separately analyzing both the left and right halves of the image.

![](writeup_images/p04.png)
Fig8. Lane fitting

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
