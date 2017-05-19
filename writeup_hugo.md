## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[//]: # (Image References)

[image1]: ./output_images/undistorted_chessboard.jpg "Undistorted"
[image2]: ./output_images/undistorted_test6.jpg "Undistorted"
[image3]: ./output_images/binary_threshold_testimage.jpg "Binary Example"
[image4]: ./output_images/binary_warped_testimage.jpg "Warp Example"
[image5]: ./output_images/mask_lane_testimage.jpg "Fit Visual"
[image6]: ./output_images/final_testimage.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is composed of a calibrate_camera() function. The function will look for a calibration file and in case it 
doesnt find it it will calibrate the camera based on all images in ".\camera_cal" directory. It will start by preparing "object points", which will be the (x, y, z) 
coordinates of the chessboard corners in the world. The chessboard is assumed to be fixed on the (x, y) plane at z=0, such that the object points 
are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time.
All chessboard corners are found in the image.   The`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane 
with each successful chessboard detection.  

The arrays `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

The calibration function returns the calibration matrix and the distortion coeficients that will be saved, so that this operation does not have
to be performed each time the calibration data is required. Below extract of the calibrate_camera function defined in lines 61 to 98.


```python
def calibrate_camera(calibration_file = 'calibration.pickle', calibration_imgdir = '.\\camera_cal\\calibration*.jpg', verbose = False):
    CalibrationInFile = os.path.isfile(calibration_file)
    if CalibrationInFile:
        if verbose:
            print('Using calibration files.')
        with open('calibration.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
            mtx, dist = pickle.load(f)
    else:  # No calibration file yet:   
        images = glob.glob(calibration_imgdir)
        calibnx = 9
        calibny = 6
        objpoints = []
        imgpoints = []
        
        # create array of object points
        objp = np.zeros((calibnx*calibny,3), np.float32)
        objp[:,:2] = np.mgrid[0:calibnx,0:calibny].T.reshape(-1,2)
        
        ret = []
        for fname in images:
            img = mpimg.imread(fname)
            #convert to gray
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (calibnx,calibny), None)
            #Draw corners
            if ret == True:
                # Append image points for calibration
                imgpoints.append(corners)
                objpoints.append(objp)
            else:
                print('Corners not found!')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1], None, None)
        # Saving the objects:
        with open('calibration.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([mtx, dist], f)
        print('New calibration performed!')
    return mtx, dist

```


The lanefind.py file will start actual execution after line 372. It will call the calibrate_camera() function and will perform distortion correction
on a test image (calibration1.jpg). The result is given below:
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The pipeline code contains a flag (process_image_plotting) that will tell the code to plot and save all required pipeline images for this
writeup.

The main code sets the flag process_image_plotting to True, loads the test image and runs the pipeline function (process_image()):

```python
process_image_plotting = True
tesImg = '.\\test_images\\test1.jpg'
img_raw = mpimg.imread(tesImg)
process_image(img_raw)
```

The pipeline will calibrate and call undistort function. The figure below shows the original and undistorted image:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In order to detect lane pixels, a combination of color thresholding (specifically the s channel) and gradient thresholding (multiple gradients) were used.

The following were applied:  

1. Sobel in the x directions, with kernel of size=5, thresholded between 30 and 255, combined (AND logic) with '2' below:
2. Sobel in the y directions, with kernel of size=5, thresholded between 30 and 255
3. Magnitude (square root of sobex^2 and sobely^2), normalized between 0-255 and thresholded between 30 and 255, combined (AND logic) with '4' below:
4. Directional threshold, between 0.75 and 1.25.
5. Bulets 1 and 2, combined by "OR" operation with 3 and 4 also cobined by OR operation(1 OR 2 AND 3 OR 4):
6. The result of 1-4, is combined by OR operation with the s channel threshold (between 150 and 255).

The combined grandient threshold is layered with the color thresolding and presented below:   
(green for gradients threshold - combined - and blue for color threshold):

![alt text][image3]

Here is the pipeline code (line 144 to 162), and the thresholding functions may be found in the lanefind.py file lines 11 to 58:

```python
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(thresh_min, thresh_max))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(thresh_min, thresh_max))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, thresh_max))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.75, 1.25))
   
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined = np.uint8(combined)
     
    # Threshold color channel
    s_thresh_min = 150
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(s_binary), combined, s_binary))
```

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is coded straight in the pipeline. First the origin point are defined empiricly, based on a sample image. The objective of the point definition is to have straight 
vertical lines after transformation of a straight lane picture. The atempt was to find a perfect trapezoidal form, considering the runway is flat.  
The parameters topline, bottomline, bottom_start, bottom_end and reduction are defined in order to generate a perfect trapezoidal form. The parameter top_correction
was defined in order to tune the trapezoidal for so that the actual result after warp is two vertical lines (for a straight lane). Code extract from lines 178 to 193)

	
```python
    # Define points that will used for perspective transfor (birdeye view)
    topline = 489
    bottomline = 690
    bottom_start = 170
    bottom_end = bottom_start + 925 +30       
    reduction = 347
    top_shift =0
    top_correction = -11
    
    
    #Source points for birdseye view transform
    src = np.float32(
        [[bottom_end-reduction+top_shift+top_correction,topline],      #top right
         [bottom_end,bottomline],                       #bottom right
         [bottom_start,bottomline],                     #bottom left
         [bottom_start+reduction+top_shift,topline]])   #top left
```    

The destination points are the same for the bottom of the trapezoid and the top are set to make the destination shape a rectangle (lines 196 to 201):

```python   
    # Define destination points for the transform
    dst = np.float32(
        [[bottom_end,topline],
         [bottom_end,bottomline],
         [bottom_start,bottomline],
         [bottom_start,topline]])
```


The points defined are applied to the opencv warpPerspective() function, right after the identification of the transformation matrices by the opencv getPerspectiveTransform()

The inverse transfor matrix "invM" is also determined for future dewarp of the found lane (lines 206 to 210).

```python   
    # Generate perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    # Apply perspective transform matrix and get a birdseye view image
    binary_warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)
```

This the source points (dots on original image) and the warped images are shown below:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane line pixels are found by identifying the histogram of the binary image and the premise is that the two peaks
of the histogram will give the near position of the beginning of the line (lines 234 to 235).

```python   
    # Take a histogram of the image (half)
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    
    # print('histogram.shape[0]')
    # print(histogram.shape[0])
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```

After this, the initial points (leftx_base, rightx_base) are used to find thresholded points in a window with origin at the initial points and hight of 1/9 th of the 
warped image. The position of the points inside the window will define a shift for the next window and the points inside the 9 windows will be used for defining a best fit
2nd order polinomial.

Here the windows are found and the points inside it (lines 269 to 290):

```python   
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
```


Here the left and right position vectors are found so that the np.polifyt can be used and left_fit/right_fit polinomial indices are found (293 to 304):


```python   
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```

Then, the polinomial lane is marked on a new image and the inner portion, between lanes, is drawn green by creating a single channel image and
transforming it back to the image perspective using the invM matrix found before (line 306 to 323).


```python   
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #Left and right lane pixels painted
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    #Create mask image, with green channel defined as value = 1
    mask_lane = np.ones(out_img.shape).astype(np.uint8)
    for i in range(mask_lane.shape[0]-1):
        for j in range(mask_lane.shape[1]-1):
            if j>=left_fitx[i] and j<= right_fitx[i]:
                mask_lane[i,j,0]=0
                mask_lane[i,j,2]=0
    
    mask_lane_dewarped = cv2.warpPerspective(mask_lane, invM, img_size, flags=cv2.INTER_LINEAR)
```


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Below the radius of curvature is found, given Y position in the spline (bottom of the image is defined) - lines: 332 to 347

```python  
    # Define y-value where we want radius of curvature as the bottom of the image
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Example values: 632.1 m    626.2 m
    curve_rad = [left_curverad, right_curverad]

```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 348 to 360. The code below also prints the radius and distance from left lane in the image frame:


```python
    #Create combined image with green lane 
    mask_lane_dewarped[np.where((mask_lane_dewarped == [0,0,0]).all(axis = 2))] = [1,1,1]
    augmented_image = image*mask_lane_dewarped
    
    left_fitx =     left_fit_cr[0]*(y_eval*ym_per_pix)**2+      left_fit_cr[1]*(y_eval*ym_per_pix) +    left_fit_cr[2]
    right_fitx =    right_fit_cr[0]*(y_eval*ym_per_pix)**2+    right_fit_cr[1]*(y_eval*ym_per_pix) +   right_fit_cr[2]
    
    #This is the position of the car in meters from the left of the image
    midle_position = augmented_image.shape[1]*xm_per_pix/2
    #define font and print data to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(augmented_image, 'Radius of curvature is: {:06.2f}'.format(np.mean(curve_rad)), (10,30), font, 1, (255,255,255),2)
    cv2.putText(augmented_image, 'Distance from the left lane: {:03.2f}'.format(midle_position-left_fitx), (10,60), font, 1, (255,255,255),2)
```

The following image shows the result:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I believe there is lots to do in the gradient and color thresholding to make this code robust to lighting changes and shades. Also the pipeline will very likely fail to
detect lanes when car is entering the road and lanes may be near othogonal to the car direction. To make this pipeline more robust, I would try 
use optimization to find best threshold parameters and make different gradient combinations to have points detected more likelly to be part of a lane line.
I would also try to find one single polimonial that would better describe both lanelines (left and right) and that should make the found polinomial more representative of the lane.