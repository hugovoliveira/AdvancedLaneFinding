import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import os
import pickle
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Strip thresholds from tupple
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
#     print('Min = {} / Max = {}'.format(np.min(scaled_sobel),np.max(scaled_sobel)))
#     print(scaled_sobel[1:20,1:20])
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # Calculate directional gradient
    # Apply threshold
    return grad_binary

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return mag_binary

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary


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
    
# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

process_image_plotting = False

def process_image(orig_image):
    global process_image_plotting
    
    # This will either calibrate the camera or use calibration data saved in a past run
    mtx, dist = calibrate_camera()
    #work with a copy of the input image
    img_raw = np.copy(orig_image)
    # undistort image, by applying camera and distortion coeficients
    image = cv2.undistort(img_raw, mtx, dist, None, mtx)
    
    if process_image_plotting:
        output_img = '.\\output_images\\undistorted_testimage.jpg'
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Original')
        ax1.imshow(img_raw)
        ax2.set_title('Undistorted')
        ax2.imshow(image)
        f.savefig(output_img)  
        f.show()        

    # Convert original image to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # Grayscale original image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold gradient
    thresh_min = 30
    thresh_max = 255
    ksize = 5

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

    if process_image_plotting:
        output_img = '.\\output_images\\binary_threshold_testimage.jpg'
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Original')
        ax1.imshow(image)
        ax2.set_title('Binary thresholded')
        ax2.imshow(color_binary*255)
        f.show()
        f.savefig(output_img)  
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(combined)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1
    
    # Define points that will used for perspective transfor (birdeye view)
    topline = 489
    bottomline = 690
    bottom_start = 170
    bottom_end = bottom_start + 925 +30       
    reduction = 348
    top_shift =-2
    top_correction = -11
    
    
    #Source points for birdseye view transform
    src = np.float32(
        [[bottom_end-reduction+top_shift+top_correction,topline],      #top right
         [bottom_end,bottomline],                       #bottom right
         [bottom_start,bottomline],                     #bottom left
         [bottom_start+reduction+top_shift,topline]])   #top left
    
    
    # Define destination points for the transform
    dst = np.float32(
        [[bottom_end,topline],
         [bottom_end,bottomline],
         [bottom_start,bottomline],
         [bottom_start,topline]])
    
    
    img_size = (image.shape[1], image.shape[0])
    
    # Generate perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    # Apply perspective transform matrix and get a birdseye view image
    binary_warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)
    
    
    # Plotting original
    if process_image_plotting:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Original')
        ax1.imshow(image)
        # plot original points used for the birdseye view transformation
        ax1.plot(bottom_end-reduction+top_shift+top_correction,topline,'.') #Top right
        ax1.plot(bottom_end,bottomline,'.') #bottom right
        ax1.plot(bottom_start,bottomline,'.') #bottom left
        ax1.plot(bottom_start+reduction+top_shift,topline,'.') #Top left
        # plot transformmed image
        ax2.set_title('binary_warped')
        warped_drawn = binary_warped.copy()
        warped_drawn[:,1047] =1
        warped_drawn[:,257] =1
        ax2.imshow(warped_drawn)
        f.show()
        output_img = '.\\output_images\\binary_warped_testimage.jpg'
        f.savefig(output_img)  
    
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
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
     
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
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
    
    if (process_image_plotting):
        output_img = '.\\output_images\\mask_lane_testimage.jpg'
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Original')
        ax1.imshow(image)
        ax2.set_title('Lanelines drawn')
        ax2.imshow(mask_lane*255)
        f.show()
        f.savefig(output_img)  
        
    
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
    
    if (process_image_plotting):
        plt.figure()
        plt.imshow(augmented_image)
        output_img = '.\\output_images\\final_testimage.jpg'
        plt.savefig(output_img)  
        plt.show()
    return augmented_image



### Start of program
### First demonstrate the calibration feature with a chessboard image
mtx, dist = calibrate_camera()
tesImg = '.\\camera_cal\\calibration1.jpg'
undistort_output = '.\\output_images\\undistorted_chessboard.jpg'
img_raw = mpimg.imread(tesImg)
image = cv2.undistort(img_raw, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Original')
ax1.imshow(img_raw)
ax2.set_title('Undistorted')
ax2.imshow(image)
f.show()
f.savefig(undistort_output)    
    
### Test pipeline on a single image
process_image_plotting = True
tesImg = '.\\test_images\\straight_lines1.jpg'
img_raw = mpimg.imread(tesImg)
process_image(img_raw)

process_image_plotting = False  
input_file = 'C:\\Users\\Usuario\\Documents\\PythonProjects\\SelfDrivingCars\\AdvancedLaneFinding\\project_video.mp4'
project_video = VideoFileClip(input_file)
treated_video = project_video.fl_image(process_image)
treated_video.write_videofile(".\\project_video_result.mp4", audio=False)



# input_file = 'C:\\Users\\Usuario\\Documents\\PythonProjects\\SelfDrivingCars\\AdvancedLaneFinding\\project_video.mp4'
# input_file = '.\\project_video.mp4'
# project_video = VideoFileClip(input_file).subclip(0,5)
# treated_video = project_video.fl_image(process_image)
# treated_video.write_videofile(".\\project_video_result.mp4", audio=False)














