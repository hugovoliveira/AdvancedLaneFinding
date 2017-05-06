import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import os
import pickle

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Strip thresholds from tupple
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary


def calibrate_camera(calibration_file = 'calibration.pickle', calibration_imgdir = '.\\camera_cal\\calibration*.jpg'):
    CalibrationInFile = os.path.isfile(calibration_file)
    if CalibrationInFile:
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
    


mtx, dist = calibrate_camera()

tesImg = '.\\test_images\\test1.jpg'

img_raw = mpimg.imread(tesImg)
img = cv2.undistort(img_raw, mtx, dist, None, mtx)

fig = plt.figure()
plt.imshow(img_raw)
plt.figure()
plt.imshow(img)

plt.show()
    

ksize = 9 # Choose a larger odd number to smooth gradient measurements
image = mpimg.imread('C:/Users/Usuario/WorkspaceEclipse/Project4SelfDrivingCar/signs_vehicles_xygrad.png',cv2.IMREAD_COLOR)
# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 150))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(50, 150))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

