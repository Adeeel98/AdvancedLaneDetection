
# Include section
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# Load calibration images(chess boards in different angles from our camera_cal folder)
def readChessImages():
    imagePath = './camera_cal/left-*.png'
    calibrationImages = list(map(lambda imageFileName: (imageFileName, cv2.imread(imageFileName)), glob.glob(imagePath)))
    #print(calibrationImages)
    return calibrationImages

# Method for displaying the images
def showImages(calibrationImages):
  
    for img in calibrationImages:
        cv2.imshow(str(img[0]),img[1])
        cv2.waitKey(0)

# Method for detecting the points in chess images (objpoints and imgpoints)
def findingPoints(calibrationImages):
    objpoints = []
    imgpoints = []
    outimages = []
    originalImages = []
    objp = np.zeros((6*8,3), np.float32) 
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) # x and y cordinates.

    for imageAndFile in calibrationImages:
        fileName, image = imageAndFile
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            img_points = cv2.drawChessboardCorners(image.copy(), (8,6), corners, ret)
            # cv2.imwrite('chesscorner.png',img_points)
            # cv2.waitKey(500)
            outimages.append(img_points)
            originalImages.append(image)
   
    return objpoints, imgpoints, originalImages, outimages

# Method for returning camera matrix, mtx, and the distortion coefficients dist 
def getCoefficients(objpoints, imgpoints, originalImage):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, originalImage.shape[0:2], None, None)
    return mtx, dist

# Method for saving camera matrix, mtx, and the distortion coefficients dist into Pickle file
def saveCoefficientsIntoPickleFile(mtx, dist,fileDestination):
    pickle.dump({ 'mtx': mtx, 'dist': dist }, open(fileDestination, 'wb'))
    
    
def showUndistortImage(original, mtx, dist):
    undist = cv2.undistort(original, mtx, dist, None, mtx)
    resized = cv2.resize(undist, (1105,830), interpolation = cv2.INTER_AREA)
    # cv2.imwrite('frame_undist.jpg' , resized)
    # cv2.waitKey(0)

# Read images
calibrationImages = readChessImages()

# # show calibration images 
# #showImages(calibrationImages)
# Return object points and image points
objpoints, imgpoints, originalImages, outimages = findingPoints(calibrationImages)
originalImage = originalImages[5]

# create coff
mtx,dist = getCoefficients(objpoints, imgpoints, originalImage)
saveCoefficientsIntoPickleFile(mtx, dist, './pickled_data/camera_calibration.p')

# # Print mtx, and dist
# print("Camera matrix: ", mtx)
# print("Distortion coefficient: ", dist)

# print("Showing original image --> undistort result we got of the image")
# dist_image = cv2.imread('frame.jpg')
# showUndistortImage(dist_image, mtx,dist)
      

