# Author     : Nhan Dao
# Description: Detects features such as corner, edge, line and blob

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

#Detects corner using harris or shi-tomasi
def cornerdetect(img_path,method='Harris'):

    if not method in {'Harris','Shi-Tomasi'}:
        raise ValueError("No method named:" + method)

    img  = cv2.imread(img_path)
    gray = cv2.imread(img_path,0)


    #Performing Harris corner detection
    # cv.cornerHarris()
    #   img - Input image, it should be grayscale and float32 type
    #   blockSize - It is the size of neighbourhood considered for corner detection
    #   ksize - Aperture parameter of Sobel derivative used
    #   k - Harris dectector free parameter in the equation
    #
    #Performing Shi-Tomasi corner detection
    #cv.goodFeaturesToTrack()
    #   img - Input image
    #   maxCorners - Max corners to return, the strongest are returned
    #   qualityLevel - Some value to tweak for better result
    #   minDistance - Mininum possible Euclidian distance btween the returned corners

    if method == 'Harris':
        #Cast gray image to float32
        gray = np.float32(gray)
        dst  = cv2.cornerHarris(gray,2,3,0.04)

        #This step dilate the corners to be more visible
        dst  = cv2.dilate(dst,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))

        #Performing some thresholding
        img[dst>0.01*dst.max()]=[0,0,255]
    elif method == 'Shi-Tomasi':
        #cv.goodFeaturesToTrack() uses Shi-Tomasi
        corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
        corners = np.int0(corners) #Corners are <x,y> vectors

        #Now draw a circle at each corner on the original image.
        #This could be a problem for large array of corners - yikes
        for vect in corners:
            x,y = vect.ravel()
            cv2.circle(img,(x,y),2,(0,0,255),-1)

    return img

def edgedetect(img_path):
    #Perform edge detect using canny edge detection method

    #Read in gray scale image
    gray = cv2.imread(img_path,0)

    #cv2.Canny()
    #   img - gray scale image
    #   threshold1 - the lower threshold for hysteresis
    #   threshold2 - the upper threshold for hysteresis

    edges = cv2.Canny(gray,100,200)

    return edges

def linedetect(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges =  cv2.Canny(gray,50,150,apertureSize=3)


    #cv.HoughLines()
    #   img - 8bit, single-channel binary source image
    #   rho - distance resolution of the accumulator in pixels
    #   theta - angle resolution of the acumulator in pixel in radians
    #   threshold - the accumulator threshold parameter
    #
    lines = cv2.HoughLines(edges,1,np.pi/180,200)

    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)

    return img

def blobdetect(img_path):
    #Using MSERs (Maximuall Stable Extreme Regeions) to detect blob
    img = cv2.imread(img_path,0)

    mser = cv2.MSER_create()
    regions = mser.detectRegions(img)
    hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in regions[0]]
    cv2.polylines(img,hulls,1,(0,255,0))

    return img

#Pseudo test harness for this
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python {script} <image_path>".format(script=sys.argv[0]))
    else:
        #Display the corner detection
        corner_harris = cornerdetect(sys.argv[1])
        plt.figure("Harris Corner Detection")
        plt.imshow(cv2.cvtColor(corner_harris,cv2.COLOR_BGR2RGB))

        corner_shi = cornerdetect(sys.argv[1],method='Shi-Tomasi')
        plt.figure("Shi-Tomasi Corner Detection")
        plt.imshow(cv2.cvtColor(corner_shi,cv2.COLOR_BGR2RGB))

        edges = edgedetect(sys.argv[1])
        plt.figure("Canny Edge Detection")
        plt.imshow(np.hstack((cv2.imread(sys.argv[1],0),edges)),cmap='gray')

        line = cv2.cvtColor(linedetect(sys.argv[1]),cv2.COLOR_BGR2GRAY)
        plt.figure("Hough line Transform")
        plt.imshow(np.hstack((cv2.imread(sys.argv[1],0),line)),cmap='gray')

        blob = blobdetect(sys.argv[1])
        plt.figure("Blob Detection")
        plt.imshow(np.hstack((cv2.imread(sys.argv[1],0),blob)),cmap='gray')

        plt.show()
