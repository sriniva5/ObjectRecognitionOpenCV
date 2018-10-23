#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ananya Srinivasan
CSC 355: Human Computer Interaction
Homework 4: Computer Vision, Exercise 1
Description: Takes an image of cups on a black background and performs otsu thresholding, followed by feature erosion 
and dilation to determine the distinct objects within the image. Finally, the connected compenents of the image are
determined and extracted by drawing rectangles around each component (object).
"""

import numpy as np
import cv2

#Image
img = cv2.imread('./coffeecup.jpg', 0)
cv2.imshow('Orig', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

#STEP 1: Otsu Thresholding
#Otsu thresholding for bimodal distributions: returns threshold value and image, make into binary
img_blur = cv2.GaussianBlur(img, (5,5), 0) #Gaussian filter to reduce most noise
ret, otsu_thresh = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Otsu', otsu_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

#STEP 2: Erosion
#Remove any noise within the image and distinguish between features
kernel = np.ones((5,5), np.uint8)
img_erode = cv2.erode(otsu_thresh, kernel, iterations=25)
cv2.imshow('Erosion', img_erode)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

#STEP 3: Dilation
#Dilate the features of the image to distinguish between them
img_dilate = cv2.dilate(img_erode, kernel, iterations=15)
cv2.imshow('Dilation', img_dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)


#STEP 4: Connected components
#Find any the particular objects within the image
cc = cv2.connectedComponentsWithStats(img_dilate, 8, cv2.CV_32S)
labels = cc[1]
stats = cc[2]

print("Number of items: %d\n" % cc[0])

for label in range(cc[0]):
    obj = label+1
    print("Object %d:" % obj)
    
    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    print("Coordinates: (%d, %d)" % (x, y))
    
    width = stats[label, cv2.CC_STAT_WIDTH]
    print("Width: %d" % width)
    
    height = stats[label, cv2.CC_STAT_HEIGHT]
    print("Height: %d" % height)
    
    area = stats[label, cv2.CC_STAT_AREA]
    print("Area: %d" % area)
    print("\n")
    
    img_rect = cv2.rectangle(img,(x, y),(x+width, y+height),(250, 0, 0), 3)
    cv2.imshow('Rectangle', img_rect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
