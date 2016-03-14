############################################
#
# File:	  project2.py
# Author: Alison Rosenzweig and Sarah Wallace
# Date:   March 15, 2016
#
# Written for ENGR 27 - Computer Vision
#
############################################

import cv2
import numpy
import math
import cvk2
import sys

#takes in an image, outputs a list of laplacian images
def pyr_build(img):
    lp = []
    imagesToBuildFrom = [img]


    for i in range(4):
        h = imagesToBuildFrom[i].shape[0]
        w = imagesToBuildFrom[i].shape[1]
        gN = cv2.pyrDown(imagesToBuildFrom[i])
        # Issue with specifying size of gN - want it to the be same size as orig image
        # from imagesToBuildFrom[i]
        gNlarge = numpy.zeros(gN.shape, dtype=numpy.float32)
        cv2.pyrUp(gN, gNlarge)
        gNfloat = gN.astype(numpy.float32)
        gNlargeFloat = gNlarge.astype(numpy.float32)
        li = gNfloat - gNlargeFloat
        imagesToBuildFrom.append(li)
        lp.append(li)
    
    return lp


#make sure there are enough commandline args
if len(sys.argv) < 3:
    print 'usage: ', sys.argv[0], ' image1 image2'
    sys.exit(1)

imageName1 = sys.argv[1] #first image
image1 = cv2.imread(imageName1)
imageName2 = sys.argv[2] #second image
image2 = cv2.imread(imageName2)

#Generate Laplacian pyramid with grayscale version
#image1_gray = numpy.zeros((image1.shape[0],image1.shape[1]), 'uint8')
#cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY, image1_gray)
#lN = pyr_build(image1_gray)

# Generating a Laplacian pyramid
lNApple = pyr_build(image1)
lNCrab = pyr_build(image2)

for L in lNApple:
    cv2.imshow('window', 0.5 + 0.5*(L / numpy.abs(L).max()))
    while cv2.waitKey(5) < 0: pass


# reconstructs original image from Laplacian pyramid
def pyr_reconstruct(lp):
    RN = numpy.zeros(lp[-1].shape, dtype=numpy.float32)
    RN[:] = lp[-1]

    del lp[-1]
    for L in lp[::-1]:
        cv2.imshow('window', 0.5 + 0.5*(RN / numpy.abs(RN).max()))
        while cv2.waitKey(5) < 0: pass
        cv2.imshow('window', 0.5 + 0.5*(L / numpy.abs(L).max()))
        while cv2.waitKey(5) < 0: pass
        RNup = numpy.zeros(L.shape, dtype=numpy.float32)
        cv2.pyrUp(RN, RNup)
        RN = RNup + 0.5 + 0.5*(L / numpy.abs(L).max())
    
    rebuilt = []
    for L in reversed(lp):
        RN = lp[-1]
        del lp[-1]
        Lprev = lp[-1]
        RNup = numpy.zeros(Lprev.shape, dtype=numpy.float32)
        cv2.pyrUp(RN, RNup)
        RNprev = RNup + Lprev
        rebuilt.append(RNprev)
        if lp.index(Lprev) == 0: # break out of loop once gone through all images
            break
    return rebuilt[-1]

rebuiltApple = pyr_reconstruct(lNApple)
cv2.imshow('window', 0.5 + 0.5*(rebuiltApple / numpy.abs(rebuiltApple).max()))
while cv2.waitKey(5) < 0: pass
rebuiltCrab = pyr_reconstruct(lNCrab)


def alpha_blend(A, B, alpha):
    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)
    # if A and B are RGB images, we must pad
    # out alpha to be the right shape
    if len(A.shape) == 3:
        alpha = numpy.expand_dims(alpha, 2)
    return A + alpha*(B-A)
