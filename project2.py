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
    imagesToBuildFrom = [img.astype(numpy.float32)]


    for i in range(4):
        gi = imagesToBuildFrom[i];
        gi1 = cv2.pyrDown(gi)
        #initializing gi_up so it will be the same size as gi
        gi1_up = numpy.zeros(gi.shape, dtype=numpy.float32)
        cv2.pyrUp(gi1, gi1_up)
        gi_float = gi.astype(numpy.float32)
        gi1_upFloat = gi1_up.astype(numpy.float32)
        li = gi_float - gi1_upFloat
        imagesToBuildFrom.append(gi1)
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
lp1 = pyr_build(image1)
lp2 = pyr_build(image2)

for L in lp1:
    cv2.imshow('window', 0.5 + 0.5*(L / numpy.abs(L).max()))
    while cv2.waitKey(5) < 0: pass


# reconstructs original image from Laplacian pyramid
def pyr_reconstruct(lp):
    # ri = numpy.zeros(lp[-1].shape, dtype=numpy.float32)
    # ri[:] = lp[-1]

    # del lp[-1]

    # for L in lp[::-1]:
    #     #cv2.imshow('window', 0.5 + 0.5*(RN / numpy.abs(RN).max()))
    #     cv2.imshow('window', 0.5 + 0.5*(RN / numpy.abs(RN).max()))
    #     while cv2.waitKey(5) < 0: pass
    #     #cv2.imshow('window', 0.5 + 0.5*(L / numpy.abs(L).max()))
    #     #while cv2.waitKey(5) < 0: pass
    #     RNup = numpy.zeros(L.shape, dtype=numpy.float32)
    #     cv2.pyrUp(RN, RNup)
    #     RN = RNup + 0.5 + 0.5*(L / numpy.abs(L).max())
    
    rebuilt = []
    rebuilt.append(lp[-1])
    del lp[-1]
    for L in reversed(lp):
        ri = rebuilt[-1]
        #ri = lp[-1]
        
        #Lprev (L i-1 in handout) is the next to last image in pyramid, now last after deletion
        Lprev = lp[-1]
        del lp[-1]
        #declare ri to have same size as Lprev
        ri_up = numpy.zeros(Lprev.shape, dtype=numpy.float32)
        cv2.pyrUp(ri, ri_up)
        ri_prev = ri_up + Lprev
        rebuilt.append(ri_prev)
        #display for debugging purposes
        cv2.imshow('window', 0.5 + 0.5*(ri / numpy.abs(L).max()))
        while cv2.waitKey(5) < 0: pass
        #if lp[0] == Lprev: # break out of loop once gone through all images
        #   break
    return rebuilt[-1]


rebuilt1 = pyr_reconstruct(lp1)
#convert to ints before returning/displaying
r1_clipped = numpy.clip(rebuilt1, 0, 255)
r1_int = r1_clipped.astype(numpy.uint8)

cv2.imshow('window', r1_int)
while cv2.waitKey(5) < 0: pass

rebuilt2 = pyr_reconstruct(lp2)


def alpha_blend(A, B, alpha):
    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)
    # if A and B are RGB images, we must pad
    # out alpha to be the right shape
    if len(A.shape) == 3:
        alpha = numpy.expand_dims(alpha, 2)
    return A + alpha*(B-A)


