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

pyrSize = 6

win = 'window'
cv2.namedWindow(win)

def getPoints(w, filename, source):

    global win

    if w.load(filename):
        print 'loaded from', filename
        
    ok = w.start(win, source)
    
    if ok:
        w.save(filename)

def makeMask(img, e):
    mask = numpy.zeros(img.shape, 'float32')
    h = img.shape[0]
    w = img.shape[1]
    cv2.ellipse(mask, (int(e.center[0]), int(e.center[1])), (int(e.u), int(e.v)), int(e.angle), 0, 360, (1, 1, 1),-1)
    # cv2.ellipse(mask, (e.center[0], e.center[1]), (e.u, e.v), 
    #         e.ange, 0, 360, (255, 255, 255), -1)
    cv2.imshow('window', mask)
    while cv2.waitKey(5) < 0: pass
    return mask

def alignAndMakeMask(img1, img2):
    #get points from first image
    e1 = cvk2.RectWidget('ellipse')

    getPoints(e1, 'im1points.txt', img1)

    #get points from second image
    e2 = cvk2.RectWidget('ellipse')
    getPoints(e2, 'im2points.txt', img2)

    mask = makeMask(img2, e2)

    h2 = img2.shape[0]
    w2 = img2.shape[1]

    #use points to find homography between the images
    Hbig = cv2.findHomography(e1.points, e2.points, 0, 5)
    H = Hbig[0]

    dims = (w2, h2)
    white_image = numpy.zeros((h2, w2, 3), numpy.uint8)
    white_image[:,:,:] = 255

    warped1 = numpy.zeros((h2, w2, 3), numpy.uint8)
    cv2.warpPerspective(img1, H, dims, warped1, borderMode=cv2.BORDER_TRANSPARENT) 

    cv2.imshow('window', warped1)
    while cv2.waitKey(5) < 0: pass

    cv2.imshow('window', img2)
    while cv2.waitKey(5) < 0: pass
    return warped1, mask

#takes in an image, outputs a list of laplacian images
def pyr_build(img):
    #lp = []
    imagesToBuildFrom = [img.astype(numpy.float32)]
    #first build the array images to build from 
    for i in range(pyrSize):
        gi = imagesToBuildFrom[i]
        gi1 = cv2.pyrDown(gi)
        imagesToBuildFrom.append(gi1)
        cv2.imshow('window', 0.5 + 0.5*(gi1 / numpy.abs(gi1).max()))
        while cv2.waitKey(5) < 0: pass


    lp = []

    #now build the actual pyramid from that array
    for i in range(pyrSize):
        gi1_up = numpy.zeros(imagesToBuildFrom[-i-1].shape, dtype=numpy.float32)
        cv2.pyrUp(imagesToBuildFrom[-i], gi1_up)
        L = cv2.subtract(imagesToBuildFrom[-i-1], gi1_up)
        lp.append(L.astype(numpy.float32))

    lp.reverse()
    return lp

def pyr_build_old(img):
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

# reconstructs original image from Laplacian pyramid
def pyr_reconstruct(lst):

    lp = lst[:]
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
        # cv2.imshow('window', 0.5 + 0.5*(ri / numpy.abs(L).max()))
        # while cv2.waitKey(5) < 0: pass
        #if lp[0] == Lprev: # break out of loop once gone through all images
        #   break
    return rebuilt[-1]

def alpha_blend(A, B, alpha):
    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)
    # if A and B are RGB images, we must pad
    # out alpha to be the right shape

    if len(A.shape) == 3 and len(alpha.shape) == 2:
        alpha = numpy.expand_dims(alpha, 2)

    return A + alpha*(B-A)

#make sure there are enough commandline args
if len(sys.argv) < 3:
    print 'usage: ', sys.argv[0], ' image1 image2'
    sys.exit(1)

imageName1 = sys.argv[1] #first image
image1 = cv2.imread(imageName1)
imageName2 = sys.argv[2] #second image
image2 = cv2.imread(imageName2)

#now align the images, warps image 1 to match up with image 2
warped1, mask = alignAndMakeMask(image1, image2)

# Generating a Laplacian pyramid
lp1 = pyr_build(warped1)
lp2 = pyr_build(image2)

#show pyramid images for image1
for L in lp1:
    cv2.imshow('window', 0.5 + 0.5*(L / numpy.abs(L).max()))
    while cv2.waitKey(5) < 0: pass


rebuilt1 = pyr_reconstruct(lp1)
#convert to ints before returning/displaying
r1_clipped = numpy.clip(rebuilt1, 0, 255)
r1_int = r1_clipped.astype(numpy.uint8)

cv2.imshow('window', r1_int)
while cv2.waitKey(5) < 0: pass

rebuilt2 = pyr_reconstruct(lp2)

mergedPyr = []
#TODO: get rid of this once the pyramid works correctly
mask = cv2.resize(mask, (lp1[0].shape[1], lp1[0].shape[0]) )
for i in range(pyrSize):
    levelMerge = alpha_blend(lp2[i], lp1[i], mask)
    mergedPyr.append(levelMerge)
    if i < pyrSize-1:
        mask = cv2.resize(mask, (lp1[i+1].shape[1], lp1[i+1].shape[0]))


merged = pyr_reconstruct(mergedPyr)

merged_clipped = numpy.clip(merged, 0, 255)
merged_int = merged_clipped.astype(numpy.uint8)

cv2.imshow('window', merged_int)
while cv2.waitKey(5) < 0: pass



