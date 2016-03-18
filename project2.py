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

pyrSize = 8

win = 'window'
cv2.namedWindow(win)

def getPoints(w, filename, source):

	global win

	if w.load(filename):
		print 'loaded from', filename
		
	ok = w.start(win, source)
	
	if ok:
		w.save(filename)

#called in alignAndMakeMask
#makes a mask from the ellipse selected in the second image
#(because the first is mapped onto the second, so this will
#be the area of interest in both images by the end)
def makeMask(img, e):
	mask = numpy.ones(img.shape, 'float32')
	h = img.shape[0]
	w = img.shape[1]
	cv2.ellipse(mask, (int(e.center[0]), int(e.center[1])), (int(e.u), int(e.v)), int(e.angle), 0, 360, (0, 0, 0), -1)
	#cv2.imshow('window', mask)
	#save the mask 
	maskInt = (numpy.clip(mask*255, 0, 255)).astype(numpy.uint8)
	cv2.imwrite('maskUsedForLaplaceBlend.jpg', maskInt)
	while cv2.waitKey(5) < 0: pass
	return mask

#aligns the images using findHomography and selecting ellipses
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

	# cv2.imshow('window', warped1)
	# while cv2.waitKey(5) < 0: pass

	# cv2.imshow('window', img2)
	# while cv2.waitKey(5) < 0: pass
	return warped1, mask

#takes in an image, outputs a list of laplacian images
def pyr_build(img):
	#lp = []
	G = [img.astype(numpy.float32)]
	#first build the array images to build from 
	for i in range(pyrSize-1):
		gi1 = cv2.pyrDown(G[i])
		G.append(gi1)

	lp = []

	#now build the actual pyramid from that array
	for i in range(0, pyrSize-1):
		h, w, d = G[i].shape
		gi1_up = cv2.pyrUp(G[i+1], None, (h, w))
		L = G[i] - gi1_up
		lp.append(L)
	# used the 2 lines below to verify theory about the limits of
	# when the depth continued to change things - comment out because
	# the print is otherwise unnecessary 
	# print "smallest height was", h
	# print "smallest width was", w

	#L[N] = G[N]
	lp.append(G[-1]) 
	return lp

# reconstructs original image from Laplacian pyramid
def pyr_reconstruct(lst):

	lp = lst[:]
	rebuilt = [lp[-1]]

	#note that this makes it so the indexing is reverse of that in the 
	#assignment sheet - ie r0 is smallest = (l reversed)0
	lp.reverse()

	for i in range(0, len(lp)-1):
		Lprev = lp[i+1]
		ri_up = numpy.zeros(Lprev.shape, dtype=numpy.float32)
		cv2.pyrUp(rebuilt[i], ri_up, (ri_up.shape[0], ri_up.shape[1]))
		ri_prev = ri_up + Lprev
		rebuilt.append(ri_prev)

	toRet = numpy.clip(rebuilt[-1], 0, 255)
	return toRet.astype(numpy.uint8)

#alpha blend function given to us in assignment sheet
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
	print 'usage: ', sys.argv[0], ' image1 image2 where image2 should be square'
	#some non-square destination/second images work but not consistently 
	#it seems as though square destination images are guarenteed to work 
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
# for L in lp2:
# 	cv2.imshow('window', 0.5 + 0.5*(L / numpy.abs(L).max()))
# 	while cv2.waitKey(5) < 0: pass

# use this to test the pyr_reconstruct function
# rebuilt1 = pyr_reconstruct(lp1)
# #convert to ints before returning/displaying
# r1_clipped = numpy.clip(rebuilt1, 0, 255)
# r1_int = r1_clipped.astype(numpy.uint8)
# cv2.imshow('window', r1_int)
# while cv2.waitKey(5) < 0: pass

# rebuilt2 = pyr_reconstruct(lp2)

#straight alpha blend
blurMask = mask.copy()
#blur the mask slightly to make the alpha blend a little less terrible
blurMask = cv2.GaussianBlur(mask, (9,9), 0)
normalBlend = alpha_blend(warped1.astype(numpy.float32), image2.astype(numpy.float32), blurMask)
normalBlend = (numpy.clip(normalBlend, 0, 255)).astype(numpy.uint8)
cv2.imshow('straight alpha blend', normalBlend)
while cv2.waitKey(5) < 0: pass
cv2.imwrite('normalBlend.jpg', normalBlend)
blurMaskInt = (numpy.clip(blurMask*255, 0, 255)).astype(numpy.uint8)
cv2.imwrite('maskForNormalBlend.jpg', blurMaskInt)



#make an array of the alpha blended images at every level of the
#laplacian pyramids, treat it as a pyramid itself, and  
#reconstruct an image from it.
mergedPyr = []
for i in range(len(lp1)):
	mask = cv2.resize(mask, (lp1[i].shape[1], lp1[i].shape[0]))
	levelMerge = alpha_blend(lp1[i], lp2[i], mask)
	mergedPyr.append(levelMerge)


merged = pyr_reconstruct(mergedPyr)

#display and save merged image
cv2.imshow('laplacian pyramid blend', merged)
while cv2.waitKey(5) < 0: pass
cv2.imwrite('merged.jpg', merged)


