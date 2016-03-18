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

win = 'hw2'
cv2.namedWindow(win)
sigmaSmall = 5 
sigmaBig = 15 

def lopass(img, sigma):
	img = img.astype(numpy.float32)
	#lo = numpy.zeros(img.shape, dtype=numpy.float32)
	lo = cv2.GaussianBlur(img, (0, 0), sigma, sigma)
	return lo

def hipass(img, sigma):
	img = img.astype(numpy.float32)
	hi = img - lopass(img, sigma)
	return hi

def getPoints(w, filename, source):
	global win

	if w.load(filename):
		print 'loaded from', filename
	
	ok = w.start(win, source)

	if ok:
		w.save(filename)

def align(img1, img2):
	#get points from first image
	p1 = cvk2.MultiPointWidget('points')
	getPoints(p1, 'im1points.txt', img1)

	#get points from second image
	p2 = cvk2.MultiPointWidget('points')
	getPoints(p2, 'im2points.txt', img2)

	h2 = img2.shape[0]
	w2 = img2.shape[1]
	#construct an array of points on the border of the second image
	border2 = numpy.array( [ [[  0, 0  ]],
                   	  	  [[ w2, 0  ]],
                          [[ w2, h2 ]],
                  	      [[  0, h2 ]] ], dtype='float32' )

	#use points to find homography between the images
	Hbig = cv2.findHomography(p1.points, p2.points, 0, 5)
	H = Hbig[0]
	# origin = box[0:2]
	#create a translation to put it at the correct origin
	# Tnice = numpy.eye(3)
	# Tnice[0,2] -= origin[0]
	# Tnice[1,2] -= origin[1]

	#compose the final homography with transpose
	#Hnice = numpy.matrix(Tnice) * numpy.matrix(H)
	dims = (w2, h2)
	white_image = numpy.zeros((h2, w2, 3), numpy.uint8)
	white_image[:,:,:] = 255
	warped1 = cv2.warpPerspective(img1, H, dims, white_image, borderMode=cv2.BORDER_TRANSPARENT) #may need to change back to Hnice
	#not sure if below is needed
	#translated2 = cv2.warpPerspective(img2,Tnice,dims)

	cv2.imshow('window', warped1)
	while cv2.waitKey(5) < 0: pass

	cv2.imshow('window', img2)
	while cv2.waitKey(5) < 0: pass
	return warped1

#make sure there are enough commandline args
if len(sys.argv) < 3:
    print 'usage: ', sys.argv[0], ' image1 image2'
    sys.exit(1)

imageName1 = sys.argv[1] #first image
image1 = cv2.imread(imageName1)
imageName2 = sys.argv[2] #second image
image2 = cv2.imread(imageName2)

warped = align(image1, image2)

far = 1.1*lopass(warped, sigmaBig)
close = 1.25*hipass(image2, sigmaSmall)

I = far+close
I_clipped = numpy.clip(I, 0, 255)
I_int = I_clipped.astype(numpy.uint8)

cv2.imshow('window', I_int)
while cv2.waitKey(5) < 0: pass

cv2.imwrite('hybrid.jpg', I_int)
