import cv2
import numpy as np,sys

A = cv2.imread('einstein.jpeg')
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in xrange(6):
	G = cv2.pyrDown(G)
	gpA.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in xrange(5,0,-1):
	GE = np.zeros(gpA[i-1].shape, dtype=np.float32)
	cv2.pyrUp(gpA[i], GE)
	L = gpA[i-1] - GE
	lpA.append(L)

lpA.reverse()
for L in lpA:
	cv2.imshow('window', 0.5 + 0.5*(L / np.abs(L).max()))
	while cv2.waitKey(5) < 0: 
		pass