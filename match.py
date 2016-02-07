# IMPLEMENTING HARRIS CORNER
# G13, CS676-2016, IITK
###################################################################################################

import Image
import numpy as np
import cv2
import sys
import math

###################################################################################################
# Configuration variables
k = 0.04
window_size = 5
offset = window_size/2
threshold = 100000
filename = ""

if len(sys.argv) == 2:
	test_set = str(sys.argv[1])
else:
	exit()

###################################################################################################
# This function returns the harris corners in the image
def harris_corners(img, filename):
	# Gradients in y and x direction, [columns reprsent the x and rows represent the y]
	Iy, Ix = np.gradient(np.array(img, dtype=np.float))
	Ixx = Ix*Ix
	Ixy = Iy*Iy
	Iyy = Iy*Iy
	height = img.shape[0]
	width = img.shape[1]

	R = [[0 for r in range(width)] for c in range(height)]
	cornerList = []
	suppressed_cornerList = []

	# Creating the image with corners indicated as red points with the help of opencv
	img = cv2.imread(filename, 0)
	newImg = img.copy()
	color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)

	# print "Finding Corners..."
	for y in range(offset, height-offset):
		for x in range(offset, width-offset):
			windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
			windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
			windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
			Sxx = windowIxx.sum()
			Sxy = windowIxy.sum()
			Syy = windowIyy.sum()
			det = (Sxx * Syy) - (Sxy**2)
			trace = Sxx + Syy
			r = det - k*(trace**2)
			R[y][x] = r
			if r > threshold:
				cornerList.append([x, y, r])
	# print "Number of corners found: ", len(cornerList)
	# print "Non Maximum Suppression ... "
	for corner in cornerList:
		x = corner[0]
		y = corner[1]
		r = corner[2]
		if r > R[y+1][x+1]:
			if r > R[y+1][x]:
				if r > R[y+1][x-1]:
					if r > R[y][x-1]:
						if r > R[y][x+1]:
							if r > R[y-1][x+1]:
								if r > R[y-1][x]:
									if r > R[y-1][x-1]:
										suppressed_cornerList.append(corner)
										color_img.itemset((y, x, 0), 0)
										color_img.itemset((y, x, 1), 0)
										color_img.itemset((y, x, 2), 255)
	# print "Number of corners after Suppression: ", len(suppressed_cornerList)
	return color_img, suppressed_cornerList

###################################################################################################

# Find out the corners for the first image
file1 = "img/set" + test_set + "/img1.png"
print "Loading : [", file1, "]"
img1 = Image.open(file1).convert('L')
img1 = np.array(img1)
print "Finding Harris Corners ... "
harris_image1, corners1 = harris_corners(img1, file1)
cv2.imwrite("img/set" + test_set + "/img1_corners.png", harris_image1)

# Find out the corners for the first image
file2 = "img/set" + test_set + "/img2.png"
print "Loading : [", file2, "]"
img2 = Image.open(file2).convert('L')
img2 = np.array(img2)
print "Finding Harris Corners ... "
harris_image2, corners2 = harris_corners(img2, file2)
cv2.imwrite("img/set" + test_set + "/img2_corners.png", harris_image2)

# Generate the descriptors for each corner in the first image
offset = 8
Iy, Ix = np.gradient(np.array(img1, dtype=np.float))
features1 = [[] for i in range(len(corners1))]
for i in range(len(corners1)):
	x = corners1[i][0]
	y = corners1[i][1]
	# Getting the histogram for the first patch
	windowIx = Ix[y:y+offset, x:x+offset].flatten()
	windowIy = Iy[y:y+offset, x:x+offset].flatten()
	thetas = []
	n = len(windowIx)
	for j in range(n):
		thetas.append(math.degrees(math.atan2(windowIy[j],windowIx[j])))
	hist = np.histogram(thetas, bins=[-180, -135, -90, -45, 0, 45, 90, 135, 180])
	features1[i].append(hist)
	# Getting the histogram for the second patch
	windowIx = Ix[y:y+offset, x-offset:x].flatten()
	windowIy = Iy[y:y+offset, x-offset:x].flatten()
	thetas = []
	n = len(windowIx)
	for j in range(n):
		thetas.append(math.degrees(math.atan2(windowIy[j],windowIx[j])))
	hist = np.histogram(thetas, bins=[-180, -135, -90, -45, 0, 45, 90, 135, 180])
	features1[i].append(hist)
	# Getting the histogram for the third patch
	windowIx = Ix[y-offset:y, x-offset:x].flatten()
	windowIy = Iy[y-offset:y, x-offset:x].flatten()
	thetas = []
	n = len(windowIx)
	for j in range(n):
		thetas.append(math.degrees(math.atan2(windowIy[j],windowIx[j])))
	hist = np.histogram(thetas, bins=[-180, -135, -90, -45, 0, 45, 90, 135, 180])
	features1[i].append(hist)
	# Getting the histogram for the fourth patch
	windowIx = Ix[y-offset:y, x:x+offset].flatten()
	windowIy = Iy[y-offset:y, x:x+offset].flatten()
	thetas = []
	n = len(windowIx)
	for j in range(n):
		thetas.append(math.degrees(math.atan2(windowIy[j],windowIx[j])))
	hist = np.histogram(thetas, bins=[-180, -135, -90, -45, 0, 45, 90, 135, 180])
	features1[i].append(hist)
features1 = np.array(features1)

Iy, Ix = np.gradient(np.array(img2, dtype=np.float))
features2 = [[] for i in range(len(corners2))]
for i in range(len(corners2)):
	x = corners2[i][0]
	y = corners2[i][1]
	# Getting the histogram for the first patch
	windowIx = Ix[y:y+offset, x:x+offset].flatten()
	windowIy = Iy[y:y+offset, x:x+offset].flatten()
	thetas = []
	n = len(windowIx)
	for j in range(n):
		thetas.append(math.degrees(math.atan2(windowIy[j],windowIx[j])))
	hist = np.histogram(thetas, bins=[-180, -135, -90, -45, 0, 45, 90, 135, 180])
	features2[i].append(hist)
	# Getting the histogram for the second patch
	windowIx = Ix[y:y+offset, x-offset:x].flatten()
	windowIy = Iy[y:y+offset, x-offset:x].flatten()
	thetas = []
	n = len(windowIx)
	for j in range(n):
		thetas.append(math.degrees(math.atan2(windowIy[j],windowIx[j])))
	hist = np.histogram(thetas, bins=[-180, -135, -90, -45, 0, 45, 90, 135, 180])
	features2[i].append(hist)
	# Getting the histogram for the third patch
	windowIx = Ix[y-offset:y, x-offset:x].flatten()
	windowIy = Iy[y-offset:y, x-offset:x].flatten()
	thetas = []
	n = len(windowIx)
	for j in range(n):
		thetas.append(math.degrees(math.atan2(windowIy[j],windowIx[j])))
	hist = np.histogram(thetas, bins=[-180, -135, -90, -45, 0, 45, 90, 135, 180])
	features2[i].append(hist)
	# Getting the histogram for the fourth patch
	windowIx = Ix[y-offset:y, x:x+offset].flatten()
	windowIy = Iy[y-offset:y, x:x+offset].flatten()
	thetas = []
	n = len(windowIx)
	for j in range(n):
		thetas.append(math.degrees(math.atan2(windowIy[j],windowIx[j])))
	hist = np.histogram(thetas, bins=[-180, -135, -90, -45, 0, 45, 90, 135, 180])
	features2[i].append(hist)
features2 = np.array(features2)