# IMPLEMENTING HARRIS CORNER
# G13, CS676-2016, IITK
###################################################################################################

import Image
import numpy as np
import cv2
import sys

###################################################################################################
# Configuration variables
k = 0.04
window_size = 5
offset = window_size/2
threshold = 100000
filename = ""
corner_set = []

if len(sys.argv) == 2:
	test_set = str(sys.argv[1])
else:
	exit()

###################################################################################################
# This function returns the harris corners in the image
def harris_corners(img):
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

# Find out the corners
for instance in range(1,3):
	filename = "img/set" + test_set + "/img" + str(instance) + ".png"
	print "Loading : [", filename, "]"
	img = Image.open(filename).convert('L')
	img = np.array(img)
	# Change the threshold for the 2 instance in set3
	if test_set == '3' and instance == 2:
		threshold = threshold/100
	print "Finding Harris Corners ... "
	harris_image, corners = harris_corners(img)
	corner_set.append(corners)
	cv2.imwrite("img/set" + test_set + "/img" + str(instance) + "_corners.png", harris_image)


# Generate the descriptors for each corner in the images

