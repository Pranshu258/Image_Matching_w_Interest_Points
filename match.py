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
threshold = 1000000
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
	Ixy = Ix*Iy
	Iyy = Iy*Iy
	height, width = img1.shape

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
	print "Number of corners found: ", len(cornerList)
	print "Applying Non Maximum Suppression ... "
	for corner in cornerList:
		x, y, r = corner[0], corner[1], corner[2]
		if not (x >= 8 and y >= 8 and y < height-8 and x < width-8):
			continue
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
	print "Number of corners after Suppression: ", len(suppressed_cornerList)
	return color_img, suppressed_cornerList

# This function returns the SIFT feature descriptor for each corner as a numpy array
def getFeatures(img, corners):
	features = [[[0 for j in range(8)] for K in range(16)] for i in range(len(corners))]
	Iy, Ix = np.gradient(np.array(img, dtype=np.float))
	height, width = img.shape
	count = 0
	for corner in corners:
		x, y = corner[0], corner[1]
		offy = -8
		for i in range(4):
			offx = -8
			for j in range(4):
				# Get the 4x4 patch
				patchIx = Ix[y+offy:y+offy+4, x+offx:x+offx+4].flatten()
				patchIy = Iy[y+offy:y+offy+4, x+offx:x+offx+4].flatten()
				# Calculate the 8 bit histogram
				thetas, weightList = [], []
				for K in range(16):
					thetas.append(math.degrees(math.atan2(patchIy[K]-Iy[y][x],patchIx[K]-Ix[y][x])))
					weightList.append(np.linalg.norm(np.array([patchIy[K]-Iy[y][x],patchIx[K]-Ix[y][x]])))
				weightList = weightList/np.sum(weightList)
				features[count][i*4 + j] = np.histogram(thetas, bins=[-180, -135, -90, -45, 0, 45, 90, 135, 180], weights=weightList)
				# Move to the next patch
				offx = offx + 4
			offy = offy + 4
		count = count + 1
	return (np.array(features))[:,:,0]


###################################################################################################

# Find out the corners for the first image
print "-------------------------------------------------"
file1 = "img/set" + test_set + "/img1.png"
print "Loading : [", file1, "]"
img1 = Image.open(file1).convert('L')
img1 = np.array(img1)

print "Finding Harris Corners ... "
harris_image1, corners1 = harris_corners(img1, file1)
cv2.imwrite("img/set" + test_set + "/img1_corners.png", harris_image1)

print "Constructing SIFT feature descriptors ... "
features1 = getFeatures(img1, corners1)
print "-------------------------------------------------"

# Find out the corners for the second image
file2 = "img/set" + test_set + "/img2.png"
print "Loading : [", file2, "]"
img2 = Image.open(file2).convert('L')
img2 = np.array(img2)
if test_set == '3':
	threshold = threshold/400

print "Finding Harris Corners ... "
harris_image2, corners2 = harris_corners(img2, file2)
cv2.imwrite("img/set" + test_set + "/img2_corners.png", harris_image2)

print "Constructing SIFT feature descriptors ... "
features2 = getFeatures(img2, corners2)
print "-------------------------------------------------"

# Image matching starts here
###################################################################################################
for i in range(len(features1)):
	D1 = 100.0
	D2 = 100.0
	match1, match2 = -1, -1
	for j in range(len(features2)):
		d = 0
		# Calculate the distance between the descriptors
		for r in range(16):
			d = d + np.linalg.norm(features2[j][r]-features1[i][r])/16.0
		if D1 > d:
			D2 = D1
			D1 = d
			match2 = match1
			match1 = j
		elif D2 > d:
			D2 = d
			match2 = j
	if D1/D2 < 0.9:
		print corners1[i][:2], corners2[match1][:2], D1
