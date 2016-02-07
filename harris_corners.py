# IMPLEMENTING HARRIS CORNER
# G13, CS676-2016, IITK
###################################################################################################

import Image
import numpy as np
import cv2

###################################################################################################

filename = "img/set1/img1.png"

# Loading the image as a greyscale numpy array
print "Loading the Image ... "
img = Image.open(filename).convert('L')
img = np.array(img)

# Gradients in y and x direction, [columns reprsent the x and rows represent the y]
Iy, Ix = np.gradient(np.array(img, dtype=np.float))


Ixx = Ix**2
Ixy = Iy*Iy
Iyy = Iy**2
height = img.shape[0]
width = img.shape[1]

cornerList = []
k = 0.04
window_size = 5
offset = window_size/2
thresh = 1000000

img = cv2.imread(filename, 0)
newImg = img.copy()
color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)

print "Finding Corners..."
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
		if r > thresh:
			cornerList.append([x, y, r])
			color_img.itemset((y, x, 0), 0)
			color_img.itemset((y, x, 1), 0)
			color_img.itemset((y, x, 2), 255)
cv2.imwrite("finalimage.png", color_img)