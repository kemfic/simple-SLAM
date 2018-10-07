'''
	ficiciSLAM, Kemal Ficici, 2018

	LICENSE
	do whatever you want with this, don't sue me
	attributing me would be pretty nice tho
'''
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from params import *
from utils import *

if __name__ == '__main__':
	img = cv2.imread('test-img/1.jpg', 0)
	img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
	corners = getFeatures(img, shi_tomasi)


	out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	bg = np.zeros_like(img)
	for corner in corners: # unravel and plot features on image as circles
		x,y = corner.ravel()
		out = cv2.circle(out, (x,y), 1, (0,255,0), 1)
		bg = cv2.circle(bg, (x,y), 1, 255, 1)	
	print(len(corners))
	cv2.imshow('blackground', bg)
	cv2.imshow('image', out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()