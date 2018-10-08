'''
	shitSLAM, Kemal Ficici, 2018

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
	cap = cv2.VideoCapture('test-vid/0.mp4')


	ret, frame1 = cap.read()
	frame1 = cv2.resize(frame1, (0,0), fx=0.5, fy=0.5)
	frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

	corners = []
	corner = getFeatures(frame1)
	kp, des = getORB(frame1, corner)
	corners.append(kp)
	prev = frame1

	while(cap.isOpened()):
		ret, frame = cap.read()
		frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
		out = frame
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


		# Track new features if we lose too many corners
		if (len(corners[-1]) < 2000):
			corner = getFeatures(prev)
			corners[-1] = corner#getORB(prev, corner)

		# Matching corners and adding to multi-frame matrix for comparison
		pt1, pt2 = match_frames_lk(prev, frame)


		# Convert to ORB Features
		kp, des = getORB(frame, pt2)
		corners[-1] = pt1
		corners.append(pt2)
		#corners[-1] = p.pt for p in kp
		print('inliers: ' + str(len(kp)))
		# Limit frames in array to 15
		if len(corners) > 15:
			corners = corners[1:]

		prev = frame

		#print('Features Detected: ' + str(len(corners[-1])))


		for i in range(len(corners[-1])):
			x, y = corners[-1][i].ravel()
			try:
				x_prev, y_prev = corners[-2][i].ravel()
				out = cv2.line(out, (x_prev, y_prev), (x,y), (255, 0, 150), 2, 1)
			except:
				None
			out = cv2.circle(out, (x,y), 3, (0,255,0), 1)
		#print(len(corners))
		cv2.imshow('image', out)



		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
