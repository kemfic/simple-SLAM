import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


from params import *

def getFeatures(img, params=shi_tomasi):
	'''
	Shi-Tomasi Corner Detection
	    IN:
	    	- Grayscale Image
	    	- GoodFeaturesToTrack Params
	'''
	corners = cv2.goodFeaturesToTrack(img, **params)
	# corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDist)
	return corners

def getEssential(pt1, pt2, params=em_params):
	'''

	https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
	'''
	E, mask = cv2.findEssentialMat(pt2, pt1, **params)
	return E

def getPose(E, pt1, pt2):
	'''
	https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
	'''
	pts, R, t, mask = cv2.recoverPose(E, pt2, pt1)
	return None

def matchFeatures(img, img2, pt1, params=lk_params):
	'''
	Optical Flow between two frames using Lucas-Kanade Method
	'''

	pt2, status, err = cv2.calcOpticalFlowPyrLK(
		img,
		img2,
		pt1,
		None,
		**params
		)

	pt2 = pt2[status == 1]
	pt1 = pt1[status == 1]

	return pt1, pt2

def getORB(img, kps):

	kp = [cv2.KeyPoint(x=f.ravel()[0], y=f.ravel()[1], _size=20) for f in kps]
	orb = cv2.ORB_create()

	kp, des = orb.compute(img, kp)

	return kp, des

def getFundamentalMatrix(kp1, kp2, params=ransac_params):
	'''
	# Uses RANSAC to get fundamental matrix and inliers
	# http://scikit-image.org/docs/dev/auto_examples/transform/plot_fundamental_matrix.html
	'''
	model, inliers = ransac((kp1, kp2), FundamentalMatrixTransform, **params)
	kp1_in = kp1[inliers]
	kp2_in = kp2[inliers]

	return model, kp1_in, kp2_in
