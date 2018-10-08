import cv2
import numpy as np
import skimage.transform as transform
from skimage.measure import ransac

#from skimage.transform import FundamentalMatrixTransform


from params import *

def hamming_distance(a, b):
	'''
	Computes Hamming Distance/Bit Set P
	github.com/geohot/twitchslam
	'''
	r = (1 << np.arange(8))[:,None]
	return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)

def match_frames_knn(frame1, frame2):
	'''
	Matches descriptors between two frames.
	docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
	'''
	bf = cv2.bfMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.knnMatch(frame1.des, frame2.des, k=2)

	# ratio test
	f1_match, f2_match = [], []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			f1_match.append([m])

	# TODO Finish Implementation
	return None

def match_frames_lk(frame1, frame2, params=param.get('lk_params')):
	'''
	Optical Flow between two frames using Lucas-Kanade Method
	'''

	pt2, status, err = cv2.calcOpticalFlowPyrLK(
		frame1.img,
		frame2.img,
		frame1.pt,
		None,
		**params
		)

	pt2 = frame1.pt[status == 1]
	pt1 = frame2.pt[status == 1]

	return pt1, pt2

def getFeatures(img, params=param.get('shi_tomasi')):
	'''
	Shi-Tomasi Corner Detection
	    IN:
	    	- Grayscale Image
	    	- GoodFeaturesToTrack Params
	'''
	corners = cv2.goodFeaturesToTrack(img, **params)
	# corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDist)
	return corners

def getEssential(pt1, pt2, params=param.get('em_params')):
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

def getORB(img, kps):

	kp = [cv2.KeyPoint(x=f.ravel()[0], y=f.ravel()[1], _size=20) for f in kps]
	orb = cv2.ORB_create()

	kp, des = orb.compute(img, kp)

	return kp, des

def getFundamentalMatrix(kp1, kp2, params=param.get('ransac_params')):
	'''
	Uses RANSAC to get fundamental matrix and inliers
	http://scikit-image.org/docs/dev/auto_examples/transform/plot_fundamental_matrix.html
	'''
	model, inliers = ransac((kp1, kp2), transform.FundamentalMatrixTransform, **params)
	kp1_in = kp1[inliers]
	kp2_in = kp2[inliers]

	return model, kp1_in, kp2_in
