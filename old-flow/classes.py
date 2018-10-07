import cv2
import numpy as np

class Frame(object):
	'''
		Frame object:
			- Image
			- Id
			- Features
			- Pose
	'''	
	def __init__(self, img, id, resize=True)
		
		return None

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

	def trackFeatures(self, prevFrame):


	def annotate(self):
