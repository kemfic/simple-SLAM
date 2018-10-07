import cv2
import numpy as np
from params import *
#from utils import *
param_ = param
class Frame(object):
	'''
		Frame object:
			- Image
			- Id
			- Features
			- Pose
	'''	
	bgr2gray = lambda self, img : cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	resize_img = lambda self, img: cv2.resize(img, (0,0), fx=0.5, fy=0.5)

	def __init__(self, img, frame_id, prevImg=None,  params=param_):
		if (np.prod((img.shape)) > (3000 * 1000)):
			self.img = self.resize_img(img)
		else:
			self.img = img
		self.gray = self.bgr2gray(self.img)
		#self.prevImg = self.preprocess(prevImg)
		self.id = frame_id
		self.size = self.img.shape
		self.params = params
		self.createFeatures()
		#if self.id < 2:
			#self.features = initFrame

	def createFeatures(self):
		'''
			Detects and compares features
			 - Shi-Tomasi Corner Detection
			 - ORB creation

		'''
		kps = cv2.goodFeaturesToTrack(self.gray, **self.params.get('shi_tomasi'))
		kp = [cv2.KeyPoint(x=f.ravel()[0], y=f.ravel()[1], _size=10) for f in kps]
		orb = cv2.ORB_create()

		self.kp, self.des = orb.compute(self.gray, kp)




	def annotate(self):
		out = self.img

		# Draw flow vectors
		print(len(self.kp))
		# Add detected corners
		coords = [self.kp[f].pt for f in range(len(self.kp))]
		#cv2.drawKeypoints(out, self.kp, out, (0,255,0), 0)
		for coord in coords:
			out = cv2.circle(out,(int(coord[0]), int(coord[1])), 2, (0,255,0))
		# Add frame number to bottom corner

		out = cv2.putText(out, 
			('frame: ' + str(int(self.id))), 
			(0, self.size[0]-10),
			cv2.FONT_HERSHEY_SIMPLEX, 
			0.5, 
			(0,255,0), 
			1)

		out = cv2.putText(out,
			("features detected: " + str(len(self.kp))),
			 (0, self.size[0] - 30), 
			 cv2.FONT_HERSHEY_SIMPLEX,
			 0.5,
			 (0,255,0),
			 1)

		return out