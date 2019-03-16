import cv2
import numpy as np
from skimage.transform import EssentialMatrixTransform as T

class Params(object):
  def __init__(self):
    NotImplemented

class Cam(object):
  def __init__(self):
    self.fx = 1000
    self.fy = 1000
    self.cx = 500
    self.cy = 151

param = dict(
	shi_tomasi = dict(
		maxCorners = 3000,
		qualityLevel = 0.01,
		minDistance = 5),

	ransac_params = dict(
        model_class = T,
		min_samples = 10,
		residual_threshold = 0.02,
		max_trials = 100
		)

)

