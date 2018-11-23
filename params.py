import cv2
import numpy as np
from skimage.transform import EssentialMatrixTransform as T


param = dict(
	shi_tomasi = dict(
		maxCorners = 10000,
		qualityLevel = 0.01,
		minDistance =3),

	ransac_params = dict(
        model_class = T,
		min_samples = 8,
		residual_threshold = 0.01,
		max_trials = 100
		)

)

