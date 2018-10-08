# ficiciSLAM Parameters
import cv2
import numpy as np
from skimage.transform import EssentialMatrixTransform as T

param = dict(
	intrinsic_params = dict(
		alpha = 1000,
		k = 1/6,
		focal_length = 10
		),

	shi_tomasi = dict(
		maxCorners = 5000,
		qualityLevel = 0.01,
		minDistance = 3),

	lk_params = dict(
		winSize = (21,21),
		maxLevel = 3,
		criteria = (
			cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
			5,
			0.3
			)
		),

	ransac_params = dict(
        model_class = T,
		min_samples = 8,
		residual_threshold = 0.2,
		max_trials = 100
		),

	em_params = dict(
		focal = 1.0,
		pp = (0.,0.),
		method = cv2.RANSAC,
		prob = 0.995,
		threshold=3.0
		)
)
