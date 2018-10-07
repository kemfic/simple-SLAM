# ficiciSLAM Parameters
import cv2
import numpy as np
param = dict(
	intrinsic_params = dict(
		alpha = 1000,
		k = 1/6,
		focal_length = 10
		),

	shi_tomasi = dict(
		maxCorners = 4000,
		qualityLevel = 0.02,
		minDistance = 1
	),

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
		min_samples = 8,
		residual_threshold = 1,
		max_trials = 5000
		),

	em_params = dict(
		focal = 1.0,
		pp = (0.,0.),
		method = cv2.RANSAC,
		prob = 0.995,
		threshold=3.0
		)
)
