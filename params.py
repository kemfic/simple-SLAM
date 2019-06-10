import numpy as np

focal = 707.0912
K = np.array([
           [focal, 0.0, 601.8873],
           [0.0, focal, 183.1104],
           [0.0,0.0,1.0]])


param = dict(
  gftt = dict(maxCorners = 100000,
              qualityLevel = 0.01,
              minDistance =10)
              
              )
