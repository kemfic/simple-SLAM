import cv2
import numpy as np
import sys
from params import *
from skimage.measure import ransac

MAX_DIST  = 10000
MIN_DISPLACE = 0 # Minimum feature displacement between frames
def get_corners_st(img, params=param.get("shi_tomasi")):
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  corners = cv2.goodFeaturesToTrack(img, **params)
  return corners

def get_features_orb(img, corners):
  '''
  Computes ORB descriptors and coords for an image
  '''

  kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=15) for f in corners]

  orb = cv2.ORB_create()
  kps, des = orb.compute(img, kp)

  # Convert Keypoint to coord
  coords = np.array([kp.pt for kp in kps])
  return coords, des
def match_frames(des1, des2, pt1, pt2):
  '''
  Matches features using K-Nearest Neighbors, and returns the indexes of the matches
  '''

  kp1 = coords_to_kp(pt1)
  kp2 = coords_to_kp(pt2)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(des1, des2, k=2)
  # Lowe's Ratio Test
  good = []
  des_idxs = []
  for m, n in matches:
    if m.distance < 0.75*n.distance and MIN_DISPLACE < np.linalg.norm(np.subtract(kp2[m.trainIdx].pt, kp1[m.queryIdx].pt)) < 300: #m.distance < 32
      good.append(m.trainIdx)
      des_idxs.append((m.queryIdx, m.trainIdx))

  return des_idxs

def estimate_f_matrix(idxs, coord1, coord2, K, param=param.get("ransac_params")):
  '''
  Estimate fundamental matrix and compute inliers using RANSAC
  Edit: changed to Essential Matrix Estimation
  http://scikit-image.org/docs/dev/auto_examples/transform/plot_fundamental_matrix.html#sphx-glr-auto-examples-transform-plot-fundamental-matrix-py
  '''
  pt1 = normalize(K, coord1[idxs[:,0]])
  pt2 = normalize(K, coord2[idxs[:,1]])
  #model, inliers = ransac((pt1, pt2), **param)

  #return idxs[inliers], model.params
  E, mask = cv2.findEssentialMat(coord1[idxs[:,0]], coord2[idxs[:,1]], cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
  mask = mask.flatten()
  idxs = idxs[(mask==1), :]

  return idxs, E


def get_R_t(E, pts2, pts1, K):
  '''
  Decomposes Essential Matrix into Pose
  '''
  #pts2 = normalize(K, pts2)
  #pts1 = normalize(K, pts1)

  ret, R, t, mask, points = cv2.recoverPose(E, pts2, pts1, cameraMatrix=K, distanceThresh=MAX_DIST)
  points = points / points[-1, :] # change scale to 1

  points = points[:, points[2,:] >0.0] #remove points located behind camera
  return points.T, R, t

def cvt2Rt(R, t):
  Rt = np.eye(4)
  Rt[:3, :3] = R
  Rt[:3, 3] = np.squeeze(t)
  return Rt

def add_ones(x):
  if len(x.shape) == 1:
    return np.concatenate([x,np.array([1.0])], axis=0)
  else:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def normalize(K, pts):
  Kinv = np.linalg.inv(K)
  return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def coords_to_kp(coords):
  kp = [cv2.KeyPoint(x=f[0], y=f[1], _size=15) for f in coords]
  return np.array(kp)

class Frame(object):
  '''
  Frame class contains image with features
  '''
  F = np.eye(3)
  pose = np.eye(4)
  Rt = np.eye(4)
  pts4d = np.array([[0., 0., 0., 1.]])
  # Each frame contains indexes for matches between itself and previous frame
  des_match_idx = None # descriptor match[i] = (current_idx, prev_idx)

  def __init__(self, img):
    self.img = img
    self.coords = get_corners_st(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY))
    self.coords, self.des = get_features_orb(self.img, self.coords)
    self.size = self.img.shape

  @property
  def annotate(self):
    out = self.img
    coords = self.coords
    for coord in coords:
      out = cv2.circle(out, (int(coord[0]), int(coord[1])), 2, (0, 255, 0))
    '''
    out = cv2.putText(out,
      ('frame: ' + str(int(self.id))),
      (0, self.size[0]-10),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (0,255,0),
      1)
    '''
    out = cv2.putText(out,
      ("inliers: " + str(len(coords))),
      (0, self.size[0] - 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (0,255,0),
      1)
    return out

if __name__ == '__main__':
  # Can pick between different input videos
  if len(sys.argv) > 1:
    vid = './vids/' +  str(sys.argv[1]) + '.mp4'
  else:
    vid = './vids/1.mp4'

  cap = cv2.VideoCapture(vid)
  cv2.namedWindow('stream', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('stream', 1280, 720)
  ret, prevframe = cap.read()
  while(cap.isOpened()):
    ret, frame = cap.read()

    frame_current = Frame(frame)
    cv2.imshow('stream', frame_current.annotate)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

