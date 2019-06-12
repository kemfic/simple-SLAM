import cv2
import numpy as np
from params import param
from utils import *

class Frame(object):
  def __init__(self, img, focal=1000., K=None):
    self.Rt = np.eye(4)
    self.R = np.eye(3)
    self.img = img
    self.coords = getCorners(img)
    
    self.focal = focal
    self.K = K
    
    self.coords, self.des = get_features_orb(self.img, self.coords)

  def match_frames(self, prev):
    MIN_DISPLACE = 0
    kp1 = self.coords
    kp2 = prev.coords
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(self.des, prev.des, k=2)

    # Lowe's Ratio Test
    good = []
    des_idxs = []
    for m, n in matches:
      if (m.distance < 0.75*n.distance and MIN_DISPLACE < np.linalg.norm(np.subtract(kp2[m.trainIdx], kp1[m.queryIdx])) < 200): #m.distance < 32
        good.append(m.trainIdx)
        des_idxs.append((m.queryIdx, m.trainIdx))
    
    #print(len(des_idxs))
    self.des_idxs = np.array(des_idxs)
    self.kp1 = kp1

  def get_essential_matrix(self, prev):
    idxs = self.des_idxs
    
    coord1 = np.array(self.coords)
    coord2 = np.array(prev.coords)
  
    pt1 = coord1[idxs[:,0]]
    pt2 = coord2[idxs[:,1]]

    E, mask = cv2.findEssentialMat(pt1, pt2, cameraMatrix=self.K, method=cv2.RANSAC, prob=0.9, threshold=1.0)
  
    mask = mask.flatten()
    idxs = idxs[(mask==1), :]

    self.des_idxs = idxs

    self.E = E

  def get_Rt(self, prev, scale=1.0):
    prev_pts = prev.coords[self.des_idxs[:,1]]
    cur_pts = self.coords[self.des_idxs[:,0]]
    ret, R, t, mask, pts = cv2.recoverPose(self.E, prev_pts, cur_pts, cameraMatrix=self.K, distanceThresh=1000)
    print(t[-1])
    
    if abs(t[-1]) > 0.001 and np.argmax(np.abs(scale*t/t[-1])) == 2:
      t = scale*t/t[-1]
      Rt = np.eye(4)
      Rt[:3, :3] = R
      Rt[:3, 3] = np.squeeze(t)
      self.Rt = prev.Rt.dot(Rt)
    else:
      self.Rt = prev.Rt
    
