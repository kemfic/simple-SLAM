#!/usr/bin/env python3
"""
Usage:
  ./simple_vo.py <sequence> [<gt>]
"""

from docopt import docopt
import cv2
import numpy as np
from frame import Frame
from viewer import Viewer3D, vt_done
from utils import getError
from params import focal, K

class SimpleVO(object):
  def __init__(self, img, focal, K):

    self.poses = []
    self.gt = []
    self.errors = []
    self.gt.append(np.eye(4))
    self.scale = 1.0
    
    self.focal = focal
    self.K = K
    self.prevFrame = Frame(img, self.focal, self.K)
    self.curFrame = Frame(img, self.focal, self.K)
    self.poses.append(self.curFrame.Rt)

  def update(self, img, gt=None):
    self.curFrame = Frame(img, self.focal, self.K)
    self.curFrame.match_frames(self.prevFrame)
    self.curFrame.get_essential_matrix(self.prevFrame)


    #TODO: set scale to 1.0 if there is no gt
    if gt is not None:
      self.gt.append(gt)
      self.scale = np.sqrt(np.sum((self.gt[-1]-self.gt[-2])**2) )    
    
    self.curFrame.get_Rt(self.prevFrame, self.scale)
    self.poses.append(self.curFrame.Rt)
    
    if gt is not None:
      error_r, error_t = getError(vo.poses[-1],vo.poses[-2],vo.gt[-1], vo.gt[-2])

      self.errors.append((error_r, error_t))

  def annotate_frames(self):
    """
    a is current frame
    b is prev frame
    """
    a = self.curFrame
    b = self.prevFrame
    out = np.copy(a.img)
    old_coords = b.coords
    
    [cv2.line(out, tuple(np.int0(a.coords[i_a])), tuple(np.int0(b.coords[i_b])), (255, 0, 255), 2) for i_a, i_b in a.des_idxs]

    [cv2.circle(out, tuple(np.int0(a.coords[i_a])), 4,(0,255,0), 2) for i_a, i_b in a.des_idxs]
    return out

if __name__ == "__main__":
  
  args = docopt(__doc__)


  cap = cv2.VideoCapture(args['<sequence>'])
  #cap = cv2.VideoCapture('/home/kemfic/projects/ficicislam/dataset/vids/15.mp4')
  
  ret, frame = cap.read()



  vo = SimpleVO(frame, focal, K)#, np.eye(4))
  
  viewer = Viewer3D()
  
  if args['<gt>'] is not None:
    txt = np.loadtxt("vid/06.txt")
    gt_prev = np.eye(4)
    error = []
  
  while not vt_done.is_set() and cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
    ret, frame = cap.read()
    #cv2.imshow("frame", vo.annotate_frames())
    
    framenum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    if args['<gt>'] is not None:
      gt = np.eye(4)
      gt[:3, :] = txt[framenum].reshape(3,4)
      gt_tform = gt * np.linalg.inv(gt_prev)


      gt_prev = gt
      vo.update(frame, gt)
    else:
      vo.update(frame)
    
    if framenum > 2:
      viewer.update(vo)
      if args['<gt>'] is not None:
        p_tform = vo.poses[-1] * np.linalg.inv(vo.poses[-2])
        error.append((p_tform * np.linalg.inv(gt_tform))[:3, -1])

    vo.prevFrame = vo.curFrame
  
  
  print("exiting...")
  viewer.stop()
  cap.release()
  #cv2.destroyAllWindows()
