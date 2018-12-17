import numpy as np
import cv2
from frame import *

class Point(object):
  '''
  Contains:
   - Descriptor
   - Color
   - Location (3d Homogenous coords)
   '''
   def __init__(self, des, coord=np.ones((1,4)), color=np.zeros((1,3)) ):
     self.descriptor = des
     self.color = color
     self.coor
