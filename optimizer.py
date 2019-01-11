import cv2
import numpy as np

import glob
import sys
import time

from stream import Stream
from viewer import MapViewer
from ba import BundleAdjustment

class Optimizer(object):
  def __init__(self, stream):
    self.ba = BundleAdjustment():
