import sys
sys.path.append('lib/')

import pangolin as pango
import OpenGL.GL as gl
import numpy as np
from multiprocessing import Process, Queue

SCALE = -5.

class MapViewer(object):
  '''
  3D viewer for ficiciSLAM written with pangolin/OpenGL
    - Heavily based off twitchslam's pangolin viewer (george, you a real G), and uoip's viewer from their S-PTAM implementation
      - https://github.com/geohot/twitchslam/blob/master/display.py
      - https://github.com/uoip/stereo_ptam/blob/master/viewer.py

  TODO:
    - write keyframe viewer
    - write pcl viewer
      - add color to pcl

  '''
  def __init__(self):
    self.state = None
    self.q = Queue()
    self.vt = Process(target=self.viewer_thread, args=(self.q,))
    self.vt.daemon = True
    self.vt.start()


    self.poses, self.colors = [], np.empty((0,3))
    self.pts = np.empty((0,4))

  def viewer_thread(self, q):
    self.viewer_init()

    while True:
      self.viewer_refresh(q)

  def viewer_init(self):
    w, h = (1024,768)
    f = 2000 #420

    pango.CreateWindowAndBind("ficiciSLAM Map Viewer", w, h)
    gl.glEnable(gl.GL_DEPTH_TEST) #prevents point overlapping issue, check out fake-stereo's issues for more info

    # Projection and ModelView Matrices
    self.scam = pango.OpenGlRenderState(
        pango.ProjectionMatrix(w, h, f, f, w //2, h//2, 0.1, 100000),
        pango.ModelViewLookAt(0, -50.0, -10.0,
                              0.0, 0.0, 0.0,
                              0.0, -1.0, 0.0))#pango.AxisDirection.AxisY))
    self.handler = pango.Handler3D(self.scam)

    # Interactive View in Window
    self.dcam = pango.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
    self.dcam.SetHandler(self.handler)
    self.dcam.Activate()

  def viewer_refresh(self, q):
    while not q.empty():
      self.state = q.get()
    # Clear and Activate Screen (we got a real nice shade of gray
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.15, 0.15, 0.15, 0.0)
    #gl.glClearColor(1.0, 1.0, 1.0, 0.0)
    self.dcam.Activate(self.scam)

    # Render
    if self.state is not None:
      gl.glLineWidth(1)
      # Render previous keyframes
      if self.state[0].shape[0] >= 2:
        gl.glColor3f(1.0, 0.0, 1.0)
        pango.DrawCameras(self.state[0][:-1])

      # Render current pose
      if self.state[0].shape[0] >= 1:
        gl.glColor3f(0.2, 1.0, 0.2)
        pango.DrawCameras(self.state[0][-1:])


      # Render keypoints
      if self.state[1].shape[0] != 0:
        gl.glPointSize(1)

        gl.glColor3f(1.0, 1.0, 1.0)
        pango.DrawPoints(self.state[1])
    pango.FinishFrame()

  def update(self, stream=None):
    '''
    Add new data to queue
    '''
    if self.q is None:
      return


    self.poses.append(stream.frames[-1].pose)

    # transform frame points to be displayed in world view
    cur_pts = np.dot(stream.frames[-1].pts4d, self.poses[-1].T)
    self.colors = np.append(self.colors, stream.frames[-1].color[...,::-1], axis=0)
    # add to the entire map
    self.pts = np.append(self.pts, cur_pts, axis=0)
    self.q.put((np.array(self.poses), np.squeeze(self.pts),np.squeeze(self.colors)))

  def stop(self):
    self.vt.terminate()
    self.vt.join()

    for x in self.__dict__.values():
      if isinstance(x, type(Queue())):
        while not x.empty():
          _ = x.get()

    print("viewer stopped")
