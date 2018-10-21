import sys
import OpenGL.GL as gl
import numpy as np
import time


sys.path.append('../lib/')
import pangolin as pango

def main():
  pango.CreateWindowAndBind('frustrum render', 640, 480)

  # Projection and ModelView Matrices
  scam = pango.OpenGlRenderState(
                  pango.ProjectionMatrix(640, 480, 2000, 2000, 320, 240, 0.1, 5000),
                  pango.ModelViewLookAt(0, -50, -10, 0, 0, 0, 0, -1, 0))#pango.AxisDirection.AxisY))
  handler = pango.Handler3D(scam)

  # Interactive View in Window
  disp_cam = pango.CreateDisplay()
  disp_cam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
  disp_cam.SetHandler(handler)

  # create and append pose matrices for cameras
  pose = np.identity(4)
  poses = []
  for i in range(3):
    poses.append(np.linalg.inv(pose))
    pose[2, 3] -= 1


  while not pango.ShouldQuit():
    # Clear screen
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.15, 0.15, 0.15, 0.0)
    disp_cam.Activate(scam)

    # Render Cameras
    gl.glLineWidth(2)

    gl.glColor3f(1.0, 0.0, 1.0)
    pango.DrawCamera(poses[0])

    gl.glColor3f(0.2, 1.0, 0.2)
    pango.DrawCameras(poses[1:-1])

    gl.glColor3f(1.0, 1.0, 1.0)
    pango.DrawCamera(poses[-1])


    # End frame update
    pango.FinishFrame()

if __name__ == '__main__':
  main()
