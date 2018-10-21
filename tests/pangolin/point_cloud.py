import sys
sys.path.append('../lib/')
import OpenGL.GL as gl
import pangolin as pango
import numpy as np

def main():
  pango.CreateWindowAndBind('point cloud cube render', 640, 480)

  # Projection and ModelView Matrices
  scam = pango.OpenGlRenderState(
                  pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
                  pango.ModelViewLookAt(-1, -1, -1, 0, 0, 0, pango.AxisDirection.AxisY))
  handler = pango.Handler3D(scam)

  # Interactive View in Window
  disp_cam = pango.CreateDisplay()
  disp_cam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
  disp_cam.SetHandler(handler)


  # Create a random point cloud
  pts = np.random.random((100, 3))* 10

  # Color matrix based on point location
  colors = np.zeros((len(pts), 3))
  colors[:, :] = 1. - pts[:,:]/10
  #colors[:] = [1.0, 0.0, 0.0]
  while not pango.ShouldQuit():
    # Clear screen
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.15, 0.15, 0.15, 0.0)
    disp_cam.Activate(scam)

    # Draw Points
    gl.glPointSize(5)
    gl.glColor3f(0.0, 1.0, 0.0)
    pango.DrawPoints(pts, colors)

    # Finish Drawing
    pango.FinishFrame()






if __name__ == '__main__':
  main()
