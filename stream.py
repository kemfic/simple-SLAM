from frame import *
import glob
import OpenGL.GL as gl
from viewer import *
import time
traj_size = (700, 700)
class Stream(object):
  '''
  Stream Class contains consecutive frames and their associated data.
  '''
  frames = []
  res = 1000
  traj_scale = 0.5
  focal = res
  def __init__(self, img, K=None):
    '''
    Adds first frame, initializes trajectory image, and camera intrinsics
    '''
    self.traj = np.zeros(traj_size)

    self.scale = float(self.res) / max(img.shape)
    img = cv2.resize(img, (0,0), fx=self.scale, fy=self.scale)
    if K is not None:
      self.K = np.array(K)
    else:
      self.K = np.array([
        [self.focal, 0, img.shape[1]//2],
        [0, self.focal, img.shape[0]//2],
        [0, 0, 1]])
    self.frames.append(Frame(img))
    '''
    Camera Intrinsics Matrix
    [fx, 0, cx]
    [0, fy, cy]
    [0,  0,  1]
    '''


  def update(self, img):
    '''
    adds new frames, calculates correspondences, essential matrices, and pose matrices between frames.
    '''
    # TODO: fix ugly code

    # resize
    img = cv2.resize(img, (0,0), fx=self.scale, fy=self.scale)

    # init a new frame
    # self.frames.append(Frame(img))
    f_cur = Frame(img)
    f_prev = self.frames[-1]

    # initial correspondence (ORB)
    # TODO: add option between ORB and LK tracking
    idxs = match_frames(f_cur.des,
        f_prev.des,
        f_cur.coords,
        f_prev.coords)

    idxs = np.array(idxs)
    # estimates essential matrix using RANSAC, and filters outliers
    # TODO: get inlier filtering to work
    f_cur.des_match_idx, f_cur.F = estimate_f_matrix(idxs,
                                                     f_cur.coords,
                                                     f_prev.coords,
                                                     self.K)

    # returns R and t (incremental), and triangulated points
    # TODO: understand how and why this works
    f_cur.pts4d, f_cur.R, f_cur.t, f_cur.rt_pts = get_R_t(f_cur.F,
                                             f_cur.coords[f_cur.des_match_idx[:,0]],
                                             f_prev.coords[f_cur.des_match_idx[:,1]],
                                             self.K)
    f_cur.color = f_cur.img[f_cur.rt_pts[:,0],f_cur.rt_pts[:,1]]
    '''
    print(np.shape(idxs))
    print(np.shape(f_cur.des_match_idx))
    print(np.shape(f_cur.pts4d))
    print(" ------------- ")
    '''
    # converts R and t to pose matrix in homogenous coords (global coords)
    Rt = np.array(cvt2Rt(f_cur.R, f_cur.t))
    f_cur.pose = f_prev.pose.dot(Rt)
    # add frames to stream array
    self.frames[-1] = f_prev
    self.frames.append(f_cur)

  @property
  def annotate(self):
    '''
    Plots flow vectors, detected features, and labels frames
     - TODO: add this as part of pangolin viewer?
    '''
    a = self.frames[-2]
    b = self.frames[-1]

    out = b.img
    coords = b.coords
    idxs = b.des_match_idx
    for i_b, i_a in idxs:
      out = cv2.line(out,
                (int(b.coords[i_b][0]), int(b.coords[i_b][1])),
                (int(a.coords[i_a][0]), int(a.coords[i_a][1])),
                (255,0,255),
                1)

      out = cv2.circle(out, (int(b.coords[i_b][0]), int(b.coords[i_b][1])), 2, (0, 255, 0))



    out = cv2.putText(out,
      (' frame: ' + str(len(self.frames))),
      (0, self.frames[-1].size[0]-10),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0,255,0),
      2)

    out = cv2.putText(out,
      (" inliers: " + str(len(idxs))),
      (0, b.size[0] - 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0,255,0),
      2)
    return out

  @property
  def annotate_traj(self):
    '''
    Plots trajectories
      - This is nowhere near as good as matplolib or an opengl/pangolin viewer
      - Use the pangolin viewer if you can
    '''
    self.traj = cv2.circle(self.traj,
        ((self.traj.shape[0]//2)+int(self.frames[-1].pose[0, -1] * self.traj_scale), (self.traj.shape[1]//2)-int(self.frames[-1].pose[-2,-1] * self.traj_scale)),
        3,
        (255), -1)
    self.traj = cv2.circle(self.traj,
        ((self.traj.shape[0]//2)+int(self.frames[-1].pose[0, -1] * self.traj_scale), (self.traj.shape[1]//2)-int(self.frames[-1].pose[-2,-1] * self.traj_scale)),
        1,
        (0),-1)
    return self.traj


if __name__ == '__main__':
  # Can pick between different input videos
  if len(sys.argv) > 1:
    vid = glob.glob('./vids/'+ str(sys.argv[1]) + '.*')[0]
  else:
    vid = './vids/1.mp4'

  cap = cv2.VideoCapture(vid)
  cv2.namedWindow('stream', cv2.WINDOW_NORMAL)
  cv2.namedWindow('traj', cv2.WINDOW_NORMAL)
  cv2.moveWindow("stream", 1920+640, 1)


  ret, frame = cap.read()
  stream = Stream(frame)
  cv2.resizeWindow('traj', traj_size[1], traj_size[0])


  disp_view = MapViewer()
  while(cap.isOpened()):
    ret, frame = cap.read()


    # add new image frame to stream
    stream.update(frame)

    if cap.get(cv2.CAP_PROP_POS_FRAMES) > 2:
      disp_view.update(stream)

    cv2.imshow('stream', stream.annotate)
    cv2.resizeWindow('stream', 640, 640)
    cv2.imshow('traj', stream.annotate_traj)


    if cv2.waitKey(1) & 0xFF == ord('q'):
      print("exiting...")
      break

  disp_view.stop()
  cap.release()
  cv2.destroyAllWindows()

