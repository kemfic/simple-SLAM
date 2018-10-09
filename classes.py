import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from params import *
from utils import *
param_ = param

class Frame(object):
    '''
        Frame object:
            - Image
            - Id
            - Features
            - Pose
    '''
    bgr2gray = lambda self, img : cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize_img = lambda self, img: cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    kps = None
    prevFrame = None
    focal = 718.8560
    pp = (0., 0.)
    distance = 500.

    R = None
    t = None
    T_accum = None
    R_accum = None
    cameraMatrix = []
    def __init__(self, img, frame_id, prevFrame=None,  param_=param_):
        self.params = param_

        if (np.prod((img.shape)) > (3000 * 1000)):
            self.img = self.resize_img(img)
        else:
            self.img = img


        self.pp = [0.,0.]#np.array([self.img.shape[0] / 2., self.img.shape[1]/2.])
        self.cameraMatrix = np.array([[self.focal, 0., self.pp[0]],
                                [0., self.focal, self.pp[1]],
                                [0., 0., 1.]])

        self.K = self.cameraMatrix
        self.Kinv = np.linalg.inv(self.K)
        self.gray = self.bgr2gray(self.img)
        self.createFeatures()
        if prevFrame is not None:
            self.prevFrame = prevFrame
            self.matchFeatures()
            self.traj = self.prevFrame.traj
        else:
            self.traj = np.zeros((1000,1000), np.uint8)
        self.id = frame_id
        self.size = self.img.shape
        self.createFeatures()

        #if self.id < 2:
            #self.features = initFrame

    def createFeatures(self):
        '''
            Detects features
             - Shi-Tomasi Corner Detection
             - ORB creation

        '''
        kps = cv2.goodFeaturesToTrack(self.gray, **self.params.get('shi_tomasi'))
        kp = [cv2.KeyPoint(x=f.ravel()[0], y=f.ravel()[1], _size=10) for f in kps]
        orb = cv2.ORB_create()

        self.corners = kps
        self.kp, self.des = orb.compute(self.gray, kp)

    def matchFeatures(self):
        '''
            Tracks features between two frames
                - Matching using Hamming Distance
        '''

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(self.prevFrame.des, self.des, k=2)
        self.matches = []
        self.prev_matches = []
        samples = []
        for m, n in matches:
             if m.distance < 0.75*n.distance and m.distance < 32 and np.linalg.norm(np.subtract(self.kp[m.trainIdx].pt, self.prevFrame.kp[m.queryIdx].pt)) < 500:
                self.matches.append((self.kp[m.trainIdx]))
                self.prev_matches.append(self.prevFrame.kp[m.queryIdx])
                samples.append( ( self.kp[m.trainIdx].pt, self.prevFrame.kp[m.queryIdx].pt ) )

        self.matches = np.array(self.matches)
        self.prev_matches = np.array(self.prev_matches)

        samples = np.array(samples)

        samples[:, 0, :] = self.normalize(samples[:, 0, :])
        samples[:, 1, :] = self.normalize(samples[:, 1, :])


        #model, inliers = ransac((samples[:,0], samples[:,1]), **self.params.get('ransac_params'))

        ##print("Matches:  %d -> %d -> %d -> %d" % (len(self.corners), len(self.matches), len(inliers), sum(inliers)))
        #samples = samples[inliers]
        #Rt = self.extractRt(np.array(model.params))

        #print(Rt)
        #if self.prevFrame.pose is not None:
        #    self.pose = np.dot(Rt, self.prevFrame.pose)
        #else:
        #    pose = np.eye(4)

        #self.matches = np.array(self.matches)
        #self.prev_matches = np.array(self.prev_matches)
        #self.matches = self.matches[inliers.nonzero()]
        #self.prev_matches = self.prev_matches[inliers.nonzero()]

        #_, self.R, self.t, mask, triangulatedPoints = cv2.recoverPose(model.params, np.array([self.kp_to_coords(self.matches)]), np.array([self.kp_to_coords(self.prev_matches)]), cameraMatrix=self.cameraMatrix, distanceThresh=self.distance)
        #tp = np.array([])


        self.matches = np.array(self.matches)
        self.prev_matches = np.array(self.prev_matches)

        F, mask = cv2.findFundamentalMat(np.array([self.kp_to_coords(self.matches)]), np.array([self.kp_to_coords(self.prev_matches)]), cv2.FM_RANSAC, 1, 0.99)

        mask = np.squeeze(mask)
        #print(mask.shape)
        #print(self.matches.shape)
        self.matches = self.matches[mask == 1]
        self.prev_matches = self.prev_matches[mask == 1]

        _, self.R, self.t, tp= cv2.recoverPose(F, np.array([self.kp_to_coords(self.matches)]), np.array([self.kp_to_coords(self.prev_matches)]), cameraMatrix=self.cameraMatrix)

        # get from skimage ransac
        #_, self.R, self.t, tp= cv2.recoverPose(model.params, np.array([self.kp_to_coords(self.matches)]), np.array([self.kp_to_coords(self.prev_matches)]), cameraMatrix=self.cameraMatrix)

        #print(_)
        #print(self.t)
        if self.prevFrame.T_accum is not None:
            self.T_accum = self.prevFrame.T_accum + np.dot(self.prevFrame.R_accum, self.t)
            self.R_accum = self.R.dot(self.prevFrame.R_accum)
        else:
            self.T_accum = self.t
            self.R_accum = self.R
        self.T_accum = np.array(self.T_accum)


    #github.com/geohot/twitchslam
    def normalize(self, pts):
        return np.dot(self.Kinv, np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T).T[:, 0:2]

    def poseRt(R, t):
        ret = np.eye(4)
        ret[:3, :3] = R
        ret[:3, 3] = t
        return ret

    # pose
    def extractRt(F):
        W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
        U,d,Vt = np.linalg.svd(F)
        if np.linalg.det(U) < 0:
            U *= -1.0
        if np.linalg.det(Vt) < 0:
            Vt *= -1.0
        R = np.dot(np.dot(U, W), Vt)
        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(U, W.T), Vt)
        t = U[:, 2]

        # TODO: Resolve ambiguities in better ways. This is wrong.
        if t[2] < 0:
            t *= -1

        # TODO: UGLY!
        if os.getenv("REVERSE") is not None:
            t *= -1
        return np.linalg.inv(poseRt(R, t))


    def kp_to_coords(self, kps):
        return [f.pt for f in kps]

    def annotate(self):
        out = self.img

        # Draw flow vectors
        # Add detected corners
        if self.prevFrame is not None:
            coords = self.kp_to_coords(self.matches)
            prev_coords = self.kp_to_coords(self.prev_matches)

            for coord, prev_coord in zip(coords, prev_coords):
                out = cv2.line(out,
                            (int(prev_coord[0]), int(prev_coord[1])),
                                (int(coord[0]),int(coord[1])),
                                (255, 0, 255),
                                1)
        else:
            coords = self.kp_to_coords(self.kp)
        #cv2.drawKeypoints(out, self.kp, out, (0,255,0), 0)
        for coord in coords:
            out = cv2.circle(out,(int(coord[0]), int(coord[1])), 2, (0,255,0))
        # Add frame number to bottom corner
        out = cv2.putText(out,
            ('frame: ' + str(int(self.id))),
            (0, self.size[0]-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1)

        out = cv2.putText(out,
            ("inliers: " + str(len(coords))),
             (0, self.size[0] - 30),
             cv2.FONT_HERSHEY_SIMPLEX,
             0.5,
             (0,255,0),
             1)
        return out

    def annotate_traj(self):
        #print(self.T_accum)
        self.traj = cv2.circle(self.prevFrame.traj, (500-2*self.T_accum[0], 500-2*self.T_accum[2]), 4, (255), 1)

    @property
    def pt(self):
        return self.kp_to_coords(self.kp)
