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

    def __init__(self, img, frame_id, prevFrame=None,  param_=param_):
        self.params = param_
        if (np.prod((img.shape)) > (3000 * 1000)):
            self.img = self.resize_img(img)
        else:
            self.img = img
        self.gray = self.bgr2gray(self.img)
        self.createFeatures()
        if prevFrame is not None:
            self.prevFrame = prevFrame
            self.matchFeatures()

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

        samples = np.array(samples)

        model, inliers = ransac((samples[:,0], samples[:,1]), **self.params.get('ransac_params'))

        print("Matches:  %d -> %d -> %d -> %d" % (len(self.des), len(matches), len(inliers), sum(inliers)))

        self.matches = np.array(self.matches)
        self.prev_matches = np.array(self.prev_matches)
        self.matches = self.matches[inliers.nonzero()]
        self.prev_matches = self.prev_matches[inliers.nonzero()]
        '''
        pt2, status, err = cv2.calcOpticalFlowPyrLK(
                self.img,
                self.prevFrame.img,
                self.corners,
                None,
                **self.params.get('lk_params')
                )

        self.matches = [cv2.KeyPoint(x=f.ravel()[0], y=f.ravel()[1], _size=10) for f in self.corners[status==1]]
        self.prev_matches = [cv2.KeyPoint(x=f.ravel()[0], y=f.ravel()[1], _size=10) for f in pt2[status==1]]
        '''

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

    @property
    def pt(self):
        return self.kp_to_coords(self.kp)
