from classes import *

import cv2
import numpy as numpy
import matplotlib
from matplotlib import pyplot as plt

if __name__ == '__main__':
    cap = cv2.VideoCapture('test-vid/1.mp4')
    prevFrame = None
    while(True):
        ret, frame = cap.read()

        currentFrame = Frame(frame, prevFrame=prevFrame, frame_id=cap.get(cv2.CAP_PROP_POS_FRAMES))

        cv2.imshow('current frame', currentFrame.annotate())
        prevFrame = currentFrame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
