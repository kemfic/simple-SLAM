from classes import *

import cv2
import numpy as numpy
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    cap = cv2.VideoCapture('test-vid/0.wmv')
    prevFrame = None

    position = []
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.ion()
    plt.show()
    while(True):
        ret, frame = cap.read()
        currentFrame = Frame(frame, prevFrame=prevFrame, frame_id=cap.get(cv2.CAP_PROP_POS_FRAMES))

        if currentFrame.id > 3:
            currentFrame.annotate_traj()
            cv2.imshow('test', currentFrame.traj)


        if currentFrame.id > 3:
            position.append(currentFrame.T_accum)
            #print(position)

        positions=np.flip(position)
        if currentFrame.id > 4:
            ax.plot(positions[:,0,0], positions[:,1,0], positions[:,2,0])
            plt.plot(positions[:,0,0], positions[:,1,0])
            fig.canvas.draw()
        cv2.imshow('current frame', currentFrame.annotate())
        prevFrame = currentFrame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
