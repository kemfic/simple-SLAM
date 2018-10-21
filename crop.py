import cv2
import numpy as np
import os

FILE_OUTPUT = 'out.mp4'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture('bleh.mp4')

currentFrame = 0

# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

width = int(width - 450)
height = int(height - 450)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
out = cv2.VideoWriter('testvideo.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30.0, (int(width),int(height)))

# while(True):
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        # Handles the mirroring of the current frame
        frame = frame[:(height), :(width)]

        # Saves for video
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame',frame)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
