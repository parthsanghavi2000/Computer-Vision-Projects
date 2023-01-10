import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import os

from optical_flow import *

def objectTracking(rawVideo, Bounding_box=False):
    # initilize
    frame_cnt = 300
    frames = np.empty((frame_cnt,), dtype=np.ndarray)
    frame_N = np.empty((frame_cnt,), dtype=np.ndarray)
    bboxs = np.empty((frame_cnt,), dtype=np.ndarray)
    for f in range(frame_cnt):
        _, frames[f] = rawVideo.read()


    if Bounding_box:
        num = int(input(" Tracking "))
        bboxs[0] = np.empty((num, 4, 2), dtype=float)
        for i in range(num):
            (x_box, y_box, boxw, boxh) = cv2.selectROI("Select Object %d" % (i), frames[0])
            cv2.destroyWindow("Select Object %d" % (i))
            bboxs[0][i, :, :] = np.array(
                [[x_box, y_box], [x_box + boxw, y_box], [x_box, y_box + boxh], [x_box + boxw, y_box + boxh]]).astype(float)


    # Start from the first frame, do optical flow for every two consecutive frames.
    X, Y = getFeatures(cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY), bboxs[0])
    for i in range(1, frame_cnt):
        print('Processing Frame', i)
        newXs, newYs = estimateAllTranslation(X, Y, frames[i - 1], frames[i])
        Xs, Ys, bboxs[i] = applyGeometricTransformation(X, Y, newXs, newYs, bboxs[i - 1])

        # update coordinates
        X = Xs
        Y = Ys

        # update feature points as required
        f_left = np.sum(Xs != -1)
        print('# of Features: %d' % f_left)
        if f_left < 15:
            print('Generate New Features')
            X, startYs = getFeatures(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), bboxs[i])

        frame_N[i] = frames[i].copy()
        for j in range(num):
            (x_box, y_box, boxw, boxh) = cv2.boundingRect(bboxs[i][j, :, :].astype(int))
            frame_N[i] = cv2.rectangle(frame_N[i], (x_box, y_box), (x_box + boxw, y_box + boxh), (0, 255, 0), 1)
            for k in range(X.shape[0]):
                frame_N[i] = cv2.circle(frame_N[i], (int(X[k, j]), int(Y[k, j])), 1, (0, 255, 0),
                                            thickness=1)


        cv2.imshow("win", frame_N[i])
        cv2.waitKey(10)



if __name__ == "__main__":
    cap = cv2.VideoCapture("Easy.mp4")
    objectTracking(cap, Bounding_box=True)
    cap.release()





