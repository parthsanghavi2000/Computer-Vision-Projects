from skimage import transform as tf
from skimage.feature import corner_harris, peak_local_max

import numpy as np
from numpy.linalg import inv
import cv2
from helpers import interp2

win = 25


def getFeatures(img, bbox):
    num = np.shape(bbox)[0]
    number = 0
    storage = np.empty((num,), dtype=np.ndarray)
    for n in range(num):
        (X, Y, width, height) = cv2.boundingRect(
            bbox[n, :, :].astype(int))  # creating a rectangle around the object to be tracked
        roi = img[Y:Y + height, X:X + width]  # selecting region of Interest
        corner_response = corner_harris(roi)  # detecting features
        max_feat = peak_local_max(corner_response, num_peaks=25, exclude_border=2)  # out of all features selecting few
        max_feat[:, 1] = max_feat[:, 1] + X  # shifting all the features inside bounding box
        max_feat[:, 0] = max_feat[:, 0] + Y
        storage[n] = max_feat
        if max_feat.shape[0] > number:
            number = max_feat.shape[0]
    features_x = np.full((number, num), -1)  # assigning -1 to the coordinates which are not featues
    features_y = np.full((number, num), -1)
    for n in range(num):
        feat = storage[n].shape[0]
        features_x[0:feat, n] = storage[n][:, 1]
        features_y[0:feat, n] = storage[n][:, 0]


    return features_x, features_y







def estimateFeatureTranslation(feature_x, feature_y, Ix, Iy, img1, img2):
    features_X = feature_x
    features_Y = feature_y

    mx, my = np.meshgrid(np.arange(win), np.arange(win))

    im1_G = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    im2_G = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    flat_x = mx.flatten() + features_X - np.floor(win / 2)
    flat_y = my.flatten() + features_Y - np.floor(win / 2)

    fea_stack = np.vstack((flat_x, flat_y))
    i1_val = interp2(im1_G, fea_stack[[0], :], fea_stack[[1], :])

    Ix_ = interp2(Ix, fea_stack[[0], :], fea_stack[[1], :])
    Iy_ = interp2(Iy, fea_stack[[0], :], fea_stack[[1], :])
    I = np.vstack((Ix_, Iy_))
    M = I.dot(I.T)

    for _ in range(15):
        flat_x1 = mx.flatten() + features_X - np.floor(win / 2)
        flat_y1 = my.flatten() + features_Y - np.floor(win / 2)
        fea = np.vstack((flat_x1, flat_y1))

        i2_val = interp2(im2_G, fea[[0], :], fea[[1], :])
        Ip = (i2_val - i1_val).reshape((-1, 1))

        b = -I.dot(Ip)
        s = inv(M).dot(b)

        features_X += s[0, 0]
        features_Y += s[1, 0]

    return features_X, features_Y

#
def estimateAllTranslation(XS, YS, img1, img2):
    I = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    I = cv2.GaussianBlur(I, (5, 5), 0.2)
    Iy, Ix = np.gradient(I.astype(float))

    Xs_flat = XS.flatten()
    Ys_flat = YS.flatten()
    Nfeature_X = np.full(Xs_flat.shape, -1, dtype=float)
    Nfeature_Y = np.full(Ys_flat.shape, -1, dtype=float)
    for i in range(np.size(XS)):
        if Xs_flat[i] != -1:
            Nfeature_X[i], Nfeature_Y[i] = estimateFeatureTranslation(Xs_flat[i], Ys_flat[i], Ix, Iy, img1, img2)


    Nfeature_X = np.reshape(Nfeature_X, XS.shape)
    Nfeature_Y = np.reshape(Nfeature_Y, YS.shape)
    return Nfeature_X, Nfeature_Y



def applyGeometricTransformation(fea_X, fea_Y, new_fea_X, new_fea_y, bbox):
    num = bbox.shape[0]
    nbbox = np.zeros_like(bbox)
    X_new = new_fea_X.copy()
    Y_new = new_fea_y.copy()
    for n in range(num):
        x_obj = fea_X[:, [n]]
        y_obj = fea_Y[:, [n]]

        nX_obj = new_fea_X[:, [n]]
        nY_obj = new_fea_y[:, [n]]

        D = np.hstack((x_obj, y_obj))
        A = np.hstack((nX_obj, nY_obj))
        t = tf.SimilarityTransform()
        t.estimate(dst=A, src=D)
        Z = t.params

        desired_thres = 1

        projected = Z.dot(np.vstack((D.T.astype(float), np.ones([1, np.shape(D)[0]]))))
        dst = np.square(projected[0:2, :].T - A).sum(axis=1)

        A_I = A[dst < desired_thres]
        D_I = D[dst < desired_thres]

        if np.shape(D_I)[0] < 4:

            A_I = A
            D_I = D

        t.estimate(dst=A_I, src=D_I)
        Z = t.params
        cC = np.vstack((bbox[n, :, :].T, np.array([1, 1, 1, 1])))

        New_C = Z.dot(cC)
        nbbox[n, :, :] = New_C[0:2, :].T

        X_new[dst >= desired_thres, n] = -1  # Coordinates of all feature points in first frame after eliminating outliers, (F, N1, 2)
        Y_new[dst >= desired_thres, n] = -1

    return X_new, Y_new, nbbox

