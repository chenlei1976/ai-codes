#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging
import os
import sys
import time

from matplotlib import pyplot as plt
import tools
from PIL import Image
import psutil
import pytesseract
from nltk.corpus import brown
import enchant
import cv2
import pyodbc
import numpy as np
import pymssql
import h5py
import pdf2image
import pdfhelper

K_MIN_IMAGE = 400
K_SIFT_FOLDER = 'c:\\siftFeature'
K_SURF_FOLDER = 'c:\\surfFeature'
K_ORB_FOLDER = 'c:\\surfFeature'
K_USE_H5 = True


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # (b, g, r) = cv2.split(img)
    # return cv2.merge([r, g, b])


def orbDetect(image1, image2):
    # feature match
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(image1, None)
    print('image1 keypoints[{}], descriptors[{}]'.format(len(kp1), len(des1)))
    kp2, des2 = orb.detectAndCompute(image2, None)
    print('image2 keypoints[{}], descriptors[{}]'.format(len(kp2), len(des2)))

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)
    print('matches[{}]'.format(len(matches)))

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:100], None, flags=2)

    return bgr2rgb(img3)


def siftDetect(img1, img2, detector='surf'):
    if detector.startswith('si'):
        print("sift detector......")
        sift = cv2.xfeatures2d.SIFT_create()
    else:
        print("surf detector......")
        sift = cv2.xfeatures2d.SURF_create()

    print("detectAndCompute......")
    # find the keypoints and descriptors with SIFT
    start_time = time.time()
    kp1, des1 = sift.detectAndCompute(img1, None)
    logging.critical(
        'img[%s], %s took %fs, descriptors[%d]!' % (str(img1.shape), detector, time.time() - start_time, len(des1)))
    start_time = time.time()
    kp2, des2 = sift.detectAndCompute(img2, None)
    logging.critical(
        'img[%s], %s took %fs, descriptors[%d]!' % (str(img2.shape), detector, time.time() - start_time, len(des2)))

    # draw keypoints
    tmp1 = img1.copy()
    cv2.drawKeypoints(image=img1, outImage=tmp1, keypoints=kp1, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                      color=(255, 0, 0))
    cv2.imwrite("/home/chenlei/images/feature_detection/kp1.jpg", tmp1)
    np.save("/home/chenlei/images/feature_detection/des1.npy", des1)

    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
    # cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

    tmp2 = img2.copy()
    cv2.drawKeypoints(image=img2, outImage=tmp2, keypoints=kp2, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                      color=(255, 0, 0))
    cv2.imwrite("/home/chenlei/images/feature_detection/kp2.jpg", tmp2)
    np.save("/home/chenlei/images/feature_detection/des2.npy", des2)

    # BFMatcher with default params

    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # bf = cv2.FlannBasedMatcher(index_params, search_params)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = [[m] for m, n in matches if m.distance < 0.4 * n.distance]

    logging.critical('similar[{}]'.format(len(good)))

    # cv2.drawMatchesKnn expects list of lists as matches.
    # good = []
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    return img3
    # return bgr2rgb(img3)


def createDescriptors(img, detector='surf'):

    if detector.startswith('si'):
        extractor = cv2.xfeatures2d.SIFT_create()
    else:
        extractor = cv2.xfeatures2d.SURF_create()
    _, des = extractor.detectAndCompute(img, None)

    return des


def createFeature(fileName, detector='surf'):

    if str(fileName).endswith('pdf') or str(fileName).endswith('PDF'):
        img = pdfhelper.pdfToImage(fileName)
    else:
        img = cv2.imread(fileName)
        if img is None:
            logging.critical('invalid image[%s]!' % (fileName))
            raise ValueError('invalid image[%s]!' % (fileName))

    img = resizeImage(img)

    des = createDescriptors(img)

    if detector.startswith('si'):
        path = K_SIFT_FOLDER
    else:
        path = K_SURF_FOLDER

    if K_USE_H5:
        featureName = os.path.splitext(os.path.basename(fileName))[0] + '.h5'
        featureName = os.path.join(path, featureName)

        f = h5py.File(featureName, 'w')
        f.create_dataset('features', data=des)
        f.close()

        # Load hdf5 dataset
        # f = h5py.File(featureName, 'r')
        # des = f['features']
        # f.close()
    else:
        # use npy file
        featureName = os.path.splitext(os.path.basename(fileName))[0]+'.npy'
        featureName = os.path.join(path, featureName)
        np.save(featureName, des)


def _getRatio(w, h):
    v = float(min(h, w))
    if v < K_MIN_IMAGE:
        return 1
    return v / K_MIN_IMAGE


# @tools.timeit
def resizeImage(image):
    h, w, c = image.shape
    ratio = _getRatio(w, h)
    return cv2.resize(image, (int(w / ratio), int(h / ratio)), interpolation=cv2.INTER_CUBIC)


if __name__ == '__main__':
    tools.init_log('./feature_detection.log')

    print('cv version: {}'.format(cv2.__version__))
    print('sys version: {}'.format(sys.version))

    # load image
    image1 = cv2.imread('/home/chenlei/images/feature_detection/4114-luggage.jpg')
    image2 = cv2.imread('/home/chenlei/images/feature_detection/5395-luggage.jpg')

    # image1 = cv2.imread('/home/chenlei/images/feature_detection/7473-luggage.jpg')
    # image2 = cv2.imread('/home/chenlei/images/feature_detection/7553-luggage.jpeg')

    # image1 = cv2.imread('/home/chenlei/images/feature_detection/6812-luggage 1.jpg')
    # image2 = cv2.imread('/home/chenlei/images/feature_detection/6812-luggage 2.jpg')

    # image1 = cv2.imread('/home/chenlei/images/feature_detection/hospital1.jpg')
    # image2 = cv2.imread('/home/chenlei/images/feature_detection/hospital3.jpg')

    # image1 = cv2.imread('/home/chenlei/images/feature_detection/clinic1.png')
    # image2 = cv2.imread('/home/chenlei/images/feature_detection/clinic2.png')

    # image1 = cv2.imread('/home/chenlei/images/feature_detection/6817-luggage 1.JPG')
    # image2 = cv2.imread('/home/chenlei/images/feature_detection/6817-luggage 2.JPG')

    image1 = resizeImage(image1)
    image2 = resizeImage(image2)

    # ORB
    # img = orbDetect(image1, image2)

    # SIFT or SURF
    img = siftDetect(image1, image2)

    # cv2.imwrite("/home/chenlei/images/feature_detection/luggage.jpg", img)
    cv2.imwrite("/home/chenlei/images/feature_detection/test1.jpg", img)

    # cv2.namedWindow("Image")
    # cv2.imshow("Image", img)
    #
    # cv2.waitKey(0)

    # plt.imshow(img)
    # plt.show()

    # if image1 is None:
    #     print("img1 is none")
    # else:
    #     print("img1 is not none")
    #
    #
    # pdf1 = cv2.imread('/home/chenlei/images/p1.pdf')
    # if pdf1 is None:
    #     print("pdf1 is none")
    # else:
    #     print("pdf1 is not none")

