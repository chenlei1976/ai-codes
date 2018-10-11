#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging
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


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # (b, g, r) = cv2.split(img)
    # return cv2.merge([r, g, b])


def orb_detect(image1, image2):
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


def sift_detect(img1, img2, detector='surf'):
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


k_min_image = 400


def get_ratio(w, h):
    v = float(min(h, w))
    if v < k_min_image:
        return 1
    return v / k_min_image


# @tools.timeit
def get_resize_image(image):
    h, w, c = image.shape
    ratio = get_ratio(w, h)
    return cv2.resize(image, (int(w / ratio), int(h / ratio)), interpolation=cv2.INTER_CUBIC)


if __name__ == '__main__':
    tools.init_log('./feature_detection.log')

    print('cv version: {}'.format(cv2.__version__))
    print('sys version: {}'.format(sys.version))

    # load image
    image1 = cv2.imread('/home/chenlei/images/feature_detection/4114-luggage.jpg')
    image2 = cv2.imread('/home/chenlei/images/feature_detection/5395-luggage.jpg')

    image1 = get_resize_image(image1)
    image2 = get_resize_image(image2)

    # ORB
    # img = orb_detect(image1, image2)

    # SIFT or SURF
    img = sift_detect(image1, image2)

    cv2.imwrite("/home/chenlei/images/feature_detection/test1.jpg", img)

    # cv2.namedWindow("Image")
    # cv2.imshow("Image", img)
    #
    # cv2.waitKey(0)

    # plt.imshow(img)
    # plt.show()
