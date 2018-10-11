#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging
import os
from matplotlib import pyplot as plt
import tools
from PIL import Image
import psutil
import numpy as np
from nltk.corpus import brown
import enchant
import cv2 as cv
import time

def equalHist_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    # cv.imshow("equalHist_demo", dst)


def clahe_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    # cv.imshow("clahe_demo", dst)


# for loop version take 9s
# def create_rgb_hist(image):
#     h, w, c = image.shape
#     rgbHist = np.zeros([16*16*16, 1], np.float32)
#     bsize = 256 / 16
#     for row in range(h):
#         for col in range(w):
#             b = image[row, col, 0]
#             g = image[row, col, 1]
#             r = image[row, col, 2]
#             index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
#             rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1
#     return rgbHist


def create_rgb_hist(image):
    b, g, r = cv.split(image)
    bsize = 256 / 16
    # tmp = (b/bsize).astype(np.int)*16*16 + (g/bsize).astype(np.int)*16 + (r / bsize).astype(np.int)
    tmp = (b / bsize).astype(int) * 16 * 16 + (g / bsize).astype(int) * 16 + (r / bsize).astype(int)
    tmp = tmp.ravel()
    rgbHist = np.zeros([16 * 16 * 16, 1], np.float32)

    for index in tmp:
        rgbHist[index, 0] += 1
    return rgbHist


def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    np.save("/home/chenlei/images/feature_detection/hist1.npy", hist1)
    np.save("/home/chenlei/images/feature_detection/hist2.npy", hist2)
    # match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    # match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    # logging.critical("Bha distance: %s, Correlation: %s, Chi-Square: %s" % (match1, match2, match3))
    logging.critical("Correlation: %s" % (match2))


k_min_image = 200


def get_ratio(w, h):
    v = float(min(h, w))
    if v < k_min_image:
        return 1
    return v / k_min_image


def get_resize_image(image):
    h, w, c = image.shape
    ratio = get_ratio(w, h)
    return cv.resize(image, (int(w / ratio), int(h / ratio)), interpolation=cv.INTER_CUBIC)


if __name__ == '__main__':
    tools.init_log('./hist_detection.log')
    print('cv version: {}'.format(cv.__version__))

    # load image
    image1 = cv.imread('/home/chenlei/images/feature_detection/7473-luggage.jpg')
    image2 = cv.imread('/home/chenlei/images/feature_detection/7553-luggage.jpeg')

    # image1 = get_resize_image(image1)
    # image2 = get_resize_image(image2)

    print(image1.size)
    print(image2.size)

    start_time = time.time()
    hist_compare(image1, image2)
    logging.critical('training took %fs!' % (time.time() - start_time))
    # cv2.namedWindow("Image")
    # cv2.imshow("Image", img)
    #
    # cv2.waitKey(0)

    # print(os.path.join('/home', "me", "mywork"))
    # plt.imshow(img)
    # plt.show()
