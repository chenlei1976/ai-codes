#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import logging
import os
import re
import shutil
import tools
from PIL import Image
import psutil
import pytesseract
import cv2


def ocr(fileName):
    ref = cv2.imread(fileName)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    code = pytesseract.image_to_string(ref)  # return unicode
    print(code)


if __name__ == '__main__':
    files = tools.get_files('C:\\Users\\Public\\ic-web')
    for fileName in files:
        print('===========',fileName)
        ocr(fileName)
    # print(files)
