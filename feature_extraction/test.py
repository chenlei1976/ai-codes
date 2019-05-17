# -*- coding: UTF-8 -*-
import time
import numpy as np
import cv2
import psutil
import pdfhelper
from dbhelper import DBHelper
import filetypecheck
import os
import tools
import logging
import psutil
import datetime
import datetime
import configparser

k_config_file = 'C:\\duplicatecheck\\duplicate_config.ini'
k_file = 'C:\\duplicatecheck\\duplicate.txt'

class ExtractionConfig(object):
    def __init__(self):
        files = [k_config_file]
        cfg = configparser.ConfigParser()
        dataset = cfg.read(files)
        if len(dataset) != len(files):
            raise ValueError("Failed to open/find config file")
        # cfg.sections()
        try:
            self.image_size = cfg.get('Extraction', 'ImageSize')
            self.image_base_path = cfg.get('Extraction', 'ImageBasePath')
            self.new_image_path_format = cfg.get('Extraction', 'NewImagePathFormat')
            self.feature_folder = cfg.get('Extraction', 'FeatureFolder')
            self.pdf_to_image_file = cfg.get('Extraction', 'PdfToImageFile')
            self.use_h5 = cfg.get('Extraction', 'UseH5')
            self.include_pdf = cfg.get('Extraction', 'IncludePdf')
            self.insert_batch_size = int(cfg.get('Extraction', 'InsertBatchSize'))
        except configparser.NoSectionError:
            raise ValueError("No section[Extraction] in ini files")


conf = ExtractionConfig()

if __name__ == "__main__":
    # for p in psutil.process_iter():
    #     if p.name().startswith('t'):
    #         print(p.name())

    # file = open(k_file1, 'w')
    # file.close()

    # try:
    #     file = open(k_file1, 'w')
    #     file.close()
    # except Exception:
    #     print('create file failed')
    #
    # if os.path.exists(k_file):
    #     print('exist')
    #     os.remove(k_file)
    print(datetime.date.today().strftime('%Y%m%d'))
    print(conf.pdf_to_image_file)
    print(datetime.datetime.now().strftime('%Y%m%d%H%M'))


